"""
Enhanced secure API endpoints with comprehensive error handling and security.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import asyncio
import uuid
from datetime import datetime
import os
import time

from ..core.async_diffusion import AsyncOdorDiffusion, BatchRequest, GenerationResult
from ..safety.filter import SafetyFilter
from ..models.molecule import Molecule
from ..utils.logging import get_logger
from ..utils.validation import InputValidator
from ..utils.security import (
    SecurityMiddleware, SecurityValidator, APIKeyManager, 
    SecurityError, get_security_middleware, get_api_key_manager
)
from ..utils.error_handling import (
    retry_with_backoff, timeout, safe_execute_async, 
    ErrorSeverity, ExponentialBackoffStrategy, 
    OdorDiffError, ValidationError, GenerationError
)

logger = get_logger(__name__)

# Security setup
security = HTTPBearer(auto_error=False)


class SecureGenerationRequest(BaseModel):
    """Secure request model for molecule generation."""
    prompt: str = Field(..., description="Text description of desired scent", max_length=1000)
    num_molecules: int = Field(5, ge=1, le=20, description="Number of molecules to generate")
    safety_threshold: float = Field(0.1, ge=0, le=1, description="Safety filtering threshold")
    synthesizability_min: float = Field(0.0, ge=0, le=1, description="Minimum synthesizability score")
    use_cache: bool = Field(True, description="Use cached results if available")
    
    @validator('prompt')
    def validate_prompt_security(cls, v):
        if not SecurityValidator.validate_input(v, max_length=1000):
            raise ValueError("Prompt contains potentially harmful content")
        return InputValidator.validate_prompt(v)


class SecureAPIResponse(BaseModel):
    """Base secure API response with metadata."""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SecurityMetrics(BaseModel):
    """Security metrics response."""
    blocked_ips: int
    rate_limited_requests: int
    api_key_validations: int
    suspicious_activities: int
    uptime_seconds: float


# Create secure FastAPI app
app = FastAPI(
    title="OdorDiff-2 Secure API",
    description="Safe Text-to-Scent Molecule Diffusion API with Enhanced Security",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.terragonlabs.ai"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.terragonlabs.ai", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global state
async_model: Optional[AsyncOdorDiffusion] = None
safety_filter: SafetyFilter = SafetyFilter()
api_start_time = time.time()

# Request tracking
active_requests = {}
request_history = []


async def get_client_info(request: Request) -> Dict[str, str]:
    """Extract client information from request."""
    return {
        'client_ip': request.client.host,
        'user_agent': request.headers.get('user-agent', ''),
        'referer': request.headers.get('referer', ''),
        'forwarded_for': request.headers.get('x-forwarded-for', '')
    }


async def validate_security(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Comprehensive security validation."""
    client_info = await get_client_info(request)
    
    # Security validation
    security_middleware = get_security_middleware()
    try:
        security_context = security_middleware.validate_request(client_info)
        
        # API key validation if provided
        if credentials:
            api_key_manager = get_api_key_manager()
            key_info = api_key_manager.validate_api_key(credentials.credentials)
            if not key_info:
                raise SecurityError("Invalid API key")
            security_context['user_id'] = key_info.get('user_id')
            security_context['api_key_valid'] = True
        else:
            security_context['api_key_valid'] = False
        
        return security_context
        
    except SecurityError as e:
        logger.warning(f"Security validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS if "rate limit" in str(e).lower() else status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


async def get_secure_model() -> AsyncOdorDiffusion:
    """Get async model with error handling."""
    global async_model
    if async_model is None:
        try:
            async_model = AsyncOdorDiffusion(
                device="cpu",
                max_workers=4,
                batch_size=8,
                enable_caching=True
            )
            await async_model.start()
            logger.info("Secure async model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service temporarily unavailable"
            )
    return async_model


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Global security middleware."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Track active requests
    active_requests[request_id] = {
        'path': request.url.path,
        'method': request.method,
        'start_time': start_time,
        'client_ip': request.client.host
    }
    
    try:
        # Basic security checks
        client_info = await get_client_info(request)
        
        # Check for suspicious patterns in URL
        if not SecurityValidator.validate_input(str(request.url), max_length=2000):
            logger.warning(f"Suspicious URL from {client_info['client_ip']}: {request.url}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Log successful request
        processing_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path}",
            request_id=request_id,
            processing_time=processing_time,
            status_code=response.status_code,
            **client_info
        )
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} - {str(e)}",
            request_id=request_id,
            processing_time=processing_time,
            **client_info
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id
            }
        )
        
    finally:
        # Clean up tracking
        active_requests.pop(request_id, None)
        
        # Keep limited history
        request_history.append({
            'request_id': request_id,
            'timestamp': start_time,
            'processing_time': time.time() - start_time,
            'path': request.url.path
        })
        
        # Limit history size
        if len(request_history) > 1000:
            request_history[:] = request_history[-500:]


@app.on_event("startup")
async def secure_startup_event():
    """Secure startup initialization."""
    logger.info("Starting OdorDiff-2 Secure API")
    
    try:
        # Initialize model
        await get_secure_model()
        
        # Initialize security components
        security_middleware = get_security_middleware()
        api_key_manager = get_api_key_manager()
        
        # Generate admin API key (in production, do this securely)
        admin_key = api_key_manager.generate_api_key(
            "admin", 
            permissions=["generate", "assess", "admin", "batch"]
        )
        logger.info(f"Admin API key: {admin_key}")
        
        logger.info("Secure API startup completed successfully")
        
    except Exception as e:
        logger.critical(f"Failed to start secure API: {e}")
        raise


@app.on_event("shutdown")
async def secure_shutdown_event():
    """Secure shutdown with cleanup."""
    logger.info("Shutting down OdorDiff-2 Secure API")
    
    try:
        if async_model:
            await async_model.stop()
        logger.info("Secure API shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@app.get("/", response_class=HTMLResponse)
async def secure_root():
    """Secure root endpoint with security information."""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>OdorDiff-2 Secure API</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .security-badge { color: #28a745; font-weight: bold; }
                .warning { color: #dc3545; background-color: #f8d7da; padding: 10px; border-radius: 4px; margin: 10px 0; }
                .info { color: #0c5460; background-color: #d1ecf1; padding: 10px; border-radius: 4px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🛡️ OdorDiff-2 Secure API</h1>
                <p class="security-badge">✅ Enhanced Security Mode Active</p>
                
                <h2>Security Features</h2>
                <ul>
                    <li>🔒 Rate limiting and IP blocking</li>
                    <li>🔑 API key authentication</li>
                    <li>🛡️ Input validation and sanitization</li>
                    <li>📊 Request monitoring and logging</li>
                    <li>⚡ Circuit breaker protection</li>
                    <li>🔄 Automatic retry mechanisms</li>
                </ul>
                
                <div class="warning">
                    <strong>⚠️ Notice:</strong> All API requests are monitored and logged for security purposes.
                </div>
                
                <div class="info">
                    <strong>📚 Documentation:</strong>
                    <ul>
                        <li><a href="/docs">OpenAPI Documentation</a></li>
                        <li><a href="/redoc">ReDoc Documentation</a></li>
                        <li><a href="/health">Health Check</a></li>
                        <li><a href="/security/metrics">Security Metrics</a></li>
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
@timeout(30.0)
@retry_with_backoff(max_attempts=2, exceptions=(Exception,))
async def secure_health_check(
    security_context: Dict[str, Any] = Depends(validate_security)
):
    """Enhanced health check with security context."""
    try:
        model = await get_secure_model()
        health_data = await model.health_check()
        
        # Add security information
        health_data['security'] = {
            'rate_limiting': True,
            'api_key_auth': security_context.get('api_key_valid', False),
            'client_ip': security_context.get('client_ip'),
            'remaining_requests': security_context.get('remaining_requests', 0)
        }
        
        return SecureAPIResponse(
            data=health_data,
            metadata={'uptime_seconds': time.time() - api_start_time}
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@app.post("/generate", response_model=SecureAPIResponse)
@timeout(120.0)
@retry_with_backoff(
    max_attempts=3, 
    strategy=ExponentialBackoffStrategy(max_delay=30.0),
    exceptions=(GenerationError, ConnectionError),
    severity=ErrorSeverity.MEDIUM
)
async def secure_generate_molecules(
    request: SecureGenerationRequest,
    security_context: Dict[str, Any] = Depends(validate_security)
):
    """Secure molecule generation endpoint."""
    try:
        logger.info(
            f"Secure generation request: {request.prompt[:50]}...",
            client_ip=security_context.get('client_ip'),
            num_molecules=request.num_molecules
        )
        
        model = await get_secure_model()
        
        # Create safety filter with enhanced settings
        safety = SafetyFilter(
            toxicity_threshold=min(request.safety_threshold, 0.2),  # Cap at 0.2 for security
            irritant_check=True,
            eco_threshold=0.3,
            ifra_compliance=True
        )
        
        # Generate molecules with timeout
        result = await asyncio.wait_for(
            model.generate_async(
                prompt=request.prompt,
                num_molecules=min(request.num_molecules, 10),  # Cap for security
                safety_filter=safety,
                synthesizability_min=request.synthesizability_min,
                use_cache=request.use_cache
            ),
            timeout=90.0  # 90 second timeout
        )
        
        if result.error:
            raise GenerationError(f"Generation failed: {result.error}")
        
        # Format response
        molecules_data = []
        for mol in result.molecules:
            if mol.is_valid:
                molecules_data.append({
                    'smiles': mol.smiles,
                    'confidence': mol.confidence,
                    'odor_profile': {
                        'primary_notes': mol.odor_profile.primary_notes,
                        'secondary_notes': mol.odor_profile.secondary_notes,
                        'intensity': mol.odor_profile.intensity,
                        'longevity_hours': mol.odor_profile.longevity_hours,
                        'character': mol.odor_profile.character
                    },
                    'safety_score': mol.safety_score,
                    'synth_score': mol.synth_score,
                    'estimated_cost': mol.estimated_cost
                })
        
        response_data = {
            'prompt': result.prompt,
            'molecules': molecules_data,
            'processing_time': result.processing_time,
            'cache_hit': result.cache_hit,
            'total_generated': len(molecules_data)
        }
        
        return SecureAPIResponse(
            data=response_data,
            metadata={
                'safety_filtered': len(result.molecules) - len(molecules_data),
                'generation_parameters': request.dict()
            }
        )
        
    except asyncio.TimeoutError:
        logger.warning("Generation timeout")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Generation request timed out"
        )
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generation service temporarily unavailable"
        )


@app.get("/security/metrics")
async def get_security_metrics(
    security_context: Dict[str, Any] = Depends(validate_security)
) -> SecurityMetrics:
    """Get security metrics (admin only)."""
    
    # Check admin access
    if not security_context.get('api_key_valid'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required"
        )
    
    security_middleware = get_security_middleware()
    
    metrics = SecurityMetrics(
        blocked_ips=len(security_middleware.blocked_ips),
        rate_limited_requests=sum(security_middleware.suspicious_activity.values()),
        api_key_validations=len(get_api_key_manager().keys),
        suspicious_activities=len([
            ip for ip, count in security_middleware.suspicious_activity.items() 
            if count > 10
        ]),
        uptime_seconds=time.time() - api_start_time
    )
    
    return metrics


@app.post("/security/block-ip")
async def block_ip_address(
    ip_address: str,
    security_context: Dict[str, Any] = Depends(validate_security)
):
    """Block IP address (admin only)."""
    
    if not security_context.get('api_key_valid'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required"
        )
    
    if not SecurityValidator.validate_ip_address(ip_address):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid IP address format"
        )
    
    security_middleware = get_security_middleware()
    security_middleware.block_ip(ip_address)
    
    return {"message": f"IP {ip_address} has been blocked"}


@app.get("/admin/status")
async def admin_status(
    security_context: Dict[str, Any] = Depends(validate_security)
):
    """Admin status endpoint."""
    
    if not security_context.get('api_key_valid'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin access required"
        )
    
    return {
        'active_requests': len(active_requests),
        'request_history_size': len(request_history),
        'uptime_seconds': time.time() - api_start_time,
        'model_loaded': async_model is not None,
        'recent_requests': request_history[-10:] if request_history else []
    }


# Error handlers
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


@app.exception_handler(SecurityError)
async def security_error_handler(request: Request, exc: SecurityError):
    """Handle security errors."""
    return JSONResponse(
        status_code=403,
        content={
            "error": "Security error",
            "detail": "Access denied",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    logger.error(f"Unhandled exception: {exc}", request_id=request_id)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )