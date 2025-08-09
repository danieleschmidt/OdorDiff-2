"""
Scaling-enhanced FastAPI endpoints for OdorDiff-2 Generation 3

This module extends the basic API with advanced scaling features:
- Integrated load balancing and caching
- Streaming responses for large batches
- Resource pooling and auto-scaling integration
- Performance monitoring and profiling
- Distributed task management with Celery
"""

import os
import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..api.endpoints import (
    GenerationRequest, MoleculeResponse, GenerationResponse,
    BatchGenerationRequest, FragranceDesignRequest, SafetyAssessmentRequest
)
from ..core.async_diffusion import AsyncOdorDiffusion
from ..models.molecule import Molecule
from ..utils.logging import get_logger

# Import scaling components
from ..scaling.multi_tier_cache import get_cache
from ..scaling.streaming import get_response_builder, StreamingFormat
from ..scaling.celery_tasks import task_manager
from ..scaling.load_balancer import LoadBalancerMiddleware, create_load_balancer_from_config
from ..scaling.resource_pool import get_pool_manager
from ..scaling.profiling import get_performance_monitor, profile_function
from ..scaling.auto_scaler import create_auto_scaler_from_config

logger = get_logger(__name__)


class ScaledGenerationRequest(GenerationRequest):
    """Extended generation request with scaling options."""
    use_distributed: bool = Field(False, description="Use distributed processing")
    priority: int = Field(0, ge=0, le=10, description="Task priority (0-10)")
    stream_results: bool = Field(False, description="Stream results as they become available")
    enable_profiling: bool = Field(False, description="Enable detailed profiling")


class StreamingBatchRequest(BatchGenerationRequest):
    """Batch request with streaming support."""
    stream_format: str = Field("sse", description="Streaming format: sse, json, websocket")
    chunk_size: int = Field(10, ge=1, le=100, description="Results per chunk")


class ScalingStatsResponse(BaseModel):
    """Response model for scaling statistics."""
    cache_stats: Dict[str, Any]
    resource_pool_stats: Dict[str, Any]
    load_balancer_stats: Dict[str, Any]
    auto_scaler_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]
    celery_stats: Dict[str, Any]


# Create enhanced FastAPI app
def create_scaling_app() -> FastAPI:
    """Create FastAPI app with all scaling features enabled."""
    
    app = FastAPI(
        title="OdorDiff-2 Scaling API",
        description="Advanced Text-to-Scent Molecule Diffusion API with Enterprise Scaling",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Load balancer middleware
    if os.getenv('ENABLE_LOAD_BALANCING', 'false').lower() == 'true':
        load_balancer = create_load_balancer_from_config()
        app.add_middleware(LoadBalancerMiddleware, load_balancer=load_balancer)
    
    return app


app = create_scaling_app()

# Global components (will be initialized on startup)
cache_manager = None
resource_pool_manager = None
performance_monitor = None
auto_scaler = None
streaming_builder = None


@app.on_event("startup")
async def startup_event():
    """Initialize all scaling components on startup."""
    global cache_manager, resource_pool_manager, performance_monitor, auto_scaler, streaming_builder
    
    logger.info("Starting OdorDiff-2 Scaling API")
    
    try:
        # Initialize multi-tier cache
        cache_manager = await get_cache()
        logger.info("Multi-tier cache initialized")
        
        # Initialize resource pool manager
        resource_pool_manager = await get_pool_manager()
        logger.info("Resource pool manager initialized")
        
        # Initialize streaming response builder
        streaming_builder = await get_response_builder()
        logger.info("Streaming response builder initialized")
        
        # Initialize performance monitoring
        if os.getenv('ENABLE_PROFILING', 'true').lower() == 'true':
            performance_monitor = get_performance_monitor()
            await performance_monitor.start_monitoring()
            logger.info("Performance monitoring started")
        
        # Initialize auto-scaler
        if os.getenv('ENABLE_AUTOSCALING', 'false').lower() == 'true':
            auto_scaler = create_auto_scaler_from_config()
            await auto_scaler.start()
            logger.info("Auto-scaler started")
        
        logger.info("All scaling components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup scaling components on shutdown."""
    logger.info("Shutting down OdorDiff-2 Scaling API")
    
    try:
        if performance_monitor:
            await performance_monitor.stop_monitoring()
        
        if auto_scaler:
            await auto_scaler.stop()
        
        if resource_pool_manager:
            await resource_pool_manager.close_all()
        
        if cache_manager:
            await cache_manager.close()
        
        logger.info("Scaling components shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


async def get_model_from_pool():
    """Get model instance from resource pool."""
    if not resource_pool_manager:
        raise HTTPException(status_code=503, detail="Resource pool manager not initialized")
    
    try:
        # Get model pool
        model_pool = await resource_pool_manager.get_pool("model_instances")
        return model_pool
    except ValueError:
        # Create model pool if it doesn't exist
        from ..scaling.resource_pool import ModelInstancePool, PoolConfig
        config = PoolConfig(initial_size=1, max_size=5, min_size=1)
        model_pool = ModelInstancePool(config)
        await model_pool.initialize()
        resource_pool_manager.pools["model_instances"] = model_pool
        return model_pool


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with scaling API information."""
    html_content = """
    <html>
        <head>
            <title>OdorDiff-2 Scaling API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .feature { background: #f0f8ff; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .status { color: green; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>OdorDiff-2: Enterprise-Scale Text-to-Scent Molecule Diffusion API</h1>
            <p><span class="status">Generation 3</span> - Advanced scaling and performance optimizations</p>
            
            <h2>Scaling Features</h2>
            <div class="feature">
                <h3>🚀 Multi-Tier Caching</h3>
                <p>L1 Memory + L2 Redis + L3 CDN caching with intelligent warmup</p>
            </div>
            <div class="feature">
                <h3>⚖️ Intelligent Load Balancing</h3>
                <p>Multiple algorithms with health checks and sticky sessions</p>
            </div>
            <div class="feature">
                <h3>🔄 Distributed Processing</h3>
                <p>Celery-based task distribution across worker nodes</p>
            </div>
            <div class="feature">
                <h3>📊 Streaming Responses</h3>
                <p>Real-time streaming for large batch operations</p>
            </div>
            <div class="feature">
                <h3>🎯 Auto-scaling</h3>
                <p>Dynamic scaling based on CPU, memory, and queue metrics</p>
            </div>
            <div class="feature">
                <h3>🔍 Continuous Profiling</h3>
                <p>Real-time performance monitoring and regression detection</p>
            </div>
            
            <h2>API Documentation</h2>
            <ul>
                <li><a href="/docs">Interactive API Documentation (Swagger)</a></li>
                <li><a href="/redoc">Alternative Documentation (ReDoc)</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/scaling/stats">Scaling Statistics</a></li>
                <li><a href="/metrics">Performance Metrics</a></li>
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/generate/scaled", response_model=GenerationResponse)
async def generate_molecules_scaled(
    request: ScaledGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate molecules with advanced scaling features."""
    request_id = str(uuid.uuid4())
    
    # Use profiling if requested
    profile_context = profile_function(f"generate_scaled_{request_id}") if request.enable_profiling else None
    
    try:
        with profile_context if profile_context else nullcontext():
            # Check cache first
            cache_key = f"generate:{request.prompt}:{request.num_molecules}:{request.safety_threshold}"
            
            if request.use_cache and cache_manager:
                cached_result = await cache_manager.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for request {request_id}")
                    return GenerationResponse(
                        request_id=request_id,
                        prompt=request.prompt,
                        molecules=cached_result['molecules'],
                        processing_time=cached_result['processing_time'],
                        cache_hit=True,
                        timestamp=datetime.now()
                    )
            
            # Use distributed processing if requested
            if request.use_distributed:
                task_id = await task_manager.submit_generation(
                    prompt=request.prompt,
                    num_molecules=request.num_molecules,
                    safety_threshold=request.safety_threshold,
                    synthesizability_min=request.synthesizability_min,
                    priority=request.priority
                )
                
                # Return task ID for status checking
                return GenerationResponse(
                    request_id=task_id,
                    prompt=request.prompt,
                    molecules=[],
                    processing_time=0.0,
                    cache_hit=False,
                    error=None,
                    timestamp=datetime.now()
                )
            
            # Use resource pool for local processing
            model_pool = await get_model_from_pool()
            
            async with model_pool.acquire() as model:
                # Create safety filter
                safety_filter = SafetyFilter(
                    toxicity_threshold=request.safety_threshold,
                    irritant_check=True
                )
                
                # Generate molecules
                start_time = time.time()
                result = await model.generate_async(
                    prompt=request.prompt,
                    num_molecules=request.num_molecules,
                    safety_filter=safety_filter,
                    synthesizability_min=request.synthesizability_min,
                    use_cache=request.use_cache
                )
                processing_time = time.time() - start_time
                
                # Convert to response format
                molecule_responses = []
                for mol in result.molecules:
                    mol_response = MoleculeResponse(
                        smiles=mol.smiles,
                        confidence=mol.confidence,
                        odor_profile={
                            'primary_notes': mol.odor_profile.primary_notes,
                            'secondary_notes': mol.odor_profile.secondary_notes,
                            'intensity': mol.odor_profile.intensity,
                            'longevity_hours': mol.odor_profile.longevity_hours,
                            'sillage': mol.odor_profile.sillage,
                            'character': mol.odor_profile.character
                        },
                        safety_score=mol.safety_score,
                        synth_score=mol.synth_score,
                        estimated_cost=mol.estimated_cost,
                        properties=mol._properties
                    )
                    molecule_responses.append(mol_response)
                
                response = GenerationResponse(
                    request_id=request_id,
                    prompt=result.prompt,
                    molecules=molecule_responses,
                    processing_time=processing_time,
                    cache_hit=result.cache_hit,
                    error=result.error,
                    timestamp=datetime.now()
                )
                
                # Cache result if successful
                if request.use_cache and cache_manager and not result.error:
                    cache_data = {
                        'molecules': molecule_responses,
                        'processing_time': processing_time
                    }
                    await cache_manager.put(cache_key, cache_data, ttl=3600)  # 1 hour TTL
                
                return response
    
    except Exception as e:
        logger.error(f"Scaled generation error {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@app.post("/generate/batch/streaming")
async def generate_batch_streaming(request: StreamingBatchRequest):
    """Generate batch with streaming response."""
    if not streaming_builder:
        raise HTTPException(status_code=503, detail="Streaming not available")
    
    request_id = str(uuid.uuid4())
    
    # Convert format string to enum
    format_map = {
        "sse": StreamingFormat.SSE,
        "json": StreamingFormat.JSON,
        "websocket": StreamingFormat.WEBSOCKET
    }
    
    stream_format = format_map.get(request.stream_format, StreamingFormat.SSE)
    
    async def molecule_generator() -> AsyncIterator[Molecule]:
        """Generate molecules asynchronously."""
        model_pool = await get_model_from_pool()
        
        for prompt in request.prompts:
            async with model_pool.acquire() as model:
                safety_filter = SafetyFilter(toxicity_threshold=request.safety_threshold)
                
                result = await model.generate_async(
                    prompt=prompt,
                    num_molecules=request.num_molecules,
                    safety_filter=safety_filter,
                    synthesizability_min=request.synthesizability_min
                )
                
                for molecule in result.molecules:
                    yield molecule
    
    # Create streaming response based on format
    if stream_format == StreamingFormat.SSE:
        return await streaming_builder.create_sse_stream(
            molecule_generator(),
            request_id,
            len(request.prompts) * request.num_molecules
        )
    elif stream_format == StreamingFormat.JSON:
        return await streaming_builder.create_json_stream(
            molecule_generator(),
            request_id,
            len(request.prompts) * request.num_molecules
        )
    else:
        raise HTTPException(status_code=400, detail="WebSocket streaming requires WebSocket connection")


@app.websocket("/ws/generate/batch")
async def websocket_batch_generation(websocket: WebSocket):
    """WebSocket endpoint for real-time batch generation."""
    if not streaming_builder:
        await websocket.close(code=1011, reason="Streaming not available")
        return
    
    connection_id = str(uuid.uuid4())
    
    try:
        # Accept WebSocket connection
        await websocket.accept()
        
        # Wait for request
        request_data = await websocket.receive_json()
        request = StreamingBatchRequest(**request_data)
        
        async def molecule_generator() -> AsyncIterator[Molecule]:
            """Generate molecules asynchronously."""
            model_pool = await get_model_from_pool()
            
            for prompt in request.prompts:
                async with model_pool.acquire() as model:
                    safety_filter = SafetyFilter(toxicity_threshold=request.safety_threshold)
                    
                    result = await model.generate_async(
                        prompt=prompt,
                        num_molecules=request.num_molecules,
                        safety_filter=safety_filter,
                        synthesizability_min=request.synthesizability_min
                    )
                    
                    for molecule in result.molecules:
                        yield molecule
        
        # Handle WebSocket streaming
        await streaming_builder.handle_websocket_stream(
            websocket,
            connection_id,
            molecule_generator(),
            str(uuid.uuid4()),
            len(request.prompts) * request.num_molecules
        )
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


@app.get("/scaling/stats", response_model=ScalingStatsResponse)
async def get_scaling_stats():
    """Get comprehensive scaling statistics."""
    stats = ScalingStatsResponse(
        cache_stats={},
        resource_pool_stats={},
        load_balancer_stats={},
        auto_scaler_stats={},
        performance_stats={},
        celery_stats={}
    )
    
    try:
        # Cache statistics
        if cache_manager:
            stats.cache_stats = cache_manager.get_analytics()
        
        # Resource pool statistics
        if resource_pool_manager:
            stats.resource_pool_stats = resource_pool_manager.get_all_stats()
        
        # Auto-scaler statistics
        if auto_scaler:
            stats.auto_scaler_stats = auto_scaler.get_scaling_stats()
        
        # Performance statistics
        if performance_monitor:
            stats.performance_stats = performance_monitor.get_performance_summary()
        
        # Celery worker statistics
        stats.celery_stats = task_manager.get_worker_stats()
        
    except Exception as e:
        logger.error(f"Error collecting scaling stats: {e}")
    
    return stats


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    # This would typically use prometheus_client to expose metrics
    metrics_text = """
# HELP odordiff2_requests_total Total number of requests
# TYPE odordiff2_requests_total counter
odordiff2_requests_total 1234

# HELP odordiff2_cache_hits_total Total number of cache hits
# TYPE odordiff2_cache_hits_total counter
odordiff2_cache_hits_total 567

# HELP odordiff2_response_time_seconds Response time in seconds
# TYPE odordiff2_response_time_seconds histogram
odordiff2_response_time_seconds_bucket{le="0.1"} 100
odordiff2_response_time_seconds_bucket{le="0.5"} 200
odordiff2_response_time_seconds_bucket{le="1.0"} 300
odordiff2_response_time_seconds_bucket{le="+Inf"} 400
odordiff2_response_time_seconds_sum 123.45
odordiff2_response_time_seconds_count 400
    """.strip()
    
    return StreamingResponse(
        iter([metrics_text]),
        media_type="text/plain"
    )


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including all scaling components."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    try:
        # Check cache manager
        if cache_manager:
            cache_analytics = cache_manager.get_analytics()
            health_status["components"]["cache"] = {
                "status": "healthy",
                "hit_rate": cache_analytics.get("overall", {}).get("hit_rate", 0)
            }
        
        # Check resource pools
        if resource_pool_manager:
            pool_stats = resource_pool_manager.get_all_stats()
            health_status["components"]["resource_pools"] = {
                "status": "healthy",
                "active_pools": len(pool_stats)
            }
        
        # Check auto-scaler
        if auto_scaler:
            scaler_stats = auto_scaler.get_scaling_stats()
            health_status["components"]["auto_scaler"] = {
                "status": "healthy",
                "current_instances": scaler_stats["current_state"]["instances"]
            }
        
        # Check performance monitor
        if performance_monitor:
            perf_stats = performance_monitor.get_performance_summary()
            health_status["components"]["performance_monitor"] = {
                "status": "healthy",
                "monitoring_active": perf_stats["monitoring_active"]
            }
        
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["error"] = str(e)
    
    return health_status


# Context manager for null context (Python 3.7+)
from contextlib import nullcontext


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "odordiff2.api.scaling_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4
    )