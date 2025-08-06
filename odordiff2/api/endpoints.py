"""
FastAPI endpoints for OdorDiff-2 REST API.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import asyncio
import uuid
from datetime import datetime
import os

from ..core.async_diffusion import AsyncOdorDiffusion, BatchRequest, GenerationResult
from ..safety.filter import SafetyFilter
from ..models.molecule import Molecule
from ..utils.logging import get_logger
from ..utils.validation import InputValidator

logger = get_logger(__name__)

# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for molecule generation."""
    prompt: str = Field(..., description="Text description of desired scent")
    num_molecules: int = Field(5, ge=1, le=100, description="Number of molecules to generate")
    safety_threshold: float = Field(0.1, ge=0, le=1, description="Safety filtering threshold")
    synthesizability_min: float = Field(0.0, ge=0, le=1, description="Minimum synthesizability score")
    use_cache: bool = Field(True, description="Use cached results if available")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        return InputValidator.validate_prompt(v)


class MoleculeResponse(BaseModel):
    """Response model for molecule data."""
    smiles: str
    confidence: float
    odor_profile: Dict[str, Any]
    safety_score: float
    synth_score: float
    estimated_cost: float
    properties: Dict[str, float]


class GenerationResponse(BaseModel):
    """Response model for generation results."""
    request_id: str
    prompt: str
    molecules: List[MoleculeResponse]
    processing_time: float
    cache_hit: bool
    error: Optional[str] = None
    timestamp: datetime


class BatchGenerationRequest(BaseModel):
    """Request model for batch generation."""
    prompts: List[str] = Field(..., min_items=1, max_items=50)
    num_molecules: int = Field(5, ge=1, le=20)
    safety_threshold: float = Field(0.1, ge=0, le=1)
    synthesizability_min: float = Field(0.0, ge=0, le=1)
    priority: int = Field(0, ge=0, le=10)


class FragranceDesignRequest(BaseModel):
    """Request model for fragrance design."""
    base_notes: str = Field(..., description="Base notes description")
    heart_notes: str = Field(..., description="Heart/middle notes description")
    top_notes: str = Field(..., description="Top notes description")
    style: str = Field(..., description="Overall fragrance style")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Molecular constraints")
    iterations: int = Field(3, ge=1, le=10, description="Optimization iterations")


class SafetyAssessmentRequest(BaseModel):
    """Request model for safety assessment."""
    smiles: str = Field(..., description="SMILES string of molecule")
    
    @validator('smiles')
    def validate_smiles(cls, v):
        return InputValidator.validate_smiles(v)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    response_time: float
    memory_usage_mb: float
    worker_count: int
    cache_enabled: bool
    stats: Dict[str, Any]


# Global API instance
app = FastAPI(
    title="OdorDiff-2 API",
    description="Safe Text-to-Scent Molecule Diffusion API",
    version="1.0.0",
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

# Global state
async_model: Optional[AsyncOdorDiffusion] = None
safety_filter: SafetyFilter = SafetyFilter()

# Job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}


async def get_model() -> AsyncOdorDiffusion:
    """Dependency to get the async model."""
    global async_model
    if async_model is None:
        async_model = AsyncOdorDiffusion(
            device="cpu",  # Configure based on available hardware
            max_workers=4,
            batch_size=8,
            enable_caching=True
        )
        await async_model.start()
    return async_model


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting OdorDiff-2 API")
    
    # Initialize model
    global async_model
    async_model = AsyncOdorDiffusion(
        device="cpu",
        max_workers=4,
        batch_size=8,
        enable_caching=True
    )
    await async_model.start()
    
    # Preload common prompts
    common_prompts = [
        "fresh citrus scent",
        "warm vanilla fragrance",
        "floral rose bouquet",
        "woody cedar aroma",
        "clean aquatic breeze"
    ]
    await async_model.preload_cache(common_prompts)
    
    logger.info("OdorDiff-2 API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down OdorDiff-2 API")
    if async_model:
        await async_model.stop()


def molecule_to_response(molecule: Molecule) -> MoleculeResponse:
    """Convert Molecule to response model."""
    return MoleculeResponse(
        smiles=molecule.smiles,
        confidence=molecule.confidence,
        odor_profile={
            'primary_notes': molecule.odor_profile.primary_notes,
            'secondary_notes': molecule.odor_profile.secondary_notes,
            'intensity': molecule.odor_profile.intensity,
            'longevity_hours': molecule.odor_profile.longevity_hours,
            'sillage': molecule.odor_profile.sillage,
            'character': molecule.odor_profile.character
        },
        safety_score=molecule.safety_score,
        synth_score=molecule.synth_score,
        estimated_cost=molecule.estimated_cost,
        properties=molecule._properties
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    html_content = """
    <html>
        <head>
            <title>OdorDiff-2 API</title>
        </head>
        <body>
            <h1>OdorDiff-2: Safe Text-to-Scent Molecule Diffusion API</h1>
            <p>Welcome to the OdorDiff-2 API for generating novel scent molecules from text descriptions.</p>
            <ul>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/redoc">Alternative Documentation (ReDoc)</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/stats">Performance Statistics</a></li>
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check(model: AsyncOdorDiffusion = Depends(get_model)):
    """Health check endpoint."""
    try:
        health_data = await model.health_check()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/stats")
async def get_stats(model: AsyncOdorDiffusion = Depends(get_model)):
    """Get performance statistics."""
    try:
        stats = model.get_performance_stats()
        return {
            "stats": stats,
            "active_jobs": len(active_jobs),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/generate", response_model=GenerationResponse)
async def generate_molecules(
    request: GenerationRequest,
    model: AsyncOdorDiffusion = Depends(get_model)
):
    """Generate scent molecules from text description."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Generation request {request_id}: {request.prompt}")
        
        # Create safety filter
        safety = SafetyFilter(
            toxicity_threshold=request.safety_threshold,
            irritant_check=True
        )
        
        # Generate molecules
        result = await model.generate_async(
            prompt=request.prompt,
            num_molecules=request.num_molecules,
            safety_filter=safety,
            synthesizability_min=request.synthesizability_min,
            use_cache=request.use_cache
        )
        
        # Convert to response format
        molecule_responses = [molecule_to_response(mol) for mol in result.molecules]
        
        response = GenerationResponse(
            request_id=request_id,
            prompt=result.prompt,
            molecules=molecule_responses,
            processing_time=result.processing_time,
            cache_hit=result.cache_hit,
            error=result.error,
            timestamp=datetime.now()
        )
        
        logger.info(f"Generation completed {request_id}: {len(result.molecules)} molecules")
        return response
        
    except Exception as e:
        logger.error(f"Generation error {request_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Generation failed: {str(e)}"
        )


@app.post("/generate/batch")
async def generate_batch(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    model: AsyncOdorDiffusion = Depends(get_model)
):
    """Submit batch generation request."""
    job_id = str(uuid.uuid4())
    
    try:
        # Validate prompts
        validated_prompts = [InputValidator.validate_prompt(p) for p in request.prompts]
        
        # Create batch request
        batch_req = BatchRequest(
            prompts=validated_prompts,
            params={
                'num_molecules': request.num_molecules,
                'safety_filter': SafetyFilter(toxicity_threshold=request.safety_threshold),
                'synthesizability_min': request.synthesizability_min
            },
            priority=request.priority
        )
        
        # Initialize job tracking
        active_jobs[job_id] = {
            'status': 'submitted',
            'progress': 0,
            'total': len(validated_prompts),
            'results': [],
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        # Submit for processing
        success = await model.submit_batch(batch_req)
        
        if not success:
            raise HTTPException(status_code=503, detail="Batch queue is full")
        
        active_jobs[job_id]['status'] = 'processing'
        
        logger.info(f"Batch job submitted {job_id}: {len(validated_prompts)} prompts")
        
        return {
            "job_id": job_id,
            "status": "submitted",
            "prompts_count": len(validated_prompts),
            "estimated_completion": "5-10 minutes",
            "check_url": f"/jobs/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Batch submission error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch submission failed: {str(e)}")


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get batch job status."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = active_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job_data['status'],
        "progress": job_data['progress'],
        "total": job_data['total'],
        "completion_percentage": (job_data['progress'] / max(1, job_data['total'])) * 100,
        "results_available": len(job_data['results']),
        "created_at": job_data['created_at'],
        "updated_at": job_data['updated_at']
    }


@app.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get batch job results."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = active_jobs[job_id]
    
    if job_data['status'] != 'completed':
        raise HTTPException(status_code=202, detail="Job not yet completed")
    
    return {
        "job_id": job_id,
        "results": job_data['results'],
        "total_results": len(job_data['results']),
        "completed_at": job_data['updated_at']
    }


@app.post("/design/fragrance")
async def design_fragrance(
    request: FragranceDesignRequest,
    model: AsyncOdorDiffusion = Depends(get_model)
):
    """Design a complete fragrance formulation."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Fragrance design request {request_id}: {request.style}")
        
        # Validate constraints
        validated_constraints = None
        if request.constraints:
            validated_constraints = InputValidator.validate_molecular_constraints(request.constraints)
        
        # Design fragrance
        formulation = await model.optimize_fragrance_async(
            base_notes=request.base_notes,
            heart_notes=request.heart_notes,
            top_notes=request.top_notes,
            style=request.style,
            constraints=validated_constraints,
            iterations=request.iterations
        )
        
        # Convert to response format
        response = {
            "request_id": request_id,
            "style": formulation.style_descriptor,
            "accords": {
                "base": [molecule_to_response(mol) for mol in formulation.base_accord],
                "heart": [molecule_to_response(mol) for mol in formulation.heart_accord],
                "top": [molecule_to_response(mol) for mol in formulation.top_accord]
            },
            "formula": formulation.to_perfume_formula(),
            "timestamp": datetime.now()
        }
        
        logger.info(f"Fragrance design completed {request_id}")
        return response
        
    except Exception as e:
        logger.error(f"Fragrance design error {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Fragrance design failed: {str(e)}")


@app.post("/assess/safety")
async def assess_safety(request: SafetyAssessmentRequest):
    """Assess safety of a molecule."""
    try:
        from ..models.molecule import Molecule
        
        # Create molecule object
        molecule = Molecule(request.smiles)
        
        if not molecule.is_valid:
            raise HTTPException(status_code=400, detail="Invalid SMILES structure")
        
        # Perform safety assessment
        report = safety_filter.assess_molecule(molecule)
        
        return {
            "smiles": request.smiles,
            "assessment": {
                "toxicity_score": report.toxicity,
                "skin_sensitizer": report.skin_sensitizer,
                "eco_score": report.eco_score,
                "ifra_compliant": report.ifra_compliant,
                "regulatory_flags": report.regulatory_flags
            },
            "recommendation": "safe" if report.toxicity <= 0.1 and report.ifra_compliant else "caution",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Safety assessment error: {e}")
        raise HTTPException(status_code=500, detail=f"Safety assessment failed: {str(e)}")


@app.get("/molecules/{smiles}/visualize")
async def visualize_molecule(smiles: str):
    """Generate 3D visualization for a molecule."""
    try:
        # Validate SMILES
        validated_smiles = InputValidator.validate_smiles(smiles)
        
        from ..models.molecule import Molecule
        molecule = Molecule(validated_smiles)
        
        if not molecule.is_valid:
            raise HTTPException(status_code=400, detail="Invalid SMILES structure")
        
        # Generate visualization file
        viz_file = f"temp_{uuid.uuid4()}.html"
        molecule.visualize_3d(viz_file)
        
        if os.path.exists(viz_file):
            return FileResponse(
                viz_file,
                media_type="text/html",
                filename=f"molecule_{smiles}.html"
            )
        else:
            raise HTTPException(status_code=500, detail="Visualization generation failed")
            
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)