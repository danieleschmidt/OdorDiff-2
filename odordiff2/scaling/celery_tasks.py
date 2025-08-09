"""
Celery Distributed Task Processing for OdorDiff-2 Scaling

This module provides distributed molecular generation using Celery workers
with Redis as the message broker and result backend.
"""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
import uuid

from celery import Celery, Task, group, chord, chain
from celery.result import AsyncResult, GroupResult
from celery.exceptions import Retry, WorkerLostError
from kombu import Queue
from kombu.serialization import register

from ..core.async_diffusion import AsyncOdorDiffusion, GenerationResult
from ..models.molecule import Molecule
from ..safety.filter import SafetyFilter
from ..utils.logging import get_logger
from .redis_config import get_redis_manager, get_redis_serializer

logger = get_logger(__name__)


# Celery configuration
def create_celery_app() -> Celery:
    """Create and configure Celery application."""
    # Get Redis configuration
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = os.getenv('REDIS_PORT', '6379')
    redis_db = os.getenv('REDIS_DB', '0')
    redis_password = os.getenv('REDIS_PASSWORD', '')
    
    # Construct Redis URL
    if redis_password:
        redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
    else:
        redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
    
    # Create Celery app
    app = Celery('odordiff2-tasks')
    
    # Configuration
    app.conf.update(
        # Broker and result backend
        broker_url=redis_url,
        result_backend=redis_url,
        
        # Task routing
        task_routes={
            'odordiff2.scaling.celery_tasks.generate_molecules': {
                'queue': 'generation',
                'routing_key': 'generation.molecules'
            },
            'odordiff2.scaling.celery_tasks.generate_batch': {
                'queue': 'batch',
                'routing_key': 'batch.generation'
            },
            'odordiff2.scaling.celery_tasks.optimize_fragrance': {
                'queue': 'optimization',
                'routing_key': 'optimization.fragrance'
            },
            'odordiff2.scaling.celery_tasks.preprocess_models': {
                'queue': 'preprocessing',
                'routing_key': 'preprocessing.models'
            }
        },
        
        # Task queues
        task_queues=(
            Queue('generation', routing_key='generation.molecules'),
            Queue('batch', routing_key='batch.generation'),
            Queue('optimization', routing_key='optimization.fragrance'),
            Queue('preprocessing', routing_key='preprocessing.models'),
            Queue('priority', routing_key='priority.*'),
        ),
        
        # Worker configuration
        worker_concurrency=4,  # Adjust based on available CPU cores
        worker_max_tasks_per_child=100,  # Prevent memory leaks
        worker_prefetch_multiplier=1,  # Disable prefetching for fair distribution
        task_acks_late=True,  # Acknowledge tasks only after completion
        
        # Task configuration
        task_serializer='msgpack',
        result_serializer='msgpack',
        accept_content=['msgpack', 'json'],
        result_expires=3600,  # Results expire after 1 hour
        task_soft_time_limit=300,  # 5 minutes soft limit
        task_time_limit=600,  # 10 minutes hard limit
        task_reject_on_worker_lost=True,
        
        # Performance optimizations
        broker_transport_options={
            'priority_steps': list(range(10)),
            'sep': ':',
            'queue_order_strategy': 'priority',
            'visibility_timeout': 3600,
            'fanout_prefix': True,
            'fanout_patterns': True
        },
        
        # Result backend optimization
        result_backend_transport_options={
            'master_name': 'mymaster',
            'retry_on_timeout': True,
            'socket_keepalive': True,
            'socket_keepalive_options': {
                'TCP_KEEPINTVL': 1,
                'TCP_KEEPCNT': 3,
                'TCP_KEEPIDLE': 1,
            },
        },
        
        # Monitoring
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Timezone
        timezone='UTC',
        enable_utc=True,
    )
    
    return app


# Create global Celery app
celery_app = create_celery_app()


class BaseOdorTask(Task):
    """Base task class with common functionality."""
    
    def __init__(self):
        self._model: Optional[AsyncOdorDiffusion] = None
        self._model_lock = asyncio.Lock()
    
    async def get_model(self) -> AsyncOdorDiffusion:
        """Get or create the async diffusion model."""
        if self._model is None:
            async with self._model_lock:
                if self._model is None:
                    # Initialize model with GPU if available
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    self._model = AsyncOdorDiffusion(
                        device=device,
                        max_workers=2,  # Limit per-task workers
                        batch_size=4,
                        enable_caching=True
                    )
                    await self._model.start()
                    logger.info(f"Initialized AsyncOdorDiffusion on {device}")
        
        return self._model
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(f"Task {task_id} retrying: {exc}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(f"Task {task_id} completed successfully")


@celery_app.task(bind=True, base=BaseOdorTask, max_retries=3, default_retry_delay=60)
def generate_molecules(
    self,
    prompt: str,
    num_molecules: int = 5,
    safety_threshold: float = 0.1,
    synthesizability_min: float = 0.0,
    use_cache: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Distributed molecular generation task.
    
    Args:
        prompt: Text description of desired scent
        num_molecules: Number of molecules to generate
        safety_threshold: Safety filtering threshold
        synthesizability_min: Minimum synthesizability score
        use_cache: Whether to use caching
        **kwargs: Additional generation parameters
    
    Returns:
        Dictionary containing generation results
    """
    try:
        # Set up async event loop for this worker
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _generate():
            model = await self.get_model()
            
            # Create safety filter
            safety_filter = SafetyFilter(
                toxicity_threshold=safety_threshold,
                irritant_check=True
            )
            
            # Generate molecules
            result = await model.generate_async(
                prompt=prompt,
                num_molecules=num_molecules,
                safety_filter=safety_filter,
                synthesizability_min=synthesizability_min,
                use_cache=use_cache,
                **kwargs
            )
            
            # Convert to serializable format
            molecules_data = []
            for mol in result.molecules:
                molecules_data.append({
                    'smiles': mol.smiles,
                    'confidence': mol.confidence,
                    'odor_profile': {
                        'primary_notes': mol.odor_profile.primary_notes,
                        'secondary_notes': mol.odor_profile.secondary_notes,
                        'intensity': mol.odor_profile.intensity,
                        'longevity_hours': mol.odor_profile.longevity_hours,
                        'sillage': mol.odor_profile.sillage,
                        'character': mol.odor_profile.character
                    },
                    'safety_score': mol.safety_score,
                    'synth_score': mol.synth_score,
                    'estimated_cost': mol.estimated_cost,
                    'properties': mol._properties
                })
            
            return {
                'task_id': self.request.id,
                'prompt': result.prompt,
                'molecules': molecules_data,
                'processing_time': result.processing_time,
                'cache_hit': result.cache_hit,
                'error': result.error,
                'worker_id': self.request.hostname,
                'timestamp': time.time()
            }
        
        # Run the async generation
        result = loop.run_until_complete(_generate())
        loop.close()
        
        return result
        
    except Exception as exc:
        logger.error(f"Error in generate_molecules task: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(bind=True, base=BaseOdorTask)
def generate_batch(
    self,
    prompts: List[str],
    num_molecules: int = 5,
    safety_threshold: float = 0.1,
    synthesizability_min: float = 0.0,
    priority: int = 0
) -> str:
    """
    Distribute batch generation across multiple workers.
    
    Args:
        prompts: List of text descriptions
        num_molecules: Number of molecules per prompt
        safety_threshold: Safety filtering threshold
        synthesizability_min: Minimum synthesizability score
        priority: Task priority (0-10)
    
    Returns:
        Group result ID for tracking batch progress
    """
    try:
        # Create individual generation tasks
        job_group = group(
            generate_molecules.s(
                prompt=prompt,
                num_molecules=num_molecules,
                safety_threshold=safety_threshold,
                synthesizability_min=synthesizability_min,
                use_cache=True
            ).set(priority=priority)
            for prompt in prompts
        )
        
        # Execute group
        result = job_group.apply_async()
        
        logger.info(f"Batch generation started: {len(prompts)} tasks, group_id: {result.id}")
        
        return {
            'batch_id': result.id,
            'task_count': len(prompts),
            'status': 'submitted',
            'timestamp': time.time()
        }
        
    except Exception as exc:
        logger.error(f"Error in generate_batch task: {exc}")
        raise


@celery_app.task(bind=True, base=BaseOdorTask, max_retries=2)
def optimize_fragrance(
    self,
    base_notes: str,
    heart_notes: str,
    top_notes: str,
    style: str,
    constraints: Optional[Dict[str, Any]] = None,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Distributed fragrance optimization task.
    
    Args:
        base_notes: Base notes description
        heart_notes: Heart notes description  
        top_notes: Top notes description
        style: Overall fragrance style
        constraints: Molecular constraints
        iterations: Number of optimization iterations
    
    Returns:
        Optimized fragrance formulation
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _optimize():
            model = await self.get_model()
            
            formulation = await model.optimize_fragrance_async(
                base_notes=base_notes,
                heart_notes=heart_notes,
                top_notes=top_notes,
                style=style,
                constraints=constraints,
                iterations=iterations
            )
            
            # Convert to serializable format
            def molecules_to_dict(molecules):
                return [
                    {
                        'smiles': mol.smiles,
                        'confidence': mol.confidence,
                        'safety_score': mol.safety_score,
                        'synth_score': mol.synth_score,
                        'estimated_cost': mol.estimated_cost
                    }
                    for mol in molecules
                ]
            
            return {
                'task_id': self.request.id,
                'style': formulation.style_descriptor,
                'accords': {
                    'base': molecules_to_dict(formulation.base_accord),
                    'heart': molecules_to_dict(formulation.heart_accord),
                    'top': molecules_to_dict(formulation.top_accord)
                },
                'formula': formulation.to_perfume_formula(),
                'worker_id': self.request.hostname,
                'timestamp': time.time()
            }
        
        result = loop.run_until_complete(_optimize())
        loop.close()
        
        return result
        
    except Exception as exc:
        logger.error(f"Error in optimize_fragrance task: {exc}")
        raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))


@celery_app.task(bind=True)
def preprocess_models(self, model_path: str, optimization_type: str = "quantization") -> Dict[str, Any]:
    """
    Preprocess and optimize models for faster inference.
    
    Args:
        model_path: Path to model files
        optimization_type: Type of optimization (quantization, onnx, tensorrt)
    
    Returns:
        Preprocessing results
    """
    try:
        start_time = time.time()
        
        if optimization_type == "quantization":
            # Implement model quantization
            result = _quantize_model(model_path)
        elif optimization_type == "onnx":
            # Convert to ONNX format
            result = _convert_to_onnx(model_path)
        elif optimization_type == "tensorrt":
            # Optimize with TensorRT
            result = _optimize_with_tensorrt(model_path)
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
        
        processing_time = time.time() - start_time
        
        return {
            'task_id': self.request.id,
            'optimization_type': optimization_type,
            'processing_time': processing_time,
            'result': result,
            'worker_id': self.request.hostname,
            'timestamp': time.time()
        }
        
    except Exception as exc:
        logger.error(f"Error in preprocess_models task: {exc}")
        raise


def _quantize_model(model_path: str) -> Dict[str, Any]:
    """Quantize PyTorch model for faster inference."""
    import torch
    from torch.quantization import quantize_dynamic
    
    try:
        # Load model
        model = torch.load(model_path, map_location='cpu')
        
        # Apply dynamic quantization
        quantized_model = quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        # Save quantized model
        quantized_path = model_path.replace('.pth', '_quantized.pth')
        torch.save(quantized_model, quantized_path)
        
        return {
            'status': 'success',
            'quantized_path': quantized_path,
            'compression_ratio': 'approximately 4x'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def _convert_to_onnx(model_path: str) -> Dict[str, Any]:
    """Convert PyTorch model to ONNX format."""
    import torch
    import torch.onnx
    
    try:
        # Load model
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)  # Adjust based on model input
        
        # Convert to ONNX
        onnx_path = model_path.replace('.pth', '.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        return {
            'status': 'success',
            'onnx_path': onnx_path,
            'format': 'ONNX v11'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def _optimize_with_tensorrt(model_path: str) -> Dict[str, Any]:
    """Optimize model with NVIDIA TensorRT."""
    try:
        # This would require TensorRT installation and GPU
        # Implementation would depend on model architecture
        
        return {
            'status': 'not_implemented',
            'message': 'TensorRT optimization requires GPU and TensorRT installation'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


class CeleryTaskManager:
    """High-level interface for managing distributed tasks."""
    
    def __init__(self):
        self.app = celery_app
    
    async def submit_generation(
        self,
        prompt: str,
        num_molecules: int = 5,
        priority: int = 0,
        **kwargs
    ) -> str:
        """Submit single molecule generation task."""
        task = generate_molecules.apply_async(
            args=[prompt, num_molecules],
            kwargs=kwargs,
            priority=priority
        )
        
        logger.info(f"Submitted generation task: {task.id}")
        return task.id
    
    async def submit_batch(
        self,
        prompts: List[str],
        num_molecules: int = 5,
        priority: int = 0,
        **kwargs
    ) -> str:
        """Submit batch generation task."""
        task = generate_batch.apply_async(
            args=[prompts, num_molecules],
            kwargs=kwargs,
            priority=priority
        )
        
        logger.info(f"Submitted batch task: {task.id}")
        return task.id
    
    async def submit_fragrance_optimization(
        self,
        base_notes: str,
        heart_notes: str,
        top_notes: str,
        style: str,
        priority: int = 5,
        **kwargs
    ) -> str:
        """Submit fragrance optimization task."""
        task = optimize_fragrance.apply_async(
            args=[base_notes, heart_notes, top_notes, style],
            kwargs=kwargs,
            priority=priority
        )
        
        logger.info(f"Submitted fragrance optimization task: {task.id}")
        return task.id
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result by ID."""
        try:
            result = AsyncResult(task_id, app=self.app)
            
            if result.ready():
                if result.successful():
                    return {
                        'status': 'completed',
                        'result': result.result,
                        'task_id': task_id
                    }
                else:
                    return {
                        'status': 'failed',
                        'error': str(result.result),
                        'task_id': task_id
                    }
            else:
                return {
                    'status': 'pending',
                    'task_id': task_id,
                    'state': result.state
                }
                
        except Exception as e:
            logger.error(f"Error getting task result {task_id}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'task_id': task_id
            }
    
    def get_batch_results(self, batch_id: str) -> Dict[str, Any]:
        """Get batch task results."""
        try:
            result = GroupResult.restore(batch_id, app=self.app)
            
            if result is None:
                return {'status': 'not_found', 'batch_id': batch_id}
            
            completed = result.completed_count()
            total = len(result.results)
            
            if result.ready():
                successful = result.successful()
                results = []
                
                for task_result in result.results:
                    if task_result.successful():
                        results.append(task_result.result)
                    else:
                        results.append({
                            'error': str(task_result.result),
                            'task_id': task_result.id
                        })
                
                return {
                    'status': 'completed',
                    'batch_id': batch_id,
                    'successful': successful,
                    'completed': completed,
                    'total': total,
                    'results': results
                }
            else:
                return {
                    'status': 'processing',
                    'batch_id': batch_id,
                    'completed': completed,
                    'total': total,
                    'progress_percentage': (completed / total) * 100 if total > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting batch results {batch_id}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'batch_id': batch_id
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            self.app.control.revoke(task_id, terminate=True)
            logger.info(f"Cancelled task: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics about active workers."""
        try:
            inspect = self.app.control.inspect()
            
            stats = inspect.stats()
            active = inspect.active()
            reserved = inspect.reserved()
            
            return {
                'workers': list(stats.keys()) if stats else [],
                'worker_count': len(stats) if stats else 0,
                'stats': stats,
                'active_tasks': active,
                'reserved_tasks': reserved,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting worker stats: {e}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }


# Global task manager instance
task_manager = CeleryTaskManager()