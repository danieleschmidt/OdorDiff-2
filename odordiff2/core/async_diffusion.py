"""
Asynchronous and optimized diffusion processing.
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable, Awaitable
import torch
import numpy as np
from dataclasses import dataclass
import time

from .diffusion import OdorDiffusion, FragranceFormulation
from ..models.molecule import Molecule
from ..safety.filter import SafetyFilter
from ..data.cache import get_molecule_cache
from ..utils.logging import get_logger, log_function_call
from ..utils.validation import InputValidator

logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """Represents a batch processing request."""
    prompts: List[str]
    params: Dict[str, Any]
    callback: Optional[Callable] = None
    priority: int = 0
    

@dataclass
class GenerationResult:
    """Result from molecule generation."""
    prompt: str
    molecules: List[Molecule]
    processing_time: float
    cache_hit: bool
    error: Optional[str] = None


class AsyncOdorDiffusion:
    """
    Asynchronous and optimized version of OdorDiffusion with batch processing,
    caching, and concurrent execution capabilities.
    """
    
    def __init__(
        self, 
        device: str = "cpu",
        max_workers: int = 4,
        batch_size: int = 8,
        enable_caching: bool = True
    ):
        self.device = device
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        
        # Core model
        self.model = OdorDiffusion(device=device)
        
        # Caching
        self.cache = get_molecule_cache() if enable_caching else None
        
        # Async processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.batch_queue: asyncio.Queue = None
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.processing_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'batch_requests': 0,
            'avg_processing_time': 0.0
        }
        
        # Load balancing
        self.worker_load = [0] * max_workers
        
        logger.info(f"AsyncOdorDiffusion initialized with {max_workers} workers, batch_size={batch_size}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self):
        """Start async processing."""
        if self.batch_queue is None:
            self.batch_queue = asyncio.Queue(maxsize=100)
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            logger.info("Async processing started")
    
    async def stop(self):
        """Stop async processing."""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        if self.executor:
            self.executor.shutdown(wait=True)
            
        logger.info("Async processing stopped")
    
    async def generate_async(
        self,
        prompt: str,
        num_molecules: int = 5,
        safety_filter: Optional[SafetyFilter] = None,
        synthesizability_min: float = 0.0,
        use_cache: bool = True,
        **kwargs
    ) -> GenerationResult:
        """
        Asynchronously generate molecules for a single prompt.
        """
        start_time = time.time()
        
        try:
            # Validate input
            prompt = InputValidator.validate_prompt(prompt)
            params = InputValidator.validate_generation_parameters(
                num_molecules=num_molecules, **kwargs
            )
            
            # Check cache first
            cache_key_params = {
                'num_molecules': num_molecules,
                'synthesizability_min': synthesizability_min,
                'safety_threshold': getattr(safety_filter, 'toxicity_threshold', 0.1) if safety_filter else None
            }
            
            cached_result = None
            if use_cache and self.cache:
                cached_result = self.cache.get_generation_result(prompt, cache_key_params)
            
            if cached_result:
                processing_time = time.time() - start_time
                self.processing_stats['cache_hits'] += 1
                logger.info(f"Cache hit for prompt: {prompt[:50]}...")
                
                return GenerationResult(
                    prompt=prompt,
                    molecules=cached_result,
                    processing_time=processing_time,
                    cache_hit=True
                )
            
            # Generate molecules asynchronously
            molecules = await self._generate_molecules_async(
                prompt, num_molecules, safety_filter, synthesizability_min, **kwargs
            )
            
            # Cache result
            if use_cache and self.cache:
                self.cache.cache_generation_result(prompt, cache_key_params, molecules)
            
            processing_time = time.time() - start_time
            self.processing_stats['total_requests'] += 1
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] * (self.processing_stats['total_requests'] - 1) + 
                 processing_time) / self.processing_stats['total_requests']
            )
            
            return GenerationResult(
                prompt=prompt,
                molecules=molecules,
                processing_time=processing_time,
                cache_hit=False
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in async generation: {e}", prompt=prompt)
            
            return GenerationResult(
                prompt=prompt,
                molecules=[],
                processing_time=processing_time,
                cache_hit=False,
                error=str(e)
            )
    
    async def _generate_molecules_async(
        self,
        prompt: str,
        num_molecules: int,
        safety_filter: Optional[SafetyFilter],
        synthesizability_min: float,
        **kwargs
    ) -> List[Molecule]:
        """Generate molecules using thread pool."""
        loop = asyncio.get_event_loop()
        
        # Select least loaded worker
        worker_idx = self.worker_load.index(min(self.worker_load))
        self.worker_load[worker_idx] += 1
        
        try:
            molecules = await loop.run_in_executor(
                self.executor,
                self._sync_generate,
                prompt,
                num_molecules,
                safety_filter,
                synthesizability_min,
                kwargs
            )
            return molecules
            
        finally:
            self.worker_load[worker_idx] -= 1
    
    def _sync_generate(
        self,
        prompt: str,
        num_molecules: int,
        safety_filter: Optional[SafetyFilter],
        synthesizability_min: float,
        kwargs: Dict[str, Any]
    ) -> List[Molecule]:
        """Synchronous generation wrapper."""
        return self.model.generate(
            prompt=prompt,
            num_molecules=num_molecules,
            safety_filter=safety_filter,
            synthesizability_min=synthesizability_min,
            **kwargs
        )
    
    async def generate_batch_async(
        self,
        batch_request: BatchRequest,
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> List[GenerationResult]:
        """
        Process a batch of prompts asynchronously with progress tracking.
        """
        results = []
        total = len(batch_request.prompts)
        
        # Process prompts concurrently
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single(idx: int, prompt: str) -> GenerationResult:
            async with semaphore:
                result = await self.generate_async(prompt, **batch_request.params)
                
                if progress_callback:
                    await progress_callback(idx + 1, total)
                
                return result
        
        # Create tasks for all prompts
        tasks = [
            process_single(i, prompt) 
            for i, prompt in enumerate(batch_request.prompts)
        ]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(GenerationResult(
                    prompt=batch_request.prompts[i],
                    molecules=[],
                    processing_time=0.0,
                    cache_hit=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        self.processing_stats['batch_requests'] += 1
        logger.info(f"Batch processing completed: {len(batch_request.prompts)} prompts")
        
        return processed_results
    
    async def _batch_processor(self):
        """Background batch processor."""
        while True:
            try:
                # Collect requests for batching
                batch_requests = []
                timeout = 1.0  # Wait 1 second for batch to fill
                
                try:
                    # Get first request (blocking)
                    first_request = await asyncio.wait_for(
                        self.batch_queue.get(), 
                        timeout=timeout
                    )
                    batch_requests.append(first_request)
                    
                    # Collect additional requests (non-blocking)
                    while len(batch_requests) < self.batch_size:
                        try:
                            request = self.batch_queue.get_nowait()
                            batch_requests.append(request)
                        except asyncio.QueueEmpty:
                            break
                            
                except asyncio.TimeoutError:
                    continue
                
                # Process batch
                if batch_requests:
                    await self._process_batch(batch_requests)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
    
    async def _process_batch(self, batch_requests: List[BatchRequest]):
        """Process a batch of requests."""
        # Sort by priority
        batch_requests.sort(key=lambda x: x.priority, reverse=True)
        
        for request in batch_requests:
            try:
                results = await self.generate_batch_async(request)
                if request.callback:
                    await request.callback(results)
                    
            except Exception as e:
                logger.error(f"Error processing batch request: {e}")
    
    async def submit_batch(self, batch_request: BatchRequest) -> bool:
        """Submit a batch request for processing."""
        try:
            await self.batch_queue.put(batch_request)
            return True
        except asyncio.QueueFull:
            logger.warning("Batch queue is full, request rejected")
            return False
    
    async def optimize_fragrance_async(
        self,
        base_notes: str,
        heart_notes: str,
        top_notes: str,
        style: str,
        constraints: Optional[Dict[str, Any]] = None,
        iterations: int = 3
    ) -> FragranceFormulation:
        """
        Asynchronously optimize a fragrance formulation with multiple iterations.
        """
        logger.info(f"Starting fragrance optimization: {style}")
        
        best_formulation = None
        best_score = 0.0
        
        # Run multiple optimization iterations concurrently
        tasks = []
        for i in range(iterations):
            task = self._optimize_single_iteration(
                base_notes, heart_notes, top_notes, style, constraints, i
            )
            tasks.append(task)
        
        formulations = await asyncio.gather(*tasks)
        
        # Select best formulation
        for formulation in formulations:
            score = self._score_formulation(formulation)
            if score > best_score:
                best_score = score
                best_formulation = formulation
        
        logger.info(f"Fragrance optimization completed, best score: {best_score:.2f}")
        return best_formulation or formulations[0]
    
    async def _optimize_single_iteration(
        self,
        base_notes: str,
        heart_notes: str,
        top_notes: str,
        style: str,
        constraints: Optional[Dict[str, Any]],
        iteration: int
    ) -> FragranceFormulation:
        """Single optimization iteration."""
        # Add variation to prompts for diversity
        variation_prompts = [
            f"{base_notes} {style} variation {iteration}",
            f"{heart_notes} {style} variation {iteration}",
            f"{top_notes} {style} variation {iteration}"
        ]
        
        # Generate molecules for each accord
        base_result = await self.generate_async(variation_prompts[0], num_molecules=3)
        heart_result = await self.generate_async(variation_prompts[1], num_molecules=3)
        top_result = await self.generate_async(variation_prompts[2], num_molecules=3)
        
        return FragranceFormulation(
            base_accord=base_result.molecules,
            heart_accord=heart_result.molecules,
            top_accord=top_result.molecules,
            style_descriptor=f"{style} (iteration {iteration})"
        )
    
    def _score_formulation(self, formulation: FragranceFormulation) -> float:
        """Score a fragrance formulation."""
        score = 0.0
        
        # Score based on molecule quality
        all_molecules = (
            formulation.base_accord + 
            formulation.heart_accord + 
            formulation.top_accord
        )
        
        if all_molecules:
            avg_safety = sum(mol.safety_score for mol in all_molecules) / len(all_molecules)
            avg_synth = sum(mol.synth_score for mol in all_molecules) / len(all_molecules)
            avg_confidence = sum(mol.confidence for mol in all_molecules) / len(all_molecules)
            
            score = (avg_safety * 0.4 + avg_synth * 0.3 + avg_confidence * 0.3)
        
        return score
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.processing_stats,
            'worker_load': self.worker_load,
            'cache_stats': self.cache.get_cache_stats() if self.cache else {},
            'queue_size': self.batch_queue.qsize() if self.batch_queue else 0
        }
    
    async def preload_cache(self, common_prompts: List[str]) -> int:
        """Preload cache with common prompts."""
        logger.info(f"Preloading cache with {len(common_prompts)} prompts")
        
        loaded = 0
        for prompt in common_prompts:
            try:
                await self.generate_async(prompt, num_molecules=5, use_cache=True)
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to preload prompt '{prompt}': {e}")
        
        logger.info(f"Cache preloaded with {loaded} prompts")
        return loaded
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        start_time = time.time()
        
        try:
            # Test generation
            test_result = await self.generate_async("test lavender scent", num_molecules=1)
            generation_ok = len(test_result.molecules) > 0 and test_result.error is None
            
            # Check memory usage
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            health_status = {
                'status': 'healthy' if generation_ok else 'degraded',
                'response_time': time.time() - start_time,
                'generation_test': generation_ok,
                'memory_usage_mb': memory_mb,
                'worker_count': self.max_workers,
                'cache_enabled': self.enable_caching,
                'stats': self.get_performance_stats()
            }
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time': time.time() - start_time
            }