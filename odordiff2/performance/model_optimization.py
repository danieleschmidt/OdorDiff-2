"""
Model inference and batch processing optimization for OdorDiff-2.
Provides advanced model optimization techniques for production deployment.
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import numpy as np
from pathlib import Path
import pickle
import json
import gc
import weakref
from contextlib import asynccontextmanager

from ..utils.logging import get_logger
from ..utils.error_handling import retry_with_backoff, handle_errors
from .optimization import ModelOptimizer, BatchProcessor, OptimizationConfig

logger = get_logger(__name__)


class OptimizationTechnique(Enum):
    """Model optimization techniques."""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    FUSION = "fusion"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    DYNAMIC_BATCHING = "dynamic_batching"
    MODEL_PARALLEL = "model_parallel"


class PrecisionMode(Enum):
    """Precision modes for optimization."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    MIXED = "mixed"


@dataclass
class OptimizationProfile:
    """Profile for model optimization settings."""
    name: str
    techniques: List[OptimizationTechnique]
    precision_mode: PrecisionMode
    batch_size_range: Tuple[int, int]
    memory_limit_mb: int
    latency_target_ms: float
    throughput_target_rps: float
    quality_threshold: float = 0.95
    
    # Advanced settings
    enable_caching: bool = True
    enable_prefetching: bool = True
    enable_pipeline_parallel: bool = False
    max_sequence_length: int = 512
    
    # Hardware-specific
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    num_cpu_threads: int = 4


@dataclass
class InferenceRequest:
    """Request for model inference."""
    id: str
    inputs: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Higher = more priority
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        if 'request_size' not in self.metadata:
            self.metadata['request_size'] = self._estimate_size()
    
    def _estimate_size(self) -> int:
        """Estimate request size for batching decisions."""
        try:
            if hasattr(self.inputs, '__len__'):
                return len(self.inputs)
            elif isinstance(self.inputs, (list, tuple)):
                return len(self.inputs)
            elif isinstance(self.inputs, dict):
                return sum(len(str(v)) for v in self.inputs.values())
            else:
                return len(str(self.inputs))
        except:
            return 1


@dataclass
class InferenceResult:
    """Result from model inference."""
    request_id: str
    outputs: Any
    latency_ms: float
    cache_hit: bool = False
    batch_size: int = 1
    model_version: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelCache:
    """Advanced model caching with LRU and intelligent prefetching."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float, int]] = {}  # key -> (value, timestamp, access_count)
        self.access_order = deque()
        self.access_patterns = defaultdict(lambda: deque(maxlen=100))
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"ModelCache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.cache_lock:
            if key in self.cache:
                value, timestamp, access_count = self.cache[key]
                
                # Check TTL
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                # Update access info
                self.cache[key] = (value, timestamp, access_count + 1)
                self._update_access_order(key)
                self.access_patterns[key].append(time.time())
                
                self.hits += 1
                return value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put value into cache."""
        with self.cache_lock:
            current_time = time.time()
            
            # Evict if necessary
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = (value, current_time, 1)
            self._update_access_order(key)
            self.access_patterns[key].append(current_time)
    
    def _update_access_order(self, key: str):
        """Update LRU order."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_order:
            return
        
        lru_key = self.access_order.popleft()
        if lru_key in self.cache:
            del self.cache[lru_key]
            self.evictions += 1
    
    def prefetch_predictions(self) -> List[str]:
        """Predict keys that should be prefetched."""
        predictions = []
        current_time = time.time()
        
        # Analyze access patterns
        for key, timestamps in self.access_patterns.items():
            if len(timestamps) < 3:
                continue
            
            # Calculate access frequency
            recent_accesses = [t for t in timestamps if current_time - t < 3600]  # Last hour
            if len(recent_accesses) >= 2:
                # Predict next access time based on pattern
                intervals = [recent_accesses[i] - recent_accesses[i-1] for i in range(1, len(recent_accesses))]
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    last_access = recent_accesses[-1]
                    
                    # If predicted next access is soon, suggest prefetch
                    predicted_next = last_access + avg_interval
                    if predicted_next - current_time < 300:  # Within 5 minutes
                        predictions.append(key)
        
        return predictions[:10]  # Limit prefetch suggestions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / max(total_requests, 1)) * 100
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': (len(self.cache) / max(self.max_size, 1)) * 100,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'ttl_seconds': self.ttl_seconds
        }


class DynamicBatcher:
    """Dynamic batching system with intelligent batching decisions."""
    
    def __init__(self, 
                 min_batch_size: int = 1,
                 max_batch_size: int = 32,
                 max_wait_time_ms: int = 10,
                 size_based_batching: bool = True):
        
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.size_based_batching = size_based_batching
        
        self.pending_requests: Dict[int, deque] = defaultdict(deque)  # priority -> requests
        self.batch_statistics = defaultdict(lambda: {'count': 0, 'total_latency': 0.0})
        self.optimal_batch_sizes: Dict[str, int] = {}
        
        self.batching_lock = asyncio.Lock()
        
        logger.info(f"DynamicBatcher initialized: batch_size=[{min_batch_size}, {max_batch_size}], wait_time={max_wait_time_ms}ms")
    
    async def add_request(self, request: InferenceRequest) -> asyncio.Future:
        """Add request to batching queue."""
        future = asyncio.Future()
        request.callback = lambda result: future.set_result(result)
        
        async with self.batching_lock:
            self.pending_requests[request.priority].appendleft(request)
        
        return future
    
    async def get_next_batch(self) -> List[InferenceRequest]:
        """Get the next batch of requests to process."""
        start_time = time.time()
        batch = []
        
        async with self.batching_lock:
            # Process by priority (higher first)
            for priority in sorted(self.pending_requests.keys(), reverse=True):
                request_queue = self.pending_requests[priority]
                
                while len(batch) < self.max_batch_size and request_queue:
                    # Check timeout
                    if (time.time() - start_time) * 1000 > self.max_wait_time_ms:
                        break
                    
                    request = request_queue.pop()
                    
                    # Check individual request timeout
                    if time.time() - request.created_at > request.timeout:
                        # Request timed out, skip it
                        if request.callback:
                            error_result = InferenceResult(
                                request_id=request.id,
                                outputs=None,
                                latency_ms=0,
                                metadata={'error': 'Request timeout'}
                            )
                            request.callback(error_result)
                        continue
                    
                    # Size compatibility check for batching
                    if self.size_based_batching and batch:
                        if not self._compatible_for_batching(batch[0], request):
                            # Put request back and break
                            request_queue.append(request)
                            break
                    
                    batch.append(request)
                    
                    if len(batch) >= self.max_batch_size:
                        break
                
                if len(batch) >= self.min_batch_size:
                    break
        
        # Remove empty queues
        empty_priorities = [p for p, q in self.pending_requests.items() if not q]
        for p in empty_priorities:
            del self.pending_requests[p]
        
        return batch
    
    def _compatible_for_batching(self, request1: InferenceRequest, request2: InferenceRequest) -> bool:
        """Check if two requests can be batched together."""
        # Simple size-based compatibility
        size1 = request1.metadata.get('request_size', 1)
        size2 = request2.metadata.get('request_size', 1)
        
        # Requests should be within 50% of each other's size
        ratio = max(size1, size2) / max(min(size1, size2), 1)
        return ratio <= 1.5
    
    def record_batch_performance(self, batch_size: int, latency_ms: float):
        """Record batch performance for optimization."""
        key = str(batch_size)
        stats = self.batch_statistics[key]
        stats['count'] += 1
        stats['total_latency'] += latency_ms
        
        # Update optimal batch size
        avg_latency_per_request = latency_ms / batch_size
        self.optimal_batch_sizes[key] = avg_latency_per_request
    
    def get_optimal_batch_size(self) -> int:
        """Get the currently optimal batch size based on performance."""
        if not self.optimal_batch_sizes:
            return self.min_batch_size
        
        # Find batch size with lowest latency per request
        best_size = self.min_batch_size
        best_latency = float('inf')
        
        for size_str, latency_per_req in self.optimal_batch_sizes.items():
            if latency_per_req < best_latency:
                best_latency = latency_per_req
                best_size = int(size_str)
        
        return max(self.min_batch_size, min(best_size, self.max_batch_size))
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get batching queue statistics."""
        total_pending = sum(len(q) for q in self.pending_requests.values())
        
        return {
            'total_pending': total_pending,
            'queues_by_priority': {
                str(p): len(q) for p, q in self.pending_requests.items()
            },
            'batch_statistics': dict(self.batch_statistics),
            'optimal_batch_size': self.get_optimal_batch_size(),
            'max_batch_size': self.max_batch_size,
            'min_batch_size': self.min_batch_size
        }


class ModelOptimizationEngine:
    """Advanced model optimization and inference engine."""
    
    def __init__(self, 
                 profile: OptimizationProfile,
                 cache_size: int = 10000,
                 max_concurrent_requests: int = 100):
        
        self.profile = profile
        self.cache = ModelCache(cache_size)
        self.batcher = DynamicBatcher(
            min_batch_size=profile.batch_size_range[0],
            max_batch_size=profile.batch_size_range[1]
        )
        
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.optimization_stats = defaultdict(int)
        
        # Threading and async handling
        self.thread_pool = ThreadPoolExecutor(
            max_workers=profile.num_cpu_threads,
            thread_name_prefix="ModelOptim"
        )
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        self.is_running = False
        self.processing_task = None
        self.prefetch_task = None
        
        # Performance tracking
        self.request_latencies = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        self.last_throughput_calculation = time.time()
        self.processed_requests = 0
        
        logger.info(f"ModelOptimizationEngine initialized with profile: {profile.name}")
    
    async def start(self):
        """Start the optimization engine."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        self.prefetch_task = asyncio.create_task(self._prefetch_loop())
        
        logger.info("Model optimization engine started")
    
    async def stop(self):
        """Stop the optimization engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
        if self.prefetch_task:
            self.prefetch_task.cancel()
        
        try:
            await asyncio.gather(self.processing_task, self.prefetch_task, return_exceptions=True)
        except:
            pass
        
        self.thread_pool.shutdown(wait=True)
        logger.info("Model optimization engine stopped")
    
    def register_model(self, 
                      name: str,
                      model: Any,
                      metadata: Dict[str, Any] = None):
        """Register a model for optimization."""
        self.models[name] = model
        self.model_metadata[name] = metadata or {}
        
        # Apply optimization techniques
        if self.profile.techniques:
            self._apply_optimizations(name, model)
        
        logger.info(f"Registered model: {name}")
    
    def _apply_optimizations(self, model_name: str, model: Any):
        """Apply optimization techniques to a model."""
        for technique in self.profile.techniques:
            try:
                if technique == OptimizationTechnique.QUANTIZATION:
                    self._apply_quantization(model_name, model)
                elif technique == OptimizationTechnique.PRUNING:
                    self._apply_pruning(model_name, model)
                elif technique == OptimizationTechnique.FUSION:
                    self._apply_fusion(model_name, model)
                # Add more techniques as needed
                
                self.optimization_stats[f'{technique.value}_applied'] += 1
                logger.info(f"Applied {technique.value} to model {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to apply {technique.value} to {model_name}: {e}")
    
    def _apply_quantization(self, model_name: str, model: Any):
        """Apply quantization optimization (stub implementation)."""
        # In practice, this would use framework-specific quantization
        # For now, just mark as quantized
        self.model_metadata[model_name]['quantized'] = True
        self.model_metadata[model_name]['precision'] = self.profile.precision_mode.value
    
    def _apply_pruning(self, model_name: str, model: Any):
        """Apply pruning optimization (stub implementation)."""
        self.model_metadata[model_name]['pruned'] = True
    
    def _apply_fusion(self, model_name: str, model: Any):
        """Apply layer fusion optimization (stub implementation)."""
        self.model_metadata[model_name]['fused'] = True
    
    async def infer_async(self, 
                         model_name: str,
                         inputs: Any,
                         request_id: str = None,
                         priority: int = 1,
                         use_cache: bool = True) -> InferenceResult:
        """Perform asynchronous model inference."""
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}_{id(inputs)}"
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(model_name, inputs)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return InferenceResult(
                    request_id=request_id,
                    outputs=cached_result,
                    latency_ms=0.1,  # Minimal cache access time
                    cache_hit=True,
                    model_version=self.model_metadata.get(model_name, {}).get('version', 'unknown')
                )
        
        # Create inference request
        request = InferenceRequest(
            id=request_id,
            inputs={'model_name': model_name, 'data': inputs},
            priority=priority
        )
        
        # Submit to batcher
        result_future = await self.batcher.add_request(request)
        return await result_future
    
    async def _processing_loop(self):
        """Main processing loop for handling batched requests."""
        while self.is_running:
            try:
                # Get next batch
                batch = await self.batcher.get_next_batch()
                
                if not batch:
                    await asyncio.sleep(0.001)  # Small delay when no requests
                    continue
                
                # Process batch
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of inference requests."""
        if not batch:
            return
        
        batch_start_time = time.time()
        
        async with self.request_semaphore:
            try:
                # Group requests by model
                model_groups = defaultdict(list)
                for request in batch:
                    model_name = request.inputs.get('model_name', 'default')
                    model_groups[model_name].append(request)
                
                # Process each model group
                for model_name, requests in model_groups.items():
                    await self._process_model_batch(model_name, requests, batch_start_time)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                
                # Send error results to all requests
                for request in batch:
                    if request.callback:
                        error_result = InferenceResult(
                            request_id=request.id,
                            outputs=None,
                            latency_ms=0,
                            metadata={'error': str(e)}
                        )
                        request.callback(error_result)
    
    async def _process_model_batch(self, 
                                  model_name: str, 
                                  requests: List[InferenceRequest],
                                  batch_start_time: float):
        """Process a batch of requests for a specific model."""
        if model_name not in self.models:
            logger.error(f"Model not found: {model_name}")
            return
        
        try:
            # Extract inputs
            batch_inputs = [req.inputs['data'] for req in requests]
            
            # Run inference
            batch_outputs = await self._run_model_inference(
                model_name, 
                batch_inputs
            )
            
            batch_latency = (time.time() - batch_start_time) * 1000
            
            # Send results to requests
            for i, request in enumerate(requests):
                output = batch_outputs[i] if i < len(batch_outputs) else None
                
                result = InferenceResult(
                    request_id=request.id,
                    outputs=output,
                    latency_ms=batch_latency,
                    batch_size=len(requests),
                    model_version=self.model_metadata.get(model_name, {}).get('version', 'unknown')
                )
                
                # Cache result if applicable
                if self.profile.enable_caching and output is not None:
                    cache_key = self._generate_cache_key(model_name, request.inputs['data'])
                    self.cache.put(cache_key, output)
                
                # Send result
                if request.callback:
                    request.callback(result)
            
            # Record performance
            self.batcher.record_batch_performance(len(requests), batch_latency)
            self.request_latencies.extend([batch_latency / len(requests)] * len(requests))
            self.processed_requests += len(requests)
            
            # Update throughput
            self._update_throughput_stats()
            
        except Exception as e:
            logger.error(f"Model batch processing error for {model_name}: {e}")
            
            # Send error to all requests
            for request in requests:
                if request.callback:
                    error_result = InferenceResult(
                        request_id=request.id,
                        outputs=None,
                        latency_ms=0,
                        metadata={'error': str(e)}
                    )
                    request.callback(error_result)
    
    async def _run_model_inference(self, model_name: str, batch_inputs: List[Any]) -> List[Any]:
        """Run actual model inference."""
        model = self.models[model_name]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _inference():
            try:
                # Simulate model inference
                # In practice, this would call the actual model
                results = []
                for inputs in batch_inputs:
                    # Mock inference result
                    result = {
                        'prediction': f"result_for_{inputs}",
                        'confidence': 0.95,
                        'processing_time': 0.05
                    }
                    results.append(result)
                    
                    # Simulate some processing time
                    time.sleep(0.001)
                
                return results
                
            except Exception as e:
                logger.error(f"Model inference error: {e}")
                return [None] * len(batch_inputs)
        
        return await loop.run_in_executor(self.thread_pool, _inference)
    
    async def _prefetch_loop(self):
        """Background prefetching loop."""
        while self.is_running:
            try:
                if self.profile.enable_prefetching:
                    predictions = self.cache.prefetch_predictions()
                    
                    for cache_key in predictions:
                        # In practice, you would decode the cache key and prefetch
                        pass
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch loop error: {e}")
                await asyncio.sleep(60)
    
    def _generate_cache_key(self, model_name: str, inputs: Any) -> str:
        """Generate cache key for inputs."""
        # Simple hash-based cache key
        import hashlib
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        return f"{model_name}:{input_hash}"
    
    def _update_throughput_stats(self):
        """Update throughput statistics."""
        current_time = time.time()
        time_diff = current_time - self.last_throughput_calculation
        
        if time_diff >= 1.0:  # Update every second
            throughput = self.processed_requests / time_diff
            self.throughput_history.append(throughput)
            
            self.processed_requests = 0
            self.last_throughput_calculation = current_time
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        avg_latency = 0.0
        if self.request_latencies:
            avg_latency = sum(self.request_latencies) / len(self.request_latencies)
        
        current_throughput = 0.0
        if self.throughput_history:
            current_throughput = self.throughput_history[-1]
        
        avg_throughput = 0.0
        if self.throughput_history:
            avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
        
        return {
            'profile_name': self.profile.name,
            'is_running': self.is_running,
            'registered_models': list(self.models.keys()),
            'optimization_techniques': [t.value for t in self.profile.techniques],
            'precision_mode': self.profile.precision_mode.value,
            
            # Performance metrics
            'average_latency_ms': avg_latency,
            'current_throughput_rps': current_throughput,
            'average_throughput_rps': avg_throughput,
            'total_requests_processed': sum(self.request_latencies),
            
            # Cache stats
            'cache_stats': self.cache.get_stats(),
            
            # Batching stats
            'batching_stats': self.batcher.get_queue_stats(),
            
            # Optimization stats
            'optimization_applications': dict(self.optimization_stats),
            
            # Resource utilization
            'memory_usage_mb': self._get_memory_usage(),
            'thread_pool_active': self.thread_pool._threads,
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0


# Pre-configured optimization profiles
OPTIMIZATION_PROFILES = {
    'high_throughput': OptimizationProfile(
        name='high_throughput',
        techniques=[OptimizationTechnique.DYNAMIC_BATCHING, OptimizationTechnique.QUANTIZATION],
        precision_mode=PrecisionMode.FP16,
        batch_size_range=(8, 64),
        memory_limit_mb=4096,
        latency_target_ms=100.0,
        throughput_target_rps=1000.0,
        enable_caching=True,
        enable_prefetching=True
    ),
    
    'low_latency': OptimizationProfile(
        name='low_latency',
        techniques=[OptimizationTechnique.QUANTIZATION, OptimizationTechnique.FUSION],
        precision_mode=PrecisionMode.FP16,
        batch_size_range=(1, 8),
        memory_limit_mb=2048,
        latency_target_ms=10.0,
        throughput_target_rps=500.0,
        enable_caching=True,
        enable_prefetching=False
    ),
    
    'balanced': OptimizationProfile(
        name='balanced',
        techniques=[OptimizationTechnique.DYNAMIC_BATCHING, OptimizationTechnique.QUANTIZATION],
        precision_mode=PrecisionMode.FP16,
        batch_size_range=(4, 32),
        memory_limit_mb=3072,
        latency_target_ms=50.0,
        throughput_target_rps=750.0,
        enable_caching=True,
        enable_prefetching=True
    ),
    
    'memory_efficient': OptimizationProfile(
        name='memory_efficient',
        techniques=[OptimizationTechnique.PRUNING, OptimizationTechnique.QUANTIZATION],
        precision_mode=PrecisionMode.INT8,
        batch_size_range=(1, 16),
        memory_limit_mb=1024,
        latency_target_ms=200.0,
        throughput_target_rps=200.0,
        enable_caching=False,
        enable_prefetching=False
    )
}


# Global optimization engine
_optimization_engine: Optional[ModelOptimizationEngine] = None

def get_optimization_engine(profile_name: str = 'balanced') -> ModelOptimizationEngine:
    """Get global optimization engine instance."""
    global _optimization_engine
    if _optimization_engine is None:
        profile = OPTIMIZATION_PROFILES.get(profile_name, OPTIMIZATION_PROFILES['balanced'])
        _optimization_engine = ModelOptimizationEngine(profile)
    return _optimization_engine

def create_optimization_engine(profile: OptimizationProfile) -> ModelOptimizationEngine:
    """Create a new optimization engine with custom profile."""
    return ModelOptimizationEngine(profile)