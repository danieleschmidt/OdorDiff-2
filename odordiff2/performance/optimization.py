"""
Performance optimization and caching components for OdorDiff-2.
Integrates with existing caching system and adds model inference optimization.
"""

import asyncio
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
import logging
import psutil
from pathlib import Path
import json
import pickle
import hashlib
from functools import wraps
import weakref

# Import existing cache system
from ..data.cache import DatabaseConnectionPool, RedisConnectionPool, EnhancedMoleculeCache
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking optimization effectiveness."""
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    queue_length: int = 0
    error_rate: float = 0.0
    throughput_improvement: float = 0.0
    latency_reduction: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_model_caching: bool = True
    enable_request_batching: bool = True
    enable_connection_pooling: bool = True
    enable_result_compression: bool = True
    
    # Batch processing
    batch_size: int = 8
    batch_timeout_ms: int = 100
    max_batch_wait: int = 32
    
    # Caching
    model_cache_size: int = 1000
    result_cache_ttl: int = 3600
    preload_popular_models: bool = True
    
    # Resource management
    max_concurrent_requests: int = 100
    worker_threads: int = 8
    memory_threshold_mb: int = 8192
    cpu_threshold_percent: float = 85.0
    
    # Monitoring
    metrics_collection_interval: int = 60
    performance_logging: bool = True
    adaptive_optimization: bool = True


class ModelOptimizer:
    """Optimizes model inference performance with caching and batching."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.model_cache = {}
        self.result_cache = EnhancedMoleculeCache()
        self.batch_queue = asyncio.Queue(maxsize=self.config.max_batch_wait)
        self.processing_batches = set()
        self.metrics = PerformanceMetrics()
        self.request_times = deque(maxlen=1000)
        self.cache_stats = defaultdict(int)
        self._optimization_lock = threading.Lock()
        
        # Start batch processing if enabled
        if self.config.enable_request_batching:
            asyncio.create_task(self._batch_processor())
        
        logger.info(f"ModelOptimizer initialized with config: {self.config}")
    
    def cache_model(self, model_key: str, model: Any) -> None:
        """Cache a model for faster access."""
        if not self.config.enable_model_caching:
            return
            
        with self._optimization_lock:
            # Check memory usage before caching
            if len(self.model_cache) >= self.config.model_cache_size:
                # Remove least recently used model
                oldest_key = min(self.model_cache.keys(), 
                               key=lambda k: self.model_cache[k]['last_used'])
                del self.model_cache[oldest_key]
                logger.debug(f"Evicted cached model: {oldest_key}")
            
            self.model_cache[model_key] = {
                'model': model,
                'last_used': time.time(),
                'use_count': 0
            }
            logger.debug(f"Cached model: {model_key}")
    
    def get_cached_model(self, model_key: str) -> Optional[Any]:
        """Retrieve cached model."""
        if not self.config.enable_model_caching:
            return None
            
        with self._optimization_lock:
            if model_key in self.model_cache:
                entry = self.model_cache[model_key]
                entry['last_used'] = time.time()
                entry['use_count'] += 1
                self.cache_stats['model_hits'] += 1
                return entry['model']
            
            self.cache_stats['model_misses'] += 1
            return None
    
    async def optimize_inference(self, 
                               inference_func: Callable,
                               inputs: List[Any],
                               cache_key: str = None) -> List[Any]:
        """Optimize inference with caching, batching, and resource management."""
        start_time = time.time()
        
        try:
            # Check result cache first
            if cache_key and self.config.enable_model_caching:
                cached_result = await self.result_cache.get(cache_key)
                if cached_result is not None:
                    self.cache_stats['result_hits'] += 1
                    self._update_metrics(start_time, cache_hit=True)
                    return cached_result
                
                self.cache_stats['result_misses'] += 1
            
            # Use batch processing for multiple inputs
            if self.config.enable_request_batching and len(inputs) > 1:
                results = await self._batch_inference(inference_func, inputs)
            else:
                # Direct inference for single inputs or when batching disabled
                results = await self._direct_inference(inference_func, inputs)
            
            # Cache results if enabled
            if cache_key and self.config.enable_model_caching:
                await self.result_cache.set(
                    cache_key, 
                    results,
                    ttl=self.config.result_cache_ttl
                )
            
            self._update_metrics(start_time, cache_hit=False)
            return results
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"Inference optimization failed: {e}")
            raise
    
    async def _batch_inference(self, inference_func: Callable, inputs: List[Any]) -> List[Any]:
        """Process inputs in optimized batches."""
        if len(inputs) <= self.config.batch_size:
            return await inference_func(inputs)
        
        # Split into batches
        batches = [
            inputs[i:i + self.config.batch_size]
            for i in range(0, len(inputs), self.config.batch_size)
        ]
        
        # Process batches concurrently
        tasks = [inference_func(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    async def _direct_inference(self, inference_func: Callable, inputs: List[Any]) -> List[Any]:
        """Direct inference without batching."""
        return await inference_func(inputs)
    
    async def _batch_processor(self):
        """Background batch processor for queued requests."""
        while True:
            try:
                batch = []
                batch_start = time.time()
                
                # Collect batch
                while (len(batch) < self.config.batch_size and 
                       time.time() - batch_start < self.config.batch_timeout_ms / 1000):
                    try:
                        item = await asyncio.wait_for(
                            self.batch_queue.get(),
                            timeout=0.01
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Process batch
                    batch_id = id(batch)
                    self.processing_batches.add(batch_id)
                    
                    try:
                        # Process all items in batch concurrently
                        tasks = [item['func'](*item['args']) for item in batch]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Set results for each request
                        for item, result in zip(batch, results):
                            item['future'].set_result(result)
                            
                    except Exception as e:
                        # Set exception for all items in batch
                        for item in batch:
                            item['future'].set_exception(e)
                    
                    finally:
                        self.processing_batches.discard(batch_id)
                
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)
    
    def _update_metrics(self, start_time: float, cache_hit: bool = False):
        """Update performance metrics."""
        duration = time.time() - start_time
        self.request_times.append(duration)
        
        # Update cache hit rate
        total_requests = sum(self.cache_stats.values())
        if total_requests > 0:
            hits = self.cache_stats['result_hits'] + self.cache_stats['model_hits']
            self.metrics.cache_hit_rate = hits / total_requests
        
        # Update response time
        if self.request_times:
            self.metrics.average_response_time = sum(self.request_times) / len(self.request_times)
            
            # Calculate RPS from recent requests
            recent_times = [t for t in self.request_times if time.time() - start_time < 60]
            if recent_times:
                self.metrics.requests_per_second = len(recent_times) / 60
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # Update system metrics
        self.metrics.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.metrics.cpu_usage_percent = psutil.cpu_percent()
        self.metrics.queue_length = self.batch_queue.qsize()
        
        return self.metrics


class ResourcePool:
    """Advanced resource pooling for database connections and compute resources."""
    
    def __init__(self, 
                 resource_factory: Callable,
                 min_size: int = 5,
                 max_size: int = 50,
                 idle_timeout: int = 300,
                 health_check_interval: int = 60):
        
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.health_check_interval = health_check_interval
        
        self.available = asyncio.Queue(maxsize=max_size)
        self.in_use = set()
        self.resource_metadata = weakref.WeakKeyDictionary()
        self.pool_lock = asyncio.Lock()
        self.total_created = 0
        self.total_errors = 0
        
        # Start background tasks
        asyncio.create_task(self._health_checker())
        asyncio.create_task(self._idle_cleanup())
        
        logger.info(f"ResourcePool initialized: min={min_size}, max={max_size}")
    
    async def acquire(self, timeout: float = 30.0) -> Any:
        """Acquire a resource from the pool."""
        start_time = time.time()
        
        try:
            # Try to get available resource
            try:
                resource = await asyncio.wait_for(
                    self.available.get(),
                    timeout=min(timeout, 5.0)
                )
                
                async with self.pool_lock:
                    self.in_use.add(resource)
                    self.resource_metadata[resource] = {
                        'acquired_at': time.time(),
                        'use_count': self.resource_metadata.get(resource, {}).get('use_count', 0) + 1
                    }
                
                return resource
                
            except asyncio.TimeoutError:
                # No available resource, try to create new one
                if self.total_created - len(self.in_use) < self.max_size:
                    resource = await self._create_resource()
                    if resource:
                        async with self.pool_lock:
                            self.in_use.add(resource)
                            self.resource_metadata[resource] = {
                                'acquired_at': time.time(),
                                'use_count': 1
                            }
                        return resource
                
                # Pool is full, wait longer
                remaining_timeout = timeout - (time.time() - start_time)
                if remaining_timeout > 0:
                    resource = await asyncio.wait_for(
                        self.available.get(),
                        timeout=remaining_timeout
                    )
                    
                    async with self.pool_lock:
                        self.in_use.add(resource)
                        self.resource_metadata[resource] = {
                            'acquired_at': time.time(),
                            'use_count': self.resource_metadata.get(resource, {}).get('use_count', 0) + 1
                        }
                    
                    return resource
                else:
                    raise TimeoutError("Resource acquisition timeout")
                    
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Resource acquisition failed: {e}")
            raise
    
    async def release(self, resource: Any):
        """Release a resource back to the pool."""
        try:
            async with self.pool_lock:
                if resource in self.in_use:
                    self.in_use.remove(resource)
                    
                    # Check if resource is healthy
                    if await self._is_healthy(resource):
                        await self.available.put(resource)
                        if resource in self.resource_metadata:
                            self.resource_metadata[resource]['released_at'] = time.time()
                    else:
                        await self._destroy_resource(resource)
                        # Create replacement if below min size
                        if len(self.in_use) + self.available.qsize() < self.min_size:
                            asyncio.create_task(self._ensure_min_size())
        
        except Exception as e:
            logger.error(f"Resource release failed: {e}")
    
    async def _create_resource(self) -> Optional[Any]:
        """Create a new resource."""
        try:
            resource = await self.resource_factory()
            self.total_created += 1
            self.resource_metadata[resource] = {
                'created_at': time.time(),
                'use_count': 0
            }
            logger.debug(f"Created new resource (total: {self.total_created})")
            return resource
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Resource creation failed: {e}")
            return None
    
    async def _destroy_resource(self, resource: Any):
        """Destroy a resource."""
        try:
            if hasattr(resource, 'close'):
                await resource.close()
            elif hasattr(resource, 'disconnect'):
                await resource.disconnect()
            
            if resource in self.resource_metadata:
                del self.resource_metadata[resource]
            
            logger.debug("Resource destroyed")
            
        except Exception as e:
            logger.error(f"Resource destruction failed: {e}")
    
    async def _is_healthy(self, resource: Any) -> bool:
        """Check if a resource is healthy."""
        try:
            if hasattr(resource, 'ping'):
                return await resource.ping()
            elif hasattr(resource, 'is_connected'):
                return resource.is_connected()
            else:
                return True  # Assume healthy if no health check available
                
        except Exception:
            return False
    
    async def _health_checker(self):
        """Background health checker."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check available resources
                healthy_resources = []
                while not self.available.empty():
                    try:
                        resource = await asyncio.wait_for(self.available.get(), timeout=0.1)
                        if await self._is_healthy(resource):
                            healthy_resources.append(resource)
                        else:
                            await self._destroy_resource(resource)
                    except asyncio.TimeoutError:
                        break
                
                # Put healthy resources back
                for resource in healthy_resources:
                    await self.available.put(resource)
                
                logger.debug(f"Health check completed: {len(healthy_resources)} healthy resources")
                
            except Exception as e:
                logger.error(f"Health checker error: {e}")
    
    async def _idle_cleanup(self):
        """Clean up idle resources."""
        while True:
            try:
                await asyncio.sleep(self.idle_timeout)
                
                current_time = time.time()
                resources_to_remove = []
                
                # Check for idle resources
                for resource, metadata in self.resource_metadata.items():
                    if (resource not in self.in_use and 
                        'released_at' in metadata and
                        current_time - metadata['released_at'] > self.idle_timeout):
                        resources_to_remove.append(resource)
                
                # Remove idle resources (keep minimum)
                current_total = len(self.in_use) + self.available.qsize()
                for resource in resources_to_remove:
                    if current_total > self.min_size:
                        try:
                            # Remove from available queue
                            temp_resources = []
                            while not self.available.empty():
                                temp_resource = await asyncio.wait_for(self.available.get(), timeout=0.1)
                                if temp_resource != resource:
                                    temp_resources.append(temp_resource)
                            
                            # Put back non-idle resources
                            for temp_resource in temp_resources:
                                await self.available.put(temp_resource)
                            
                            await self._destroy_resource(resource)
                            current_total -= 1
                            
                        except Exception as e:
                            logger.error(f"Idle cleanup error: {e}")
                
            except Exception as e:
                logger.error(f"Idle cleanup task error: {e}")
    
    async def _ensure_min_size(self):
        """Ensure minimum pool size."""
        current_total = len(self.in_use) + self.available.qsize()
        while current_total < self.min_size:
            resource = await self._create_resource()
            if resource:
                await self.available.put(resource)
                current_total += 1
            else:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'total_created': self.total_created,
            'total_errors': self.total_errors,
            'in_use': len(self.in_use),
            'available': self.available.qsize(),
            'total_resources': len(self.in_use) + self.available.qsize(),
            'min_size': self.min_size,
            'max_size': self.max_size
        }


class BatchProcessor:
    """Advanced batch processing for improved throughput."""
    
    def __init__(self, 
                 batch_size: int = 16,
                 max_wait_time: float = 0.1,
                 max_concurrent_batches: int = 4):
        
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches
        
        self.pending_requests = asyncio.Queue()
        self.processing_batches = 0
        self.batch_stats = defaultdict(int)
        self.processing_lock = asyncio.Semaphore(max_concurrent_batches)
        
        # Start batch processor
        asyncio.create_task(self._batch_processor_loop())
        
        logger.info(f"BatchProcessor initialized: batch_size={batch_size}, max_wait={max_wait_time}")
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a request for batch processing."""
        future = asyncio.Future()
        request = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'future': future,
            'submitted_at': time.time()
        }
        
        await self.pending_requests.put(request)
        self.batch_stats['submitted'] += 1
        
        return await future
    
    async def _batch_processor_loop(self):
        """Main batch processing loop."""
        while True:
            try:
                batch = await self._collect_batch()
                if batch:
                    asyncio.create_task(self._process_batch(batch))
                
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Batch processor loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[Dict[str, Any]]:
        """Collect requests into a batch."""
        batch = []
        batch_start = time.time()
        
        # Collect requests until batch is full or timeout
        while (len(batch) < self.batch_size and 
               time.time() - batch_start < self.max_wait_time):
            
            try:
                request = await asyncio.wait_for(
                    self.pending_requests.get(),
                    timeout=0.01
                )
                batch.append(request)
                
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests."""
        async with self.processing_lock:
            self.processing_batches += 1
            batch_id = id(batch)
            
            try:
                logger.debug(f"Processing batch {batch_id} with {len(batch)} requests")
                
                # Group requests by function for better batching
                func_groups = defaultdict(list)
                for request in batch:
                    func_key = f"{request['func'].__name__}_{id(request['func'])}"
                    func_groups[func_key].append(request)
                
                # Process each function group
                for func_key, requests in func_groups.items():
                    await self._process_function_group(requests)
                
                self.batch_stats['processed_batches'] += 1
                self.batch_stats['processed_requests'] += len(batch)
                
            except Exception as e:
                logger.error(f"Batch {batch_id} processing failed: {e}")
                # Set exception for all requests in batch
                for request in batch:
                    if not request['future'].done():
                        request['future'].set_exception(e)
                
                self.batch_stats['failed_batches'] += 1
            
            finally:
                self.processing_batches -= 1
    
    async def _process_function_group(self, requests: List[Dict[str, Any]]):
        """Process requests with the same function."""
        if len(requests) == 1:
            # Single request
            request = requests[0]
            try:
                result = await request['func'](*request['args'], **request['kwargs'])
                request['future'].set_result(result)
            except Exception as e:
                request['future'].set_exception(e)
        
        else:
            # Multiple requests - try to batch if function supports it
            func = requests[0]['func']
            
            if hasattr(func, 'supports_batching') and func.supports_batching:
                # Function supports native batching
                try:
                    all_args = [req['args'] for req in requests]
                    all_kwargs = [req['kwargs'] for req in requests]
                    
                    results = await func.batch_process(all_args, all_kwargs)
                    
                    for request, result in zip(requests, results):
                        request['future'].set_result(result)
                        
                except Exception as e:
                    for request in requests:
                        request['future'].set_exception(e)
            
            else:
                # Process concurrently
                tasks = [
                    func(*req['args'], **req['kwargs'])
                    for req in requests
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for request, result in zip(requests, results):
                    if isinstance(result, Exception):
                        request['future'].set_exception(result)
                    else:
                        request['future'].set_result(result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        stats = dict(self.batch_stats)
        stats.update({
            'processing_batches': self.processing_batches,
            'pending_requests': self.pending_requests.qsize(),
            'batch_size': self.batch_size,
            'max_concurrent_batches': self.max_concurrent_batches
        })
        
        if stats['submitted'] > 0:
            stats['success_rate'] = (
                stats['processed_requests'] / stats['submitted']
            ) * 100
        
        return stats


# Performance monitoring decorator
def monitor_performance(cache_key_func: Callable = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


# Global optimization instances
_model_optimizer: Optional[ModelOptimizer] = None
_batch_processor: Optional[BatchProcessor] = None

def get_model_optimizer(config: OptimizationConfig = None) -> ModelOptimizer:
    """Get global model optimizer instance."""
    global _model_optimizer
    if _model_optimizer is None:
        _model_optimizer = ModelOptimizer(config)
    return _model_optimizer

def get_batch_processor(batch_size: int = 16) -> BatchProcessor:
    """Get global batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor(batch_size=batch_size)
    return _batch_processor