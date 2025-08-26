"""
Advanced performance optimization for OdorDiff-2 scaling.
"""

import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps, lru_cache
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Track and report performance metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.lock = threading.Lock()
    
    def record(self, operation: str, duration: float, success: bool = True):
        """Record performance metric."""
        with self.lock:
            self.metrics[operation].append({
                'duration': duration,
                'success': success,
                'timestamp': time.time()
            })
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics."""
        with self.lock:
            records = self.metrics.get(operation, [])
            if not records:
                return {}
            
            durations = [r['duration'] for r in records if r['success']]
            if not durations:
                return {}
            
            return {
                'count': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'success_rate': len(durations) / len(records)
            }

# Global performance tracker
performance_tracker = PerformanceMetrics()

def performance_monitor(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                performance_tracker.record(operation_name, duration, success)
                logger.debug(f"Operation {operation_name} took {duration:.3f}s (success: {success})")
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                performance_tracker.record(operation_name, duration, success)
                logger.debug(f"Async operation {operation_name} took {duration:.3f}s (success: {success})")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class BatchProcessor:
    """High-performance batch processing for molecule generation."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    @performance_monitor("batch_generation")
    def process_batch(self, prompts: List[str], processing_func: Callable) -> List[Any]:
        """Process batch of prompts with parallel execution."""
        if len(prompts) <= self.batch_size:
            # Small batch - use threading
            futures = []
            for prompt in prompts:
                future = self.thread_pool.submit(processing_func, prompt)
                futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Batch item failed: {e}")
                    results.append(None)
            
            return results
        else:
            # Large batch - use multiprocessing
            batch_chunks = [prompts[i:i+self.batch_size] for i in range(0, len(prompts), self.batch_size)]
            
            futures = []
            for chunk in batch_chunks:
                future = self.process_pool.submit(self._process_chunk, chunk, processing_func)
                futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_results = future.result(timeout=60)  # 60 second timeout
                    results.extend(chunk_results)
                except Exception as e:
                    logger.warning(f"Batch chunk failed: {e}")
            
            return results
    
    def _process_chunk(self, chunk: List[str], processing_func: Callable) -> List[Any]:
        """Process a chunk of prompts."""
        return [processing_func(prompt) for prompt in chunk]
    
    def shutdown(self):
        """Shutdown thread and process pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class IntelligentCache:
    """Intelligent caching system with LRU and performance-based eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.hit_counts = defaultdict(int)
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.cache[key]['timestamp'] > self.ttl:
                    del self.cache[key]
                    del self.access_times[key]
                    return None
                
                # Update access metrics
                self.access_times[key] = time.time()
                self.hit_counts[key] += 1
                
                return self.cache[key]['data']
            
            return None
    
    def put(self, key: str, data: Any) -> None:
        """Put item in cache with intelligent eviction."""
        with self.lock:
            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_least_valuable()
            
            self.cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
            self.access_times[key] = time.time()
    
    def _evict_least_valuable(self):
        """Evict least valuable cache entry based on access pattern."""
        if not self.cache:
            return
        
        # Calculate value score (combination of recency and frequency)
        current_time = time.time()
        scores = {}
        
        for key in self.cache.keys():
            recency_score = 1.0 / (current_time - self.access_times.get(key, 0) + 1)
            frequency_score = self.hit_counts[key]
            scores[key] = recency_score * frequency_score
        
        # Remove lowest scoring item
        least_valuable_key = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[least_valuable_key]
        del self.access_times[least_valuable_key]
        del self.hit_counts[least_valuable_key]
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_counts.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_counts': dict(self.hit_counts),
                'total_hits': sum(self.hit_counts.values())
            }

class AsyncMoleculeGenerator:
    """High-performance async molecule generation system."""
    
    def __init__(self):
        self.cache = IntelligentCache(max_size=500, ttl=1800)  # 30 min TTL
        self.batch_processor = BatchProcessor(max_workers=4)
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
    
    @performance_monitor("async_generation")
    async def generate_molecules_async(self, prompts: List[str]) -> List[List[str]]:
        """Generate molecules asynchronously with high performance."""
        tasks = []
        
        for prompt in prompts:
            task = self._generate_single_async(prompt)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Generation failed: {result}")
                successful_results.append([])  # Empty result for failed generation
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def _generate_single_async(self, prompt: str) -> List[str]:
        """Generate molecules for single prompt asynchronously."""
        async with self.semaphore:
            # Check cache first
            cache_key = f"molecules_{hash(prompt)}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached_result
            
            # Simulate molecule generation (would be actual ML inference)
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Template-based generation (placeholder for actual ML model)
            molecules = self._template_generate(prompt)
            
            # Cache result
            self.cache.put(cache_key, molecules)
            
            return molecules
    
    def _template_generate(self, prompt: str) -> List[str]:
        """Template-based molecule generation (placeholder for ML model)."""
        templates = {
            'floral': ['CC(C)=CCO', 'CCCC(C)=CCO', 'CC(C)(O)CCO'],
            'citrus': ['CC(C)=CC', 'C=C(C)CCC=C(C)C', 'CC1=CCC(CC1)C(C)C'],
            'woody': ['CC(C)(C)c1ccccc1', 'c1ccc2c(c1)cccc2'],
            'fresh': ['CCO', 'CCCO', 'CC(C)O'],
            'spicy': ['COc1ccccc1', 'c1ccc(cc1)C=O', 'CC(=O)c1ccccc1']
        }
        
        prompt_lower = prompt.lower()
        selected_molecules = []
        
        for category, molecules in templates.items():
            if category in prompt_lower:
                selected_molecules.extend(molecules[:2])
        
        if not selected_molecules:
            selected_molecules = templates['fresh'][:2]  # Default
        
        return selected_molecules[:3]  # Return top 3
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'cache_stats': self.cache.stats(),
            'performance_metrics': {
                op: performance_tracker.get_stats(op) 
                for op in ['async_generation', 'batch_generation']
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.batch_processor.shutdown()
        self.cache.clear()

# Global high-performance generator
async_generator = AsyncMoleculeGenerator()
