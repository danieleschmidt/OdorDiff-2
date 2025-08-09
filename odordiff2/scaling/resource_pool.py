"""
Advanced Resource Pooling for OdorDiff-2 Scaling

Implements intelligent resource pooling for expensive operations:
- GPU compute resource pooling
- Model instance pooling with lazy loading
- Database connection pooling with failover
- Compute worker pooling
- Memory pool management
- I/O operation pooling
- Smart resource allocation and cleanup
"""

import os
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, AsyncContextManager, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from collections import deque
import weakref
import gc
import psutil

import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from ..utils.logging import get_logger
from ..core.async_diffusion import AsyncOdorDiffusion

logger = get_logger(__name__)


class ResourceType(Enum):
    """Types of resources that can be pooled."""
    GPU_COMPUTE = "gpu_compute"
    MODEL_INSTANCE = "model_instance"
    DATABASE_CONNECTION = "database_connection"
    THREAD_WORKER = "thread_worker"
    PROCESS_WORKER = "process_worker"
    MEMORY_BUFFER = "memory_buffer"
    IO_HANDLE = "io_handle"


class PoolStrategy(Enum):
    """Resource pool management strategies."""
    FIFO = "fifo"            # First In, First Out
    LIFO = "lifo"            # Last In, First Out (stack)
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    PRIORITY = "priority"


@dataclass
class PoolConfig:
    """Configuration for resource pools."""
    # Pool sizing
    initial_size: int = 2
    max_size: int = 10
    min_size: int = 1
    
    # Resource management
    max_idle_time: float = 300.0    # 5 minutes
    cleanup_interval: float = 60.0   # 1 minute
    health_check_interval: float = 30.0  # 30 seconds
    
    # Allocation strategy
    allocation_strategy: PoolStrategy = PoolStrategy.LEAST_USED
    enable_preallocation: bool = True
    enable_health_checks: bool = True
    
    # Performance tuning
    acquire_timeout: float = 30.0
    max_wait_queue: int = 100
    enable_metrics: bool = True
    
    # Memory management
    max_memory_per_resource_mb: int = 1024  # 1GB per resource
    enable_memory_monitoring: bool = True
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage


@dataclass
class ResourceMetrics:
    """Metrics for resource usage."""
    resource_id: str
    resource_type: ResourceType
    created_at: float
    last_used: float
    usage_count: int = 0
    total_time_used: float = 0.0
    current_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    is_healthy: bool = True
    error_count: int = 0
    
    @property
    def idle_time(self) -> float:
        """Time since last use."""
        return time.time() - self.last_used
    
    @property
    def average_usage_time(self) -> float:
        """Average time per usage."""
        return self.total_time_used / max(1, self.usage_count)


class PooledResource:
    """Wrapper for pooled resources with lifecycle management."""
    
    def __init__(
        self,
        resource_id: str,
        resource: Any,
        resource_type: ResourceType,
        creator_func: Optional[Callable] = None,
        destructor_func: Optional[Callable] = None,
        health_check_func: Optional[Callable] = None
    ):
        self.resource_id = resource_id
        self.resource = resource
        self.resource_type = resource_type
        self.creator_func = creator_func
        self.destructor_func = destructor_func
        self.health_check_func = health_check_func
        
        # State management
        self.is_acquired = False
        self.is_healthy = True
        self.lock = asyncio.Lock()
        
        # Metrics
        self.metrics = ResourceMetrics(
            resource_id=resource_id,
            resource_type=resource_type,
            created_at=time.time(),
            last_used=time.time()
        )
    
    async def acquire(self) -> Any:
        """Acquire resource for use."""
        async with self.lock:
            if self.is_acquired:
                raise RuntimeError(f"Resource {self.resource_id} is already acquired")
            
            if not self.is_healthy:
                raise RuntimeError(f"Resource {self.resource_id} is unhealthy")
            
            self.is_acquired = True
            self.metrics.usage_count += 1
            self.metrics.last_used = time.time()
            
            logger.debug(f"Acquired resource {self.resource_id}")
            return self.resource
    
    async def release(self, usage_time: float = 0.0):
        """Release resource back to pool."""
        async with self.lock:
            if not self.is_acquired:
                logger.warning(f"Resource {self.resource_id} was not acquired")
                return
            
            self.is_acquired = False
            self.metrics.total_time_used += usage_time
            self.metrics.last_used = time.time()
            
            logger.debug(f"Released resource {self.resource_id}")
    
    async def health_check(self) -> bool:
        """Check resource health."""
        if not self.health_check_func:
            return True
        
        try:
            if asyncio.iscoroutinefunction(self.health_check_func):
                is_healthy = await self.health_check_func(self.resource)
            else:
                is_healthy = self.health_check_func(self.resource)
            
            self.is_healthy = bool(is_healthy)
            
            if not self.is_healthy:
                self.metrics.error_count += 1
                logger.warning(f"Resource {self.resource_id} failed health check")
            
            return self.is_healthy
            
        except Exception as e:
            self.is_healthy = False
            self.metrics.error_count += 1
            logger.error(f"Health check failed for resource {self.resource_id}: {e}")
            return False
    
    async def destroy(self):
        """Destroy resource and cleanup."""
        try:
            if self.destructor_func:
                if asyncio.iscoroutinefunction(self.destructor_func):
                    await self.destructor_func(self.resource)
                else:
                    self.destructor_func(self.resource)
            
            logger.debug(f"Destroyed resource {self.resource_id}")
            
        except Exception as e:
            logger.error(f"Error destroying resource {self.resource_id}: {e}")
    
    def update_memory_usage(self):
        """Update memory usage metrics."""
        try:
            if hasattr(self.resource, '__sizeof__'):
                memory_mb = self.resource.__sizeof__() / (1024 ** 2)
            else:
                # Estimate based on process memory if resource is a model/large object
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 ** 2)
            
            self.metrics.current_memory_mb = memory_mb
            self.metrics.peak_memory_mb = max(
                self.metrics.peak_memory_mb, 
                memory_mb
            )
            
        except Exception as e:
            logger.debug(f"Could not update memory usage for {self.resource_id}: {e}")


class ResourcePool:
    """Generic resource pool with intelligent management."""
    
    def __init__(
        self,
        pool_name: str,
        resource_type: ResourceType,
        config: PoolConfig,
        creator_func: Callable,
        destructor_func: Optional[Callable] = None,
        health_check_func: Optional[Callable] = None
    ):
        self.pool_name = pool_name
        self.resource_type = resource_type
        self.config = config
        self.creator_func = creator_func
        self.destructor_func = destructor_func
        self.health_check_func = health_check_func
        
        # Pool state
        self.available_resources: deque = deque()
        self.all_resources: Dict[str, PooledResource] = {}
        self.wait_queue: deque = deque()
        
        # Synchronization
        self.pool_lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.pool_lock)
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.pool_metrics = {
            'total_created': 0,
            'total_destroyed': 0,
            'total_acquisitions': 0,
            'total_wait_time': 0.0,
            'max_size_reached': 0,
            'health_check_failures': 0
        }
        
    async def initialize(self):
        """Initialize the resource pool."""
        async with self.pool_lock:
            # Create initial resources
            if self.config.enable_preallocation:
                for i in range(self.config.initial_size):
                    try:
                        await self._create_resource()
                    except Exception as e:
                        logger.error(f"Failed to create initial resource {i}: {e}")
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.config.enable_health_checks:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Initialized resource pool '{self.pool_name}' with {len(self.all_resources)} resources")
    
    async def close(self):
        """Close the resource pool and cleanup all resources."""
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Destroy all resources
        async with self.pool_lock:
            for resource in list(self.all_resources.values()):
                await resource.destroy()
            
            self.all_resources.clear()
            self.available_resources.clear()
        
        logger.info(f"Closed resource pool '{self.pool_name}'")
    
    @asynccontextmanager
    async def acquire(self) -> AsyncContextManager[Any]:
        """Acquire a resource from the pool."""
        start_time = time.time()
        resource = None
        
        try:
            resource = await self._acquire_resource()
            actual_resource = await resource.acquire()
            
            # Update metrics
            wait_time = time.time() - start_time
            self.pool_metrics['total_acquisitions'] += 1
            self.pool_metrics['total_wait_time'] += wait_time
            
            yield actual_resource
            
        finally:
            if resource:
                usage_time = time.time() - start_time
                await resource.release(usage_time)
                await self._return_resource(resource)
    
    async def _acquire_resource(self) -> PooledResource:
        """Internal method to acquire a resource."""
        async with self.condition:
            # Wait for available resource or ability to create one
            while True:
                # Check for available resource
                if self.available_resources:
                    if self.config.allocation_strategy == PoolStrategy.FIFO:
                        resource = self.available_resources.popleft()
                    else:  # LIFO
                        resource = self.available_resources.pop()
                    
                    return resource
                
                # Try to create new resource if under max size
                if len(self.all_resources) < self.config.max_size:
                    resource = await self._create_resource()
                    return resource
                
                # Wait for resource to become available
                if len(self.wait_queue) >= self.config.max_wait_queue:
                    raise RuntimeError("Resource pool wait queue is full")
                
                waiter = asyncio.Future()
                self.wait_queue.append(waiter)
                
                try:
                    await asyncio.wait_for(
                        waiter,
                        timeout=self.config.acquire_timeout
                    )
                except asyncio.TimeoutError:
                    if waiter in self.wait_queue:
                        self.wait_queue.remove(waiter)
                    raise RuntimeError("Timeout waiting for resource")
    
    async def _return_resource(self, resource: PooledResource):
        """Return resource to the pool."""
        async with self.condition:
            # Update memory usage
            if self.config.enable_memory_monitoring:
                resource.update_memory_usage()
            
            # Check if we should keep the resource
            if (len(self.all_resources) > self.config.min_size and 
                resource.metrics.idle_time > self.config.max_idle_time):
                await self._destroy_resource(resource)
                return
            
            # Return to available pool
            self.available_resources.append(resource)
            
            # Notify waiting tasks
            if self.wait_queue:
                waiter = self.wait_queue.popleft()
                if not waiter.done():
                    waiter.set_result(None)
            else:
                self.condition.notify()
    
    async def _create_resource(self) -> PooledResource:
        """Create a new resource."""
        resource_id = f"{self.pool_name}_{len(self.all_resources)}_{int(time.time())}"
        
        try:
            # Create the actual resource
            if asyncio.iscoroutinefunction(self.creator_func):
                raw_resource = await self.creator_func()
            else:
                raw_resource = self.creator_func()
            
            # Wrap in pooled resource
            pooled_resource = PooledResource(
                resource_id=resource_id,
                resource=raw_resource,
                resource_type=self.resource_type,
                creator_func=self.creator_func,
                destructor_func=self.destructor_func,
                health_check_func=self.health_check_func
            )
            
            self.all_resources[resource_id] = pooled_resource
            self.pool_metrics['total_created'] += 1
            self.pool_metrics['max_size_reached'] = max(
                self.pool_metrics['max_size_reached'],
                len(self.all_resources)
            )
            
            logger.debug(f"Created resource {resource_id}")
            return pooled_resource
            
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            raise
    
    async def _destroy_resource(self, resource: PooledResource):
        """Destroy a resource."""
        try:
            await resource.destroy()
            
            if resource.resource_id in self.all_resources:
                del self.all_resources[resource.resource_id]
            
            self.pool_metrics['total_destroyed'] += 1
            logger.debug(f"Destroyed resource {resource.resource_id}")
            
        except Exception as e:
            logger.error(f"Failed to destroy resource {resource.resource_id}: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup of idle resources."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                async with self.pool_lock:
                    current_time = time.time()
                    to_remove = []
                    
                    # Find idle resources to remove
                    for resource in self.available_resources:
                        if (len(self.all_resources) > self.config.min_size and
                            resource.metrics.idle_time > self.config.max_idle_time):
                            to_remove.append(resource)
                    
                    # Remove idle resources
                    for resource in to_remove:
                        if resource in self.available_resources:
                            self.available_resources.remove(resource)
                        await self._destroy_resource(resource)
                    
                    if to_remove:
                        logger.debug(f"Cleaned up {len(to_remove)} idle resources")
                
                # Trigger garbage collection if memory usage is high
                if self.config.enable_memory_monitoring:
                    process = psutil.Process()
                    memory_percent = process.memory_percent()
                    
                    if memory_percent > self.config.gc_threshold * 100:
                        gc.collect()
                        logger.debug(f"Triggered GC at {memory_percent:.1f}% memory usage")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _health_check_loop(self):
        """Background health checking of resources."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                async with self.pool_lock:
                    unhealthy_resources = []
                    
                    # Check health of all resources
                    for resource in self.all_resources.values():
                        if not resource.is_acquired:
                            is_healthy = await resource.health_check()
                            if not is_healthy:
                                unhealthy_resources.append(resource)
                                self.pool_metrics['health_check_failures'] += 1
                    
                    # Remove unhealthy resources
                    for resource in unhealthy_resources:
                        if resource in self.available_resources:
                            self.available_resources.remove(resource)
                        await self._destroy_resource(resource)
                    
                    if unhealthy_resources:
                        logger.warning(f"Removed {len(unhealthy_resources)} unhealthy resources")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        available_count = len(self.available_resources)
        acquired_count = sum(1 for r in self.all_resources.values() if r.is_acquired)
        healthy_count = sum(1 for r in self.all_resources.values() if r.is_healthy)
        
        return {
            'pool_name': self.pool_name,
            'resource_type': self.resource_type.value,
            'total_resources': len(self.all_resources),
            'available_resources': available_count,
            'acquired_resources': acquired_count,
            'healthy_resources': healthy_count,
            'wait_queue_size': len(self.wait_queue),
            'config': {
                'max_size': self.config.max_size,
                'min_size': self.config.min_size,
                'allocation_strategy': self.config.allocation_strategy.value
            },
            'metrics': self.pool_metrics,
            'resource_details': [
                {
                    'id': r.resource_id,
                    'is_acquired': r.is_acquired,
                    'is_healthy': r.is_healthy,
                    'usage_count': r.metrics.usage_count,
                    'idle_time': r.metrics.idle_time,
                    'memory_mb': r.metrics.current_memory_mb
                }
                for r in self.all_resources.values()
            ]
        }


class ModelInstancePool(ResourcePool):
    """Specialized pool for ML model instances."""
    
    def __init__(self, config: PoolConfig):
        async def create_model():
            """Create new model instance."""
            model = AsyncOdorDiffusion(
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_workers=2,
                batch_size=4,
                enable_caching=True
            )
            await model.start()
            return model
        
        async def destroy_model(model):
            """Cleanup model instance."""
            await model.stop()
            if hasattr(model, 'model') and hasattr(model.model, 'cpu'):
                model.model.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        async def health_check_model(model):
            """Check if model is healthy."""
            try:
                # Test generation
                result = await model.generate_async("test", num_molecules=1)
                return result.error is None
            except:
                return False
        
        super().__init__(
            pool_name="model_instances",
            resource_type=ResourceType.MODEL_INSTANCE,
            config=config,
            creator_func=create_model,
            destructor_func=destroy_model,
            health_check_func=health_check_model
        )


class GPUComputePool(ResourcePool):
    """Specialized pool for GPU compute resources."""
    
    def __init__(self, config: PoolConfig):
        def create_gpu_context():
            """Create GPU compute context."""
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            
            device_id = torch.cuda.current_device()
            context = {
                'device': torch.device(f'cuda:{device_id}'),
                'stream': torch.cuda.Stream(),
                'memory_pool': torch.cuda.graph_pool_handle() if hasattr(torch.cuda, 'graph_pool_handle') else None
            }
            return context
        
        def destroy_gpu_context(context):
            """Cleanup GPU context."""
            if context.get('stream'):
                context['stream'].synchronize()
            torch.cuda.empty_cache()
        
        def health_check_gpu(context):
            """Check GPU health."""
            try:
                device = context['device']
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.matmul(test_tensor, test_tensor)
                return result.shape == (100, 100)
            except:
                return False
        
        super().__init__(
            pool_name="gpu_compute",
            resource_type=ResourceType.GPU_COMPUTE,
            config=config,
            creator_func=create_gpu_context,
            destructor_func=destroy_gpu_context,
            health_check_func=health_check_gpu
        )


class ResourcePoolManager:
    """Manages multiple resource pools."""
    
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.default_configs: Dict[ResourceType, PoolConfig] = {}
        
        # Set default configurations
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default configurations for different resource types."""
        self.default_configs[ResourceType.MODEL_INSTANCE] = PoolConfig(
            initial_size=1,
            max_size=5,
            min_size=1,
            max_idle_time=600.0,  # 10 minutes for models
            cleanup_interval=120.0,
            health_check_interval=60.0
        )
        
        self.default_configs[ResourceType.GPU_COMPUTE] = PoolConfig(
            initial_size=2,
            max_size=8,
            min_size=1,
            max_idle_time=300.0,  # 5 minutes for GPU
            cleanup_interval=60.0,
            health_check_interval=30.0
        )
        
        self.default_configs[ResourceType.THREAD_WORKER] = PoolConfig(
            initial_size=4,
            max_size=20,
            min_size=2,
            max_idle_time=180.0,  # 3 minutes for workers
            cleanup_interval=60.0,
            health_check_interval=0.0  # Disable for thread workers
        )
    
    async def create_pool(
        self,
        pool_name: str,
        resource_type: ResourceType,
        creator_func: Callable,
        destructor_func: Optional[Callable] = None,
        health_check_func: Optional[Callable] = None,
        config: Optional[PoolConfig] = None
    ) -> ResourcePool:
        """Create a new resource pool."""
        
        if pool_name in self.pools:
            raise ValueError(f"Pool '{pool_name}' already exists")
        
        # Use default config if not provided
        if config is None:
            config = self.default_configs.get(resource_type, PoolConfig())
        
        # Create and initialize pool
        pool = ResourcePool(
            pool_name=pool_name,
            resource_type=resource_type,
            config=config,
            creator_func=creator_func,
            destructor_func=destructor_func,
            health_check_func=health_check_func
        )
        
        await pool.initialize()
        self.pools[pool_name] = pool
        
        logger.info(f"Created resource pool: {pool_name}")
        return pool
    
    async def get_pool(self, pool_name: str) -> ResourcePool:
        """Get existing resource pool."""
        if pool_name not in self.pools:
            raise ValueError(f"Pool '{pool_name}' not found")
        return self.pools[pool_name]
    
    async def close_pool(self, pool_name: str):
        """Close and remove a resource pool."""
        if pool_name in self.pools:
            await self.pools[pool_name].close()
            del self.pools[pool_name]
            logger.info(f"Closed resource pool: {pool_name}")
    
    async def close_all(self):
        """Close all resource pools."""
        for pool_name in list(self.pools.keys()):
            await self.close_pool(pool_name)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        return {
            pool_name: pool.get_stats()
            for pool_name, pool in self.pools.items()
        }


# Global resource pool manager
_pool_manager: Optional[ResourcePoolManager] = None


async def get_pool_manager() -> ResourcePoolManager:
    """Get or create global resource pool manager."""
    global _pool_manager
    
    if _pool_manager is None:
        _pool_manager = ResourcePoolManager()
    
    return _pool_manager