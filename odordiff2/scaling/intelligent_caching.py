"""
Intelligent multi-tier caching system with AI-driven optimization.
"""

import asyncio
import hashlib
import json
import pickle
import time
import threading
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import weakref

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    priority: int = 1  # 1=low, 2=medium, 3=high
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all values."""
        pass
    
    @abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_memory = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                if entry.is_expired():
                    await self._remove_entry(key)
                    self._misses += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.update_access()
                self._hits += 1
                return entry.value
            
            self._misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in memory cache."""
        async with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Fallback estimate
            
            # Remove existing entry if present
            if key in self._cache:
                await self._remove_entry(key)
            
            # Check if we need to evict entries
            await self._evict_if_needed(size_bytes)
            
            # Create and store entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            self._cache[key] = entry
            self._total_memory += size_bytes
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all values."""
        async with self._lock:
            self._cache.clear()
            self._total_memory = 0
            return True
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "backend": "memory",
                "entries": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_bytes": self._total_memory,
                "max_memory_bytes": self.max_memory_bytes,
                "memory_usage_percent": (self._total_memory / self.max_memory_bytes) * 100,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions
            }
    
    async def _remove_entry(self, key: str):
        """Remove entry and update memory tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_memory -= entry.size_bytes
    
    async def _evict_if_needed(self, new_size: int):
        """Evict entries if needed to make space."""
        # Check size limit
        while len(self._cache) >= self.max_size and self._cache:
            oldest_key = next(iter(self._cache))
            await self._remove_entry(oldest_key)
            self._evictions += 1
        
        # Check memory limit
        while (self._total_memory + new_size) > self.max_memory_bytes and self._cache:
            oldest_key = next(iter(self._cache))
            await self._remove_entry(oldest_key)
            self._evictions += 1


class RedisCacheBackend(CacheBackend):
    """Redis cache backend with compression."""
    
    def __init__(self, redis_client, compression: bool = True):
        self.redis_client = redis_client
        self.compression = compression
        self.prefix = "odordiff_cache:"
        
        # Statistics
        self._local_hits = 0
        self._local_misses = 0
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize and optionally compress value."""
        data = pickle.dumps(value)
        
        if self.compression:
            data = zlib.compress(data)
        
        return data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize and optionally decompress value."""
        if self.compression:
            data = zlib.decompress(data)
        
        return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)
            
            if data is None:
                self._local_misses += 1
                return None
            
            value = self._deserialize(data)
            self._local_hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self._local_misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = self._serialize(value)
            
            if ttl:
                await self.redis_client.setex(redis_key, int(ttl), data)
            else:
                await self.redis_client.set(redis_key, data)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all values."""
        try:
            pattern = f"{self.prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                await self.redis_client.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            return False
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = await self.redis_client.info()
            
            total_requests = self._local_hits + self._local_misses
            hit_rate = self._local_hits / total_requests if total_requests > 0 else 0
            
            return {
                "backend": "redis",
                "connected": True,
                "memory_usage_bytes": info.get("used_memory", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "local_hits": self._local_hits,
                "local_misses": self._local_misses,
                "local_hit_rate": hit_rate,
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {
                "backend": "redis",
                "connected": False,
                "error": str(e)
            }


class SmartCacheOrchestrator:
    """Intelligent multi-tier cache orchestrator with ML-driven optimization."""
    
    def __init__(
        self,
        l1_backend: CacheBackend,  # Fast tier (memory)
        l2_backend: Optional[CacheBackend] = None,  # Persistent tier (Redis)
        enable_prediction: bool = True
    ):
        self.l1_backend = l1_backend
        self.l2_backend = l2_backend
        self.enable_prediction = enable_prediction
        
        # Access pattern analysis
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._key_predictions: Dict[str, float] = {}  # Key -> predicted next access time
        self._pattern_lock = asyncio.Lock()
        
        # Cache policies
        self._promotion_threshold = 3  # Promote to L1 after N accesses
        self._demotion_threshold = 0.1  # Demote from L1 if access rate < 0.1/hour
        
        # Statistics
        self._l1_hits = 0
        self._l2_hits = 0
        self._total_misses = 0
        self._promotions = 0
        self._demotions = 0
        
        # Start background optimization
        self._optimization_task = None
        if enable_prediction:
            asyncio.create_task(self._start_optimization_loop())
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-tier cache."""
        # Record access for pattern analysis
        await self._record_access(key)
        
        # Try L1 cache first
        value = await self.l1_backend.get(key)
        if value is not None:
            self._l1_hits += 1
            return value
        
        # Try L2 cache if available
        if self.l2_backend:
            value = await self.l2_backend.get(key)
            if value is not None:
                self._l2_hits += 1
                
                # Consider promoting to L1
                await self._consider_promotion(key, value)
                return value
        
        self._total_misses += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tier_preference: str = "auto"
    ) -> bool:
        """Set value in appropriate cache tier."""
        success = True
        
        # Determine which tiers to use
        if tier_preference == "auto":
            use_l1, use_l2 = await self._decide_cache_tiers(key, value)
        elif tier_preference == "l1_only":
            use_l1, use_l2 = True, False
        elif tier_preference == "l2_only":
            use_l1, use_l2 = False, True
        else:  # "both"
            use_l1, use_l2 = True, True
        
        # Set in appropriate tiers
        if use_l1:
            success &= await self.l1_backend.set(key, value, ttl)
        
        if use_l2 and self.l2_backend:
            success &= await self.l2_backend.set(key, value, ttl)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers."""
        success = True
        
        success &= await self.l1_backend.delete(key)
        
        if self.l2_backend:
            success &= await self.l2_backend.delete(key)
        
        return success
    
    async def clear(self) -> bool:
        """Clear all cache tiers."""
        success = True
        
        success &= await self.l1_backend.clear()
        
        if self.l2_backend:
            success &= await self.l2_backend.clear()
        
        return success
    
    async def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = await self.l1_backend.stats()
        l2_stats = await self.l2_backend.stats() if self.l2_backend else {}
        
        total_requests = self._l1_hits + self._l2_hits + self._total_misses
        
        return {
            "total_requests": total_requests,
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "total_misses": self._total_misses,
            "overall_hit_rate": (self._l1_hits + self._l2_hits) / total_requests if total_requests > 0 else 0,
            "l1_hit_rate": self._l1_hits / total_requests if total_requests > 0 else 0,
            "promotions": self._promotions,
            "demotions": self._demotions,
            "l1_backend": l1_stats,
            "l2_backend": l2_stats,
            "tracked_patterns": len(self._access_patterns)
        }
    
    async def _record_access(self, key: str):
        """Record access pattern for key."""
        if not self.enable_prediction:
            return
        
        current_time = time.time()
        
        async with self._pattern_lock:
            # Keep last 100 accesses for pattern analysis
            if len(self._access_patterns[key]) >= 100:
                self._access_patterns[key].pop(0)
            
            self._access_patterns[key].append(current_time)
    
    async def _decide_cache_tiers(self, key: str, value: Any) -> Tuple[bool, bool]:
        """Decide which cache tiers to use for a key."""
        # For new keys or simple policy, cache in both tiers
        if key not in self._access_patterns:
            return True, True
        
        # Analyze access pattern
        accesses = self._access_patterns[key]
        if len(accesses) < 2:
            return True, True
        
        # Calculate access frequency (accesses per hour)
        time_span = accesses[-1] - accesses[0]
        if time_span == 0:
            return True, True
        
        access_rate = len(accesses) / (time_span / 3600)
        
        # High-frequency keys go to L1, all keys go to L2
        use_l1 = access_rate > self._promotion_threshold
        use_l2 = True
        
        return use_l1, use_l2
    
    async def _consider_promotion(self, key: str, value: Any):
        """Consider promoting key from L2 to L1."""
        if key not in self._access_patterns:
            return
        
        accesses = self._access_patterns[key]
        if len(accesses) < self._promotion_threshold:
            return
        
        # Check recent access frequency
        recent_accesses = [a for a in accesses if time.time() - a < 3600]  # Last hour
        
        if len(recent_accesses) >= self._promotion_threshold:
            await self.l1_backend.set(key, value)
            self._promotions += 1
            logger.debug(f"Promoted key to L1: {key}")
    
    async def _start_optimization_loop(self):
        """Start background optimization loop."""
        try:
            while True:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._optimize_cache()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cache optimization error: {e}")
    
    async def _optimize_cache(self):
        """Perform cache optimization based on access patterns."""
        if not self.enable_prediction:
            return
        
        current_time = time.time()
        
        async with self._pattern_lock:
            # Identify keys for demotion from L1
            l1_stats = await self.l1_backend.stats()
            
            # Get keys that haven't been accessed recently
            stale_keys = []
            for key, accesses in self._access_patterns.items():
                if not accesses:
                    continue
                
                # Check if key hasn't been accessed in the last hour
                last_access = accesses[-1]
                if current_time - last_access > 3600:  # 1 hour
                    stale_keys.append(key)
            
            # Demote stale keys from L1
            for key in stale_keys[:10]:  # Limit demotions per cycle
                success = await self.l1_backend.delete(key)
                if success:
                    self._demotions += 1
                    logger.debug(f"Demoted key from L1: {key}")
        
        logger.debug("Cache optimization cycle completed")


class CacheManager:
    """High-level cache manager with automatic setup and optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._orchestrator: Optional[SmartCacheOrchestrator] = None
        self._setup_lock = asyncio.Lock()
    
    async def setup(self, redis_url: Optional[str] = None):
        """Setup cache backends and orchestrator."""
        async with self._setup_lock:
            if self._orchestrator is not None:
                return
            
            # Setup L1 (memory) backend
            l1_backend = MemoryCacheBackend(
                max_size=self.config.get("l1_max_size", 1000),
                max_memory_mb=self.config.get("l1_max_memory_mb", 100)
            )
            
            # Setup L2 (Redis) backend if URL provided
            l2_backend = None
            if redis_url:
                try:
                    import aioredis
                    redis_client = aioredis.from_url(redis_url)
                    await redis_client.ping()
                    l2_backend = RedisCacheBackend(
                        redis_client,
                        compression=self.config.get("l2_compression", True)
                    )
                    logger.info("Connected to Redis for L2 cache")
                except Exception as e:
                    logger.warning(f"Could not setup Redis cache: {e}")
            
            # Create orchestrator
            self._orchestrator = SmartCacheOrchestrator(
                l1_backend=l1_backend,
                l2_backend=l2_backend,
                enable_prediction=self.config.get("enable_prediction", True)
            )
            
            logger.info("Cache manager setup complete")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self._orchestrator is None:
            await self.setup()
        return await self._orchestrator.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        if self._orchestrator is None:
            await self.setup()
        return await self._orchestrator.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if self._orchestrator is None:
            await self.setup()
        return await self._orchestrator.delete(key)
    
    async def clear(self) -> bool:
        """Clear all caches."""
        if self._orchestrator is None:
            await self.setup()
        return await self._orchestrator.clear()
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._orchestrator is None:
            await self.setup()
        return await self._orchestrator.stats()


# Global cache manager
cache_manager = CacheManager()


# Decorator for easy caching
def cached(ttl: Optional[float] = None, key_prefix: str = ""):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{key_prefix}{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try cache first
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Call function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await cache_manager.set(cache_key, result, ttl)
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator