"""
Multi-Tier Caching System for OdorDiff-2 Scaling

Implements a sophisticated caching hierarchy:
1. L1: In-memory LRU cache (fastest)
2. L2: Redis distributed cache (shared across instances)  
3. L3: CDN edge cache (for static content and frequent responses)

Features:
- Cache warmup strategies
- Intelligent cache eviction policies
- Cache hit/miss analytics
- Automatic cache coherence
- Compression and serialization
"""

import os
import time
import hashlib
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import threading
import weakref
from collections import OrderedDict
from enum import Enum

from ..models.molecule import Molecule
from ..utils.logging import get_logger
from .redis_config import get_redis_manager, get_redis_serializer

logger = get_logger(__name__)


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_CDN = "l3_cdn"


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    data: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    compression_ratio: float = 1.0
    source_level: CacheLevel = CacheLevel.L1_MEMORY
    
    @property
    def age(self) -> float:
        """Age of the entry in seconds."""
        return time.time() - self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics and metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


class LRUCache:
    """High-performance LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired:
                    del self._cache[key]
                    self._stats.misses += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                
                self._stats.hits += 1
                self._stats.update_hit_rate()
                
                return entry
            
            self._stats.misses += 1
            self._stats.update_hit_rate()
            return None
    
    def put(self, key: str, data: Any, ttl: Optional[float] = None) -> bool:
        """Put item into cache."""
        with self._lock:
            # Estimate size
            import sys
            size_bytes = sys.getsizeof(data)
            
            # Create entry
            entry = CacheEntry(
                data=data,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            # Check if we need to evict
            self._evict_if_needed(size_bytes)
            
            # Add new entry
            if key in self._cache:
                # Update existing
                old_entry = self._cache[key]
                self._stats.size_bytes -= old_entry.size_bytes
                
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Update stats
            self._stats.size_bytes += size_bytes
            self._stats.entry_count = len(self._cache)
            
            return True
    
    def _evict_if_needed(self, new_entry_size: int):
        """Evict entries if needed based on size and memory constraints."""
        # Check size limit
        while len(self._cache) >= self.max_size:
            self._evict_lru()
        
        # Check memory limit
        while (self._stats.size_bytes + new_entry_size) > self.max_memory_bytes:
            if not self._evict_lru():
                break  # Can't evict anymore
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False
        
        # Get LRU key
        lru_key = next(iter(self._cache))
        entry = self._cache[lru_key]
        
        # Remove and update stats
        del self._cache[lru_key]
        self._stats.size_bytes -= entry.size_bytes
        self._stats.evictions += 1
        self._stats.entry_count = len(self._cache)
        
        logger.debug(f"Evicted LRU entry: {lru_key}")
        return True
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return self._stats


class RedisCache:
    """Redis-based distributed cache layer."""
    
    def __init__(self, key_prefix: str = "odordiff2:cache:"):
        self.key_prefix = key_prefix
        self._redis_manager = None
        self._serializer = None
        self._stats = CacheStats()
    
    async def initialize(self):
        """Initialize Redis connections."""
        self._redis_manager = await get_redis_manager()
        self._serializer = get_redis_serializer()
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from Redis cache."""
        try:
            if not self._redis_manager:
                await self.initialize()
            
            redis_key = self._make_key(key)
            client = self._redis_manager.get_async_client()
            
            # Get data and metadata
            start_time = time.time()
            data = await client.hgetall(redis_key)
            
            if not data:
                self._stats.misses += 1
                return None
            
            # Deserialize
            entry_data = self._serializer.deserialize(data.get('data', b''))
            created_at = float(data.get('created_at', 0))
            ttl = float(data.get('ttl', 0)) if data.get('ttl') else None
            access_count = int(data.get('access_count', 0))
            
            # Check expiration
            if ttl and time.time() > (created_at + ttl):
                await client.delete(redis_key)
                self._stats.misses += 1
                return None
            
            # Update access metadata
            await client.hset(redis_key, mapping={
                'last_accessed': time.time(),
                'access_count': access_count + 1
            })
            
            # Create cache entry
            entry = CacheEntry(
                data=entry_data,
                created_at=created_at,
                last_accessed=time.time(),
                access_count=access_count + 1,
                ttl=ttl,
                source_level=CacheLevel.L2_REDIS
            )
            
            # Update stats
            access_time = time.time() - start_time
            self._stats.hits += 1
            self._stats.avg_access_time = (
                (self._stats.avg_access_time * (self._stats.hits - 1) + access_time) 
                / self._stats.hits
            )
            self._stats.update_hit_rate()
            
            return entry
            
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self._stats.misses += 1
            return None
    
    async def put(self, key: str, data: Any, ttl: Optional[float] = None) -> bool:
        """Put item into Redis cache."""
        try:
            if not self._redis_manager:
                await self.initialize()
            
            redis_key = self._make_key(key)
            client = self._redis_manager.get_async_client()
            
            # Serialize data
            serialized_data = self._serializer.serialize(data)
            
            # Prepare metadata
            now = time.time()
            metadata = {
                'data': serialized_data,
                'created_at': now,
                'last_accessed': now,
                'access_count': 0
            }
            
            if ttl:
                metadata['ttl'] = ttl
            
            # Store in Redis
            await client.hset(redis_key, mapping=metadata)
            
            # Set expiration if specified
            if ttl:
                await client.expire(redis_key, int(ttl))
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache put error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        try:
            if not self._redis_manager:
                await self.initialize()
            
            redis_key = self._make_key(key)
            client = self._redis_manager.get_async_client()
            
            result = await client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str = "*") -> int:
        """Clear keys matching pattern."""
        try:
            if not self._redis_manager:
                await self.initialize()
            
            client = self._redis_manager.get_async_client()
            pattern_key = self._make_key(pattern)
            
            keys = await client.keys(pattern_key)
            if keys:
                return await client.delete(*keys)
            
            return 0
            
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            return 0
    
    def get_stats(self) -> CacheStats:
        """Get Redis cache statistics."""
        return self._stats


class CDNCache:
    """CDN integration for static content and frequent responses."""
    
    def __init__(self, cdn_base_url: Optional[str] = None):
        self.cdn_base_url = cdn_base_url or os.getenv('CDN_BASE_URL', '')
        self.enabled = bool(self.cdn_base_url)
        self._stats = CacheStats()
    
    def is_cacheable(self, key: str, data: Any) -> bool:
        """Determine if content should be cached on CDN."""
        # Cache molecular visualizations, images, static responses
        cacheable_types = ['visualization', 'image', 'static', 'frequent']
        
        return any(cache_type in key.lower() for cache_type in cacheable_types)
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from CDN cache."""
        if not self.enabled:
            return None
        
        try:
            import aiohttp
            
            cdn_url = f"{self.cdn_base_url}/{self._hash_key(key)}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(cdn_url) as response:
                    if response.status == 200:
                        data = await response.read()
                        
                        entry = CacheEntry(
                            data=data,
                            created_at=time.time(),
                            last_accessed=time.time(),
                            source_level=CacheLevel.L3_CDN
                        )
                        
                        self._stats.hits += 1
                        self._stats.update_hit_rate()
                        
                        return entry
                    
                    self._stats.misses += 1
                    return None
                    
        except Exception as e:
            logger.error(f"CDN cache get error: {e}")
            self._stats.misses += 1
            return None
    
    async def put(self, key: str, data: Any, ttl: Optional[float] = None) -> bool:
        """Put item into CDN cache."""
        if not self.enabled or not self.is_cacheable(key, data):
            return False
        
        try:
            # Implementation would depend on CDN provider API
            # This is a placeholder for CDN upload logic
            logger.debug(f"Would upload to CDN: {key}")
            return True
            
        except Exception as e:
            logger.error(f"CDN cache put error: {e}")
            return False
    
    def _hash_key(self, key: str) -> str:
        """Create hash-based CDN key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def get_stats(self) -> CacheStats:
        """Get CDN cache statistics."""
        return self._stats


class MultiTierCache:
    """Multi-tier caching system coordinator."""
    
    def __init__(
        self,
        l1_max_size: int = 1000,
        l1_max_memory_mb: int = 512,
        default_ttl: int = 3600,
        enable_cdn: bool = True
    ):
        # Cache layers
        self.l1_cache = LRUCache(l1_max_size, l1_max_memory_mb)
        self.l2_cache = RedisCache()
        self.l3_cache = CDNCache() if enable_cdn else None
        
        # Configuration
        self.default_ttl = default_ttl
        
        # Analytics
        self._cache_analytics = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0,
            'total_requests': 0
        }
        
        # Cache warming
        self._warmup_tasks: Dict[str, asyncio.Task] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache hierarchy (L1 -> L2 -> L3)."""
        self._cache_analytics['total_requests'] += 1
        
        # Try L1 first (fastest)
        entry = self.l1_cache.get(key)
        if entry:
            self._cache_analytics['l1_hits'] += 1
            logger.debug(f"L1 cache hit: {key}")
            return entry.data
        
        self._cache_analytics['l1_misses'] += 1
        
        # Try L2 (Redis)
        entry = await self.l2_cache.get(key)
        if entry:
            self._cache_analytics['l2_hits'] += 1
            logger.debug(f"L2 cache hit: {key}")
            
            # Populate L1 for future requests
            self.l1_cache.put(key, entry.data, entry.ttl)
            return entry.data
        
        self._cache_analytics['l2_misses'] += 1
        
        # Try L3 (CDN) if available
        if self.l3_cache:
            entry = await self.l3_cache.get(key)
            if entry:
                self._cache_analytics['l3_hits'] += 1
                logger.debug(f"L3 cache hit: {key}")
                
                # Populate L1 and L2 for future requests
                self.l1_cache.put(key, entry.data, self.default_ttl)
                await self.l2_cache.put(key, entry.data, self.default_ttl)
                return entry.data
            
            self._cache_analytics['l3_misses'] += 1
        
        logger.debug(f"Cache miss across all tiers: {key}")
        return None
    
    async def put(
        self, 
        key: str, 
        data: Any, 
        ttl: Optional[float] = None,
        levels: List[CacheLevel] = None
    ) -> bool:
        """Put item into specified cache levels."""
        if ttl is None:
            ttl = self.default_ttl
        
        if levels is None:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
            if self.l3_cache and self.l3_cache.is_cacheable(key, data):
                levels.append(CacheLevel.L3_CDN)
        
        success = True
        
        # Store in specified levels
        if CacheLevel.L1_MEMORY in levels:
            success &= self.l1_cache.put(key, data, ttl)
        
        if CacheLevel.L2_REDIS in levels:
            success &= await self.l2_cache.put(key, data, ttl)
        
        if CacheLevel.L3_CDN in levels and self.l3_cache:
            success &= await self.l3_cache.put(key, data, ttl)
        
        if success:
            logger.debug(f"Cached item in levels {levels}: {key}")
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete item from all cache levels."""
        results = []
        
        # Delete from L1
        try:
            # L1 cache doesn't have explicit delete, just clear entry
            self.l1_cache._cache.pop(key, None)
            results.append(True)
        except:
            results.append(False)
        
        # Delete from L2
        results.append(await self.l2_cache.delete(key))
        
        # Delete from L3 (if available)
        if self.l3_cache:
            # CDN deletion would require provider-specific API
            results.append(True)  # Placeholder
        
        return any(results)
    
    async def warm_up(
        self,
        prompts: List[str],
        generator_func: Callable[[str], Any],
        concurrency: int = 5
    ):
        """Warm up cache with common prompts."""
        logger.info(f"Starting cache warmup with {len(prompts)} prompts")
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def warm_single(prompt: str):
            async with semaphore:
                try:
                    cache_key = self._generate_cache_key(prompt)
                    
                    # Check if already cached
                    if await self.get(cache_key):
                        return
                    
                    # Generate and cache
                    data = await generator_func(prompt)
                    await self.put(cache_key, data)
                    
                    logger.debug(f"Warmed cache for: {prompt}")
                    
                except Exception as e:
                    logger.error(f"Cache warmup failed for '{prompt}': {e}")
        
        # Create warmup tasks
        tasks = [warm_single(prompt) for prompt in prompts]
        self._warmup_tasks = {f"warmup_{i}": task for i, task in enumerate(tasks)}
        
        # Execute warmup
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Cache warmup completed")
    
    def _generate_cache_key(self, prompt: str, **params) -> str:
        """Generate standardized cache key."""
        # Create hash from prompt and parameters
        key_data = f"{prompt}:{sorted(params.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive cache analytics."""
        total_hits = (
            self._cache_analytics['l1_hits'] + 
            self._cache_analytics['l2_hits'] + 
            self._cache_analytics['l3_hits']
        )
        total_misses = (
            self._cache_analytics['l1_misses'] + 
            self._cache_analytics['l2_misses'] + 
            self._cache_analytics['l3_misses']
        )
        
        total_requests = self._cache_analytics['total_requests']
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'overall': {
                'total_requests': total_requests,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'hit_rate': overall_hit_rate
            },
            'l1_memory': {
                'hits': self._cache_analytics['l1_hits'],
                'misses': self._cache_analytics['l1_misses'],
                'stats': self.l1_cache.get_stats().__dict__
            },
            'l2_redis': {
                'hits': self._cache_analytics['l2_hits'],
                'misses': self._cache_analytics['l2_misses'],
                'stats': self.l2_cache.get_stats().__dict__
            },
            'l3_cdn': {
                'hits': self._cache_analytics['l3_hits'],
                'misses': self._cache_analytics['l3_misses'],
                'stats': self.l3_cache.get_stats().__dict__ if self.l3_cache else {}
            },
            'warmup_tasks': len(self._warmup_tasks)
        }
    
    async def close(self):
        """Close all cache connections."""
        # Cancel warmup tasks
        for task in self._warmup_tasks.values():
            if not task.done():
                task.cancel()
        
        # Clear L1 cache
        self.l1_cache.clear()
        
        logger.info("Multi-tier cache closed")


# Global cache instance
_global_cache: Optional[MultiTierCache] = None


async def get_cache() -> MultiTierCache:
    """Get or create global multi-tier cache instance."""
    global _global_cache
    
    if _global_cache is None:
        # Configure based on environment
        l1_max_size = int(os.getenv('CACHE_L1_MAX_SIZE', '1000'))
        l1_max_memory = int(os.getenv('CACHE_L1_MAX_MEMORY_MB', '512'))
        default_ttl = int(os.getenv('CACHE_DEFAULT_TTL', '3600'))
        enable_cdn = os.getenv('CACHE_ENABLE_CDN', 'true').lower() == 'true'
        
        _global_cache = MultiTierCache(
            l1_max_size=l1_max_size,
            l1_max_memory_mb=l1_max_memory,
            default_ttl=default_ttl,
            enable_cdn=enable_cdn
        )
    
    return _global_cache