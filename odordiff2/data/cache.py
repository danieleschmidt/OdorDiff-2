"""
Enhanced data management and caching system with optimized connection pooling.
"""

import os
import json
import pickle
import hashlib
import time
import asyncio
import gzip
from typing import Any, Dict, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager, asynccontextmanager
from queue import Queue, Empty
import atexit

try:
    import redis
    import redis.asyncio as aioredis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    aioredis = None

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

from ..models.molecule import Molecule
from ..utils.logging import get_logger
from ..utils.error_handling import retry_with_backoff, ExponentialBackoffStrategy, safe_execute_async

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl: float  # Time to live in seconds
    size: int   # Size in bytes
    tags: List[str]
    access_count: int = 0
    compression: Optional[str] = None


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling."""
    max_connections: int = 20
    min_connections: int = 5
    connection_timeout: float = 5.0
    socket_timeout: float = 3.0
    retry_on_timeout: bool = True
    max_idle_time: float = 300.0  # 5 minutes
    health_check_interval: float = 60.0  # 1 minute
    max_retries: int = 3


class DatabaseConnectionPool:
    """Thread-safe database connection pool for SQLite."""
    
    def __init__(self, db_path: str, config: ConnectionPoolConfig = None):
        self.db_path = db_path
        self.config = config or ConnectionPoolConfig()
        
        self._pool: Queue = Queue(maxsize=self.config.max_connections)
        self._created_connections = 0
        self._lock = threading.RLock()
        self._closed = False
        
        # Connection health tracking
        self._connection_health: Dict[int, float] = {}
        self._failed_connections: set = set()
        
        # Initialize minimum connections
        self._init_connections()
        
        # Start health check thread
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, 
            daemon=True
        )
        self._health_check_thread.start()
        
        # Register cleanup on exit
        atexit.register(self.close)
    
    def _init_connections(self):
        """Initialize minimum number of connections."""
        for _ in range(self.config.min_connections):
            conn = self._create_connection()
            self._pool.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        with self._lock:
            if self._created_connections >= self.config.max_connections:
                raise RuntimeError("Maximum connections reached")
            
            try:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=self.config.connection_timeout,
                    check_same_thread=False
                )
                
                # Configure connection
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                conn.execute('PRAGMA cache_size=10000')
                conn.execute('PRAGMA temp_store=MEMORY')
                
                conn_id = id(conn)
                self._connection_health[conn_id] = time.time()
                self._created_connections += 1
                
                logger.debug(f"Created database connection {conn_id}")
                return conn
                
            except Exception as e:
                logger.error(f"Failed to create database connection: {e}")
                raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        try:
            # Try to get connection from pool with timeout
            try:
                conn = self._pool.get(timeout=self.config.connection_timeout)
            except Empty:
                # Pool is empty, try to create new connection
                if self._created_connections < self.config.max_connections:
                    conn = self._create_connection()
                else:
                    # Wait a bit more for a connection to be returned
                    conn = self._pool.get(timeout=self.config.connection_timeout * 2)
            
            conn_id = id(conn)
            
            # Check if connection is healthy
            if conn_id in self._failed_connections:
                conn.close()
                self._created_connections -= 1
                self._failed_connections.remove(conn_id)
                conn = self._create_connection()
                conn_id = id(conn)
            
            # Update last used time
            self._connection_health[conn_id] = time.time()
            
            yield conn
            
        except Exception as e:
            # Mark connection as failed
            if conn:
                self._failed_connections.add(id(conn))
            raise
        finally:
            # Return connection to pool
            if conn and not self._closed:
                try:
                    self._pool.put_nowait(conn)
                except:
                    # Pool might be full, close connection
                    conn.close()
                    with self._lock:
                        self._created_connections -= 1
    
    def _health_check_loop(self):
        """Background thread to check connection health."""
        while not self._closed:
            try:
                time.sleep(self.config.health_check_interval)
                self._check_connection_health()
            except Exception as e:
                logger.error(f"Error in connection health check: {e}")
    
    def _check_connection_health(self):
        """Check and clean up unhealthy connections."""
        current_time = time.time()
        
        # Check for idle connections
        idle_connections = []
        with self._lock:
            for conn_id, last_used in self._connection_health.items():
                if current_time - last_used > self.config.max_idle_time:
                    idle_connections.append(conn_id)
        
        # Remove idle connections if we have more than minimum
        if len(idle_connections) > 0 and self._created_connections > self.config.min_connections:
            # This is a simplified approach - in practice, you'd need more sophisticated tracking
            logger.debug(f"Found {len(idle_connections)} idle connections")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                'total_connections': self._created_connections,
                'pool_size': self._pool.qsize(),
                'max_connections': self.config.max_connections,
                'min_connections': self.config.min_connections,
                'failed_connections': len(self._failed_connections),
                'healthy_connections': len(self._connection_health)
            }
    
    def close(self):
        """Close all connections in the pool."""
        if self._closed:
            return
        
        self._closed = True
        logger.info("Closing database connection pool")
        
        # Close all connections in pool
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        with self._lock:
            self._created_connections = 0
            self._connection_health.clear()
            self._failed_connections.clear()


class RedisConnectionPool:
    """Enhanced Redis connection pool with async support."""
    
    def __init__(self, config: Dict[str, Any]):
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis not available - install redis-py")
        
        self.config = config
        
        # Sync Redis pool
        self._sync_pool = redis.ConnectionPool(
            host=config.get('host', 'localhost'),
            port=config.get('port', 6379),
            password=config.get('password'),
            db=config.get('db', 0),
            max_connections=config.get('max_connections', 20),
            socket_timeout=config.get('socket_timeout', 3.0),
            socket_connect_timeout=config.get('connection_timeout', 5.0),
            retry_on_timeout=config.get('retry_on_timeout', True),
            health_check_interval=config.get('health_check_interval', 60)
        )
        
        # Async Redis pool
        self._async_pool = aioredis.ConnectionPool(
            host=config.get('host', 'localhost'),
            port=config.get('port', 6379),
            password=config.get('password'),
            db=config.get('db', 0),
            max_connections=config.get('max_connections', 20),
            socket_timeout=config.get('socket_timeout', 3.0),
            socket_connect_timeout=config.get('connection_timeout', 5.0),
            retry_on_timeout=config.get('retry_on_timeout', True)
        )
        
        self._sync_client = redis.Redis(connection_pool=self._sync_pool)
        self._async_client = aioredis.Redis(connection_pool=self._async_pool)
        
        # Health monitoring
        self._last_health_check = 0
        self._health_status = True
        
        logger.info("Redis connection pool initialized")
    
    def get_sync_client(self) -> redis.Redis:
        """Get synchronous Redis client."""
        return self._sync_client
    
    def get_async_client(self) -> aioredis.Redis:
        """Get asynchronous Redis client."""
        return self._async_client
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            current_time = time.time()
            if current_time - self._last_health_check < 30:  # Cache health for 30 seconds
                return self._health_status
            
            # Test ping
            await self._async_client.ping()
            self._health_status = True
            self._last_health_check = current_time
            return True
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            self._health_status = False
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis connection pool statistics."""
        try:
            info = self._sync_client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'health_status': self._health_status,
                'pool_created_connections': self._sync_pool.created_connections,
                'pool_available_connections': len(self._sync_pool._available_connections),
                'pool_in_use_connections': len(self._sync_pool._in_use_connections)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {'error': str(e)}
    
    async def close(self):
        """Close Redis connection pools."""
        try:
            await self._async_client.close()
            self._sync_client.close()
            logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection pool: {e}")


class SerializationManager:
    """Manages different serialization formats with compression."""
    
    def __init__(self, default_format: str = "json", compression: bool = True):
        self.default_format = default_format
        self.compression = compression
        
        self.serializers = {
            'json': (json.dumps, json.loads),
            'pickle': (pickle.dumps, pickle.loads),
        }
        
        if MSGPACK_AVAILABLE:
            self.serializers['msgpack'] = (msgpack.packb, msgpack.unpackb)
    
    def serialize(self, data: Any, format: str = None, compress: bool = None) -> bytes:
        """Serialize data with optional compression."""
        format = format or self.default_format
        compress = compress if compress is not None else self.compression
        
        if format not in self.serializers:
            raise ValueError(f"Unsupported serialization format: {format}")
        
        serializer, _ = self.serializers[format]
        
        try:
            if format == 'json':
                serialized = serializer(data).encode('utf-8')
            else:
                serialized = serializer(data)
            
            if compress:
                serialized = gzip.compress(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization error with {format}: {e}")
            raise
    
    def deserialize(self, data: bytes, format: str = None, compressed: bool = None) -> Any:
        """Deserialize data with optional decompression."""
        format = format or self.default_format
        compressed = compressed if compressed is not None else self.compression
        
        if format not in self.serializers:
            raise ValueError(f"Unsupported serialization format: {format}")
        
        _, deserializer = self.serializers[format]
        
        try:
            if compressed:
                data = gzip.decompress(data)
            
            if format == 'json':
                return deserializer(data.decode('utf-8'))
            else:
                return deserializer(data)
                
        except Exception as e:
            logger.error(f"Deserialization error with {format}: {e}")
            raise
    

class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._total_size = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
                
            entry = self._cache[key]
            
            # Check TTL
            if time.time() - entry.created_at > entry.ttl:
                self._remove(key)
                return None
                
            # Update access time and order
            entry.accessed_at = time.time()
            self._access_order.remove(key)
            self._access_order.append(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> None:
        """Set value in cache."""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
                
            # Calculate size (rough estimate)
            size = len(str(value))
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove(key)
                
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl,
                size=size,
                tags=tags or []
            )
            
            # Add to cache
            self._cache[key] = entry
            self._access_order.append(key)
            self._total_size += size
            
            # Evict if necessary
            self._evict_if_needed()
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._access_order.remove(key)
            self._total_size -= entry.size
    
    def _evict_if_needed(self) -> None:
        """Evict least recently used entries if cache is full."""
        while len(self._cache) > self.max_size:
            oldest_key = self._access_order[0]
            self._remove(oldest_key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._total_size = 0
    
    def clear_by_tags(self, tags: List[str]) -> int:
        """Clear entries that match any of the provided tags."""
        with self._lock:
            keys_to_remove = []
            for key, entry in self._cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove(key)
                
            return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_size_bytes': self._total_size,
                'hit_rate': getattr(self, '_hits', 0) / max(1, getattr(self, '_requests', 1))
            }


class PersistentCache:
    """Enhanced persistent cache using SQLite backend with connection pooling."""
    
    def __init__(self, db_path: str = "cache.db", max_size_mb: int = 100, 
                 pool_config: ConnectionPoolConfig = None, serialization: str = "json",
                 compression: bool = True):
        self.db_path = db_path
        self.max_size_mb = max_size_mb
        self.pool_config = pool_config or ConnectionPoolConfig()
        
        # Initialize connection pool and serialization
        self.connection_pool = DatabaseConnectionPool(db_path, self.pool_config)
        self.serializer = SerializationManager(serialization, compression)
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._requests = 0
        self._lock = threading.RLock()
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with optimized schema."""
        with self.connection_pool.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    accessed_at REAL,
                    ttl REAL,
                    size INTEGER,
                    tags TEXT,
                    access_count INTEGER DEFAULT 0,
                    compression TEXT,
                    format TEXT DEFAULT 'json'
                )
            ''')
            
            # Optimized indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_ttl ON cache_entries(created_at, ttl)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_size ON cache_entries(size)')
            
            # Enable query optimization
            conn.execute('ANALYZE cache_entries')
            conn.commit()
    
    @retry_with_backoff(max_attempts=3, strategy=ExponentialBackoffStrategy(max_delay=5.0))
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache with retry logic."""
        with self._lock:
            self._requests += 1
            
            try:
                with self.connection_pool.get_connection() as conn:
                    cursor = conn.execute('''
                        SELECT value, created_at, ttl, compression, format, access_count 
                        FROM cache_entries WHERE key = ?
                    ''', (key,))
                    row = cursor.fetchone()
                    
                    if not row:
                        self._misses += 1
                        return None
                    
                    value_blob, created_at, ttl, compression, format_type, access_count = row
                    
                    # Check TTL
                    if time.time() - created_at > ttl:
                        self._remove_with_connection(conn, key)
                        self._misses += 1
                        return None
                    
                    # Update access time and count
                    conn.execute('''
                        UPDATE cache_entries 
                        SET accessed_at = ?, access_count = access_count + 1
                        WHERE key = ?
                    ''', (time.time(), key))
                    conn.commit()
                    
                    # Deserialize value
                    compressed = compression is not None
                    value = self.serializer.deserialize(
                        value_blob, 
                        format=format_type or 'json', 
                        compressed=compressed
                    )
                    
                    self._hits += 1
                    return value
                    
            except Exception as e:
                logger.error(f"Error retrieving from persistent cache: {e}")
                self._misses += 1
                return None
    
    @retry_with_backoff(max_attempts=3, strategy=ExponentialBackoffStrategy(max_delay=5.0))
    def set(self, key: str, value: Any, ttl: float = 3600, tags: List[str] = None, 
            format: str = None, compress: bool = None) -> None:
        """Set value in persistent cache with optimized serialization."""
        with self._lock:
            try:
                # Serialize value with optimal settings
                value_blob = self.serializer.serialize(value, format, compress)
                size = len(value_blob)
                tags_str = json.dumps(tags or [])
                current_time = time.time()
                
                # Determine compression and format info
                actual_compress = compress if compress is not None else self.serializer.compression
                actual_format = format or self.serializer.default_format
                compression_str = 'gzip' if actual_compress else None
                
                with self.connection_pool.get_connection() as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, created_at, accessed_at, ttl, size, tags, 
                         access_count, compression, format)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    ''', (key, value_blob, current_time, current_time, ttl, size, 
                          tags_str, compression_str, actual_format))
                    conn.commit()
                
                # Clean up if needed
                self._cleanup_if_needed()
                
            except Exception as e:
                logger.error(f"Error storing to persistent cache: {e}")
                raise
    
    def _remove_with_connection(self, conn: sqlite3.Connection, key: str) -> None:
        """Remove entry using existing connection."""
        conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
        conn.commit()
    
    def _remove(self, key: str) -> None:
        """Remove entry from persistent cache."""
        try:
            with self.connection_pool.get_connection() as conn:
                self._remove_with_connection(conn, key)
        except Exception as e:
            logger.error(f"Error removing from persistent cache: {e}")
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds size limits using optimized queries."""
        try:
            with self.connection_pool.get_connection() as conn:
                # Check total size
                cursor = conn.execute('SELECT SUM(size), COUNT(*) FROM cache_entries')
                result = cursor.fetchone()
                total_size = result[0] or 0
                total_entries = result[1] or 0
                
                max_size_bytes = self.max_size_mb * 1024 * 1024
                
                if total_size > max_size_bytes and total_entries > 0:
                    # Use smart cleanup strategy - remove least accessed and oldest entries
                    cleanup_count = max(1, int(0.2 * total_entries))
                    
                    conn.execute('''
                        DELETE FROM cache_entries 
                        WHERE key IN (
                            SELECT key FROM cache_entries 
                            ORDER BY access_count ASC, accessed_at ASC 
                            LIMIT ?
                        )
                    ''', (cleanup_count,))
                    conn.commit()
                    
                    logger.info(f"Cleaned up {cleanup_count} cache entries")
                    
        except Exception as e:
            logger.error(f"Error cleaning up persistent cache: {e}")
    
    def clear_expired(self) -> int:
        """Clear expired entries with connection pooling."""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(
                    'DELETE FROM cache_entries WHERE ? - created_at > ttl',
                    (time.time(),)
                )
                conn.commit()
                count = cursor.rowcount
                if count > 0:
                    logger.info(f"Cleared {count} expired cache entries")
                return count
        except Exception as e:
            logger.error(f"Error clearing expired entries: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            with self.connection_pool.get_connection() as conn:
                # Get basic stats
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(size) as total_size,
                        AVG(size) as avg_size,
                        MAX(access_count) as max_access_count,
                        AVG(access_count) as avg_access_count,
                        COUNT(CASE WHEN compression IS NOT NULL THEN 1 END) as compressed_entries
                    FROM cache_entries
                ''')
                stats = cursor.fetchone()
                
                # Get format distribution
                cursor = conn.execute('''
                    SELECT format, COUNT(*) 
                    FROM cache_entries 
                    GROUP BY format
                ''')
                format_stats = dict(cursor.fetchall())
                
                hit_rate = self._hits / max(1, self._requests) * 100
                
                cache_stats = {
                    'total_entries': stats[0] or 0,
                    'total_size_bytes': stats[1] or 0,
                    'total_size_mb': (stats[1] or 0) / (1024 * 1024),
                    'avg_entry_size': stats[2] or 0,
                    'max_access_count': stats[3] or 0,
                    'avg_access_count': stats[4] or 0,
                    'compressed_entries': stats[5] or 0,
                    'format_distribution': format_stats,
                    'hit_rate_percent': round(hit_rate, 2),
                    'total_requests': self._requests,
                    'total_hits': self._hits,
                    'total_misses': self._misses,
                    'connection_pool_stats': self.connection_pool.get_stats()
                }
                
                return cache_stats
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    async def close(self):
        """Close cache and connection pool."""
        try:
            self.connection_pool.close()
            logger.info("Persistent cache closed")
        except Exception as e:
            logger.error(f"Error closing persistent cache: {e}")
    
    # Async methods for better integration
    async def get_async(self, key: str) -> Optional[Any]:
        """Async version of get method."""
        return await safe_execute_async(self.get, key)
    
    async def set_async(self, key: str, value: Any, ttl: float = 3600, 
                       tags: List[str] = None, format: str = None, 
                       compress: bool = None) -> None:
        """Async version of set method."""
        await safe_execute_async(self.set, key, value, ttl, tags, format, compress)


class EnhancedMoleculeCache:
    """Enhanced specialized cache for molecule-related data with connection pooling."""
    
    def __init__(self, cache_dir: str = "molecule_cache", config: Dict[str, Any] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = config or {}
        pool_config = ConnectionPoolConfig(
            max_connections=self.config.get('max_connections', 10),
            min_connections=self.config.get('min_connections', 3),
            connection_timeout=self.config.get('connection_timeout', 5.0)
        )
        
        # Enhanced caches with different optimization strategies
        self.generation_cache = LRUCache(
            max_size=self.config.get('generation_cache_size', 500), 
            default_ttl=self.config.get('generation_cache_ttl', 7200)  # 2 hours
        )
        
        self.safety_cache = LRUCache(
            max_size=self.config.get('safety_cache_size', 1000), 
            default_ttl=self.config.get('safety_cache_ttl', 86400)  # 24 hours
        )
        
        self.synthesis_cache = LRUCache(
            max_size=self.config.get('synthesis_cache_size', 200), 
            default_ttl=self.config.get('synthesis_cache_ttl', 3600)  # 1 hour
        )
        
        # Enhanced persistent cache for stable data
        self.property_cache = PersistentCache(
            str(self.cache_dir / "properties.db"), 
            max_size_mb=self.config.get('property_cache_mb', 50),
            pool_config=pool_config,
            serialization='msgpack' if MSGPACK_AVAILABLE else 'json',
            compression=True
        )
        
        # Redis cache for distributed scenarios (optional)
        self.redis_cache: Optional[RedisConnectionPool] = None
        if REDIS_AVAILABLE and self.config.get('redis_enabled', False):
            try:
                self.redis_cache = RedisConnectionPool(self.config.get('redis', {}))
                logger.info("Redis cache initialized for molecule cache")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
        
        # Performance metrics
        self._cache_metrics = {
            'generation_requests': 0,
            'generation_hits': 0,
            'safety_requests': 0,
            'safety_hits': 0,
            'property_requests': 0,
            'property_hits': 0
        }
        
    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        data = str(args) + str(sorted(kwargs.items()))
        hash_obj = hashlib.md5(data.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get_generation_result(self, prompt: str, params: Dict[str, Any]) -> Optional[List[Molecule]]:
        """Get cached generation result."""
        key = self._make_key("gen", prompt, **params)
        cached_data = self.generation_cache.get(key)
        
        if cached_data:
            # Deserialize molecules
            return [Molecule.from_dict(mol_data) for mol_data in cached_data]
        return None
    
    def cache_generation_result(self, prompt: str, params: Dict[str, Any], molecules: List[Molecule]) -> None:
        """Cache generation result."""
        key = self._make_key("gen", prompt, **params)
        # Serialize molecules
        mol_data = [mol.to_dict() for mol in molecules]
        self.generation_cache.set(key, mol_data, tags=["generation"])
    
    def get_safety_assessment(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Get cached safety assessment."""
        key = self._make_key("safety", smiles)
        return self.safety_cache.get(key)
    
    def cache_safety_assessment(self, smiles: str, assessment: Dict[str, Any]) -> None:
        """Cache safety assessment."""
        key = self._make_key("safety", smiles)
        self.safety_cache.set(key, assessment, tags=["safety"])
    
    def get_synthesis_routes(self, smiles: str, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Get cached synthesis routes."""
        key = self._make_key("synth", smiles, **params)
        return self.synthesis_cache.get(key)
    
    def cache_synthesis_routes(self, smiles: str, params: Dict[str, Any], routes: List[Dict[str, Any]]) -> None:
        """Cache synthesis routes."""
        key = self._make_key("synth", smiles, **params)
        self.synthesis_cache.set(key, routes, tags=["synthesis"])
    
    def get_molecular_properties(self, smiles: str) -> Optional[Dict[str, float]]:
        """Get cached molecular properties."""
        key = self._make_key("props", smiles)
        return self.property_cache.get(key)
    
    def cache_molecular_properties(self, smiles: str, properties: Dict[str, float]) -> None:
        """Cache molecular properties."""
        key = self._make_key("props", smiles)
        # Properties are usually stable, so longer TTL
        self.property_cache.set(key, properties, ttl=86400 * 7, tags=["properties"])  # 1 week
    
    def clear_generation_cache(self) -> None:
        """Clear generation cache (e.g., when model is updated)."""
        self.generation_cache.clear()
        logger.info("Generation cache cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "generation_cache": self.generation_cache.get_stats(),
            "safety_cache": self.safety_cache.get_stats(),
            "synthesis_cache": self.synthesis_cache.get_stats(),
            "property_cache": self.property_cache.get_stats(),
            "metrics": self._cache_metrics.copy()
        }
        
        # Add Redis stats if available
        if self.redis_cache:
            try:
                stats["redis_cache"] = self.redis_cache.get_stats()
            except Exception as e:
                stats["redis_cache"] = {"error": str(e)}
        
        # Calculate hit rates
        for cache_type in ['generation', 'safety', 'property']:
            requests_key = f'{cache_type}_requests'
            hits_key = f'{cache_type}_hits'
            
            if self._cache_metrics[requests_key] > 0:
                hit_rate = (self._cache_metrics[hits_key] / self._cache_metrics[requests_key]) * 100
                stats['metrics'][f'{cache_type}_hit_rate_percent'] = round(hit_rate, 2)
        
        return stats
    
    async def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries across all caches."""
        results = {
            "persistent_expired": self.property_cache.clear_expired()
        }
        
        # Clear expired from Redis if available
        if self.redis_cache:
            try:
                # Redis handles TTL automatically, but we can get stats
                client = self.redis_cache.get_async_client()
                info = await client.info()
                results["redis_expired_keys"] = info.get('expired_keys', 0)
            except Exception as e:
                logger.error(f"Error getting Redis expired keys info: {e}")
        
        return results
    
    async def clear_generation_cache(self) -> None:
        """Clear generation cache (e.g., when model is updated)."""
        self.generation_cache.clear()
        
        # Clear from Redis if available
        if self.redis_cache:
            try:
                client = self.redis_cache.get_async_client()
                # Use pattern matching to clear generation-related keys
                async for key in client.scan_iter(match="gen:*"):
                    await client.delete(key)
            except Exception as e:
                logger.error(f"Error clearing Redis generation cache: {e}")
        
        logger.info("Generation cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all cache backends."""
        health = {
            'memory_caches': 'healthy',
            'persistent_cache': 'healthy',
            'redis_cache': 'not_available'
        }
        
        # Check persistent cache
        try:
            self.property_cache.get_stats()
        except Exception as e:
            health['persistent_cache'] = f'unhealthy: {str(e)}'
        
        # Check Redis cache
        if self.redis_cache:
            try:
                is_healthy = await self.redis_cache.health_check()
                health['redis_cache'] = 'healthy' if is_healthy else 'unhealthy'
            except Exception as e:
                health['redis_cache'] = f'unhealthy: {str(e)}'
        
        return health
    
    async def close(self):
        """Close all cache connections."""
        try:
            await self.property_cache.close()
            if self.redis_cache:
                await self.redis_cache.close()
            logger.info("Enhanced molecule cache closed")
        except Exception as e:
            logger.error(f"Error closing enhanced molecule cache: {e}")


# Maintain backward compatibility
class MoleculeCache(EnhancedMoleculeCache):
    """Backward compatible molecule cache."""
    
    def __init__(self, cache_dir: str = "molecule_cache"):
        super().__init__(cache_dir)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Sync version of get_cache_stats for backward compatibility."""
        return asyncio.run(super().get_cache_stats())
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Sync version of cleanup_expired for backward compatibility."""
        return asyncio.run(super().cleanup_expired())


class DatasetManager:
    """Manage training and reference datasets."""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset metadata
        self.metadata_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading dataset metadata: {e}")
                
        return {
            "datasets": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_metadata(self):
        """Save dataset metadata."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving dataset metadata: {e}")
    
    def register_dataset(
        self, 
        name: str, 
        file_path: str, 
        dataset_type: str,
        description: str = "",
        version: str = "1.0"
    ) -> None:
        """Register a new dataset."""
        self.metadata["datasets"][name] = {
            "file_path": file_path,
            "type": dataset_type,
            "description": description,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        self._save_metadata()
        logger.info(f"Registered dataset: {name}")
    
    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a dataset."""
        return self.metadata["datasets"].get(name)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all registered datasets."""
        return list(self.metadata["datasets"].values())
    
    def download_dataset(self, name: str, url: str, force_update: bool = False) -> bool:
        """Download a dataset from URL."""
        # This is a placeholder - would implement actual download logic
        logger.info(f"Would download dataset {name} from {url}")
        return True


# Global cache instance
_molecule_cache: Optional[EnhancedMoleculeCache] = None

def get_molecule_cache(config: Dict[str, Any] = None) -> EnhancedMoleculeCache:
    """Get global enhanced molecule cache instance."""
    global _molecule_cache
    if _molecule_cache is None:
        _molecule_cache = EnhancedMoleculeCache(config=config)
    return _molecule_cache

def get_legacy_cache() -> MoleculeCache:
    """Get backward compatible molecule cache instance."""
    return MoleculeCache()

# Convenience functions for cache management
async def setup_cache_from_config(config: Dict[str, Any]) -> EnhancedMoleculeCache:
    """Setup cache from configuration."""
    cache = EnhancedMoleculeCache(
        cache_dir=config.get('cache_dir', 'molecule_cache'),
        config=config
    )
    
    # Perform initial health check
    health = await cache.health_check()
    logger.info(f"Cache health check: {health}")
    
    return cache

async def cleanup_all_caches():
    """Cleanup all cache instances."""
    global _molecule_cache
    if _molecule_cache:
        await _molecule_cache.close()
        _molecule_cache = None
        logger.info("All cache instances cleaned up")

# Register cleanup on exit
atexit.register(lambda: asyncio.run(cleanup_all_caches()))