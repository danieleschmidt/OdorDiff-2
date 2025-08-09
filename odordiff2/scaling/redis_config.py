"""
Redis Configuration and Connection Management for Distributed Scaling
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import redis
import aioredis
from redis.sentinel import Sentinel
from redis.connection import ConnectionPool
from contextlib import asynccontextmanager

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RedisConfig:
    """Redis configuration settings."""
    # Connection settings
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    
    # Connection pooling
    max_connections: int = 50
    connection_timeout: int = 10
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=dict)
    
    # High availability
    sentinels: List[tuple] = field(default_factory=list)  # [(host, port), ...]
    master_name: str = "mymaster"
    sentinel_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tuning
    decode_responses: bool = True
    encoding: str = "utf-8"
    health_check_interval: int = 30
    retry_on_timeout: bool = True
    
    # Clustering
    cluster_nodes: List[Dict[str, Any]] = field(default_factory=list)
    skip_full_coverage_check: bool = False
    
    # Cache settings
    default_ttl: int = 3600  # 1 hour
    compression_enabled: bool = True
    compression_threshold: int = 1024  # bytes
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Basic connection
        config.host = os.getenv('REDIS_HOST', config.host)
        config.port = int(os.getenv('REDIS_PORT', str(config.port)))
        config.password = os.getenv('REDIS_PASSWORD')
        config.db = int(os.getenv('REDIS_DB', str(config.db)))
        
        # Connection pool
        config.max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', str(config.max_connections)))
        config.connection_timeout = int(os.getenv('REDIS_TIMEOUT', str(config.connection_timeout)))
        
        # High availability
        sentinels_str = os.getenv('REDIS_SENTINELS', '')
        if sentinels_str:
            config.sentinels = [
                (host.strip(), int(port.strip())) 
                for host, port in [s.split(':') for s in sentinels_str.split(',')]
            ]
        config.master_name = os.getenv('REDIS_MASTER_NAME', config.master_name)
        
        # Performance
        config.default_ttl = int(os.getenv('REDIS_DEFAULT_TTL', str(config.default_ttl)))
        config.compression_enabled = os.getenv('REDIS_COMPRESSION', 'true').lower() == 'true'
        
        return config


class RedisConnectionManager:
    """Manages Redis connections with high availability, pooling, and clustering support."""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._sync_client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._sentinel: Optional[Sentinel] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize Redis connections."""
        try:
            await self._setup_connections()
            await self._start_health_monitoring()
            logger.info("Redis connection manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis connections: {e}")
            raise
    
    async def _setup_connections(self):
        """Set up Redis connections based on configuration."""
        if self.config.sentinels:
            await self._setup_sentinel_connections()
        elif self.config.cluster_nodes:
            await self._setup_cluster_connections()
        else:
            await self._setup_single_connections()
    
    async def _setup_single_connections(self):
        """Set up single Redis instance connections."""
        # Connection pool
        self._connection_pool = ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password,
            db=self.config.db,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.connection_timeout,
            socket_keepalive=self.config.socket_keepalive,
            socket_keepalive_options=self.config.socket_keepalive_options,
            decode_responses=self.config.decode_responses,
            encoding=self.config.encoding,
            health_check_interval=self.config.health_check_interval,
            retry_on_timeout=self.config.retry_on_timeout
        )
        
        # Synchronous client
        self._sync_client = redis.Redis(connection_pool=self._connection_pool)
        
        # Asynchronous client
        self._async_client = aioredis.Redis(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password,
            db=self.config.db,
            decode_responses=self.config.decode_responses,
            encoding=self.config.encoding,
            max_connections=self.config.max_connections,
            retry_on_timeout=self.config.retry_on_timeout
        )
    
    async def _setup_sentinel_connections(self):
        """Set up Redis Sentinel connections for high availability."""
        self._sentinel = Sentinel(
            self.config.sentinels,
            **self.config.sentinel_kwargs
        )
        
        # Get master connection
        self._sync_client = self._sentinel.master_for(
            self.config.master_name,
            password=self.config.password,
            db=self.config.db,
            decode_responses=self.config.decode_responses,
            encoding=self.config.encoding
        )
        
        # Async sentinel client
        sentinel_async = aioredis.Sentinel(self.config.sentinels)
        self._async_client = sentinel_async.master_for(
            self.config.master_name,
            password=self.config.password,
            db=self.config.db,
            decode_responses=self.config.decode_responses,
            encoding=self.config.encoding
        )
    
    async def _setup_cluster_connections(self):
        """Set up Redis Cluster connections."""
        from rediscluster import RedisCluster
        
        self._sync_client = RedisCluster(
            startup_nodes=self.config.cluster_nodes,
            password=self.config.password,
            decode_responses=self.config.decode_responses,
            encoding=self.config.encoding,
            skip_full_coverage_check=self.config.skip_full_coverage_check,
            max_connections_per_node=self.config.max_connections // len(self.config.cluster_nodes)
        )
        
        # Note: aioredis cluster support is limited, fallback to single connection
        self._async_client = aioredis.Redis(
            host=self.config.cluster_nodes[0]['host'],
            port=self.config.cluster_nodes[0]['port'],
            password=self.config.password,
            db=self.config.db,
            decode_responses=self.config.decode_responses,
            encoding=self.config.encoding
        )
    
    async def _start_health_monitoring(self):
        """Start background health monitoring."""
        if self.config.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_monitor())
    
    async def _health_monitor(self):
        """Background health monitoring task."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Test connections
                if self._sync_client:
                    self._sync_client.ping()
                    
                if self._async_client:
                    await self._async_client.ping()
                    
                logger.debug("Redis health check passed")
                
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
                # Attempt reconnection
                try:
                    await self._setup_connections()
                except Exception as reconnect_error:
                    logger.error(f"Failed to reconnect to Redis: {reconnect_error}")
            
            except asyncio.CancelledError:
                break
    
    def get_sync_client(self) -> redis.Redis:
        """Get synchronous Redis client."""
        if not self._sync_client:
            raise RuntimeError("Redis connection not initialized")
        return self._sync_client
    
    def get_async_client(self) -> aioredis.Redis:
        """Get asynchronous Redis client."""
        if not self._async_client:
            raise RuntimeError("Redis async connection not initialized")
        return self._async_client
    
    @asynccontextmanager
    async def get_pipeline(self, transaction: bool = True):
        """Get Redis pipeline for batched operations."""
        client = self.get_async_client()
        pipe = client.pipeline(transaction=transaction)
        try:
            yield pipe
        finally:
            await pipe.reset()
    
    async def close(self):
        """Close all connections."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._async_client:
            await self._async_client.close()
        
        if self._connection_pool:
            self._connection_pool.disconnect()
        
        logger.info("Redis connections closed")


class RedisDataSerializer:
    """Handle data serialization for Redis with compression and protocol buffers."""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.compression_enabled = config.compression_enabled
        self.compression_threshold = config.compression_threshold
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data for Redis storage."""
        try:
            import msgpack
            import zstandard as zstd
            
            # Serialize to msgpack
            serialized = msgpack.packb(data, use_bin_type=True)
            
            # Compress if enabled and data is large enough
            if (self.compression_enabled and 
                len(serialized) > self.compression_threshold):
                compressor = zstd.ZstdCompressor(level=3)
                compressed = compressor.compress(serialized)
                # Add compression marker
                return b'ZSTD' + compressed
            
            return serialized
            
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            # Fallback to JSON
            import json
            return json.dumps(data).encode('utf-8')
    
    def deserialize(self, data: Union[bytes, str]) -> Any:
        """Deserialize data from Redis."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Check for compression marker
            if data.startswith(b'ZSTD'):
                import zstandard as zstd
                import msgpack
                
                decompressor = zstd.ZstdDecompressor()
                decompressed = decompressor.decompress(data[4:])
                return msgpack.unpackb(decompressed, raw=False)
            
            # Try msgpack first
            try:
                import msgpack
                return msgpack.unpackb(data, raw=False)
            except:
                # Fallback to JSON
                import json
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            return None


# Global Redis manager instance
_redis_manager: Optional[RedisConnectionManager] = None
_serializer: Optional[RedisDataSerializer] = None


async def get_redis_manager() -> RedisConnectionManager:
    """Get or create Redis connection manager."""
    global _redis_manager
    
    if _redis_manager is None:
        config = RedisConfig.from_env()
        _redis_manager = RedisConnectionManager(config)
        await _redis_manager.initialize()
    
    return _redis_manager


def get_redis_serializer() -> RedisDataSerializer:
    """Get or create Redis data serializer."""
    global _serializer
    
    if _serializer is None:
        config = RedisConfig.from_env()
        _serializer = RedisDataSerializer(config)
    
    return _serializer


async def close_redis_connections():
    """Close all Redis connections."""
    global _redis_manager
    
    if _redis_manager:
        await _redis_manager.close()
        _redis_manager = None