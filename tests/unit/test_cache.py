"""
Unit tests for data management and caching system.
"""

import pytest
import os
import json
import time
import tempfile
import threading
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any, List

from odordiff2.data.cache import (
    CacheEntry, ConnectionPoolConfig, DatabaseConnectionPool, 
    SerializationManager, LRUCache, PersistentCache, EnhancedMoleculeCache,
    MoleculeCache, DatasetManager, get_molecule_cache, get_legacy_cache
)
from odordiff2.models.molecule import Molecule


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test creating a CacheEntry instance."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=1234567890.0,
            accessed_at=1234567890.0,
            ttl=3600.0,
            size=100,
            tags=["test"],
            access_count=5,
            compression="gzip"
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.created_at == 1234567890.0
        assert entry.accessed_at == 1234567890.0
        assert entry.ttl == 3600.0
        assert entry.size == 100
        assert entry.tags == ["test"]
        assert entry.access_count == 5
        assert entry.compression == "gzip"
    
    def test_cache_entry_defaults(self):
        """Test CacheEntry with default values."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=1234567890.0,
            accessed_at=1234567890.0,
            ttl=3600.0,
            size=100,
            tags=["test"]
        )
        
        assert entry.access_count == 0
        assert entry.compression is None


class TestConnectionPoolConfig:
    """Test ConnectionPoolConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ConnectionPoolConfig()
        
        assert config.max_connections == 20
        assert config.min_connections == 5
        assert config.connection_timeout == 5.0
        assert config.socket_timeout == 3.0
        assert config.retry_on_timeout is True
        assert config.max_idle_time == 300.0
        assert config.health_check_interval == 60.0
        assert config.max_retries == 3
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConnectionPoolConfig(
            max_connections=50,
            min_connections=10,
            connection_timeout=10.0,
            socket_timeout=5.0,
            retry_on_timeout=False,
            max_idle_time=600.0,
            health_check_interval=120.0,
            max_retries=5
        )
        
        assert config.max_connections == 50
        assert config.min_connections == 10
        assert config.connection_timeout == 10.0
        assert config.socket_timeout == 5.0
        assert config.retry_on_timeout is False
        assert config.max_idle_time == 600.0
        assert config.health_check_interval == 120.0
        assert config.max_retries == 5


class TestDatabaseConnectionPool:
    """Test DatabaseConnectionPool class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            db_path = tmp_file.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def pool_config(self):
        """Create test pool configuration."""
        return ConnectionPoolConfig(
            max_connections=5,
            min_connections=2,
            connection_timeout=1.0,
            health_check_interval=0.1
        )
    
    def test_pool_initialization(self, temp_db_path, pool_config):
        """Test connection pool initialization."""
        pool = DatabaseConnectionPool(temp_db_path, pool_config)
        
        assert pool.db_path == temp_db_path
        assert pool.config == pool_config
        assert not pool._closed
        assert pool._created_connections >= pool_config.min_connections
        
        # Clean up
        pool.close()
    
    def test_get_connection_context_manager(self, temp_db_path, pool_config):
        """Test getting connection using context manager."""
        pool = DatabaseConnectionPool(temp_db_path, pool_config)
        
        with pool.get_connection() as conn:
            assert conn is not None
            # Test that connection works
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        pool.close()
    
    def test_connection_reuse(self, temp_db_path, pool_config):
        """Test that connections are reused."""
        pool = DatabaseConnectionPool(temp_db_path, pool_config)
        
        connections = []
        for _ in range(3):
            with pool.get_connection() as conn:
                connections.append(id(conn))
        
        # Should reuse connections from pool
        assert len(set(connections)) <= pool_config.max_connections
        
        pool.close()
    
    def test_max_connections_limit(self, temp_db_path):
        """Test maximum connections limit."""
        config = ConnectionPoolConfig(max_connections=2, min_connections=1)
        pool = DatabaseConnectionPool(temp_db_path, config)
        
        # Should be able to create up to max_connections
        connections = []
        try:
            for _ in range(3):  # Try to get more than max
                conn = pool._create_connection()
                connections.append(conn)
        except RuntimeError as e:
            assert "Maximum connections reached" in str(e)
        finally:
            # Clean up connections
            for conn in connections:
                conn.close()
            pool.close()
    
    def test_get_stats(self, temp_db_path, pool_config):
        """Test getting pool statistics."""
        pool = DatabaseConnectionPool(temp_db_path, pool_config)
        
        stats = pool.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_connections' in stats
        assert 'pool_size' in stats
        assert 'max_connections' in stats
        assert 'min_connections' in stats
        assert 'failed_connections' in stats
        assert 'healthy_connections' in stats
        
        assert stats['max_connections'] == pool_config.max_connections
        assert stats['min_connections'] == pool_config.min_connections
        assert stats['total_connections'] >= pool_config.min_connections
        
        pool.close()
    
    def test_close_pool(self, temp_db_path, pool_config):
        """Test closing connection pool."""
        pool = DatabaseConnectionPool(temp_db_path, pool_config)
        
        # Get some connections to populate pool
        with pool.get_connection():
            pass
        
        initial_stats = pool.get_stats()
        assert initial_stats['total_connections'] > 0
        
        pool.close()
        
        assert pool._closed is True
        
        # Should raise error when trying to get connection from closed pool
        with pytest.raises(RuntimeError, match="Connection pool is closed"):
            with pool.get_connection():
                pass
    
    def test_connection_failure_handling(self, temp_db_path, pool_config):
        """Test handling of connection failures."""
        pool = DatabaseConnectionPool(temp_db_path, pool_config)
        
        # Simulate connection failure by marking a connection as failed
        with pool.get_connection() as conn:
            conn_id = id(conn)
            pool._failed_connections.add(conn_id)
        
        # Next connection should be recreated
        with pool.get_connection() as conn:
            assert id(conn) != conn_id
        
        pool.close()


class TestSerializationManager:
    """Test SerializationManager class."""
    
    def test_initialization_default(self):
        """Test SerializationManager initialization with defaults."""
        manager = SerializationManager()
        
        assert manager.default_format == "json"
        assert manager.compression is True
        assert 'json' in manager.serializers
        assert 'pickle' in manager.serializers
    
    def test_initialization_custom(self):
        """Test SerializationManager initialization with custom settings."""
        manager = SerializationManager(default_format="pickle", compression=False)
        
        assert manager.default_format == "pickle"
        assert manager.compression is False
    
    def test_serialize_json_without_compression(self):
        """Test JSON serialization without compression."""
        manager = SerializationManager(compression=False)
        data = {"key": "value", "number": 42}
        
        result = manager.serialize(data)
        
        assert isinstance(result, bytes)
        # Should be able to decode as UTF-8 JSON
        decoded = json.loads(result.decode('utf-8'))
        assert decoded == data
    
    def test_serialize_json_with_compression(self):
        """Test JSON serialization with compression."""
        manager = SerializationManager(compression=True)
        data = {"key": "value" * 100}  # Larger data to see compression benefit
        
        result = manager.serialize(data)
        
        assert isinstance(result, bytes)
        # Compressed data should be smaller for repetitive content
        uncompressed = manager.serialize(data, compress=False)
        assert len(result) < len(uncompressed)
    
    def test_serialize_pickle(self):
        """Test pickle serialization."""
        manager = SerializationManager(default_format="pickle")
        data = {"key": "value", "complex": [1, 2, {"nested": True}]}
        
        result = manager.serialize(data)
        
        assert isinstance(result, bytes)
    
    def test_deserialize_json(self):
        """Test JSON deserialization."""
        manager = SerializationManager()
        original_data = {"key": "value", "number": 42}
        
        serialized = manager.serialize(original_data)
        deserialized = manager.deserialize(serialized)
        
        assert deserialized == original_data
    
    def test_deserialize_pickle(self):
        """Test pickle deserialization."""
        manager = SerializationManager(default_format="pickle")
        original_data = {"key": "value", "complex": [1, 2, {"nested": True}]}
        
        serialized = manager.serialize(original_data)
        deserialized = manager.deserialize(serialized)
        
        assert deserialized == original_data
    
    def test_serialize_unsupported_format(self):
        """Test serialization with unsupported format."""
        manager = SerializationManager()
        
        with pytest.raises(ValueError, match="Unsupported serialization format"):
            manager.serialize({"key": "value"}, format="unsupported")
    
    def test_deserialize_unsupported_format(self):
        """Test deserialization with unsupported format."""
        manager = SerializationManager()
        
        with pytest.raises(ValueError, match="Unsupported serialization format"):
            manager.deserialize(b"data", format="unsupported")
    
    def test_compression_roundtrip(self):
        """Test compression and decompression roundtrip."""
        manager = SerializationManager(compression=True)
        data = {"repeated": "content " * 1000}
        
        serialized = manager.serialize(data, compress=True)
        deserialized = manager.deserialize(serialized, compressed=True)
        
        assert deserialized == data
    
    def test_explicit_compression_override(self):
        """Test explicit compression parameter override."""
        manager = SerializationManager(compression=False)
        data = {"key": "value"}
        
        # Override compression to True
        compressed = manager.serialize(data, compress=True)
        uncompressed = manager.serialize(data, compress=False)
        
        # Should be different sizes
        assert len(compressed) != len(uncompressed)
        
        # Both should deserialize to same data
        assert manager.deserialize(compressed, compressed=True) == data
        assert manager.deserialize(uncompressed, compressed=False) == data


class TestLRUCache:
    """Test LRUCache class."""
    
    def test_cache_initialization(self):
        """Test LRUCache initialization."""
        cache = LRUCache(max_size=100, default_ttl=3600)
        
        assert cache.max_size == 100
        assert cache.default_ttl == 3600
        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0
        assert cache._total_size == 0
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = LRUCache(max_size=10)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Non-existent key should return None
        assert cache.get("nonexistent") is None
    
    def test_cache_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        cache = LRUCache(max_size=10, default_ttl=0.1)  # 0.1 second TTL
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL to expire
        time.sleep(0.15)
        assert cache.get("key1") is None
    
    def test_cache_custom_ttl(self):
        """Test cache with custom TTL per entry."""
        cache = LRUCache(max_size=10)
        
        cache.set("key1", "value1", ttl=0.1)
        cache.set("key2", "value2", ttl=3600)
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        
        # Wait for key1 to expire
        time.sleep(0.15)
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"  # Should still be valid
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=3)
        
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # All should be present
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # Still present
        assert cache.get("key4") == "value4"  # Newly added
    
    def test_cache_tags(self):
        """Test cache entry tagging."""
        cache = LRUCache(max_size=10)
        
        cache.set("key1", "value1", tags=["tag1", "tag2"])
        cache.set("key2", "value2", tags=["tag2", "tag3"])
        cache.set("key3", "value3", tags=["tag1"])
        
        # Clear entries by tags
        removed_count = cache.clear_by_tags(["tag2"])
        
        assert removed_count == 2  # key1 and key2 should be removed
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"  # Only has tag1
    
    def test_cache_clear(self):
        """Test clearing entire cache."""
        cache = LRUCache(max_size=10)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert len(cache._cache) == 2
        
        cache.clear()
        
        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0
        assert cache._total_size == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.get_stats()
        
        assert isinstance(stats, dict)
        assert stats['size'] == 2
        assert stats['max_size'] == 10
        assert stats['total_size_bytes'] > 0
        assert 'hit_rate' in stats
    
    def test_cache_thread_safety(self):
        """Test cache thread safety."""
        cache = LRUCache(max_size=100)
        results = []
        errors = []
        
        def worker(thread_id, num_operations):
            try:
                for i in range(num_operations):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    
                    cache.set(key, value)
                    retrieved = cache.get(key)
                    
                    if retrieved != value:
                        errors.append(f"Thread {thread_id}: Expected {value}, got {retrieved}")
                    else:
                        results.append(f"Thread {thread_id}: OK")
            except Exception as e:
                errors.append(f"Thread {thread_id}: Exception {e}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i, 10))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 50  # 5 threads * 10 operations


class TestPersistentCache:
    """Test PersistentCache class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            db_path = tmp_file.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return ConnectionPoolConfig(
            max_connections=3,
            min_connections=1,
            connection_timeout=1.0
        )
    
    def test_cache_initialization(self, temp_db_path, cache_config):
        """Test persistent cache initialization."""
        cache = PersistentCache(
            db_path=temp_db_path,
            max_size_mb=10,
            pool_config=cache_config,
            serialization="json",
            compression=True
        )
        
        assert cache.db_path == temp_db_path
        assert cache.max_size_mb == 10
        assert cache.pool_config == cache_config
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._requests == 0
        
        # Database should be initialized
        assert os.path.exists(temp_db_path)
    
    def test_cache_set_and_get(self, temp_db_path):
        """Test basic persistent cache operations."""
        cache = PersistentCache(db_path=temp_db_path)
        
        # Set value
        test_data = {"key": "value", "number": 42}
        cache.set("test_key", test_data, ttl=3600)
        
        # Get value
        retrieved = cache.get("test_key")
        assert retrieved == test_data
        
        # Check metrics
        assert cache._requests == 1
        assert cache._hits == 1
        assert cache._misses == 0
    
    def test_cache_miss(self, temp_db_path):
        """Test cache miss behavior."""
        cache = PersistentCache(db_path=temp_db_path)
        
        result = cache.get("nonexistent_key")
        
        assert result is None
        assert cache._requests == 1
        assert cache._hits == 0
        assert cache._misses == 1
    
    def test_cache_ttl_expiration(self, temp_db_path):
        """Test TTL expiration in persistent cache."""
        cache = PersistentCache(db_path=temp_db_path)
        
        # Set with short TTL
        cache.set("temp_key", "temp_value", ttl=0.1)
        
        # Should be available immediately
        assert cache.get("temp_key") == "temp_value"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired and removed
        assert cache.get("temp_key") is None
    
    def test_cache_different_formats(self, temp_db_path):
        """Test caching with different serialization formats."""
        cache = PersistentCache(db_path=temp_db_path)
        
        # Test JSON format
        json_data = {"format": "json", "data": [1, 2, 3]}
        cache.set("json_key", json_data, format="json")
        assert cache.get("json_key") == json_data
        
        # Test pickle format
        pickle_data = {"format": "pickle", "complex": {1, 2, 3}}  # Set is not JSON serializable
        cache.set("pickle_key", pickle_data, format="pickle")
        assert cache.get("pickle_key") == pickle_data
    
    def test_cache_compression(self, temp_db_path):
        """Test cache with compression enabled/disabled."""
        cache = PersistentCache(db_path=temp_db_path)
        
        # Large data that benefits from compression
        large_data = {"content": "repeated_content " * 1000}
        
        # Test with compression
        cache.set("compressed_key", large_data, compress=True)
        assert cache.get("compressed_key") == large_data
        
        # Test without compression
        cache.set("uncompressed_key", large_data, compress=False)
        assert cache.get("uncompressed_key") == large_data
    
    def test_cache_tags(self, temp_db_path):
        """Test caching with tags."""
        cache = PersistentCache(db_path=temp_db_path)
        
        cache.set("key1", "value1", tags=["tag1", "tag2"])
        cache.set("key2", "value2", tags=["tag2", "tag3"])
        
        # Both should be retrievable
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
    
    def test_clear_expired(self, temp_db_path):
        """Test clearing expired entries."""
        cache = PersistentCache(db_path=temp_db_path)
        
        # Set entries with different TTLs
        cache.set("short_ttl", "value1", ttl=0.1)
        cache.set("long_ttl", "value2", ttl=3600)
        
        # Wait for short TTL to expire
        time.sleep(0.15)
        
        # Clear expired entries
        cleared_count = cache.clear_expired()
        
        assert cleared_count >= 1  # At least the short_ttl entry
        assert cache.get("short_ttl") is None
        assert cache.get("long_ttl") == "value2"
    
    def test_cache_stats(self, temp_db_path):
        """Test cache statistics."""
        cache = PersistentCache(db_path=temp_db_path)
        
        # Add some data
        cache.set("key1", "value1")
        cache.set("key2", {"complex": "data"})
        cache.get("key1")  # Generate a hit
        cache.get("nonexistent")  # Generate a miss
        
        stats = cache.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_entries' in stats
        assert 'total_size_bytes' in stats
        assert 'total_size_mb' in stats
        assert 'hit_rate_percent' in stats
        assert 'total_requests' in stats
        assert 'total_hits' in stats
        assert 'total_misses' in stats
        assert 'connection_pool_stats' in stats
        
        assert stats['total_entries'] >= 2
        assert stats['total_hits'] >= 1
        assert stats['total_misses'] >= 1
    
    @pytest.mark.asyncio
    async def test_async_methods(self, temp_db_path):
        """Test async cache methods."""
        cache = PersistentCache(db_path=temp_db_path)
        
        # Test async set
        await cache.set_async("async_key", "async_value", ttl=3600)
        
        # Test async get
        result = await cache.get_async("async_key")
        assert result == "async_value"
        
        # Test async close
        await cache.close()


class TestEnhancedMoleculeCache:
    """Test EnhancedMoleculeCache class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return {
            'max_connections': 5,
            'min_connections': 2,
            'connection_timeout': 1.0,
            'generation_cache_size': 100,
            'generation_cache_ttl': 1800,
            'safety_cache_size': 200,
            'safety_cache_ttl': 3600,
            'synthesis_cache_size': 50,
            'synthesis_cache_ttl': 1800,
            'property_cache_mb': 10,
            'redis_enabled': False
        }
    
    @pytest.fixture
    def sample_molecules(self):
        """Create sample molecules for testing."""
        molecules = []
        smiles_list = ["CCO", "CC(C)O", "c1ccccc1"]
        
        for smiles in smiles_list:
            mol = Molecule(smiles, confidence=0.9)
            mol.safety_score = 0.8
            mol.synth_score = 0.7
            mol.estimated_cost = 50.0
            molecules.append(mol)
        
        return molecules
    
    def test_cache_initialization(self, temp_cache_dir, cache_config):
        """Test enhanced molecule cache initialization."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir, config=cache_config)
        
        assert cache.cache_dir == Path(temp_cache_dir)
        assert cache.config == cache_config
        assert cache.generation_cache is not None
        assert cache.safety_cache is not None
        assert cache.synthesis_cache is not None
        assert cache.property_cache is not None
        assert cache.redis_cache is None  # Redis disabled in config
        
        # Check cache sizes
        assert cache.generation_cache.max_size == cache_config['generation_cache_size']
        assert cache.safety_cache.max_size == cache_config['safety_cache_size']
        assert cache.synthesis_cache.max_size == cache_config['synthesis_cache_size']
    
    def test_make_key(self, temp_cache_dir):
        """Test cache key generation."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        # Test with simple arguments
        key1 = cache._make_key("test", "arg1", "arg2")
        key2 = cache._make_key("test", "arg1", "arg2")
        key3 = cache._make_key("test", "arg1", "arg3")
        
        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
        assert key1.startswith("test:")
        
        # Test with kwargs
        key4 = cache._make_key("test", param1="value1", param2="value2")
        key5 = cache._make_key("test", param2="value2", param1="value1")  # Different order
        
        assert key4 == key5  # Order of kwargs shouldn't matter
    
    def test_generation_cache_operations(self, temp_cache_dir, sample_molecules):
        """Test generation result caching operations."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        prompt = "Generate citrus scent"
        params = {"num_molecules": 3, "temperature": 1.0}
        
        # Initially, no cached result
        result = cache.get_generation_result(prompt, params)
        assert result is None
        
        # Cache the result
        cache.cache_generation_result(prompt, params, sample_molecules)
        
        # Should now retrieve cached result
        cached_result = cache.get_generation_result(prompt, params)
        assert cached_result is not None
        assert len(cached_result) == len(sample_molecules)
        
        # Check that molecules are properly deserialized
        for original, cached in zip(sample_molecules, cached_result):
            assert cached.smiles == original.smiles
            assert cached.confidence == original.confidence
    
    def test_safety_cache_operations(self, temp_cache_dir):
        """Test safety assessment caching operations."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        smiles = "CCO"
        assessment = {
            "toxicity_score": 0.05,
            "skin_sensitizer": False,
            "eco_score": 0.1,
            "ifra_compliant": True
        }
        
        # Initially, no cached assessment
        result = cache.get_safety_assessment(smiles)
        assert result is None
        
        # Cache the assessment
        cache.cache_safety_assessment(smiles, assessment)
        
        # Should now retrieve cached assessment
        cached_assessment = cache.get_safety_assessment(smiles)
        assert cached_assessment == assessment
    
    def test_synthesis_cache_operations(self, temp_cache_dir):
        """Test synthesis routes caching operations."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        smiles = "CCO"
        params = {"max_steps": 3, "green_chemistry": True}
        routes = [
            {
                "route_id": 1,
                "steps": ["step1", "step2"],
                "score": 0.8
            }
        ]
        
        # Initially, no cached routes
        result = cache.get_synthesis_routes(smiles, params)
        assert result is None
        
        # Cache the routes
        cache.cache_synthesis_routes(smiles, params, routes)
        
        # Should now retrieve cached routes
        cached_routes = cache.get_synthesis_routes(smiles, params)
        assert cached_routes == routes
    
    def test_molecular_properties_cache(self, temp_cache_dir):
        """Test molecular properties caching operations."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        smiles = "CCO"
        properties = {
            "molecular_weight": 46.07,
            "logP": -0.31,
            "tpsa": 20.23,
            "hbd": 1,
            "hba": 1
        }
        
        # Initially, no cached properties
        result = cache.get_molecular_properties(smiles)
        assert result is None
        
        # Cache the properties
        cache.cache_molecular_properties(smiles, properties)
        
        # Should now retrieve cached properties
        cached_properties = cache.get_molecular_properties(smiles)
        assert cached_properties == properties
    
    def test_clear_generation_cache(self, temp_cache_dir, sample_molecules):
        """Test clearing generation cache."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        # Add some generation results
        cache.cache_generation_result("prompt1", {}, sample_molecules)
        cache.cache_generation_result("prompt2", {}, sample_molecules)
        
        # Verify they exist
        assert cache.get_generation_result("prompt1", {}) is not None
        assert cache.get_generation_result("prompt2", {}) is not None
        
        # Clear generation cache
        cache.clear_generation_cache()
        
        # Should be gone
        assert cache.get_generation_result("prompt1", {}) is None
        assert cache.get_generation_result("prompt2", {}) is None
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, temp_cache_dir):
        """Test getting comprehensive cache statistics."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        # Add some data to different caches
        cache.generation_cache.set("gen_key", "gen_value")
        cache.safety_cache.set("safety_key", "safety_value")
        cache.synthesis_cache.set("synth_key", "synth_value")
        cache.property_cache.set("prop_key", {"prop": "value"})
        
        stats = await cache.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert 'generation_cache' in stats
        assert 'safety_cache' in stats
        assert 'synthesis_cache' in stats
        assert 'property_cache' in stats
        assert 'metrics' in stats
        
        # Check that each cache has stats
        for cache_name in ['generation_cache', 'safety_cache', 'synthesis_cache']:
            assert 'size' in stats[cache_name]
            assert 'max_size' in stats[cache_name]
        
        assert 'total_entries' in stats['property_cache']
        assert 'hit_rate_percent' in stats['property_cache']
    
    @pytest.mark.asyncio
    async def test_cleanup_expired(self, temp_cache_dir):
        """Test cleaning up expired entries."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        # Add entries with short TTL to persistent cache
        cache.property_cache.set("expired_key", {"data": "value"}, ttl=0.1)
        cache.property_cache.set("valid_key", {"data": "value"}, ttl=3600)
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Cleanup expired entries
        results = await cache.cleanup_expired()
        
        assert isinstance(results, dict)
        assert 'persistent_expired' in results
        assert results['persistent_expired'] >= 1  # At least one expired entry
    
    @pytest.mark.asyncio
    async def test_health_check(self, temp_cache_dir):
        """Test cache health check."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        health = await cache.health_check()
        
        assert isinstance(health, dict)
        assert 'memory_caches' in health
        assert 'persistent_cache' in health
        assert 'redis_cache' in health
        
        # Memory caches should be healthy
        assert health['memory_caches'] == 'healthy'
        
        # Persistent cache should be healthy (or have specific error)
        assert 'healthy' in health['persistent_cache'] or 'unhealthy' in health['persistent_cache']
        
        # Redis should be not available (disabled in test)
        assert health['redis_cache'] == 'not_available'
    
    @pytest.mark.asyncio
    async def test_close_cache(self, temp_cache_dir):
        """Test closing cache resources."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        # Should not raise exception
        await cache.close()


class TestMoleculeCacheBackwardCompatibility:
    """Test MoleculeCache backward compatibility."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_legacy_cache_initialization(self, temp_cache_dir):
        """Test backward compatible cache initialization."""
        cache = MoleculeCache(cache_dir=temp_cache_dir)
        
        assert isinstance(cache, EnhancedMoleculeCache)
        assert cache.cache_dir == Path(temp_cache_dir)
    
    def test_sync_methods(self, temp_cache_dir):
        """Test synchronous versions of async methods."""
        cache = MoleculeCache(cache_dir=temp_cache_dir)
        
        # Test sync get_cache_stats
        stats = cache.get_cache_stats()
        assert isinstance(stats, dict)
        
        # Test sync cleanup_expired
        results = cache.cleanup_expired()
        assert isinstance(results, dict)


class TestDatasetManager:
    """Test DatasetManager class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_dataset_file(self, temp_data_dir):
        """Create sample dataset file."""
        dataset_path = os.path.join(temp_data_dir, "sample_dataset.csv")
        with open(dataset_path, 'w') as f:
            f.write("smiles,odor_description\n")
            f.write("CCO,alcohol\n")
            f.write("c1ccccc1,benzene\n")
        return dataset_path
    
    def test_dataset_manager_initialization(self, temp_data_dir):
        """Test DatasetManager initialization."""
        manager = DatasetManager(data_dir=temp_data_dir)
        
        assert manager.data_dir == Path(temp_data_dir)
        assert manager.metadata_file == Path(temp_data_dir) / "metadata.json"
        assert isinstance(manager.metadata, dict)
        assert "datasets" in manager.metadata
        assert "created_at" in manager.metadata
        assert "last_updated" in manager.metadata
    
    def test_register_dataset(self, temp_data_dir, sample_dataset_file):
        """Test registering a dataset."""
        manager = DatasetManager(data_dir=temp_data_dir)
        
        manager.register_dataset(
            name="sample_dataset",
            file_path=sample_dataset_file,
            dataset_type="training",
            description="Sample dataset for testing",
            version="1.0"
        )
        
        # Check metadata
        dataset_info = manager.get_dataset_info("sample_dataset")
        assert dataset_info is not None
        assert dataset_info["file_path"] == sample_dataset_file
        assert dataset_info["type"] == "training"
        assert dataset_info["description"] == "Sample dataset for testing"
        assert dataset_info["version"] == "1.0"
        assert "created_at" in dataset_info
        assert dataset_info["size_bytes"] > 0
    
    def test_get_dataset_info(self, temp_data_dir, sample_dataset_file):
        """Test getting dataset information."""
        manager = DatasetManager(data_dir=temp_data_dir)
        
        # Non-existent dataset
        assert manager.get_dataset_info("nonexistent") is None
        
        # Register and retrieve
        manager.register_dataset("test_dataset", sample_dataset_file, "test")
        info = manager.get_dataset_info("test_dataset")
        
        assert info is not None
        assert info["file_path"] == sample_dataset_file
        assert info["type"] == "test"
    
    def test_list_datasets(self, temp_data_dir, sample_dataset_file):
        """Test listing all datasets."""
        manager = DatasetManager(data_dir=temp_data_dir)
        
        # Initially empty
        datasets = manager.list_datasets()
        assert len(datasets) == 0
        
        # Register some datasets
        manager.register_dataset("dataset1", sample_dataset_file, "training")
        manager.register_dataset("dataset2", sample_dataset_file, "validation")
        
        datasets = manager.list_datasets()
        assert len(datasets) == 2
        
        # Check that all datasets are included
        dataset_names = [d.get("file_path") for d in datasets]
        assert sample_dataset_file in dataset_names
    
    def test_metadata_persistence(self, temp_data_dir, sample_dataset_file):
        """Test that metadata persists across manager instances."""
        # Create first manager and register dataset
        manager1 = DatasetManager(data_dir=temp_data_dir)
        manager1.register_dataset("persistent_dataset", sample_dataset_file, "test")
        
        # Create second manager - should load existing metadata
        manager2 = DatasetManager(data_dir=temp_data_dir)
        dataset_info = manager2.get_dataset_info("persistent_dataset")
        
        assert dataset_info is not None
        assert dataset_info["file_path"] == sample_dataset_file
        assert dataset_info["type"] == "test"


class TestGlobalFunctions:
    """Test global cache functions."""
    
    def test_get_molecule_cache_singleton(self):
        """Test that get_molecule_cache returns singleton."""
        cache1 = get_molecule_cache()
        cache2 = get_molecule_cache()
        
        assert cache1 is cache2
        assert isinstance(cache1, EnhancedMoleculeCache)
    
    def test_get_legacy_cache(self):
        """Test get_legacy_cache function."""
        cache = get_legacy_cache()
        
        assert isinstance(cache, MoleculeCache)
        assert isinstance(cache, EnhancedMoleculeCache)  # MoleculeCache extends EnhancedMoleculeCache
    
    def test_get_molecule_cache_with_config(self):
        """Test get_molecule_cache with custom config."""
        config = {
            'generation_cache_size': 200,
            'safety_cache_size': 300,
            'redis_enabled': False
        }
        
        # Clear global cache instance first
        from odordiff2.data import cache as cache_module
        cache_module._molecule_cache = None
        
        cache = get_molecule_cache(config)
        
        assert isinstance(cache, EnhancedMoleculeCache)
        assert cache.generation_cache.max_size == 200
        assert cache.safety_cache.max_size == 300


class TestCacheIntegration:
    """Integration tests for cache system."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_full_molecule_cache_workflow(self, temp_cache_dir):
        """Test complete molecule cache workflow."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        # Test generation result caching
        prompt = "Generate fresh citrus scent"
        params = {"num_molecules": 2, "temperature": 1.0}
        molecules = [
            Molecule("CC(=O)c1ccc(cc1)OC", confidence=0.9),
            Molecule("CCO", confidence=0.8)
        ]
        
        # Cache and retrieve
        cache.cache_generation_result(prompt, params, molecules)
        cached_molecules = cache.get_generation_result(prompt, params)
        
        assert len(cached_molecules) == 2
        assert cached_molecules[0].smiles == molecules[0].smiles
        
        # Test safety assessment caching
        smiles = "CCO"
        assessment = {
            "toxicity_score": 0.1,
            "skin_sensitizer": False,
            "regulatory_flags": []
        }
        
        cache.cache_safety_assessment(smiles, assessment)
        cached_assessment = cache.get_safety_assessment(smiles)
        
        assert cached_assessment == assessment
        
        # Test molecular properties caching
        properties = {
            "molecular_weight": 46.07,
            "logP": -0.31,
            "vapor_pressure": 59.3
        }
        
        cache.cache_molecular_properties(smiles, properties)
        cached_properties = cache.get_molecular_properties(smiles)
        
        assert cached_properties == properties
    
    def test_cache_performance_characteristics(self, temp_cache_dir):
        """Test cache performance with larger datasets."""
        cache = EnhancedMoleculeCache(
            cache_dir=temp_cache_dir,
            config={
                'generation_cache_size': 100,
                'safety_cache_size': 100,
                'property_cache_mb': 1
            }
        )
        
        # Generate test data
        test_data = []
        for i in range(150):  # More than cache size
            test_data.append({
                'key': f'test_key_{i}',
                'value': f'test_value_{i}' * 100  # Make values larger
            })
        
        # Test LRU eviction in memory cache
        start_time = time.time()
        for item in test_data:
            cache.safety_cache.set(item['key'], item['value'])
        set_time = time.time() - start_time
        
        # Should have evicted older entries
        assert cache.safety_cache.get_stats()['size'] <= 100
        
        # Test retrieval performance
        start_time = time.time()
        hit_count = 0
        for item in test_data[-50:]:  # Test last 50 (should be in cache)
            if cache.safety_cache.get(item['key']) == item['value']:
                hit_count += 1
        get_time = time.time() - start_time
        
        # Performance assertions
        assert set_time < 1.0  # Should be reasonably fast
        assert get_time < 0.1   # Retrieval should be very fast
        assert hit_count > 0    # Should have some hits
    
    @pytest.mark.asyncio
    async def test_async_cache_operations(self, temp_cache_dir):
        """Test asynchronous cache operations."""
        cache = EnhancedMoleculeCache(cache_dir=temp_cache_dir)
        
        # Test concurrent operations
        async def cache_operation(key_suffix, value_suffix):
            key = f"async_key_{key_suffix}"
            value = f"async_value_{value_suffix}"
            
            await cache.property_cache.set_async(key, {"data": value})
            result = await cache.property_cache.get_async(key)
            
            return result["data"] if result else None
        
        # Run multiple operations concurrently
        tasks = []
        for i in range(10):
            task = cache_operation(i, i)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result == f"async_value_{i}"
        
        # Test health check
        health = await cache.health_check()
        assert health['memory_caches'] == 'healthy'