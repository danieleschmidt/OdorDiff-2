"""
Data management and caching system for OdorDiff-2.
"""

import os
import json
import pickle
import hashlib
import time
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import threading
from dataclasses import dataclass, asdict

from ..models.molecule import Molecule
from ..utils.logging import get_logger

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
    """Persistent cache using SQLite backend."""
    
    def __init__(self, db_path: str = "cache.db", max_size_mb: int = 100):
        self.db_path = db_path
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    accessed_at REAL,
                    ttl REAL,
                    size INTEGER,
                    tags TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)')
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT value, created_at, ttl FROM cache_entries WHERE key = ?',
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    value_blob, created_at, ttl = row
                    
                    # Check TTL
                    if time.time() - created_at > ttl:
                        self._remove(key)
                        return None
                    
                    # Update access time
                    conn.execute(
                        'UPDATE cache_entries SET accessed_at = ? WHERE key = ?',
                        (time.time(), key)
                    )
                    
                    # Deserialize value
                    return pickle.loads(value_blob)
                    
            except Exception as e:
                logger.error(f"Error retrieving from persistent cache: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl: float = 3600, tags: List[str] = None) -> None:
        """Set value in persistent cache."""
        with self._lock:
            try:
                # Serialize value
                value_blob = pickle.dumps(value)
                size = len(value_blob)
                tags_str = json.dumps(tags or [])
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, created_at, accessed_at, ttl, size, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (key, value_blob, time.time(), time.time(), ttl, size, tags_str))
                
                # Clean up if needed
                self._cleanup_if_needed()
                
            except Exception as e:
                logger.error(f"Error storing to persistent cache: {e}")
    
    def _remove(self, key: str) -> None:
        """Remove entry from persistent cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
        except Exception as e:
            logger.error(f"Error removing from persistent cache: {e}")
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds size limits."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check total size
                cursor = conn.execute('SELECT SUM(size) FROM cache_entries')
                total_size = cursor.fetchone()[0] or 0
                
                max_size_bytes = self.max_size_mb * 1024 * 1024
                
                if total_size > max_size_bytes:
                    # Remove oldest accessed entries
                    conn.execute('''
                        DELETE FROM cache_entries 
                        WHERE key IN (
                            SELECT key FROM cache_entries 
                            ORDER BY accessed_at ASC 
                            LIMIT ?
                        )
                    ''', (max(1, int(0.2 * conn.execute('SELECT COUNT(*) FROM cache_entries').fetchone()[0])),))
                    
        except Exception as e:
            logger.error(f"Error cleaning up persistent cache: {e}")
    
    def clear_expired(self) -> int:
        """Clear expired entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'DELETE FROM cache_entries WHERE ? - created_at > ttl',
                    (time.time(),)
                )
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error clearing expired entries: {e}")
            return 0


class MoleculeCache:
    """Specialized cache for molecule-related data."""
    
    def __init__(self, cache_dir: str = "molecule_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Different caches for different data types
        self.generation_cache = LRUCache(max_size=500, default_ttl=7200)  # 2 hours
        self.safety_cache = LRUCache(max_size=1000, default_ttl=86400)   # 24 hours
        self.synthesis_cache = LRUCache(max_size=200, default_ttl=3600)   # 1 hour
        self.property_cache = PersistentCache(
            str(self.cache_dir / "properties.db"), 
            max_size_mb=50
        )
        
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "generation_cache": self.generation_cache.get_stats(),
            "safety_cache": self.safety_cache.get_stats(),
            "synthesis_cache": self.synthesis_cache.get_stats(),
        }
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries across all caches."""
        return {
            "persistent_expired": self.property_cache.clear_expired()
        }


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
_molecule_cache: Optional[MoleculeCache] = None

def get_molecule_cache() -> MoleculeCache:
    """Get global molecule cache instance."""
    global _molecule_cache
    if _molecule_cache is None:
        _molecule_cache = MoleculeCache()
    return _molecule_cache