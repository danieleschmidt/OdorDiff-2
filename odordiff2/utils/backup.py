"""
Data backup and recovery mechanisms for cache and critical data.
"""

import os
import json
import gzip
import shutil
import sqlite3
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading
import schedule
import time
import hashlib
import tarfile
import tempfile

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

from .logging import get_logger
from .error_handling import retry_with_backoff, ExponentialBackoffStrategy

logger = get_logger(__name__)


@dataclass
class BackupConfig:
    """Configuration for backup operations."""
    backup_dir: str = "backups"
    max_backups: int = 10
    backup_interval_hours: int = 24
    compress_backups: bool = True
    verify_backups: bool = True
    backup_metadata: bool = True
    incremental_backup: bool = False
    retention_days: int = 30
    
    # Remote backup configuration
    s3_bucket: Optional[str] = None
    s3_prefix: str = "odordiff2-backups"
    s3_region: str = "us-east-1"
    
    # Encryption (placeholder for future implementation)
    encrypt_backups: bool = False
    encryption_key: Optional[str] = None


@dataclass
class BackupMetadata:
    """Metadata for backup files."""
    backup_id: str
    created_at: datetime
    backup_type: str  # full, incremental
    source_path: str
    backup_path: str
    file_count: int
    total_size_bytes: int
    checksum: str
    compression_ratio: float = 1.0
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'backup_id': self.backup_id,
            'created_at': self.created_at.isoformat(),
            'backup_type': self.backup_type,
            'source_path': self.source_path,
            'backup_path': self.backup_path,
            'file_count': self.file_count,
            'total_size_bytes': self.total_size_bytes,
            'checksum': self.checksum,
            'compression_ratio': self.compression_ratio,
            'tags': self.tags or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupMetadata':
        """Create from dictionary."""
        return cls(
            backup_id=data['backup_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            backup_type=data['backup_type'],
            source_path=data['source_path'],
            backup_path=data['backup_path'],
            file_count=data['file_count'],
            total_size_bytes=data['total_size_bytes'],
            checksum=data['checksum'],
            compression_ratio=data.get('compression_ratio', 1.0),
            tags=data.get('tags', {})
        )


class BackupStorage:
    """Abstract base for backup storage backends."""
    
    async def store_backup(self, source_path: str, backup_id: str) -> str:
        """Store backup and return storage path."""
        raise NotImplementedError
    
    async def retrieve_backup(self, backup_id: str, destination_path: str) -> bool:
        """Retrieve backup to destination path."""
        raise NotImplementedError
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup from storage."""
        raise NotImplementedError
    
    async def list_backups(self) -> List[str]:
        """List available backup IDs."""
        raise NotImplementedError


class LocalBackupStorage(BackupStorage):
    """Local filesystem backup storage."""
    
    def __init__(self, backup_dir: str):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def store_backup(self, source_path: str, backup_id: str) -> str:
        """Store backup locally."""
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        
        # Create compressed archive
        with tarfile.open(backup_path, 'w:gz') as tar:
            if os.path.isdir(source_path):
                tar.add(source_path, arcname=Path(source_path).name)
            else:
                tar.add(source_path, arcname=Path(source_path).name)
        
        logger.info(f"Backup stored locally: {backup_path}")
        return str(backup_path)
    
    async def retrieve_backup(self, backup_id: str, destination_path: str) -> bool:
        """Retrieve backup from local storage."""
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        try:
            # Extract archive
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(destination_path)
            
            logger.info(f"Backup retrieved to: {destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retrieve backup {backup_id}: {e}")
            return False
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup from local storage."""
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        
        try:
            if backup_path.exists():
                backup_path.unlink()
                logger.info(f"Backup deleted: {backup_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    async def list_backups(self) -> List[str]:
        """List available backup IDs."""
        backups = []
        for backup_file in self.backup_dir.glob("*.tar.gz"):
            backup_id = backup_file.stem
            backups.append(backup_id)
        return sorted(backups)


class S3BackupStorage(BackupStorage):
    """S3-compatible backup storage."""
    
    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1"):
        if not AWS_AVAILABLE:
            raise RuntimeError("AWS SDK not available - install boto3")
        
        self.bucket = bucket
        self.prefix = prefix
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Verify bucket access
        try:
            self.s3_client.head_bucket(Bucket=bucket)
            logger.info(f"S3 backup storage initialized: s3://{bucket}/{prefix}")
        except ClientError as e:
            logger.error(f"Cannot access S3 bucket {bucket}: {e}")
            raise
    
    def _get_s3_key(self, backup_id: str) -> str:
        """Get S3 key for backup."""
        key = f"{backup_id}.tar.gz"
        return f"{self.prefix}/{key}" if self.prefix else key
    
    async def store_backup(self, source_path: str, backup_id: str) -> str:
        """Store backup to S3."""
        # Create local compressed backup first
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create compressed archive
            with tarfile.open(temp_path, 'w:gz') as tar:
                if os.path.isdir(source_path):
                    tar.add(source_path, arcname=Path(source_path).name)
                else:
                    tar.add(source_path, arcname=Path(source_path).name)
            
            # Upload to S3
            s3_key = self._get_s3_key(backup_id)
            
            def upload_file():
                self.s3_client.upload_file(temp_path, self.bucket, s3_key)
            
            # Run upload in thread to avoid blocking
            await asyncio.get_event_loop().run_in_executor(None, upload_file)
            
            s3_path = f"s3://{self.bucket}/{s3_key}"
            logger.info(f"Backup stored to S3: {s3_path}")
            return s3_path
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    async def retrieve_backup(self, backup_id: str, destination_path: str) -> bool:
        """Retrieve backup from S3."""
        s3_key = self._get_s3_key(backup_id)
        
        # Download to temporary file first
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            def download_file():
                self.s3_client.download_file(self.bucket, s3_key, temp_path)
            
            await asyncio.get_event_loop().run_in_executor(None, download_file)
            
            # Extract archive
            with tarfile.open(temp_path, 'r:gz') as tar:
                tar.extractall(destination_path)
            
            logger.info(f"Backup retrieved from S3 to: {destination_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to retrieve backup {backup_id} from S3: {e}")
            return False
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup from S3."""
        s3_key = self._get_s3_key(backup_id)
        
        try:
            def delete_object():
                self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            
            await asyncio.get_event_loop().run_in_executor(None, delete_object)
            logger.info(f"Backup deleted from S3: s3://{self.bucket}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete backup {backup_id} from S3: {e}")
            return False
    
    async def list_backups(self) -> List[str]:
        """List available backup IDs."""
        try:
            def list_objects():
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=f"{self.prefix}/" if self.prefix else ""
                )
                return response.get('Contents', [])
            
            objects = await asyncio.get_event_loop().run_in_executor(None, list_objects)
            
            backup_ids = []
            for obj in objects:
                key = obj['Key']
                if key.endswith('.tar.gz'):
                    # Extract backup ID from key
                    backup_id = Path(key).stem
                    backup_ids.append(backup_id)
            
            return sorted(backup_ids)
            
        except ClientError as e:
            logger.error(f"Failed to list S3 backups: {e}")
            return []


class BackupManager:
    """Comprehensive backup and recovery manager."""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backends
        self.local_storage = LocalBackupStorage(config.backup_dir)
        
        self.s3_storage: Optional[S3BackupStorage] = None
        if config.s3_bucket and AWS_AVAILABLE:
            try:
                self.s3_storage = S3BackupStorage(
                    config.s3_bucket, 
                    config.s3_prefix, 
                    config.s3_region
                )
            except Exception as e:
                logger.warning(f"Failed to initialize S3 storage: {e}")
        
        # Metadata storage
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.metadata: Dict[str, Dict[str, Any]] = self._load_metadata()
        
        # Background scheduler
        self._scheduler_running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Metrics
        self.backup_metrics = {
            'total_backups': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'total_restored': 0,
            'last_backup_time': None,
            'last_restore_time': None
        }
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load backup metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading backup metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save backup metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving backup metadata: {e}")
    
    def _generate_backup_id(self, source_path: str) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = Path(source_path).name
        return f"{source_name}_{timestamp}"
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        hash_obj = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    @retry_with_backoff(max_attempts=3, strategy=ExponentialBackoffStrategy(max_delay=30.0))
    async def create_backup(
        self, 
        source_path: str, 
        backup_type: str = "full",
        tags: Dict[str, str] = None,
        remote_storage: bool = True
    ) -> Optional[str]:
        """Create a backup of the specified path."""
        
        if not os.path.exists(source_path):
            logger.error(f"Source path does not exist: {source_path}")
            return None
        
        backup_id = self._generate_backup_id(source_path)
        self.backup_metrics['total_backups'] += 1
        
        try:
            logger.info(f"Creating backup {backup_id} from {source_path}")
            start_time = time.time()
            
            # Store to local storage first
            local_backup_path = await self.local_storage.store_backup(source_path, backup_id)
            
            # Calculate metadata
            backup_stat = os.stat(local_backup_path)
            original_size = self._get_path_size(source_path)
            compression_ratio = original_size / backup_stat.st_size if backup_stat.st_size > 0 else 1.0
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                created_at=datetime.now(),
                backup_type=backup_type,
                source_path=source_path,
                backup_path=local_backup_path,
                file_count=self._get_file_count(source_path),
                total_size_bytes=backup_stat.st_size,
                checksum=self._calculate_checksum(local_backup_path),
                compression_ratio=compression_ratio,
                tags=tags or {}
            )
            
            # Store to remote if configured and requested
            if remote_storage and self.s3_storage:
                try:
                    s3_path = await self.s3_storage.store_backup(source_path, backup_id)
                    metadata.tags['s3_path'] = s3_path
                except Exception as e:
                    logger.warning(f"Failed to store backup to S3: {e}")
            
            # Save metadata
            self.metadata[backup_id] = metadata.to_dict()
            self._save_metadata()
            
            backup_time = time.time() - start_time
            self.backup_metrics['successful_backups'] += 1
            self.backup_metrics['last_backup_time'] = datetime.now().isoformat()
            
            logger.info(
                f"Backup {backup_id} created successfully in {backup_time:.2f}s "
                f"(compression: {compression_ratio:.2f}x)"
            )
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            return backup_id
            
        except Exception as e:
            self.backup_metrics['failed_backups'] += 1
            logger.error(f"Failed to create backup {backup_id}: {e}")
            return None
    
    def _get_path_size(self, path: str) -> int:
        """Get total size of path (file or directory)."""
        if os.path.isfile(path):
            return os.path.getsize(path)
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except OSError:
                    pass
        return total_size
    
    def _get_file_count(self, path: str) -> int:
        """Get total number of files in path."""
        if os.path.isfile(path):
            return 1
        
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(path):
            file_count += len(filenames)
        return file_count
    
    async def restore_backup(
        self, 
        backup_id: str, 
        destination_path: str,
        verify_checksum: bool = True,
        prefer_local: bool = True
    ) -> bool:
        """Restore a backup to the specified destination."""
        
        if backup_id not in self.metadata:
            logger.error(f"Backup {backup_id} not found in metadata")
            return False
        
        try:
            logger.info(f"Restoring backup {backup_id} to {destination_path}")
            start_time = time.time()
            
            # Try local storage first if preferred and available
            success = False
            if prefer_local:
                success = await self.local_storage.retrieve_backup(backup_id, destination_path)
            
            # Try S3 if local failed or not preferred
            if not success and self.s3_storage:
                success = await self.s3_storage.retrieve_backup(backup_id, destination_path)
            
            if not success:
                logger.error(f"Failed to restore backup {backup_id}")
                return False
            
            # Verify checksum if requested
            if verify_checksum and self.config.verify_backups:
                if not await self._verify_backup_integrity(backup_id):
                    logger.error(f"Backup {backup_id} failed integrity check")
                    return False
            
            restore_time = time.time() - start_time
            self.backup_metrics['total_restored'] += 1
            self.backup_metrics['last_restore_time'] = datetime.now().isoformat()
            
            logger.info(f"Backup {backup_id} restored successfully in {restore_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {e}")
            return False
    
    async def _verify_backup_integrity(self, backup_id: str) -> bool:
        """Verify backup integrity using checksum."""
        if backup_id not in self.metadata:
            return False
        
        metadata = self.metadata[backup_id]
        stored_checksum = metadata.get('checksum')
        
        if not stored_checksum:
            logger.warning(f"No checksum available for backup {backup_id}")
            return True  # Skip verification if no checksum
        
        # Check local backup if exists
        local_backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        if local_backup_path.exists():
            current_checksum = self._calculate_checksum(str(local_backup_path))
            return current_checksum == stored_checksum
        
        logger.warning(f"Local backup file not found for verification: {backup_id}")
        return True  # Skip if file not available locally
    
    async def delete_backup(self, backup_id: str, delete_remote: bool = True) -> bool:
        """Delete a backup from all storage locations."""
        
        if backup_id not in self.metadata:
            logger.error(f"Backup {backup_id} not found")
            return False
        
        success = True
        
        try:
            # Delete from local storage
            if not await self.local_storage.delete_backup(backup_id):
                success = False
            
            # Delete from S3 if configured
            if delete_remote and self.s3_storage:
                if not await self.s3_storage.delete_backup(backup_id):
                    success = False
            
            # Remove from metadata
            if success:
                del self.metadata[backup_id]
                self._save_metadata()
                logger.info(f"Backup {backup_id} deleted successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    async def list_backups(self, source_path: str = None) -> List[BackupMetadata]:
        """List available backups."""
        backups = []
        
        for backup_id, metadata_dict in self.metadata.items():
            metadata = BackupMetadata.from_dict(metadata_dict)
            
            if source_path is None or metadata.source_path == source_path:
                backups.append(metadata)
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x.created_at, reverse=True)
        return backups
    
    async def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy."""
        
        # Get all backups sorted by creation date
        all_backups = await self.list_backups()
        
        # Group by source path for per-source retention
        backups_by_source = {}
        for backup in all_backups:
            source = backup.source_path
            if source not in backups_by_source:
                backups_by_source[source] = []
            backups_by_source[source].append(backup)
        
        # Apply retention policy
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        for source_path, backups in backups_by_source.items():
            # Keep max_backups most recent backups
            if len(backups) > self.config.max_backups:
                backups_to_delete = backups[self.config.max_backups:]
                
                for backup in backups_to_delete:
                    logger.info(f"Deleting old backup (count limit): {backup.backup_id}")
                    await self.delete_backup(backup.backup_id)
            
            # Delete backups older than retention period
            for backup in backups:
                if backup.created_at < cutoff_date:
                    logger.info(f"Deleting old backup (age limit): {backup.backup_id}")
                    await self.delete_backup(backup.backup_id)
    
    def start_scheduled_backups(self, sources: List[Dict[str, Any]]):
        """Start scheduled backup operations."""
        if self._scheduler_running:
            logger.warning("Backup scheduler already running")
            return
        
        def run_scheduler():
            # Schedule backups for each source
            for source_config in sources:
                source_path = source_config['path']
                interval_hours = source_config.get('interval_hours', self.config.backup_interval_hours)
                
                schedule.every(interval_hours).hours.do(
                    lambda path=source_path: asyncio.run(
                        self.create_backup(path, tags={'scheduled': 'true'})
                    )
                )
            
            logger.info(f"Scheduled backups for {len(sources)} sources")
            
            # Run scheduler loop
            while self._scheduler_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._scheduler_thread.start()
        
        logger.info("Backup scheduler started")
    
    def stop_scheduled_backups(self):
        """Stop scheduled backup operations."""
        if not self._scheduler_running:
            return
        
        self._scheduler_running = False
        schedule.clear()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        logger.info("Backup scheduler stopped")
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_size = 0
        oldest_backup = None
        newest_backup = None
        
        for backup_data in self.metadata.values():
            total_size += backup_data.get('total_size_bytes', 0)
            
            created_at = datetime.fromisoformat(backup_data['created_at'])
            if oldest_backup is None or created_at < oldest_backup:
                oldest_backup = created_at
            if newest_backup is None or created_at > newest_backup:
                newest_backup = created_at
        
        return {
            'total_backups': len(self.metadata),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_backup': oldest_backup.isoformat() if oldest_backup else None,
            'newest_backup': newest_backup.isoformat() if newest_backup else None,
            'metrics': self.backup_metrics.copy(),
            'local_storage_available': True,
            's3_storage_available': self.s3_storage is not None,
            'scheduler_running': self._scheduler_running
        }


# Global backup manager instance
_backup_manager: Optional[BackupManager] = None

def get_backup_manager(config: BackupConfig = None) -> BackupManager:
    """Get global backup manager instance."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager(config or BackupConfig())
    return _backup_manager


def setup_cache_backup(cache_paths: List[str], config: BackupConfig = None):
    """Setup automatic backup for cache directories."""
    manager = get_backup_manager(config)
    
    sources = []
    for cache_path in cache_paths:
        sources.append({
            'path': cache_path,
            'interval_hours': 6,  # Backup caches every 6 hours
        })
    
    manager.start_scheduled_backups(sources)
    logger.info(f"Cache backup setup completed for {len(cache_paths)} paths")