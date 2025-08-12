"""
Comprehensive health check and system diagnostics for OdorDiff-2.
"""

import asyncio
import time
import psutil
import logging
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import socket
import subprocess
import sys

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    duration_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms
        }


class BaseHealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, warning_threshold: float = 5.0, critical_threshold: float = 10.0):
        self.name = name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        start_time = time.time()
        
        try:
            result = await self._check_implementation()
            
            # Determine status based on thresholds if not set
            if result.status == HealthStatus.UNKNOWN:
                duration_s = result.duration_ms / 1000
                if duration_s > self.critical_threshold:
                    result.status = HealthStatus.CRITICAL
                elif duration_s > self.warning_threshold:
                    result.status = HealthStatus.WARNING
                else:
                    result.status = HealthStatus.HEALTHY
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time(),
                duration_ms=duration_ms
            )
    
    async def _check_implementation(self) -> HealthCheckResult:
        """Override this method in subclasses."""
        raise NotImplementedError


class SystemResourceCheck(BaseHealthCheck):
    """Check system resource usage."""
    
    def __init__(self):
        super().__init__("system_resources", warning_threshold=80.0, critical_threshold=95.0)
    
    async def _check_implementation(self) -> HealthCheckResult:
        start_time = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Load average (Unix-like systems)
        load_avg = None
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            # Windows doesn't have load average
            pass
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        # Process count
        process_count = len(psutil.pids())
        
        details = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_total_gb': round(memory.total / (1024**3), 2),
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_percent': disk_percent,
            'disk_total_gb': round(disk.total / (1024**3), 2),
            'disk_free_gb': round(disk.free / (1024**3), 2),
            'process_count': process_count,
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv
        }
        
        if load_avg:
            details.update({
                'load_1min': load_avg[0],
                'load_5min': load_avg[1],
                'load_15min': load_avg[2]
            })
        
        # Determine overall status
        max_usage = max(cpu_percent, memory_percent, disk_percent)
        
        if max_usage > self.critical_threshold:
            status = HealthStatus.CRITICAL
            message = f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%"
        elif max_usage > self.warning_threshold:
            status = HealthStatus.WARNING
            message = f"Elevated resource usage: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Resource usage normal: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%"
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details=details,
            timestamp=time.time(),
            duration_ms=duration_ms
        )


class ModelHealthCheck(BaseHealthCheck):
    """Check ML model health and availability."""
    
    def __init__(self, model_registry: Dict[str, Any] = None):
        super().__init__("model_health")
        self.model_registry = model_registry or {}
    
    async def _check_implementation(self) -> HealthCheckResult:
        start_time = time.time()
        
        details = {
            'models_loaded': 0,
            'models_failed': 0,
            'model_status': {},
            'memory_usage_mb': 0
        }
        
        # Check if model files exist
        model_paths = [
            'checkpoints/pretrained/odordiff2-safe-v1.pt',
            'checkpoints/property_models/property_predictor.pt',
            'checkpoints/property_models/odor_predictor.pt',
            'checkpoints/property_models/synthesis_predictor.pt',
            'checkpoints/property_models/safety_predictor.pt'
        ]
        
        available_models = 0
        for model_path in model_paths:
            if Path(model_path).exists():
                available_models += 1
                details['model_status'][model_path] = 'available'
            else:
                details['model_status'][model_path] = 'missing'
        
        # Try to load a simple model check
        try:
            import torch
            # Check GPU availability
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count()
            
            details.update({
                'torch_version': torch.__version__,
                'gpu_available': gpu_available,
                'gpu_count': gpu_count
            })
            
            if gpu_available:
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    details[f'gpu_{i}'] = {
                        'name': gpu_name,
                        'memory_gb': round(gpu_memory, 2)
                    }
                    
        except ImportError:
            details['torch_available'] = False
        
        details['models_available'] = available_models
        details['models_total'] = len(model_paths)
        
        # Determine status
        if available_models == 0:
            status = HealthStatus.CRITICAL
            message = "No model files found"
        elif available_models < len(model_paths) // 2:
            status = HealthStatus.WARNING
            message = f"Only {available_models}/{len(model_paths)} models available"
        else:
            status = HealthStatus.HEALTHY
            message = f"Models available: {available_models}/{len(model_paths)}"
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details=details,
            timestamp=time.time(),
            duration_ms=duration_ms
        )


class DatabaseHealthCheck(BaseHealthCheck):
    """Check database connectivity and performance."""
    
    def __init__(self, connection_string: Optional[str] = None):
        super().__init__("database")
        self.connection_string = connection_string
    
    async def _check_implementation(self) -> HealthCheckResult:
        start_time = time.time()
        
        if not self.connection_string:
            # No database configured - this is OK for OdorDiff-2
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="No database configured (using file-based storage)",
                details={'database_type': 'none'},
                timestamp=time.time(),
                duration_ms=(time.time() - start_time) * 1000
            )
        
        # If database is configured, test connection
        try:
            # This would be database-specific logic
            # For now, just check if we can resolve the host
            details = {
                'connection_string': self.connection_string[:50] + "...",  # Truncate for security
                'connection_test': 'skipped'
            }
            
            status = HealthStatus.HEALTHY
            message = "Database health check skipped (no implementation)"
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Database connection failed: {str(e)}"
            details = {'error': str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details=details,
            timestamp=time.time(),
            duration_ms=duration_ms
        )


class CacheHealthCheck(BaseHealthCheck):
    """Check cache system health."""
    
    def __init__(self, cache_instance=None):
        super().__init__("cache")
        self.cache_instance = cache_instance
    
    async def _check_implementation(self) -> HealthCheckResult:
        start_time = time.time()
        
        # Test basic cache operations
        test_key = f"health_check_{int(time.time())}"
        test_value = {"timestamp": time.time(), "test": True}
        
        details = {
            'cache_available': False,
            'read_test': False,
            'write_test': False,
            'cache_size': 0
        }
        
        try:
            if self.cache_instance:
                # Test write
                await self.cache_instance.set(test_key, test_value, ttl=60)
                details['write_test'] = True
                
                # Test read
                retrieved = await self.cache_instance.get(test_key)
                details['read_test'] = retrieved is not None
                
                # Get cache size if possible
                try:
                    details['cache_size'] = len(self.cache_instance)
                except:
                    details['cache_size'] = 'unknown'
                
                # Cleanup
                await self.cache_instance.delete(test_key)
                
                details['cache_available'] = True
                status = HealthStatus.HEALTHY
                message = "Cache system operational"
                
            else:
                # No cache configured
                status = HealthStatus.HEALTHY
                message = "No cache system configured"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Cache system failure: {str(e)}"
            details['error'] = str(e)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details=details,
            timestamp=time.time(),
            duration_ms=duration_ms
        )


class APIEndpointCheck(BaseHealthCheck):
    """Check API endpoint availability."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__("api_endpoints")
        self.base_url = base_url
    
    async def _check_implementation(self) -> HealthCheckResult:
        start_time = time.time()
        
        endpoints_to_check = [
            "/health",
            "/api/v1/info",
            "/api/v1/generate"  # We'll do a simple GET to check if endpoint exists
        ]
        
        details = {
            'base_url': self.base_url,
            'endpoints': {}
        }
        
        healthy_endpoints = 0
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints_to_check:
                    url = f"{self.base_url}{endpoint}"
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            details['endpoints'][endpoint] = {
                                'status_code': response.status,
                                'response_time_ms': 0,  # Would measure this properly
                                'available': response.status < 500
                            }
                            if response.status < 500:
                                healthy_endpoints += 1
                    except Exception as e:
                        details['endpoints'][endpoint] = {
                            'error': str(e),
                            'available': False
                        }
                        
        except ImportError:
            # aiohttp not available, skip endpoint checks
            status = HealthStatus.WARNING
            message = "Cannot check API endpoints (aiohttp not available)"
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                duration_ms=duration_ms
            )
        
        # Determine overall status
        if healthy_endpoints == len(endpoints_to_check):
            status = HealthStatus.HEALTHY
            message = f"All {healthy_endpoints} endpoints healthy"
        elif healthy_endpoints > 0:
            status = HealthStatus.WARNING
            message = f"{healthy_endpoints}/{len(endpoints_to_check)} endpoints healthy"
        else:
            status = HealthStatus.CRITICAL
            message = "No endpoints responding"
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details=details,
            timestamp=time.time(),
            duration_ms=duration_ms
        )


class SecurityHealthCheck(BaseHealthCheck):
    """Check security-related health indicators."""
    
    def __init__(self):
        super().__init__("security")
    
    async def _check_implementation(self) -> HealthCheckResult:
        start_time = time.time()
        
        details = {
            'ssl_enabled': False,
            'auth_enabled': False,
            'rate_limiting_enabled': False,
            'input_validation_enabled': True,  # Assume enabled since we have validation
            'security_headers': {},
            'file_permissions': {}
        }
        
        warnings = []
        
        # Check file permissions on sensitive files
        sensitive_files = [
            'checkpoints/',
            'config/',
            '.env'
        ]
        
        for file_path in sensitive_files:
            path = Path(file_path)
            if path.exists():
                try:
                    stat = path.stat()
                    # Check if readable by others (Unix-like systems)
                    if hasattr(stat, 'st_mode'):
                        mode = stat.st_mode
                        details['file_permissions'][file_path] = oct(mode)[-3:]
                        
                        # Check for overly permissive permissions
                        if mode & 0o004:  # World readable
                            warnings.append(f"{file_path} is world-readable")
                except Exception as e:
                    details['file_permissions'][file_path] = f"Error: {e}"
        
        # Check environment variables for secrets
        env_warnings = []
        dangerous_env_vars = ['API_KEY', 'SECRET', 'PASSWORD', 'TOKEN']
        for var in dangerous_env_vars:
            if var in sys.environ:
                env_warnings.append(f"Environment variable {var} detected")
        
        if env_warnings:
            warnings.extend(env_warnings)
            details['environment_warnings'] = env_warnings
        
        # Overall security assessment
        if warnings:
            status = HealthStatus.WARNING
            message = f"Security warnings detected: {len(warnings)} issues"
            details['warnings'] = warnings
        else:
            status = HealthStatus.HEALTHY
            message = "No security issues detected"
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details=details,
            timestamp=time.time(),
            duration_ms=duration_ms
        )


class HealthMonitor:
    """Central health monitoring system."""
    
    def __init__(self):
        self.checks: List[BaseHealthCheck] = []
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_interval = 60  # seconds
        self.running = False
        self._monitor_task = None
        
        # Add default health checks
        self.add_check(SystemResourceCheck())
        self.add_check(ModelHealthCheck())
        self.add_check(DatabaseHealthCheck())
        self.add_check(CacheHealthCheck())
        self.add_check(SecurityHealthCheck())
    
    def add_check(self, health_check: BaseHealthCheck):
        """Add a health check."""
        self.checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_check(self, check_name: str):
        """Remove a health check by name."""
        self.checks = [check for check in self.checks if check.name != check_name]
        if check_name in self.last_results:
            del self.last_results[check_name]
        logger.info(f"Removed health check: {check_name}")
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        # Run checks concurrently
        tasks = [check.check() for check in self.checks]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for check, result in zip(self.checks, check_results):
            if isinstance(result, Exception):
                # Handle failed check
                result = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check exception: {str(result)}",
                    details={'exception': str(result)},
                    timestamp=time.time(),
                    duration_ms=0
                )
            
            results[check.name] = result
            self.last_results[check.name] = result
        
        return results
    
    async def run_check(self, check_name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        check = next((c for c in self.checks if c.name == check_name), None)
        if not check:
            return None
        
        result = await check.check()
        self.last_results[check.name] = result
        return result
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.last_results:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'No health checks have been run',
                'last_check': None,
                'checks': {}
            }
        
        # Determine overall status
        statuses = [result.status for result in self.last_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY
        
        critical_count = sum(1 for s in statuses if s == HealthStatus.CRITICAL)
        warning_count = sum(1 for s in statuses if s == HealthStatus.WARNING)
        healthy_count = sum(1 for s in statuses if s == HealthStatus.HEALTHY)
        
        return {
            'status': overall_status.value,
            'message': f"System health: {healthy_count} healthy, {warning_count} warnings, {critical_count} critical",
            'last_check': max(result.timestamp for result in self.last_results.values()),
            'summary': {
                'healthy': healthy_count,
                'warnings': warning_count,
                'critical': critical_count,
                'total': len(self.last_results)
            },
            'checks': {name: result.to_dict() for name, result in self.last_results.items()}
        }
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.running:
            logger.warning("Health monitoring is already running")
            return
        
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started health monitoring (interval: {self.check_interval}s)")
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                results = await self.run_all_checks()
                
                # Log any critical issues
                for name, result in results.items():
                    if result.status == HealthStatus.CRITICAL:
                        logger.error(f"CRITICAL health issue in {name}: {result.message}")
                    elif result.status == HealthStatus.WARNING:
                        logger.warning(f"Health warning in {name}: {result.message}")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get health monitoring metrics."""
        if not self.last_results:
            return {}
        
        metrics = {
            'health_check_count': len(self.last_results),
            'last_check_timestamp': max(result.timestamp for result in self.last_results.values()),
            'average_check_duration_ms': sum(result.duration_ms for result in self.last_results.values()) / len(self.last_results),
            'status_distribution': {}
        }
        
        # Count status distribution
        for status in HealthStatus:
            count = sum(1 for result in self.last_results.values() if result.status == status)
            metrics['status_distribution'][status.value] = count
        
        return metrics


# Global health monitor instance
health_monitor = HealthMonitor()


async def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    return health_monitor.get_overall_status()


async def run_health_check(check_name: str) -> Optional[Dict[str, Any]]:
    """Run a specific health check."""
    result = await health_monitor.run_check(check_name)
    return result.to_dict() if result else None


async def run_all_health_checks() -> Dict[str, Any]:
    """Run all health checks."""
    results = await health_monitor.run_all_checks()
    return {name: result.to_dict() for name, result in results.items()}


# Example usage and testing
if __name__ == "__main__":
    async def test_health_checks():
        """Test health check system."""
        print("Running health checks...")
        
        # Run all checks
        results = await run_all_health_checks()
        
        print("\nHealth Check Results:")
        print("=" * 50)
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Status: {result['status']}")
            print(f"  Message: {result['message']}")
            print(f"  Duration: {result['duration_ms']:.1f}ms")
            
            if result['details']:
                print("  Details:")
                for key, value in result['details'].items():
                    if isinstance(value, dict):
                        print(f"    {key}: {json.dumps(value, indent=6)}")
                    else:
                        print(f"    {key}: {value}")
        
        # Overall status
        overall = await get_system_health()
        print(f"\nOVERALL SYSTEM STATUS: {overall['status'].upper()}")
        print(f"Message: {overall['message']}")
    
    # Run the test
    asyncio.run(test_health_checks())