"""
Comprehensive health check system for OdorDiff-2.
"""

import asyncio
import time
import psutil
import platform
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from ..utils.logging import get_logger
from ..utils.error_handling import safe_execute_async, timeout

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    duration_ms: float = 0
    timestamp: float = 0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()
        if self.details is None:
            self.details = {}


class HealthCheck:
    """Base health check class."""
    
    def __init__(self, name: str, timeout_seconds: float = 30.0, critical: bool = False):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.critical = critical
        self.last_result: Optional[HealthCheckResult] = None
        self.failure_count = 0
        self.last_success = time.time()
    
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self._perform_check(),
                timeout=self.timeout_seconds
            )
            
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms
            
            # Reset failure count on success
            if result.status == HealthStatus.HEALTHY:
                self.failure_count = 0
                self.last_success = time.time()
            else:
                self.failure_count += 1
            
            self.last_result = result
            return result
            
        except asyncio.TimeoutError:
            self.failure_count += 1
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.timeout_seconds}s",
                duration_ms=duration_ms,
                details={'timeout': True, 'failure_count': self.failure_count}
            )
            
            self.last_result = result
            return result
            
        except Exception as e:
            self.failure_count += 1
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                details={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'failure_count': self.failure_count
                }
            )
            
            self.last_result = result
            return result
    
    async def _perform_check(self) -> HealthCheckResult:
        """Override this method to implement actual health check logic."""
        raise NotImplementedError


class SystemResourcesCheck(HealthCheck):
    """Check system resource usage."""
    
    def __init__(self, cpu_threshold: float = 85.0, memory_threshold: float = 85.0, 
                 disk_threshold: float = 85.0, **kwargs):
        super().__init__("system_resources", **kwargs)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check system resource usage."""
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1.0)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        details = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk_percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
        }
        
        # Determine status based on thresholds
        issues = []
        
        if cpu_percent > self.cpu_threshold:
            issues.append(f"CPU usage high: {cpu_percent:.1f}%")
        
        if memory_percent > self.memory_threshold:
            issues.append(f"Memory usage high: {memory_percent:.1f}%")
        
        if disk_percent > self.disk_threshold:
            issues.append(f"Disk usage high: {disk_percent:.1f}%")
        
        if not issues:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="System resources within acceptable limits",
                details=details
            )
        elif len(issues) == 1 and cpu_percent < 95 and memory_percent < 95:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message=f"System resources elevated: {', '.join(issues)}",
                details=details
            )
        else:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"System resources critical: {', '.join(issues)}",
                details=details
            )


class DatabaseHealthCheck(HealthCheck):
    """Check database/cache connectivity and performance."""
    
    def __init__(self, cache_manager=None, **kwargs):
        super().__init__("database", **kwargs)
        self.cache_manager = cache_manager
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check database/cache health."""
        if not self.cache_manager:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="No cache manager configured",
                details={'cache_enabled': False}
            )
        
        try:
            # Test cache connectivity
            test_key = "__health_check_test__"
            test_value = {"timestamp": time.time(), "test": True}
            
            # Write test
            start_time = time.time()
            await self.cache_manager.set(test_key, test_value, expire=60)
            write_time = time.time() - start_time
            
            # Read test
            start_time = time.time()
            retrieved_value = await self.cache_manager.get(test_key)
            read_time = time.time() - start_time
            
            # Cleanup
            await self.cache_manager.delete(test_key)
            
            # Get cache stats
            cache_stats = await self.cache_manager.get_stats()
            
            details = {
                'write_time_ms': write_time * 1000,
                'read_time_ms': read_time * 1000,
                'test_successful': retrieved_value is not None,
                'cache_stats': cache_stats
            }
            
            # Determine status
            if retrieved_value is None:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.CRITICAL,
                    message="Cache read/write test failed",
                    details=details
                )
            elif write_time > 1.0 or read_time > 1.0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Cache performance degraded (write: {write_time*1000:.1f}ms, read: {read_time*1000:.1f}ms)",
                    details=details
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Cache connectivity and performance good",
                    details=details
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Cache health check failed: {str(e)}",
                details={'error': str(e), 'error_type': type(e).__name__}
            )


class ModelHealthCheck(HealthCheck):
    """Check model availability and performance."""
    
    def __init__(self, async_model=None, **kwargs):
        super().__init__("model", critical=True, **kwargs)
        self.async_model = async_model
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check model health."""
        if not self.async_model:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message="Model not initialized",
                details={'model_loaded': False}
            )
        
        try:
            # Test model with simple generation
            start_time = time.time()
            
            # Get model health info
            health_info = await self.async_model.health_check()
            
            generation_time = time.time() - start_time
            
            details = {
                'generation_time_ms': generation_time * 1000,
                'model_health': health_info,
                'worker_count': getattr(self.async_model, 'max_workers', 0),
                'queue_size': len(getattr(self.async_model, '_request_queue', [])),
                'cache_enabled': getattr(self.async_model, 'enable_caching', False)
            }
            
            # Check if model is responsive
            if generation_time > 10.0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Model response time elevated: {generation_time*1000:.1f}ms",
                    details=details
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Model responsive and healthy",
                    details=details
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Model health check failed: {str(e)}",
                details={'error': str(e), 'error_type': type(e).__name__}
            )


class ExternalDependencyCheck(HealthCheck):
    """Check external dependencies."""
    
    def __init__(self, dependencies: List[str] = None, **kwargs):
        super().__init__("dependencies", **kwargs)
        self.dependencies = dependencies or []
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check external dependencies."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for dep in self.dependencies:
            try:
                if dep == "internet":
                    # Test internet connectivity
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get("https://httpbin.org/status/200", timeout=5) as response:
                            results[dep] = {
                                'status': 'healthy' if response.status == 200 else 'degraded',
                                'response_time': response.headers.get('X-Response-Time', 'unknown')
                            }
                            
                elif dep == "rdkit":
                    # Test RDKit functionality
                    try:
                        from rdkit import Chem
                        mol = Chem.MolFromSmiles("CCO")
                        results[dep] = {
                            'status': 'healthy' if mol is not None else 'unhealthy',
                            'version': getattr(Chem, '__version__', 'unknown')
                        }
                    except ImportError:
                        results[dep] = {'status': 'critical', 'error': 'RDKit not available'}
                        overall_status = HealthStatus.CRITICAL
                        
            except Exception as e:
                results[dep] = {
                    'status': 'critical',
                    'error': str(e)
                }
                if overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.DEGRADED
        
        # Determine overall status
        critical_count = sum(1 for r in results.values() if r.get('status') == 'critical')
        degraded_count = sum(1 for r in results.values() if r.get('status') == 'degraded')
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        
        return HealthCheckResult(
            name=self.name,
            status=overall_status,
            message=f"Dependencies check: {len(results)} checked",
            details={'dependencies': results}
        )


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.checks: List[HealthCheck] = []
        self.check_interval = 60  # seconds
        self._monitoring = False
        self._monitor_task = None
        self.last_results: Dict[str, HealthCheckResult] = {}
        
    def add_check(self, health_check: HealthCheck):
        """Add a health check."""
        self.checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_check(self, name: str):
        """Remove a health check by name."""
        self.checks = [c for c in self.checks if c.name != name]
        self.last_results.pop(name, None)
        logger.info(f"Removed health check: {name}")
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        logger.info("Running all health checks")
        results = {}
        
        # Run checks concurrently
        tasks = [check.check() for check in self.checks]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for check, result in zip(self.checks, check_results):
            if isinstance(result, Exception):
                # Handle exceptions during health checks
                result = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check exception: {str(result)}",
                    details={'exception': str(result)}
                )
            
            results[check.name] = result
            self.last_results[check.name] = result
        
        return results
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.last_results:
            # Run checks if no recent results
            await self.run_all_checks()
        
        # Calculate overall status
        statuses = [result.status for result in self.last_results.values()]
        critical_checks = [name for name, result in self.last_results.items() 
                         if result.status == HealthStatus.CRITICAL]
        unhealthy_checks = [name for name, result in self.last_results.items() 
                          if result.status == HealthStatus.UNHEALTHY]
        degraded_checks = [name for name, result in self.last_results.items() 
                         if result.status == HealthStatus.DEGRADED]
        
        # Determine overall status
        if critical_checks:
            overall_status = HealthStatus.CRITICAL
        elif unhealthy_checks:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_checks:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            'status': overall_status.value,
            'timestamp': time.time(),
            'checks_total': len(self.checks),
            'checks_healthy': len([s for s in statuses if s == HealthStatus.HEALTHY]),
            'checks_degraded': len(degraded_checks),
            'checks_unhealthy': len(unhealthy_checks),
            'checks_critical': len(critical_checks),
            'critical_checks': critical_checks,
            'unhealthy_checks': unhealthy_checks,
            'degraded_checks': degraded_checks,
            'details': {name: {
                'status': result.status.value,
                'message': result.message,
                'duration_ms': result.duration_ms,
                'timestamp': result.timestamp
            } for name, result in self.last_results.items()},
            'uptime': self._get_uptime(),
            'version': self._get_version_info()
        }
    
    def _get_uptime(self) -> Dict[str, float]:
        """Get system uptime information."""
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        return {
            'system_uptime_seconds': uptime_seconds,
            'system_boot_timestamp': boot_time,
        }
    
    def _get_version_info(self) -> Dict[str, str]:
        """Get version information."""
        return {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
        }
    
    async def start_monitoring(self, interval: int = 60):
        """Start continuous health monitoring."""
        self.check_interval = interval
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started health monitoring with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self._monitoring:
            try:
                await self.run_all_checks()
                
                # Log health summary
                overall_health = await self.get_overall_health()
                logger.info(
                    f"Health check completed: {overall_health['status'].upper()} "
                    f"({overall_health['checks_healthy']}/{overall_health['checks_total']} healthy)"
                )
                
                # Alert on critical issues
                if overall_health['status'] in ['critical', 'unhealthy']:
                    logger.error(
                        f"System health {overall_health['status'].upper()}: "
                        f"Critical: {overall_health['critical_checks']}, "
                        f"Unhealthy: {overall_health['unhealthy_checks']}"
                    )
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


def setup_default_health_checks(async_model=None, cache_manager=None):
    """Setup default health checks."""
    monitor = get_health_monitor()
    
    # System resources check
    monitor.add_check(SystemResourcesCheck(
        cpu_threshold=85.0,
        memory_threshold=85.0,
        disk_threshold=85.0
    ))
    
    # Database/Cache check
    if cache_manager:
        monitor.add_check(DatabaseHealthCheck(cache_manager=cache_manager))
    
    # Model check
    if async_model:
        monitor.add_check(ModelHealthCheck(async_model=async_model))
    
    # External dependencies check
    monitor.add_check(ExternalDependencyCheck(
        dependencies=["internet", "rdkit"]
    ))
    
    logger.info("Default health checks configured")