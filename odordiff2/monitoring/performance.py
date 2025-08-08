"""
Performance monitoring and optimization system.
"""

import time
import threading
import asyncio
import gc
import weakref
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import psutil
import resource
import tracemalloc
import cProfile
import io
import pstats

from ..utils.logging import get_logger
from .metrics import get_metrics_collector

logger = get_logger(__name__)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    threads: int
    open_files: int
    network_connections: int = 0
    gc_stats: Dict[str, int] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """Resource usage statistics."""
    peak_memory_mb: float
    avg_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    total_requests: int
    total_errors: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float


class MemoryTracker:
    """Memory usage tracking and leak detection."""
    
    def __init__(self, max_snapshots: int = 100):
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.tracemalloc_enabled = False
        self.tracked_objects: Set[weakref.ref] = set()
        self.allocation_stats = defaultdict(int)
        
    def start_tracemalloc(self):
        """Start memory tracing."""
        if not self.tracemalloc_enabled:
            tracemalloc.start(10)  # Keep top 10 frames
            self.tracemalloc_enabled = True
            logger.info("Memory tracing started")
    
    def stop_tracemalloc(self):
        """Stop memory tracing."""
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
            logger.info("Memory tracing stopped")
    
    def take_snapshot(self) -> PerformanceSnapshot:
        """Take a performance snapshot."""
        try:
            process = psutil.Process()
            
            # Basic metrics
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            cpu_percent = process.cpu_percent()
            
            # GC stats
            gc_stats = {f'gen_{i}': gc.get_count()[i] for i in range(3)}
            gc_stats['collected'] = gc.collected
            
            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                threads=process.num_threads(),
                open_files=process.num_fds() if hasattr(process, 'num_fds') else 0,
                gc_stats=gc_stats
            )
            
            self.snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            logger.error(f"Error taking performance snapshot: {e}")
            return None
    
    def get_memory_usage_trend(self, minutes: int = 10) -> Dict[str, float]:
        """Get memory usage trend over time."""
        if not self.snapshots:
            return {}
        
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [
            s for s in self.snapshots 
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {}
        
        memory_values = [s.memory_mb for s in recent_snapshots]
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        
        return {
            'memory_trend_mb': memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0,
            'memory_avg_mb': statistics.mean(memory_values),
            'memory_peak_mb': max(memory_values),
            'cpu_avg_percent': statistics.mean(cpu_values),
            'cpu_peak_percent': max(cpu_values),
            'snapshots_count': len(recent_snapshots)
        }
    
    def detect_memory_leaks(self, threshold_mb: float = 50.0) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 10:
            return {'status': 'insufficient_data'}
        
        # Check memory trend over last snapshots
        recent_memory = [s.memory_mb for s in list(self.snapshots)[-10:]]
        
        if len(recent_memory) < 10:
            return {'status': 'insufficient_data'}
        
        # Calculate trend
        x = list(range(len(recent_memory)))
        y = recent_memory
        
        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Check if slope indicates memory leak
        is_leak = slope > 1.0  # >1MB increase per snapshot
        severity = 'high' if slope > 5.0 else 'medium' if slope > 2.0 else 'low'
        
        result = {
            'status': 'leak_detected' if is_leak else 'normal',
            'memory_trend_mb_per_snapshot': slope,
            'severity': severity if is_leak else 'none',
            'current_memory_mb': recent_memory[-1],
            'memory_increase_mb': recent_memory[-1] - recent_memory[0]
        }
        
        # Get top memory allocators if tracemalloc is enabled
        if self.tracemalloc_enabled:
            try:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')[:10]
                
                result['top_allocators'] = [
                    {
                        'file': stat.traceback.format()[0],
                        'size_mb': stat.size / 1024 / 1024,
                        'count': stat.count
                    }
                    for stat in top_stats
                ]
            except Exception as e:
                logger.error(f"Error getting tracemalloc stats: {e}")
        
        return result
    
    def track_object(self, obj: Any, name: str = None):
        """Track object for memory monitoring."""
        ref = weakref.ref(obj)
        self.tracked_objects.add(ref)
        if name:
            self.allocation_stats[name] += 1
    
    def get_tracked_objects_count(self) -> int:
        """Get count of tracked objects still alive."""
        # Clean up dead references
        self.tracked_objects = {ref for ref in self.tracked_objects if ref() is not None}
        return len(self.tracked_objects)


class PerformanceProfiler:
    """Code performance profiler."""
    
    def __init__(self):
        self.profiles = {}
        self.active_profilers = {}
        
    def start_profiling(self, name: str = "default"):
        """Start profiling."""
        profiler = cProfile.Profile()
        profiler.enable()
        self.active_profilers[name] = profiler
        logger.info(f"Started profiling: {name}")
    
    def stop_profiling(self, name: str = "default") -> Optional[str]:
        """Stop profiling and return results."""
        if name not in self.active_profilers:
            return None
        
        profiler = self.active_profilers.pop(name)
        profiler.disable()
        
        # Generate profile report
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        report = s.getvalue()
        self.profiles[name] = {
            'timestamp': time.time(),
            'report': report
        }
        
        logger.info(f"Stopped profiling: {name}")
        return report
    
    def profile_function(self, name: str = None):
        """Decorator to profile a function."""
        def decorator(func):
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                self.start_profiling(profile_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.stop_profiling(profile_name)
            
            return wrapper
        return decorator


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, 
                 snapshot_interval: int = 30,
                 alert_memory_threshold: float = 80.0,
                 alert_cpu_threshold: float = 90.0):
        
        self.snapshot_interval = snapshot_interval
        self.alert_memory_threshold = alert_memory_threshold
        self.alert_cpu_threshold = alert_cpu_threshold
        
        self.memory_tracker = MemoryTracker()
        self.profiler = PerformanceProfiler()
        self.metrics = get_metrics_collector()
        
        self.response_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        
        self.alerts_sent = set()
        self.alert_callbacks: List[Callable] = []
        
        self._monitoring = False
        self._monitor_task = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self.memory_tracker.start_tracemalloc()
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        self.memory_tracker.stop_tracemalloc()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Take performance snapshot
                snapshot = self.memory_tracker.take_snapshot()
                if snapshot:
                    await self._process_snapshot(snapshot)
                
                # Check for memory leaks
                leak_info = self.memory_tracker.detect_memory_leaks()
                if leak_info['status'] == 'leak_detected':
                    await self._handle_memory_leak_alert(leak_info)
                
                # Collect metrics
                self._collect_performance_metrics(snapshot)
                
                await asyncio.sleep(self.snapshot_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.snapshot_interval)
    
    async def _process_snapshot(self, snapshot: PerformanceSnapshot):
        """Process performance snapshot."""
        # Check alert thresholds
        if snapshot.memory_percent > self.alert_memory_threshold:
            alert_key = f"memory_{int(time.time() // 300)}"  # 5-minute intervals
            if alert_key not in self.alerts_sent:
                await self._send_alert(
                    "high_memory",
                    f"Memory usage: {snapshot.memory_percent:.1f}%",
                    snapshot
                )
                self.alerts_sent.add(alert_key)
        
        if snapshot.cpu_percent > self.alert_cpu_threshold:
            alert_key = f"cpu_{int(time.time() // 300)}"
            if alert_key not in self.alerts_sent:
                await self._send_alert(
                    "high_cpu",
                    f"CPU usage: {snapshot.cpu_percent:.1f}%",
                    snapshot
                )
                self.alerts_sent.add(alert_key)
        
        # Clean up old alerts
        current_window = int(time.time() // 300)
        self.alerts_sent = {
            alert for alert in self.alerts_sent 
            if int(alert.split('_')[1]) >= current_window - 12  # Keep 1 hour
        }
    
    def _collect_performance_metrics(self, snapshot: PerformanceSnapshot):
        """Collect performance metrics."""
        if not snapshot:
            return
        
        self.metrics.set_gauge("performance_memory_mb", snapshot.memory_mb)
        self.metrics.set_gauge("performance_memory_percent", snapshot.memory_percent)
        self.metrics.set_gauge("performance_cpu_percent", snapshot.cpu_percent)
        self.metrics.set_gauge("performance_threads", snapshot.threads)
        self.metrics.set_gauge("performance_open_files", snapshot.open_files)
        
        for gen, count in snapshot.gc_stats.items():
            self.metrics.set_gauge(f"performance_gc_{gen}", count)
        
        # Response time metrics
        if self.response_times:
            recent_times = list(self.response_times)
            self.metrics.set_gauge("performance_avg_response_time", statistics.mean(recent_times))
            if len(recent_times) > 1:
                self.metrics.set_gauge("performance_p95_response_time", 
                                     statistics.quantiles(recent_times, n=20)[18])  # 95th percentile
                self.metrics.set_gauge("performance_p99_response_time", 
                                     statistics.quantiles(recent_times, n=100)[98])  # 99th percentile
    
    def record_request(self, endpoint: str, response_time: float, success: bool = True):
        """Record request performance."""
        self.response_times.append(response_time)
        self.request_counts[endpoint] += 1
        
        if not success:
            self.error_counts[endpoint] += 1
        
        # Record metrics
        self.metrics.record_histogram("request_duration", response_time, 
                                     tags={"endpoint": endpoint})
        self.metrics.increment_counter("requests_total", 
                                     tags={"endpoint": endpoint, "status": "success" if success else "error"})
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    async def _send_alert(self, alert_type: str, message: str, snapshot: PerformanceSnapshot):
        """Send performance alert."""
        alert_data = {
            'type': alert_type,
            'message': message,
            'timestamp': time.time(),
            'snapshot': snapshot
        }
        
        logger.warning(f"Performance alert: {alert_type} - {message}")
        
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def _handle_memory_leak_alert(self, leak_info: Dict[str, Any]):
        """Handle memory leak detection."""
        if leak_info['severity'] in ['medium', 'high']:
            await self._send_alert(
                "memory_leak",
                f"Memory leak detected: {leak_info['memory_increase_mb']:.1f}MB increase",
                None
            )
    
    def get_performance_summary(self) -> ResourceUsage:
        """Get performance summary."""
        snapshots = list(self.memory_tracker.snapshots)
        if not snapshots:
            return ResourceUsage(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        memory_values = [s.memory_mb for s in snapshots]
        cpu_values = [s.cpu_percent for s in snapshots]
        response_times = list(self.response_times)
        
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        
        return ResourceUsage(
            peak_memory_mb=max(memory_values) if memory_values else 0,
            avg_memory_mb=statistics.mean(memory_values) if memory_values else 0,
            peak_cpu_percent=max(cpu_values) if cpu_values else 0,
            avg_cpu_percent=statistics.mean(cpu_values) if cpu_values else 0,
            total_requests=total_requests,
            total_errors=total_errors,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
            p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else 0
        )
    
    def optimize_gc(self):
        """Optimize garbage collection."""
        # Force garbage collection
        collected = gc.collect()
        
        # Adjust GC thresholds based on memory usage
        current_memory = self.memory_tracker.snapshots[-1].memory_mb if self.memory_tracker.snapshots else 0
        
        if current_memory > 500:  # High memory usage
            # More aggressive GC
            gc.set_threshold(500, 10, 10)
        else:
            # Normal GC settings
            gc.set_threshold(700, 10, 10)
        
        logger.info(f"GC optimization: collected {collected} objects")
        return collected


# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor