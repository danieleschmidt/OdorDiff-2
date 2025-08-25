"""
Advanced metrics collection and reporting system.
"""

import time
import threading
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import psutil
import logging

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags
        }


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self, max_points: int = 10000, retention_hours: int = 24):
        self.max_points = max_points
        self.retention_seconds = retention_hours * 3600
        self.metrics = defaultdict(deque)
        self.lock = threading.RLock()
        
        # Performance counters
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
        
        # System metrics
        self.system_metrics_enabled = True
        self._last_system_collection = 0
        
    def increment(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value
            self._add_metric(name, self.counters[name], tags or {})
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        with self.lock:
            self.gauges[name] = value
            self._add_metric(name, value, tags or {})
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        with self.lock:
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._add_metric(name, value, tags or {})
    
    def timer(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, tags or {})
    
    def _add_metric(self, name: str, value: float, tags: Dict[str, str]):
        """Add metric point."""
        point = MetricPoint(name, value, tags=tags)
        
        with self.lock:
            self.metrics[name].append(point)
            
            # Enforce max points limit
            if len(self.metrics[name]) > self.max_points:
                self.metrics[name].popleft()
    
    def collect_system_metrics(self):
        """Collect system performance metrics."""
        if not self.system_metrics_enabled:
            return
            
        now = time.time()
        if now - self._last_system_collection < 60:  # Collect every minute
            return
            
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.gauge('system.cpu.usage', cpu_percent, {'type': 'percent'})
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.gauge('system.memory.usage', memory.percent, {'type': 'percent'})
            self.gauge('system.memory.used_bytes', memory.used, {'type': 'bytes'})
            self.gauge('system.memory.available_bytes', memory.available, {'type': 'bytes'})
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.gauge('system.disk.usage', (disk.used / disk.total) * 100, {'type': 'percent'})
            self.gauge('system.disk.used_bytes', disk.used, {'type': 'bytes'})
            self.gauge('system.disk.free_bytes', disk.free, {'type': 'bytes'})
            
            # Process metrics
            process = psutil.Process()
            self.gauge('process.memory.rss_bytes', process.memory_info().rss, {'type': 'bytes'})
            self.gauge('process.memory.vms_bytes', process.memory_info().vms, {'type': 'bytes'})
            self.gauge('process.cpu.usage', process.cpu_percent(), {'type': 'percent'})
            self.gauge('process.threads', process.num_threads(), {'type': 'count'})
            
            self._last_system_collection = now
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def get_metrics(self, name: str = None, since: float = None) -> List[Dict[str, Any]]:
        """Get metrics data."""
        self.collect_system_metrics()
        
        with self.lock:
            if name:
                metrics_data = self.metrics.get(name, deque())
            else:
                metrics_data = []
                for metric_name, points in self.metrics.items():
                    metrics_data.extend(points)
            
            # Filter by timestamp if specified
            if since:
                metrics_data = [point for point in metrics_data if point.timestamp >= since]
            else:
                metrics_data = list(metrics_data)
            
            return [point.to_dict() if hasattr(point, 'to_dict') else point for point in metrics_data]
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        with self.lock:
            values = self.histograms.get(name, [])
            if not values:
                return {}
            
            sorted_values = sorted(values)
            count = len(values)
            
            return {
                'count': count,
                'sum': sum(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / count,
                'median': sorted_values[count // 2],
                'p95': sorted_values[int(count * 0.95)] if count > 0 else 0,
                'p99': sorted_values[int(count * 0.99)] if count > 0 else 0
            }
    
    def cleanup_old_metrics(self):
        """Clean up old metric points."""
        cutoff = time.time() - self.retention_seconds
        
        with self.lock:
            for name in list(self.metrics.keys()):
                points = self.metrics[name]
                while points and points[0].timestamp < cutoff:
                    points.popleft()
                    
                # Remove empty metric series
                if not points:
                    del self.metrics[name]
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self.lock:
            # Counters
            for name, value in self.counters.items():
                lines.append(f'# TYPE {name} counter')
                lines.append(f'{name} {value}')
            
            # Gauges  
            for name, value in self.gauges.items():
                lines.append(f'# TYPE {name} gauge')
                lines.append(f'{name} {value}')
            
            # Histograms
            for name, values in self.histograms.items():
                if values:
                    stats = self.get_histogram_stats(name)
                    lines.append(f'# TYPE {name} histogram')
                    lines.append(f'{name}_count {stats["count"]}')
                    lines.append(f'{name}_sum {stats["sum"]}')
                    lines.append(f'{name}_bucket{{le="0.1"}} {sum(1 for v in values if v <= 0.1)}')
                    lines.append(f'{name}_bucket{{le="0.5"}} {sum(1 for v in values if v <= 0.5)}')
                    lines.append(f'{name}_bucket{{le="1.0"}} {sum(1 for v in values if v <= 1.0)}')
                    lines.append(f'{name}_bucket{{le="5.0"}} {sum(1 for v in values if v <= 5.0)}')
                    lines.append(f'{name}_bucket{{le="+Inf"}} {len(values)}')
        
        return '\n'.join(lines)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health-related metrics."""
        self.collect_system_metrics()
        
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histogram_stats': {name: self.get_histogram_stats(name) 
                                  for name in self.histograms.keys()},
                'total_metric_points': sum(len(points) for points in self.metrics.values()),
                'metric_series_count': len(self.metrics)
            }


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.histogram(f'{self.name}.duration', duration, self.tags)
            
            if exc_type:
                self.tags['status'] = 'error'
            else:
                self.tags['status'] = 'success'
                
            self.collector.increment(f'{self.name}.calls', 1, self.tags)


# Global metrics collector instance
_metrics_collector = None
_metrics_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience functions
def increment(name: str, value: int = 1, tags: Dict[str, str] = None):
    """Increment a counter metric."""
    get_metrics_collector().increment(name, value, tags)


def gauge(name: str, value: float, tags: Dict[str, str] = None):
    """Set a gauge metric."""
    get_metrics_collector().gauge(name, value, tags)


def histogram(name: str, value: float, tags: Dict[str, str] = None):
    """Record a histogram value."""
    get_metrics_collector().histogram(name, value, tags)


def timer(name: str, tags: Dict[str, str] = None):
    """Context manager for timing operations."""
    return get_metrics_collector().timer(name, tags)