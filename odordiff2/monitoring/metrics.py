"""
Advanced metrics collection and monitoring system.
"""

import time
import threading
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import psutil
import json
import os

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, timestamp: float = None, **metadata):
        """Add a point to the series."""
        if timestamp is None:
            timestamp = time.time()
        
        point = MetricPoint(
            name=self.name,
            value=value,
            timestamp=timestamp,
            tags=self.tags.copy(),
            metadata=metadata
        )
        self.points.append(point)
    
    def get_recent_points(self, seconds: int = 300) -> List[MetricPoint]:
        """Get points from last N seconds."""
        cutoff_time = time.time() - seconds
        return [p for p in self.points if p.timestamp >= cutoff_time]
    
    def get_statistics(self, seconds: int = 300) -> Dict[str, float]:
        """Get statistics for recent points."""
        recent_points = self.get_recent_points(seconds)
        if not recent_points:
            return {}
        
        values = [p.value for p in recent_points]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'latest': values[-1] if values else 0,
            'rate': len(values) / seconds  # points per second
        }


class MetricsCollector:
    """High-performance metrics collection system."""
    
    def __init__(self, max_series: int = 1000, export_interval: int = 60):
        self.max_series = max_series
        self.export_interval = export_interval
        self.series: Dict[str, MetricSeries] = {}
        self.lock = threading.RLock()
        self.exporters: List[Callable] = []
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # System metrics
        self.system_metrics_enabled = True
        self.last_system_metrics = time.time()
        
        # Background export task
        self._export_task = None
        self._running = False
        
    def start(self):
        """Start background metrics collection."""
        if not self._running:
            self._running = True
            self._export_task = asyncio.create_task(self._export_loop())
            logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop metrics collection."""
        self._running = False
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, **metadata):
        """Record a metric value."""
        tags = tags or {}
        series_key = f"{name}:{json.dumps(tags, sort_keys=True)}"
        
        with self.lock:
            if series_key not in self.series:
                if len(self.series) >= self.max_series:
                    # Remove oldest series
                    oldest_key = min(self.series.keys(), 
                                   key=lambda k: self.series[k].points[0].timestamp if self.series[k].points else 0)
                    del self.series[oldest_key]
                
                self.series[series_key] = MetricSeries(name=name, tags=tags)
            
            self.series[series_key].add_point(value, **metadata)
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        with self.lock:
            key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
            self.counters[key] += value
            self.record_metric(f"{name}_total", self.counters[key], tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        with self.lock:
            key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
            self.gauges[key] = value
            self.record_metric(name, value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        with self.lock:
            key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
            self.histograms[key].append(value)
            
            # Keep only recent values
            cutoff_time = time.time() - 300  # 5 minutes
            self.histograms[key] = [
                v for v in self.histograms[key][-1000:]  # Keep last 1000 values
            ]
            
            # Record percentiles
            values = self.histograms[key]
            if values:
                percentiles = [50, 75, 90, 95, 99]
                for p in percentiles:
                    pct_value = statistics.quantiles(values, n=100)[p-1] if len(values) > 1 else values[0]
                    self.record_metric(f"{name}_p{p}", pct_value, tags)
    
    def time_function(self, name: str, tags: Dict[str, str] = None):
        """Decorator to time function execution."""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        self.record_histogram(f"{name}_duration", execution_time, tags)
                        self.increment_counter(f"{name}_success", tags=tags)
                        return result
                    except Exception as e:
                        execution_time = time.time() - start_time
                        self.record_histogram(f"{name}_duration", execution_time, tags)
                        self.increment_counter(f"{name}_error", tags={**(tags or {}), 'error_type': type(e).__name__})
                        raise
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        self.record_histogram(f"{name}_duration", execution_time, tags)
                        self.increment_counter(f"{name}_success", tags=tags)
                        return result
                    except Exception as e:
                        execution_time = time.time() - start_time
                        self.record_histogram(f"{name}_duration", execution_time, tags)
                        self.increment_counter(f"{name}_error", tags={**(tags or {}), 'error_type': type(e).__name__})
                        raise
                return sync_wrapper
        return decorator
    
    def collect_system_metrics(self):
        """Collect system metrics."""
        if not self.system_metrics_enabled:
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.set_gauge("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_percent", memory.percent)
            self.set_gauge("system_memory_used_bytes", memory.used)
            self.set_gauge("system_memory_available_bytes", memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.set_gauge("system_disk_percent", disk.percent)
            self.set_gauge("system_disk_used_bytes", disk.used)
            self.set_gauge("system_disk_free_bytes", disk.free)
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                self.increment_counter("system_network_bytes_sent", net_io.bytes_sent - getattr(self, '_last_bytes_sent', 0))
                self.increment_counter("system_network_bytes_recv", net_io.bytes_recv - getattr(self, '_last_bytes_recv', 0))
                self._last_bytes_sent = net_io.bytes_sent
                self._last_bytes_recv = net_io.bytes_recv
            except:
                pass
            
            # Process metrics
            process = psutil.Process()
            self.set_gauge("process_cpu_percent", process.cpu_percent())
            self.set_gauge("process_memory_bytes", process.memory_info().rss)
            self.set_gauge("process_threads", process.num_threads())
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self.lock:
            metrics = {}
            
            # Time series metrics
            for series_key, series in self.series.items():
                stats = series.get_statistics()
                if stats:
                    metrics[series_key] = {
                        'name': series.name,
                        'tags': series.tags,
                        'statistics': stats
                    }
            
            # Current counters and gauges
            metrics['counters'] = dict(self.counters)
            metrics['gauges'] = dict(self.gauges)
            
            # System metrics
            self.collect_system_metrics()
            
            return metrics
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self.lock:
            # Export counters
            for key, value in self.counters.items():
                metric_name, tags_json = key.split(':', 1)
                tags = json.loads(tags_json)
                tag_str = ','.join([f'{k}="{v}"' for k, v in tags.items()]) if tags else ''
                lines.append(f'{metric_name}{{}} {value}')
            
            # Export gauges
            for key, value in self.gauges.items():
                metric_name, tags_json = key.split(':', 1)
                tags = json.loads(tags_json)
                tag_str = ','.join([f'{k}="{v}"' for k, v in tags.items()]) if tags else ''
                lines.append(f'{metric_name}{{{tag_str}}} {value}')
        
        return '\n'.join(lines)
    
    def add_exporter(self, exporter: Callable[[Dict[str, Any]], None]):
        """Add metrics exporter."""
        self.exporters.append(exporter)
    
    async def _export_loop(self):
        """Background metrics export loop."""
        while self._running:
            try:
                await asyncio.sleep(self.export_interval)
                
                metrics = self.get_all_metrics()
                
                for exporter in self.exporters:
                    try:
                        if asyncio.iscoroutinefunction(exporter):
                            await exporter(metrics)
                        else:
                            exporter(metrics)
                    except Exception as e:
                        logger.error(f"Metrics export error: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics export loop error: {e}")


class MetricsExporter:
    """Base class for metrics exporters."""
    
    def export(self, metrics: Dict[str, Any]):
        """Export metrics."""
        raise NotImplementedError


class FileExporter(MetricsExporter):
    """Export metrics to file."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def export(self, metrics: Dict[str, Any]):
        """Export metrics to JSON file."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'metrics': metrics
                }, f, indent=2)
        except Exception as e:
            logger.error(f"File export error: {e}")


class HTTPExporter(MetricsExporter):
    """Export metrics via HTTP."""
    
    def __init__(self, endpoint: str, api_key: str = None):
        self.endpoint = endpoint
        self.api_key = api_key
    
    async def export(self, metrics: Dict[str, Any]):
        """Export metrics via HTTP POST."""
        try:
            import aiohttp
            
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            payload = {
                'timestamp': time.time(),
                'metrics': metrics
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status >= 400:
                        logger.error(f"HTTP export failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"HTTP export error: {e}")


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


# Convenience functions
def record_metric(name: str, value: float, tags: Dict[str, str] = None, **metadata):
    """Record a metric using global collector."""
    get_metrics_collector().record_metric(name, value, tags, **metadata)

def increment_counter(name: str, value: float = 1.0, tags: Dict[str, str] = None):
    """Increment counter using global collector."""
    get_metrics_collector().increment_counter(name, value, tags)

def set_gauge(name: str, value: float, tags: Dict[str, str] = None):
    """Set gauge using global collector."""
    get_metrics_collector().set_gauge(name, value, tags)

def record_histogram(name: str, value: float, tags: Dict[str, str] = None):
    """Record histogram using global collector."""
    get_metrics_collector().record_histogram(name, value, tags)

def time_function(name: str, tags: Dict[str, str] = None):
    """Time function using global collector."""
    return get_metrics_collector().time_function(name, tags)