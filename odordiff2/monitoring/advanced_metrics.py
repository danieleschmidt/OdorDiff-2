"""
Advanced metrics collection and performance monitoring for OdorDiff-2.
"""

import asyncio
import time
import threading
import psutil
import gc
import functools
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import statistics
import json
import logging
from datetime import datetime, timedelta
import weakref

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Individual metric value with timestamp."""
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    sum: float
    mean: float
    median: float
    min: float
    max: float
    std_dev: float
    percentile_95: float
    percentile_99: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Advanced metrics collector with real-time aggregation."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points_per_metric = max_points_per_metric
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Performance monitoring
        self._start_time = time.time()
        self._collection_overhead = deque(maxlen=1000)
    
    def counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        start = time.perf_counter()
        
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            self._metrics[key].append(MetricValue(self._counters[key], labels=labels or {}))
        
        self._record_overhead(time.perf_counter() - start)
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        start = time.perf_counter()
        
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            self._metrics[key].append(MetricValue(value, labels=labels or {}))
        
        self._record_overhead(time.perf_counter() - start)
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a value in a histogram."""
        start = time.perf_counter()
        
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            self._metrics[key].append(MetricValue(value, labels=labels or {}))
        
        self._record_overhead(time.perf_counter() - start)
    
    def timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        self.histogram(f"{name}_duration_seconds", duration, labels)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _record_overhead(self, overhead: float):
        """Record collection overhead for monitoring."""
        self._collection_overhead.append(overhead)
    
    def get_summary(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        key = self._make_key(name, labels)
        
        with self._lock:
            if key not in self._metrics or not self._metrics[key]:
                return None
            
            values = [m.value for m in self._metrics[key]]
            
            if not values:
                return None
            
            return MetricSummary(
                name=name,
                count=len(values),
                sum=sum(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                min=min(values),
                max=max(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
                percentile_95=statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                percentile_99=statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
                labels=labels or {}
            )
    
    def get_all_summaries(self) -> List[MetricSummary]:
        """Get summaries for all metrics."""
        summaries = []
        
        with self._lock:
            for key in self._metrics:
                # Parse metric name and labels from key
                if '{' in key:
                    name = key.split('{')[0]
                    labels_str = key.split('{')[1].rstrip('}')
                    labels = dict(pair.split('=') for pair in labels_str.split(',') if pair)
                else:
                    name = key
                    labels = {}
                
                summary = self.get_summary(name, labels)
                if summary:
                    summaries.append(summary)
        
        return summaries
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        summaries = self.get_all_summaries()
        
        for summary in summaries:
            # Generate Prometheus metric name
            prom_name = summary.name.replace('-', '_').replace('.', '_')
            
            # Add help and type comments
            lines.append(f"# HELP {prom_name} Generated metric from OdorDiff-2")
            lines.append(f"# TYPE {prom_name} histogram")
            
            # Add labels
            label_str = ""
            if summary.labels:
                label_pairs = [f'{k}="{v}"' for k, v in summary.labels.items()]
                label_str = "{" + ",".join(label_pairs) + "}"
            
            # Add metric values
            lines.append(f"{prom_name}_count{label_str} {summary.count}")
            lines.append(f"{prom_name}_sum{label_str} {summary.sum}")
            lines.append(f"{prom_name}_mean{label_str} {summary.mean}")
            lines.append(f"{prom_name}_median{label_str} {summary.median}")
            lines.append(f"{prom_name}_p95{label_str} {summary.percentile_95}")
            lines.append(f"{prom_name}_p99{label_str} {summary.percentile_99}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Get Python-specific metrics
        gc_stats = gc.get_stats()
        generation_counts = [stats['collections'] for stats in gc_stats]
        
        return {
            "system_memory_percent": memory.percent,
            "system_memory_available_bytes": memory.available,
            "system_cpu_percent": cpu_percent,
            "python_gc_generation0_collections": generation_counts[0] if generation_counts else 0,
            "python_gc_generation1_collections": generation_counts[1] if len(generation_counts) > 1 else 0,
            "python_gc_generation2_collections": generation_counts[2] if len(generation_counts) > 2 else 0,
            "metrics_collection_overhead_mean": statistics.mean(self._collection_overhead) if self._collection_overhead else 0,
        }
    
    def clear_old_metrics(self, max_age_seconds: float = 3600):
        """Clear metrics older than specified age."""
        cutoff_time = time.time() - max_age_seconds
        
        with self._lock:
            for key in list(self._metrics.keys()):
                # Filter out old values
                self._metrics[key] = deque(
                    (m for m in self._metrics[key] if m.timestamp > cutoff_time),
                    maxlen=self.max_points_per_metric
                )
                
                # Remove empty metric collections
                if not self._metrics[key]:
                    del self._metrics[key]


class PerformanceMonitor:
    """Advanced performance monitoring with automatic profiling."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._active_operations: Dict[str, float] = {}
        self._operation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    @contextmanager
    def monitor_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for monitoring operation performance."""
        start_time = time.perf_counter()
        operation_id = f"{operation_name}_{id(threading.current_thread())}"
        
        self._active_operations[operation_id] = start_time
        
        try:
            yield
            
        except Exception as e:
            # Record error metrics
            error_labels = {**(labels or {}), "error_type": type(e).__name__}
            self.metrics.counter(f"{operation_name}_errors_total", labels=error_labels)
            raise
            
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Record timing
            self.metrics.timing(operation_name, duration, labels)
            
            # Record in history
            self._operation_history[operation_name].append(duration)
            
            # Clean up
            self._active_operations.pop(operation_id, None)
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        history = self._operation_history.get(operation_name, [])
        
        if not history:
            return {}
        
        return {
            "count": len(history),
            "mean_duration": statistics.mean(history),
            "median_duration": statistics.median(history),
            "min_duration": min(history),
            "max_duration": max(history),
            "p95_duration": statistics.quantiles(history, n=20)[18] if len(history) >= 20 else max(history),
            "p99_duration": statistics.quantiles(history, n=100)[98] if len(history) >= 100 else max(history),
        }
    
    def get_active_operations(self) -> Dict[str, float]:
        """Get currently active operations and their durations."""
        current_time = time.perf_counter()
        return {
            op_id: current_time - start_time
            for op_id, start_time in self._active_operations.items()
        }


class AlertManager:
    """Alert management system for metrics thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._alert_rules: List[Dict[str, Any]] = []
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        self._alert_history: deque = deque(maxlen=1000)
        
    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "greater",  # greater, less, equal
        duration_seconds: float = 60.0,
        labels: Optional[Dict[str, str]] = None
    ):
        """Add an alert rule."""
        rule = {
            "name": name,
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "duration_seconds": duration_seconds,
            "labels": labels or {},
            "created_at": time.time()
        }
        
        self._alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    def check_alerts(self):
        """Check all alert rules and trigger alerts if needed."""
        current_time = time.time()
        
        for rule in self._alert_rules:
            alert_key = f"{rule['name']}_{hash(str(rule['labels']))}"
            
            # Get current metric value
            summary = self.metrics.get_summary(rule['metric_name'], rule['labels'])
            
            if not summary:
                continue
            
            # Check threshold
            current_value = summary.mean  # Use mean for threshold comparison
            threshold_breached = self._check_threshold(
                current_value, rule['threshold'], rule['comparison']
            )
            
            if threshold_breached:
                if alert_key not in self._active_alerts:
                    # New alert
                    alert = {
                        "rule": rule,
                        "current_value": current_value,
                        "started_at": current_time,
                        "last_triggered": current_time
                    }
                    self._active_alerts[alert_key] = alert
                    self._alert_history.append({**alert, "action": "started"})
                    logger.warning(f"Alert started: {rule['name']} (value: {current_value})")
                else:
                    # Update existing alert
                    self._active_alerts[alert_key]["last_triggered"] = current_time
                    self._active_alerts[alert_key]["current_value"] = current_value
            else:
                if alert_key in self._active_alerts:
                    # Resolve alert
                    resolved_alert = self._active_alerts.pop(alert_key)
                    self._alert_history.append({
                        **resolved_alert,
                        "action": "resolved",
                        "resolved_at": current_time,
                        "duration": current_time - resolved_alert["started_at"]
                    })
                    logger.info(f"Alert resolved: {rule['name']}")
    
    def _check_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Check if value breaches threshold."""
        if comparison == "greater":
            return value > threshold
        elif comparison == "less":
            return value < threshold
        elif comparison == "equal":
            return abs(value - threshold) < 1e-6
        else:
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all currently active alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return list(self._alert_history)[-limit:]


# Global metrics system
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor(metrics_collector)
alert_manager = AlertManager(metrics_collector)


# Decorators for easy metrics integration
def measure_time(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to measure function execution time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with performance_monitor.monitor_operation(operation_name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics_collector.counter(metric_name, labels=labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Setup default system monitoring
def setup_system_monitoring():
    """Setup automatic system metrics collection."""
    def collect_system_metrics():
        while True:
            try:
                system_metrics = metrics_collector.get_system_metrics()
                
                for metric_name, value in system_metrics.items():
                    metrics_collector.gauge(metric_name, value)
                
                # Check alerts
                alert_manager.check_alerts()
                
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(30)  # Wait longer on error
    
    # Start background thread for system monitoring
    monitor_thread = threading.Thread(target=collect_system_metrics, daemon=True)
    monitor_thread.start()
    
    # Setup default alerts
    alert_manager.add_alert_rule(
        "high_memory_usage",
        "system_memory_percent",
        threshold=90.0,
        comparison="greater",
        duration_seconds=60.0
    )
    
    alert_manager.add_alert_rule(
        "high_cpu_usage",
        "system_cpu_percent",
        threshold=80.0,
        comparison="greater",
        duration_seconds=120.0
    )
    
    logger.info("System monitoring setup complete")


# Auto-start system monitoring when module is imported
setup_system_monitoring()