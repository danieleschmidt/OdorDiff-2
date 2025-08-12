"""
Performance monitoring and metrics collection for OdorDiff-2.
Provides comprehensive monitoring, alerting, and dashboard capabilities.
"""

import asyncio
import time
import json
import threading
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import psutil
import platform
from pathlib import Path
import statistics
import gzip
import pickle
from contextlib import asynccontextmanager
import weakref

from ..utils.logging import get_logger
from .optimization import PerformanceMetrics, ModelOptimizer
from .load_balancing import LoadBalancer, ServerInstance
from .auto_scaling import AutoScaler, ScalingMetrics

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str
    unit: str
    labels: List[str] = field(default_factory=list)
    help_text: str = ""
    
    def __post_init__(self):
        if not self.help_text:
            self.help_text = self.description


@dataclass
class MetricSample:
    """A single metric sample."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __str__(self) -> str:
        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        return f"{self.name}{{{label_str}}} {self.value} {int(self.timestamp * 1000)}"


@dataclass
class Alert:
    """Performance alert."""
    id: str
    name: str
    severity: AlertSeverity
    metric_name: str
    threshold: float
    current_value: float
    message: str
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    duration: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    
    def resolve(self):
        """Mark alert as resolved."""
        if not self.resolved:
            self.resolved = True
            self.resolved_at = time.time()
            self.duration = self.resolved_at - self.timestamp


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    enabled: bool = True
    collection_interval: int = 15
    retention_period: int = 86400  # 24 hours
    max_samples_per_metric: int = 10000
    
    # Export settings
    export_enabled: bool = True
    export_interval: int = 60
    export_format: str = "prometheus"  # prometheus, json, csv
    export_path: str = "metrics"
    
    # Alerting
    alerting_enabled: bool = True
    alert_evaluation_interval: int = 30
    alert_history_size: int = 1000
    
    # Dashboard
    dashboard_enabled: bool = True
    dashboard_update_interval: int = 5
    dashboard_history_window: int = 3600  # 1 hour
    
    # Storage
    persistent_storage: bool = True
    storage_compression: bool = True
    storage_path: str = "monitoring_data"


class MetricsRegistry:
    """Registry for metric definitions and samples."""
    
    def __init__(self):
        self.definitions: Dict[str, MetricDefinition] = {}
        self.samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        self.registry_lock = threading.Lock()
        
        # Register built-in metrics
        self._register_builtin_metrics()
        
        logger.info("MetricsRegistry initialized")
    
    def _register_builtin_metrics(self):
        """Register built-in system metrics."""
        builtin_metrics = [
            MetricDefinition("cpu_usage_percent", MetricType.GAUGE, "CPU usage percentage", "%"),
            MetricDefinition("memory_usage_percent", MetricType.GAUGE, "Memory usage percentage", "%"),
            MetricDefinition("disk_usage_percent", MetricType.GAUGE, "Disk usage percentage", "%"),
            MetricDefinition("network_bytes_sent", MetricType.COUNTER, "Network bytes sent", "bytes"),
            MetricDefinition("network_bytes_recv", MetricType.COUNTER, "Network bytes received", "bytes"),
            MetricDefinition("request_count", MetricType.COUNTER, "Total requests", "count"),
            MetricDefinition("request_duration", MetricType.HISTOGRAM, "Request duration", "ms"),
            MetricDefinition("error_count", MetricType.COUNTER, "Error count", "count"),
            MetricDefinition("active_connections", MetricType.GAUGE, "Active connections", "count"),
            MetricDefinition("queue_length", MetricType.GAUGE, "Queue length", "count"),
            MetricDefinition("cache_hit_rate", MetricType.GAUGE, "Cache hit rate", "ratio"),
            MetricDefinition("model_inference_time", MetricType.HISTOGRAM, "Model inference time", "ms"),
            MetricDefinition("batch_size", MetricType.HISTOGRAM, "Batch size", "count"),
            MetricDefinition("throughput", MetricType.GAUGE, "Throughput", "requests/second"),
        ]
        
        for metric_def in builtin_metrics:
            self.definitions[metric_def.name] = metric_def
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric definition."""
        with self.registry_lock:
            self.definitions[metric_def.name] = metric_def
        logger.debug(f"Registered metric: {metric_def.name}")
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Record a counter metric."""
        with self.registry_lock:
            key = self._make_key(name, labels or {})
            self.counters[key] += value
            
            sample = MetricSample(
                name=name,
                value=self.counters[key],
                timestamp=time.time(),
                labels=labels or {}
            )
            self.samples[key].append(sample)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        with self.registry_lock:
            key = self._make_key(name, labels or {})
            self.gauges[key] = value
            
            sample = MetricSample(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.samples[key].append(sample)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value."""
        with self.registry_lock:
            key = self._make_key(name, labels or {})
            self.histograms[key].append(value)
            
            # Keep only recent values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            sample = MetricSample(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.samples[key].append(sample)
    
    def start_timer(self, name: str, labels: Dict[str, str] = None) -> 'Timer':
        """Start a timer for measuring duration."""
        return Timer(self, name, labels or {})
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record a timer duration."""
        with self.registry_lock:
            key = self._make_key(name, labels or {})
            self.timers[key].append(duration)
            
            sample = MetricSample(
                name=name,
                value=duration,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.samples[key].append(sample)
    
    def get_metric_samples(self, name: str, 
                          labels: Dict[str, str] = None,
                          start_time: float = None,
                          end_time: float = None) -> List[MetricSample]:
        """Get metric samples with optional filtering."""
        with self.registry_lock:
            key = self._make_key(name, labels or {})
            samples = list(self.samples.get(key, []))
        
        # Apply time filtering
        if start_time or end_time:
            filtered_samples = []
            for sample in samples:
                if start_time and sample.timestamp < start_time:
                    continue
                if end_time and sample.timestamp > end_time:
                    continue
                filtered_samples.append(sample)
            return filtered_samples
        
        return samples
    
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        with self.registry_lock:
            key = self._make_key(name, labels or {})
            values = self.histograms.get(key, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'count': n,
            'sum': sum(sorted_values),
            'min': min(sorted_values),
            'max': max(sorted_values),
            'mean': statistics.mean(sorted_values),
            'median': statistics.median(sorted_values),
            'p95': sorted_values[int(n * 0.95)] if n > 0 else 0,
            'p99': sorted_values[int(n * 0.99)] if n > 0 else 0,
            'stddev': statistics.stdev(sorted_values) if n > 1 else 0
        }
    
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for metric + labels combination."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def get_all_metric_names(self) -> List[str]:
        """Get all registered metric names."""
        return list(self.definitions.keys())
    
    def cleanup_old_samples(self, max_age: int = 86400):
        """Clean up old metric samples."""
        cutoff_time = time.time() - max_age
        cleaned_count = 0
        
        with self.registry_lock:
            for key, sample_queue in self.samples.items():
                original_length = len(sample_queue)
                
                # Remove old samples
                while sample_queue and sample_queue[0].timestamp < cutoff_time:
                    sample_queue.popleft()
                
                cleaned_count += original_length - len(sample_queue)
        
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} old metric samples")


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, registry: MetricsRegistry, name: str, labels: Dict[str, str]):
        self.registry = registry
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (time.time() - self.start_time) * 1000  # Convert to ms
            self.registry.record_timer(self.name, duration, self.labels)


class AlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self, metrics_registry: MetricsRegistry):
        self.metrics_registry = metrics_registry
        self.alert_rules: Dict[str, Dict] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_handlers: List[Callable] = []
        
        self.evaluation_lock = asyncio.Lock()
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        logger.info("AlertManager initialized")
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        default_rules = {
            'high_cpu_usage': {
                'metric': 'cpu_usage_percent',
                'threshold': 90.0,
                'severity': AlertSeverity.WARNING,
                'duration': 300,  # 5 minutes
                'message': 'High CPU usage detected: {value}%'
            },
            'high_memory_usage': {
                'metric': 'memory_usage_percent',
                'threshold': 85.0,
                'severity': AlertSeverity.WARNING,
                'duration': 300,
                'message': 'High memory usage detected: {value}%'
            },
            'high_error_rate': {
                'metric': 'error_rate',
                'threshold': 5.0,
                'severity': AlertSeverity.CRITICAL,
                'duration': 60,
                'message': 'High error rate detected: {value}%'
            },
            'slow_response_time': {
                'metric': 'request_duration',
                'threshold': 2000.0,  # 2 seconds
                'severity': AlertSeverity.WARNING,
                'duration': 180,
                'message': 'Slow response time detected: {value}ms'
            }
        }
        
        for rule_name, config in default_rules.items():
            self.alert_rules[rule_name] = config
    
    def add_alert_rule(self, name: str, metric: str, threshold: float, 
                       severity: AlertSeverity, duration: int = 60,
                       message: str = None):
        """Add a new alert rule."""
        if not message:
            message = f"{metric} exceeded threshold: {{value}}"
        
        self.alert_rules[name] = {
            'metric': metric,
            'threshold': threshold,
            'severity': severity,
            'duration': duration,
            'message': message
        }
        
        logger.info(f"Added alert rule: {name}")
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    async def evaluate_alerts(self):
        """Evaluate all alert rules."""
        async with self.evaluation_lock:
            current_time = time.time()
            
            for rule_name, rule_config in self.alert_rules.items():
                await self._evaluate_alert_rule(rule_name, rule_config, current_time)
    
    async def _evaluate_alert_rule(self, rule_name: str, rule_config: Dict, current_time: float):
        """Evaluate a single alert rule."""
        metric_name = rule_config['metric']
        threshold = rule_config['threshold']
        severity = rule_config['severity']
        duration = rule_config.get('duration', 60)
        
        # Get recent metric samples
        samples = self.metrics_registry.get_metric_samples(
            metric_name,
            start_time=current_time - duration
        )
        
        if not samples:
            return
        
        # Check if threshold is exceeded for required duration
        exceeding_samples = [s for s in samples if s.value > threshold]
        
        if len(exceeding_samples) >= len(samples) * 0.8:  # 80% of samples exceed threshold
            # Threshold exceeded - create or update alert
            alert_id = f"{rule_name}_{metric_name}"
            current_value = samples[-1].value if samples else 0.0
            
            if alert_id not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    id=alert_id,
                    name=rule_name,
                    severity=severity,
                    metric_name=metric_name,
                    threshold=threshold,
                    current_value=current_value,
                    message=rule_config['message'].format(value=current_value),
                    timestamp=current_time
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                await self._fire_alert(alert)
                
                logger.warning(f"Alert fired: {alert.name} - {alert.message}")
            
            else:
                # Update existing alert
                self.active_alerts[alert_id].current_value = current_value
        
        else:
            # Threshold not exceeded - resolve alert if active
            alert_id = f"{rule_name}_{metric_name}"
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolve()
                
                await self._resolve_alert(alert)
                
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert.name}")
    
    async def _fire_alert(self, alert: Alert):
        """Fire an alert to all handlers."""
        for handler in self.alert_handlers:
            try:
                await handler('fire', alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert via all handlers."""
        for handler in self.alert_handlers:
            try:
                await handler('resolve', alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
        
        # Calculate MTTR (Mean Time To Resolution)
        resolved_alerts = [a for a in self.alert_history if a.resolved]
        mttr = 0.0
        if resolved_alerts:
            mttr = statistics.mean(a.duration for a in resolved_alerts) / 60  # minutes
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'resolved_alerts': len(resolved_alerts),
            'mean_time_to_resolution_minutes': mttr,
            'severity_distribution': dict(severity_counts),
            'alert_rules_count': len(self.alert_rules)
        }


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.metrics_registry = MetricsRegistry()
        self.alert_manager = AlertManager(self.metrics_registry)
        
        self.system_collector = None
        self.application_collectors: List[Callable] = []
        self.exporters: List[Callable] = []
        
        self.is_running = False
        self.collection_task = None
        self.alert_task = None
        self.export_task = None
        self.cleanup_task = None
        
        # Component integrations
        self.model_optimizer: Optional[ModelOptimizer] = None
        self.load_balancer: Optional[LoadBalancer] = None
        self.auto_scaler: Optional[AutoScaler] = None
        
        self._setup_storage()
        self._setup_default_exporters()
        
        logger.info("PerformanceMonitor initialized")
    
    def _setup_storage(self):
        """Setup persistent storage."""
        if self.config.persistent_storage:
            storage_path = Path(self.config.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_default_exporters(self):
        """Setup default metric exporters."""
        if self.config.export_enabled:
            if self.config.export_format == "prometheus":
                self.exporters.append(self._prometheus_exporter)
            elif self.config.export_format == "json":
                self.exporters.append(self._json_exporter)
    
    def integrate_model_optimizer(self, optimizer: ModelOptimizer):
        """Integrate with model optimizer for metrics."""
        self.model_optimizer = optimizer
        self.application_collectors.append(self._collect_model_metrics)
        logger.info("Integrated with ModelOptimizer")
    
    def integrate_load_balancer(self, load_balancer: LoadBalancer):
        """Integrate with load balancer for metrics."""
        self.load_balancer = load_balancer
        self.application_collectors.append(self._collect_load_balancer_metrics)
        logger.info("Integrated with LoadBalancer")
    
    def integrate_auto_scaler(self, auto_scaler: AutoScaler):
        """Integrate with auto scaler for metrics."""
        self.auto_scaler = auto_scaler
        self.application_collectors.append(self._collect_scaling_metrics)
        logger.info("Integrated with AutoScaler")
    
    async def start(self):
        """Start performance monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start collection tasks
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.alert_task = asyncio.create_task(self._alert_loop())
        self.export_task = asyncio.create_task(self._export_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop(self):
        """Stop performance monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel tasks
        for task in [self.collection_task, self.alert_task, self.export_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Performance monitoring stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.is_running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.config.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.config.collection_interval)
    
    async def _collect_all_metrics(self):
        """Collect all metrics."""
        # System metrics
        await self._collect_system_metrics()
        
        # Application metrics
        for collector in self.application_collectors:
            try:
                await collector()
            except Exception as e:
                logger.error(f"Application metrics collection error: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics_registry.set_gauge("cpu_usage_percent", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_registry.set_gauge("memory_usage_percent", memory.percent)
        self.metrics_registry.set_gauge("memory_total_bytes", memory.total)
        self.metrics_registry.set_gauge("memory_available_bytes", memory.available)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics_registry.set_gauge("disk_usage_percent", disk.percent)
        self.metrics_registry.set_gauge("disk_total_bytes", disk.total)
        self.metrics_registry.set_gauge("disk_free_bytes", disk.free)
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.metrics_registry.record_counter("network_bytes_sent", net_io.bytes_sent)
        self.metrics_registry.record_counter("network_bytes_recv", net_io.bytes_recv)
        
        # Process information
        process = psutil.Process()
        self.metrics_registry.set_gauge("process_cpu_percent", process.cpu_percent())
        self.metrics_registry.set_gauge("process_memory_bytes", process.memory_info().rss)
        self.metrics_registry.set_gauge("process_open_files", len(process.open_files()))
        
        # Load average (Unix-like systems only)
        try:
            load_avg = psutil.getloadavg()
            self.metrics_registry.set_gauge("load_average_1m", load_avg[0])
            self.metrics_registry.set_gauge("load_average_5m", load_avg[1])
            self.metrics_registry.set_gauge("load_average_15m", load_avg[2])
        except AttributeError:
            pass  # Windows doesn't have load average
    
    async def _collect_model_metrics(self):
        """Collect model optimizer metrics."""
        if not self.model_optimizer:
            return
        
        metrics = self.model_optimizer.get_performance_metrics()
        
        self.metrics_registry.set_gauge("model_requests_per_second", metrics.requests_per_second)
        self.metrics_registry.set_gauge("model_average_response_time", metrics.average_response_time)
        self.metrics_registry.set_gauge("model_cache_hit_rate", metrics.cache_hit_rate)
        self.metrics_registry.set_gauge("model_queue_length", metrics.queue_length)
        
        # Cache statistics
        cache_stats = self.model_optimizer.cache_stats
        for stat_name, value in cache_stats.items():
            self.metrics_registry.record_counter(f"model_cache_{stat_name}", value)
    
    async def _collect_load_balancer_metrics(self):
        """Collect load balancer metrics."""
        if not self.load_balancer:
            return
        
        stats = self.load_balancer.get_stats()
        
        self.metrics_registry.set_gauge("lb_total_servers", stats['total_servers'])
        self.metrics_registry.set_gauge("lb_healthy_servers", stats['healthy_servers'])
        self.metrics_registry.set_gauge("lb_unhealthy_servers", stats['unhealthy_servers'])
        self.metrics_registry.record_counter("lb_total_requests", stats['total_requests'])
        self.metrics_registry.record_counter("lb_failed_requests", stats['failed_requests'])
        self.metrics_registry.set_gauge("lb_success_rate", stats['success_rate'])
        
        # Per-server metrics
        for server_id, server_stats in stats.get('servers', {}).items():
            labels = {'server_id': server_id}
            self.metrics_registry.set_gauge("server_active_connections", 
                                          server_stats['active_connections'], labels)
            self.metrics_registry.set_gauge("server_requests_per_second",
                                          server_stats['requests_per_second'], labels)
            self.metrics_registry.set_gauge("server_average_response_time",
                                          server_stats['average_response_time'], labels)
            self.metrics_registry.set_gauge("server_success_rate",
                                          server_stats['success_rate'], labels)
    
    async def _collect_scaling_metrics(self):
        """Collect auto-scaling metrics."""
        if not self.auto_scaler:
            return
        
        stats = self.auto_scaler.get_scaling_stats()
        
        self.metrics_registry.set_gauge("scaling_current_instances", stats['current_instances'])
        self.metrics_registry.set_gauge("scaling_min_instances", stats['min_instances'])
        self.metrics_registry.set_gauge("scaling_max_instances", stats['max_instances'])
        self.metrics_registry.record_counter("scaling_total_events", stats['total_scaling_events'])
        self.metrics_registry.set_gauge("scaling_success_rate", stats['scaling_success_rate'])
        self.metrics_registry.set_gauge("scaling_cooldown_remaining", stats['cooldown_remaining'])
    
    async def _alert_loop(self):
        """Alert evaluation loop."""
        while self.is_running:
            try:
                if self.config.alerting_enabled:
                    await self.alert_manager.evaluate_alerts()
                await asyncio.sleep(self.config.alert_evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(self.config.alert_evaluation_interval)
    
    async def _export_loop(self):
        """Metrics export loop."""
        while self.is_running:
            try:
                if self.config.export_enabled:
                    for exporter in self.exporters:
                        await exporter()
                await asyncio.sleep(self.config.export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics export error: {e}")
                await asyncio.sleep(self.config.export_interval)
    
    async def _cleanup_loop(self):
        """Cleanup loop for old data."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                self.metrics_registry.cleanup_old_samples(self.config.retention_period)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _prometheus_exporter(self):
        """Export metrics in Prometheus format."""
        output_lines = []
        
        # Get all metric samples
        for metric_name in self.metrics_registry.get_all_metric_names():
            samples = self.metrics_registry.get_metric_samples(metric_name)
            
            if not samples:
                continue
            
            # Get metric definition
            metric_def = self.metrics_registry.definitions.get(metric_name)
            if metric_def:
                output_lines.append(f"# HELP {metric_name} {metric_def.help_text}")
                output_lines.append(f"# TYPE {metric_name} {metric_def.type.value}")
            
            # Add samples
            for sample in samples[-1:]:  # Only export latest sample
                output_lines.append(str(sample))
        
        # Write to file
        export_path = Path(self.config.export_path) / "metrics.prom"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            f.write('\n'.join(output_lines))
    
    async def _json_exporter(self):
        """Export metrics in JSON format."""
        export_data = {
            'timestamp': time.time(),
            'metrics': {}
        }
        
        for metric_name in self.metrics_registry.get_all_metric_names():
            samples = self.metrics_registry.get_metric_samples(metric_name)
            
            if samples:
                latest_sample = samples[-1]
                export_data['metrics'][metric_name] = {
                    'value': latest_sample.value,
                    'timestamp': latest_sample.timestamp,
                    'labels': latest_sample.labels
                }
                
                # Add histogram stats if applicable
                if metric_name in self.metrics_registry.histograms:
                    stats = self.metrics_registry.get_histogram_stats(metric_name)
                    export_data['metrics'][f"{metric_name}_stats"] = stats
        
        # Write to file
        export_path = Path(self.config.export_path) / "metrics.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def get_dashboard_data(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        end_time = time.time()
        start_time = end_time - (window_minutes * 60)
        
        dashboard_data = {
            'timestamp': end_time,
            'window_minutes': window_minutes,
            'system': {},
            'application': {},
            'alerts': {
                'active': len(self.alert_manager.get_active_alerts()),
                'recent': self.alert_manager.get_alert_history(hours=1)
            }
        }
        
        # System metrics
        system_metrics = ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent']
        for metric_name in system_metrics:
            samples = self.metrics_registry.get_metric_samples(
                metric_name, start_time=start_time, end_time=end_time
            )
            if samples:
                dashboard_data['system'][metric_name] = {
                    'current': samples[-1].value,
                    'history': [{'timestamp': s.timestamp, 'value': s.value} for s in samples[-20:]]
                }
        
        # Application metrics
        if self.model_optimizer:
            app_metrics = ['model_requests_per_second', 'model_average_response_time', 'model_cache_hit_rate']
            for metric_name in app_metrics:
                samples = self.metrics_registry.get_metric_samples(
                    metric_name, start_time=start_time, end_time=end_time
                )
                if samples:
                    dashboard_data['application'][metric_name] = {
                        'current': samples[-1].value,
                        'history': [{'timestamp': s.timestamp, 'value': s.value} for s in samples[-20:]]
                    }
        
        return dashboard_data
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get overall monitoring statistics."""
        return {
            'is_running': self.is_running,
            'metrics_count': len(self.metrics_registry.definitions),
            'total_samples': sum(len(samples) for samples in self.metrics_registry.samples.values()),
            'alert_stats': self.alert_manager.get_alert_stats(),
            'config': asdict(self.config),
            'integrations': {
                'model_optimizer': self.model_optimizer is not None,
                'load_balancer': self.load_balancer is not None,
                'auto_scaler': self.auto_scaler is not None
            }
        }


# Monitoring decorators for automatic instrumentation
def monitor_function(metric_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            if monitor and monitor.is_running:
                with monitor.metrics_registry.start_timer(f"{metric_name}_duration"):
                    result = func(*args, **kwargs)
                monitor.metrics_registry.record_counter(f"{metric_name}_calls")
                return result
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def monitor_async_function(metric_name: str = None):
    """Decorator to monitor async function performance."""
    def decorator(func):
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"{func.__module__}.{func.__name__}"
        
        async def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            if monitor and monitor.is_running:
                with monitor.metrics_registry.start_timer(f"{metric_name}_duration"):
                    result = await func(*args, **kwargs)
                monitor.metrics_registry.record_counter(f"{metric_name}_calls")
                return result
            else:
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global monitoring instance
_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor(config: MonitoringConfig = None) -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(config)
    return _performance_monitor