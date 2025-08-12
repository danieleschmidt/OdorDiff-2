"""
Auto-scaling components for OdorDiff-2.
Provides intelligent scaling based on metrics and predictions.
"""

import asyncio
import time
import math
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import psutil
import json
from pathlib import Path
import statistics

from ..utils.logging import get_logger
from .load_balancing import LoadBalancer, ServerInstance, ServerHealth, ServiceRegistry
from .optimization import PerformanceMetrics, OptimizationConfig

logger = get_logger(__name__)


class ScalingAction(Enum):
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ScalingTrigger(Enum):
    """Scaling triggers."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    response_time: float
    queue_length: int
    error_rate: float
    active_connections: int
    throughput: float
    
    # Predictive metrics
    predicted_cpu: float = 0.0
    predicted_memory: float = 0.0
    predicted_load: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingRule:
    """Configuration for a scaling rule."""
    name: str
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    min_instances: int = 1
    max_instances: int = 10
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    cooldown_period: int = 300  # 5 minutes
    evaluation_period: int = 60  # 1 minute
    data_points_to_alarm: int = 3
    enabled: bool = True
    weight: float = 1.0


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    action: ScalingAction
    trigger: str
    metric_value: float
    threshold: float
    instances_before: int
    instances_after: int
    reason: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling."""
    enabled: bool = True
    min_instances: int = 1
    max_instances: int = 20
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 75.0
    target_response_time: float = 500.0  # ms
    
    # Scaling factors
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.8
    
    # Timing
    scale_up_cooldown: int = 180  # 3 minutes
    scale_down_cooldown: int = 600  # 10 minutes
    metrics_collection_interval: int = 30
    evaluation_interval: int = 60
    
    # Thresholds
    cpu_scale_up_threshold: float = 80.0
    cpu_scale_down_threshold: float = 20.0
    memory_scale_up_threshold: float = 85.0
    memory_scale_down_threshold: float = 25.0
    response_time_threshold: float = 1000.0
    error_rate_threshold: float = 5.0
    
    # Predictive scaling
    enable_predictive_scaling: bool = True
    prediction_horizon: int = 300  # 5 minutes
    learning_period: int = 86400  # 24 hours
    
    # Instance management
    instance_startup_time: int = 120  # 2 minutes
    instance_shutdown_time: int = 60   # 1 minute
    health_check_grace_period: int = 180  # 3 minutes


class MetricsCollector:
    """Collects and aggregates metrics for scaling decisions."""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=1000)  # Keep 1000 data points
        self.custom_metric_collectors: Dict[str, Callable] = {}
        self.is_collecting = False
        self.collection_task = None
        
        logger.info(f"MetricsCollector initialized with interval: {collection_interval}s")
    
    async def start_collection(self):
        """Start metrics collection."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")
    
    def add_custom_metric_collector(self, name: str, collector: Callable):
        """Add a custom metric collector."""
        self.custom_metric_collectors[name] = collector
        logger.info(f"Added custom metric collector: {name}")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                metrics = await self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Network I/O for request rate approximation
        net_io = psutil.net_io_counters()
        
        # Calculate request rate from recent history
        request_rate = self._calculate_request_rate()
        
        # Calculate response time from recent history
        response_time = self._calculate_average_response_time()
        
        # System load metrics
        try:
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
        except AttributeError:
            load_avg = 0.0  # Windows doesn't have load average
        
        # Process metrics
        process_count = len(psutil.pids())
        
        # Custom metrics
        custom_metrics = {}
        for name, collector in self.custom_metric_collectors.items():
            try:
                value = await collector()
                custom_metrics[name] = value
            except Exception as e:
                logger.warning(f"Custom metric collection failed for {name}: {e}")
        
        return ScalingMetrics(
            timestamp=timestamp,
            cpu_utilization=cpu_percent,
            memory_utilization=memory_percent,
            request_rate=request_rate,
            response_time=response_time,
            queue_length=int(load_avg * 10),  # Approximation
            error_rate=0.0,  # Would be calculated from application metrics
            active_connections=process_count,
            throughput=request_rate,
            custom_metrics=custom_metrics
        )
    
    def _calculate_request_rate(self) -> float:
        """Calculate current request rate from metrics history."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Use last few data points
        recent_metrics = list(self.metrics_history)[-5:]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Simple approximation based on CPU and memory changes
        # In practice, this would use actual request counters
        total_load = sum(m.cpu_utilization + m.memory_utilization for m in recent_metrics)
        time_span = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        
        if time_span > 0:
            return total_load / time_span
        
        return 0.0
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time from recent metrics."""
        if not self.metrics_history:
            return 0.0
        
        # Simple approximation based on system load
        # In practice, this would use actual response time measurements
        recent_cpu = self.metrics_history[-1].cpu_utilization
        recent_memory = self.metrics_history[-1].memory_utilization
        
        # Higher utilization = higher response time
        base_time = 100.0  # Base response time in ms
        load_factor = (recent_cpu + recent_memory) / 200.0
        
        return base_time * (1 + load_factor * 5)
    
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, float]:
        """Get summary statistics for recent metrics."""
        if not self.metrics_history:
            return {}
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        summary = {
            'avg_cpu': statistics.mean(m.cpu_utilization for m in recent_metrics),
            'max_cpu': max(m.cpu_utilization for m in recent_metrics),
            'avg_memory': statistics.mean(m.memory_utilization for m in recent_metrics),
            'max_memory': max(m.memory_utilization for m in recent_metrics),
            'avg_response_time': statistics.mean(m.response_time for m in recent_metrics),
            'max_response_time': max(m.response_time for m in recent_metrics),
            'avg_request_rate': statistics.mean(m.request_rate for m in recent_metrics),
            'data_points': len(recent_metrics)
        }
        
        return summary


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, learning_period: int = 86400, prediction_horizon: int = 300):
        self.learning_period = learning_period
        self.prediction_horizon = prediction_horizon
        self.historical_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.models = {}
        
        logger.info(f"PredictiveScaler initialized: learning_period={learning_period}s, horizon={prediction_horizon}s")
    
    def add_data_point(self, metrics: ScalingMetrics):
        """Add a data point for learning."""
        hour_of_day = int((metrics.timestamp % 86400) / 3600)  # 0-23
        day_of_week = int((metrics.timestamp / 86400) % 7)     # 0-6
        
        pattern_key = f"{day_of_week}_{hour_of_day}"
        
        data_point = {
            'timestamp': metrics.timestamp,
            'cpu': metrics.cpu_utilization,
            'memory': metrics.memory_utilization,
            'request_rate': metrics.request_rate,
            'response_time': metrics.response_time
        }
        
        self.historical_patterns[pattern_key].append(data_point)
    
    def predict_metrics(self, target_time: float) -> Optional[ScalingMetrics]:
        """Predict metrics for a future time."""
        target_hour = int((target_time % 86400) / 3600)
        target_day = int((target_time / 86400) % 7)
        pattern_key = f"{target_day}_{target_hour}"
        
        if pattern_key not in self.historical_patterns:
            return None
        
        historical_data = list(self.historical_patterns[pattern_key])
        
        if len(historical_data) < 3:
            return None
        
        # Simple prediction based on historical average
        avg_cpu = statistics.mean(d['cpu'] for d in historical_data)
        avg_memory = statistics.mean(d['memory'] for d in historical_data)
        avg_request_rate = statistics.mean(d['request_rate'] for d in historical_data)
        avg_response_time = statistics.mean(d['response_time'] for d in historical_data)
        
        # Apply trend if available
        if len(historical_data) >= 5:
            recent_data = historical_data[-5:]
            older_data = historical_data[-10:-5] if len(historical_data) >= 10 else historical_data[:-5]
            
            if older_data:
                cpu_trend = statistics.mean(d['cpu'] for d in recent_data) - statistics.mean(d['cpu'] for d in older_data)
                memory_trend = statistics.mean(d['memory'] for d in recent_data) - statistics.mean(d['memory'] for d in older_data)
                
                avg_cpu += cpu_trend
                avg_memory += memory_trend
        
        return ScalingMetrics(
            timestamp=target_time,
            cpu_utilization=max(0, min(100, avg_cpu)),
            memory_utilization=max(0, min(100, avg_memory)),
            request_rate=max(0, avg_request_rate),
            response_time=max(0, avg_response_time),
            queue_length=0,
            error_rate=0.0,
            active_connections=0,
            throughput=max(0, avg_request_rate),
            predicted_cpu=avg_cpu,
            predicted_memory=avg_memory,
            predicted_load=(avg_cpu + avg_memory) / 2
        )
    
    def should_preemptively_scale(self, current_instances: int) -> Tuple[bool, ScalingAction, str]:
        """Determine if preemptive scaling is needed."""
        future_time = time.time() + self.prediction_horizon
        predicted_metrics = self.predict_metrics(future_time)
        
        if not predicted_metrics:
            return False, ScalingAction.NO_ACTION, "No prediction available"
        
        # Check if predicted load exceeds thresholds
        if predicted_metrics.predicted_cpu > 80 or predicted_metrics.predicted_memory > 85:
            return True, ScalingAction.SCALE_UP, f"Predicted high load: CPU={predicted_metrics.predicted_cpu:.1f}%, Memory={predicted_metrics.predicted_memory:.1f}%"
        
        if predicted_metrics.predicted_cpu < 20 and predicted_metrics.predicted_memory < 25 and current_instances > 1:
            return True, ScalingAction.SCALE_DOWN, f"Predicted low load: CPU={predicted_metrics.predicted_cpu:.1f}%, Memory={predicted_metrics.predicted_memory:.1f}%"
        
        return False, ScalingAction.NO_ACTION, "Predicted load within normal range"


class AutoScaler:
    """Main auto-scaling controller."""
    
    def __init__(self, 
                 config: AutoScalingConfig = None,
                 load_balancer: LoadBalancer = None,
                 service_registry: ServiceRegistry = None):
        
        self.config = config or AutoScalingConfig()
        self.load_balancer = load_balancer
        self.service_registry = service_registry
        
        self.metrics_collector = MetricsCollector(self.config.metrics_collection_interval)
        self.predictive_scaler = PredictiveScaler(
            self.config.learning_period,
            self.config.prediction_horizon
        ) if self.config.enable_predictive_scaling else None
        
        self.scaling_rules = self._create_default_rules()
        self.scaling_history = deque(maxlen=100)
        self.last_scaling_action = 0
        self.current_instances = self.config.min_instances
        self.is_running = False
        
        self.scaling_task = None
        self.instance_factories: Dict[str, Callable] = {}
        
        logger.info(f"AutoScaler initialized with {len(self.scaling_rules)} rules")
    
    def _create_default_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        return [
            ScalingRule(
                name="cpu_utilization",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                threshold_up=self.config.cpu_scale_up_threshold,
                threshold_down=self.config.cpu_scale_down_threshold,
                min_instances=self.config.min_instances,
                max_instances=self.config.max_instances,
                cooldown_period=self.config.scale_up_cooldown,
                weight=1.0
            ),
            ScalingRule(
                name="memory_utilization",
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                threshold_up=self.config.memory_scale_up_threshold,
                threshold_down=self.config.memory_scale_down_threshold,
                min_instances=self.config.min_instances,
                max_instances=self.config.max_instances,
                cooldown_period=self.config.scale_up_cooldown,
                weight=0.8
            ),
            ScalingRule(
                name="response_time",
                trigger=ScalingTrigger.RESPONSE_TIME,
                threshold_up=self.config.response_time_threshold,
                threshold_down=self.config.response_time_threshold * 0.5,
                min_instances=self.config.min_instances,
                max_instances=self.config.max_instances,
                cooldown_period=self.config.scale_up_cooldown,
                weight=0.6
            )
        ]
    
    def add_instance_factory(self, service_name: str, factory: Callable):
        """Add factory function for creating new instances."""
        self.instance_factories[service_name] = factory
        logger.info(f"Added instance factory for service: {service_name}")
    
    async def start(self):
        """Start the auto-scaler."""
        if self.is_running:
            return
        
        self.is_running = True
        await self.metrics_collector.start_collection()
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        
        logger.info("AutoScaler started")
    
    async def stop(self):
        """Stop the auto-scaler."""
        self.is_running = False
        await self.metrics_collector.stop_collection()
        
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AutoScaler stopped")
    
    async def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.evaluation_interval)
                await self._evaluate_scaling_decision()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(self.config.evaluation_interval)
    
    async def _evaluate_scaling_decision(self):
        """Evaluate whether scaling action is needed."""
        if not self.metrics_collector.metrics_history:
            logger.debug("No metrics available for scaling decision")
            return
        
        current_metrics = self.metrics_collector.metrics_history[-1]
        
        # Add to predictive scaler
        if self.predictive_scaler:
            self.predictive_scaler.add_data_point(current_metrics)
        
        # Check cooldown period
        if time.time() - self.last_scaling_action < self._get_cooldown_period():
            logger.debug("Scaling action in cooldown period")
            return
        
        # Evaluate rules
        scaling_decision = await self._evaluate_scaling_rules(current_metrics)
        
        if scaling_decision['action'] != ScalingAction.NO_ACTION:
            await self._execute_scaling_action(scaling_decision)
        
        # Check predictive scaling
        if self.predictive_scaler and self.config.enable_predictive_scaling:
            should_scale, action, reason = self.predictive_scaler.should_preemptively_scale(
                self.current_instances
            )
            
            if should_scale and action != scaling_decision['action']:
                logger.info(f"Predictive scaling recommendation: {action.value} - {reason}")
                
                predictive_decision = {
                    'action': action,
                    'reason': f"Predictive: {reason}",
                    'metric_name': 'predictive',
                    'metric_value': 0.0,
                    'threshold': 0.0,
                    'rule_name': 'predictive_scaling'
                }
                
                # Execute if not conflicting with recent reactive scaling
                if time.time() - self.last_scaling_action > 60:  # 1 minute buffer
                    await self._execute_scaling_action(predictive_decision)
    
    async def _evaluate_scaling_rules(self, metrics: ScalingMetrics) -> Dict[str, Any]:
        """Evaluate all scaling rules and return decision."""
        scale_up_votes = 0
        scale_down_votes = 0
        scale_up_weight = 0.0
        scale_down_weight = 0.0
        
        triggered_rules = []
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            metric_value = self._get_metric_value(metrics, rule.trigger)
            
            if metric_value is None:
                continue
            
            # Check if rule triggers scaling
            if metric_value > rule.threshold_up:
                if self.current_instances < rule.max_instances:
                    scale_up_votes += 1
                    scale_up_weight += rule.weight
                    triggered_rules.append({
                        'rule': rule,
                        'action': ScalingAction.SCALE_UP,
                        'metric_value': metric_value,
                        'threshold': rule.threshold_up
                    })
            
            elif metric_value < rule.threshold_down:
                if self.current_instances > rule.min_instances:
                    scale_down_votes += 1
                    scale_down_weight += rule.weight
                    triggered_rules.append({
                        'rule': rule,
                        'action': ScalingAction.SCALE_DOWN,
                        'metric_value': metric_value,
                        'threshold': rule.threshold_down
                    })
        
        # Make decision based on votes and weights
        if scale_up_weight > scale_down_weight and triggered_rules:
            # Find the most significant scale-up trigger
            scale_up_rules = [r for r in triggered_rules if r['action'] == ScalingAction.SCALE_UP]
            if scale_up_rules:
                best_rule = max(scale_up_rules, key=lambda r: r['metric_value'] - r['threshold'])
                return {
                    'action': ScalingAction.SCALE_UP,
                    'reason': f"Metric {best_rule['rule'].trigger.value} ({best_rule['metric_value']:.1f}) > threshold ({best_rule['threshold']:.1f})",
                    'metric_name': best_rule['rule'].trigger.value,
                    'metric_value': best_rule['metric_value'],
                    'threshold': best_rule['threshold'],
                    'rule_name': best_rule['rule'].name
                }
        
        elif scale_down_weight > scale_up_weight and triggered_rules:
            # Find the most significant scale-down trigger
            scale_down_rules = [r for r in triggered_rules if r['action'] == ScalingAction.SCALE_DOWN]
            if scale_down_rules:
                best_rule = max(scale_down_rules, key=lambda r: r['threshold'] - r['metric_value'])
                return {
                    'action': ScalingAction.SCALE_DOWN,
                    'reason': f"Metric {best_rule['rule'].trigger.value} ({best_rule['metric_value']:.1f}) < threshold ({best_rule['threshold']:.1f})",
                    'metric_name': best_rule['rule'].trigger.value,
                    'metric_value': best_rule['metric_value'],
                    'threshold': best_rule['threshold'],
                    'rule_name': best_rule['rule'].name
                }
        
        return {
            'action': ScalingAction.NO_ACTION,
            'reason': 'No scaling rules triggered',
            'metric_name': None,
            'metric_value': 0.0,
            'threshold': 0.0,
            'rule_name': None
        }
    
    def _get_metric_value(self, metrics: ScalingMetrics, trigger: ScalingTrigger) -> Optional[float]:
        """Get metric value for a specific trigger."""
        if trigger == ScalingTrigger.CPU_UTILIZATION:
            return metrics.cpu_utilization
        elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
            return metrics.memory_utilization
        elif trigger == ScalingTrigger.REQUEST_RATE:
            return metrics.request_rate
        elif trigger == ScalingTrigger.RESPONSE_TIME:
            return metrics.response_time
        elif trigger == ScalingTrigger.QUEUE_LENGTH:
            return float(metrics.queue_length)
        elif trigger == ScalingTrigger.ERROR_RATE:
            return metrics.error_rate
        else:
            return None
    
    async def _execute_scaling_action(self, decision: Dict[str, Any]):
        """Execute a scaling action."""
        action = decision['action']
        instances_before = self.current_instances
        
        try:
            if action == ScalingAction.SCALE_UP:
                new_instances = max(
                    instances_before + 1,
                    int(instances_before * self.config.scale_up_factor)
                )
                new_instances = min(new_instances, self.config.max_instances)
                
                if new_instances > instances_before:
                    await self._scale_up(new_instances - instances_before)
                    self.current_instances = new_instances
            
            elif action == ScalingAction.SCALE_DOWN:
                new_instances = min(
                    instances_before - 1,
                    int(instances_before * self.config.scale_down_factor)
                )
                new_instances = max(new_instances, self.config.min_instances)
                
                if new_instances < instances_before:
                    await self._scale_down(instances_before - new_instances)
                    self.current_instances = new_instances
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                action=action,
                trigger=decision['rule_name'] or 'unknown',
                metric_value=decision['metric_value'],
                threshold=decision['threshold'],
                instances_before=instances_before,
                instances_after=self.current_instances,
                reason=decision['reason'],
                success=True
            )
            
            self.scaling_history.append(event)
            self.last_scaling_action = time.time()
            
            logger.info(f"Scaling action executed: {action.value} from {instances_before} to {self.current_instances} instances. Reason: {decision['reason']}")
            
        except Exception as e:
            error_msg = f"Scaling action failed: {e}"
            logger.error(error_msg)
            
            event = ScalingEvent(
                timestamp=time.time(),
                action=action,
                trigger=decision['rule_name'] or 'unknown',
                metric_value=decision['metric_value'],
                threshold=decision['threshold'],
                instances_before=instances_before,
                instances_after=instances_before,  # No change due to failure
                reason=decision['reason'],
                success=False,
                error=error_msg
            )
            
            self.scaling_history.append(event)
    
    async def _scale_up(self, count: int):
        """Add new instances."""
        logger.info(f"Scaling up by {count} instances")
        
        for i in range(count):
            # Create new instance (stub implementation)
            instance_id = f"auto-instance-{int(time.time())}-{i}"
            
            # In practice, you would create actual instances using cloud APIs
            # For now, simulate by adding to load balancer
            if self.load_balancer:
                new_instance = ServerInstance(
                    id=instance_id,
                    host="localhost",
                    port=8000 + len(self.load_balancer.servers),
                    weight=1.0,
                    health=ServerHealth.HEALTHY
                )
                
                await self.load_balancer.add_server(new_instance)
            
            logger.info(f"Created new instance: {instance_id}")
    
    async def _scale_down(self, count: int):
        """Remove instances."""
        logger.info(f"Scaling down by {count} instances")
        
        if not self.load_balancer:
            return
        
        # Remove least utilized servers
        server_utilization = []
        for server_id, server in self.load_balancer.servers.items():
            if server.health == ServerHealth.HEALTHY:
                utilization = server.calculate_load_score()
                server_utilization.append((server_id, utilization))
        
        # Sort by utilization (ascending) and remove least utilized
        server_utilization.sort(key=lambda x: x[1])
        
        for i in range(min(count, len(server_utilization))):
            server_id = server_utilization[i][0]
            await self.load_balancer.remove_server(server_id)
            logger.info(f"Removed instance: {server_id}")
    
    def _get_cooldown_period(self) -> int:
        """Get appropriate cooldown period."""
        if not self.scaling_history:
            return self.config.scale_up_cooldown
        
        last_event = self.scaling_history[-1]
        
        if last_event.action == ScalingAction.SCALE_UP:
            return self.config.scale_up_cooldown
        elif last_event.action == ScalingAction.SCALE_DOWN:
            return self.config.scale_down_cooldown
        else:
            return self.config.scale_up_cooldown
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        total_events = len(self.scaling_history)
        successful_events = sum(1 for e in self.scaling_history if e.success)
        
        stats = {
            'is_running': self.is_running,
            'current_instances': self.current_instances,
            'min_instances': self.config.min_instances,
            'max_instances': self.config.max_instances,
            'total_scaling_events': total_events,
            'successful_scaling_events': successful_events,
            'scaling_success_rate': (successful_events / max(total_events, 1)) * 100,
            'last_scaling_action': self.last_scaling_action,
            'cooldown_remaining': max(0, self._get_cooldown_period() - (time.time() - self.last_scaling_action)),
            'active_rules': len([r for r in self.scaling_rules if r.enabled]),
            'predictive_scaling_enabled': self.config.enable_predictive_scaling
        }
        
        # Recent events
        recent_events = list(self.scaling_history)[-10:]  # Last 10 events
        stats['recent_events'] = [
            {
                'timestamp': e.timestamp,
                'action': e.action.value,
                'trigger': e.trigger,
                'instances_before': e.instances_before,
                'instances_after': e.instances_after,
                'reason': e.reason,
                'success': e.success
            }
            for e in recent_events
        ]
        
        # Metrics summary
        stats['metrics_summary'] = self.metrics_collector.get_metrics_summary()
        
        return stats


# Global instances
_auto_scaler: Optional[AutoScaler] = None

def get_auto_scaler(config: AutoScalingConfig = None) -> AutoScaler:
    """Get global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler(config)
    return _auto_scaler