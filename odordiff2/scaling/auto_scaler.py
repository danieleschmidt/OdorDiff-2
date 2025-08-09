"""
Auto-scaling System for OdorDiff-2

Implements intelligent horizontal and vertical scaling based on:
- CPU and memory utilization metrics
- Queue depth and processing backlog
- Request rate and response time patterns
- Custom business metrics (generation throughput, error rates)
- Predictive scaling based on historical patterns

Supports multiple scaling backends:
- Kubernetes Horizontal Pod Autoscaler (HPA)
- Docker Swarm scaling
- Cloud provider auto-scaling groups
- Custom container orchestration
"""

import os
import time
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import psutil
import numpy as np
from collections import deque

from ..utils.logging import get_logger
from .redis_config import get_redis_manager

logger = get_logger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    SCALE_OUT = "scale_out"  # Add instances
    SCALE_IN = "scale_in"    # Remove instances
    NO_ACTION = "no_action"


class ScalingBackend(Enum):
    """Scaling backend types."""
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    AWS_ECS = "aws_ecs"
    GCP_COMPUTE = "gcp_compute"
    AZURE_CONTAINER = "azure_container"
    CUSTOM = "custom"


@dataclass
class ScalingMetrics:
    """System metrics used for scaling decisions."""
    # Resource utilization
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_io_percent: float = 0.0
    network_io_mbps: float = 0.0
    
    # Application metrics
    request_rate: float = 0.0           # requests/second
    avg_response_time: float = 0.0      # seconds
    error_rate: float = 0.0             # percentage
    active_connections: int = 0
    
    # Queue metrics
    queue_depth: int = 0
    queue_processing_rate: float = 0.0  # items/second
    queue_wait_time: float = 0.0        # average wait time
    
    # Business metrics
    generation_throughput: float = 0.0   # molecules/second
    cache_hit_rate: float = 0.0         # percentage
    worker_utilization: float = 0.0     # percentage
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingPolicy:
    """Configuration for scaling policies."""
    # Scale-out thresholds
    cpu_scale_out_threshold: float = 70.0        # percent
    memory_scale_out_threshold: float = 80.0     # percent
    queue_scale_out_threshold: int = 100         # items
    response_time_threshold: float = 5.0         # seconds
    error_rate_threshold: float = 5.0            # percent
    
    # Scale-in thresholds
    cpu_scale_in_threshold: float = 30.0         # percent
    memory_scale_in_threshold: float = 40.0      # percent
    queue_scale_in_threshold: int = 10           # items
    
    # Instance limits
    min_instances: int = 1
    max_instances: int = 50
    
    # Scaling behavior
    scale_out_cooldown: int = 300      # seconds
    scale_in_cooldown: int = 600       # seconds
    scale_out_step: int = 2            # instances to add
    scale_in_step: int = 1             # instances to remove
    
    # Evaluation windows
    evaluation_window: int = 300       # seconds
    metric_samples: int = 6            # number of samples to evaluate
    
    # Predictive scaling
    enable_predictive: bool = True
    prediction_horizon: int = 1800     # 30 minutes ahead
    confidence_threshold: float = 0.8   # minimum confidence for predictions


class MetricsCollector:
    """Collects system and application metrics for scaling decisions."""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self._collection_task: Optional[asyncio.Task] = None
        self._redis_manager = None
        
    async def start(self):
        """Start metrics collection."""
        self._redis_manager = await get_redis_manager()
        self._collection_task = asyncio.create_task(self._collect_metrics_loop())
        logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop metrics collection."""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop."""
        while True:
            try:
                metrics = await self.collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Store in Redis for cluster-wide visibility
                await self._store_metrics_in_redis(metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system and application metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Application metrics (from various sources)
        app_metrics = await self._collect_app_metrics()
        
        # Queue metrics
        queue_metrics = await self._collect_queue_metrics()
        
        return ScalingMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_percent=self._calculate_disk_io_percent(disk_io),
            network_io_mbps=self._calculate_network_io_mbps(network_io),
            **app_metrics,
            **queue_metrics
        )
    
    async def _collect_app_metrics(self) -> Dict[str, float]:
        """Collect application-specific metrics."""
        try:
            # Get metrics from Redis or application state
            if self._redis_manager:
                client = self._redis_manager.get_async_client()
                
                # Get cached application metrics
                app_metrics = await client.hgetall("odordiff2:metrics:app")
                
                return {
                    'request_rate': float(app_metrics.get('request_rate', 0)),
                    'avg_response_time': float(app_metrics.get('avg_response_time', 0)),
                    'error_rate': float(app_metrics.get('error_rate', 0)),
                    'active_connections': int(app_metrics.get('active_connections', 0)),
                    'generation_throughput': float(app_metrics.get('generation_throughput', 0)),
                    'cache_hit_rate': float(app_metrics.get('cache_hit_rate', 0)),
                    'worker_utilization': float(app_metrics.get('worker_utilization', 0))
                }
            
        except Exception as e:
            logger.error(f"Error collecting app metrics: {e}")
        
        return {
            'request_rate': 0.0,
            'avg_response_time': 0.0, 
            'error_rate': 0.0,
            'active_connections': 0,
            'generation_throughput': 0.0,
            'cache_hit_rate': 0.0,
            'worker_utilization': 0.0
        }
    
    async def _collect_queue_metrics(self) -> Dict[str, Any]:
        """Collect queue-related metrics."""
        try:
            if self._redis_manager:
                client = self._redis_manager.get_async_client()
                
                # Get Celery queue metrics
                queue_depth = 0
                queue_processing_rate = 0.0
                queue_wait_time = 0.0
                
                # Check various queue lengths
                for queue_name in ['generation', 'batch', 'optimization']:
                    length = await client.llen(f"celery.{queue_name}")
                    queue_depth += length
                
                # Get processing rate from recent metrics
                processing_metrics = await client.hgetall("odordiff2:metrics:queue")
                if processing_metrics:
                    queue_processing_rate = float(processing_metrics.get('processing_rate', 0))
                    queue_wait_time = float(processing_metrics.get('avg_wait_time', 0))
                
                return {
                    'queue_depth': queue_depth,
                    'queue_processing_rate': queue_processing_rate,
                    'queue_wait_time': queue_wait_time
                }
        
        except Exception as e:
            logger.error(f"Error collecting queue metrics: {e}")
        
        return {
            'queue_depth': 0,
            'queue_processing_rate': 0.0,
            'queue_wait_time': 0.0
        }
    
    def _calculate_disk_io_percent(self, disk_io) -> float:
        """Calculate disk I/O utilization percentage."""
        if not hasattr(self, '_prev_disk_io'):
            self._prev_disk_io = disk_io
            return 0.0
        
        if disk_io and self._prev_disk_io:
            read_bytes = disk_io.read_bytes - self._prev_disk_io.read_bytes
            write_bytes = disk_io.write_bytes - self._prev_disk_io.write_bytes
            total_bytes = read_bytes + write_bytes
            
            # Estimate percentage based on typical disk throughput (100 MB/s baseline)
            baseline_throughput = 100 * 1024 * 1024  # 100 MB/s
            percent = min(100.0, (total_bytes / self.collection_interval) / baseline_throughput * 100)
            
            self._prev_disk_io = disk_io
            return percent
        
        return 0.0
    
    def _calculate_network_io_mbps(self, network_io) -> float:
        """Calculate network I/O in Mbps."""
        if not hasattr(self, '_prev_network_io'):
            self._prev_network_io = network_io
            return 0.0
        
        if network_io and self._prev_network_io:
            bytes_sent = network_io.bytes_sent - self._prev_network_io.bytes_sent
            bytes_recv = network_io.bytes_recv - self._prev_network_io.bytes_recv
            total_bytes = bytes_sent + bytes_recv
            
            # Convert to Mbps
            mbps = (total_bytes * 8) / (self.collection_interval * 1024 * 1024)
            
            self._prev_network_io = network_io
            return mbps
        
        return 0.0
    
    async def _store_metrics_in_redis(self, metrics: ScalingMetrics):
        """Store metrics in Redis for cluster-wide access."""
        try:
            if self._redis_manager:
                client = self._redis_manager.get_async_client()
                
                # Store metrics with expiration
                metrics_data = {
                    'cpu_percent': metrics.cpu_percent,
                    'memory_percent': metrics.memory_percent,
                    'queue_depth': metrics.queue_depth,
                    'request_rate': metrics.request_rate,
                    'avg_response_time': metrics.avg_response_time,
                    'timestamp': metrics.timestamp
                }
                
                await client.hset("odordiff2:metrics:current", mapping=metrics_data)
                await client.expire("odordiff2:metrics:current", 120)  # 2 minutes TTL
                
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    def get_recent_metrics(self, window_seconds: int = 300) -> List[ScalingMetrics]:
        """Get metrics from the last N seconds."""
        cutoff_time = time.time() - window_seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]


class PredictiveScaler:
    """Implements predictive scaling using historical patterns."""
    
    def __init__(self, prediction_horizon: int = 1800):
        self.prediction_horizon = prediction_horizon  # 30 minutes
        self.patterns: Dict[str, List[float]] = {
            'hourly': [],
            'daily': [],
            'weekly': []
        }
    
    def update_patterns(self, metrics_history: List[ScalingMetrics]):
        """Update historical patterns from metrics."""
        if len(metrics_history) < 10:
            return
        
        # Extract CPU utilization over time
        cpu_values = [m.cpu_percent for m in metrics_history[-100:]]  # Last 100 samples
        request_rates = [m.request_rate for m in metrics_history[-100:]]
        
        # Simple pattern detection (could be enhanced with ML models)
        self.patterns['hourly'] = self._detect_hourly_pattern(cpu_values)
        self.patterns['daily'] = self._detect_daily_pattern(request_rates)
    
    def _detect_hourly_pattern(self, values: List[float]) -> List[float]:
        """Detect hourly usage patterns."""
        if len(values) < 60:  # Need at least 30 minutes of data (60 samples at 30s intervals)
            return values
        
        # Simple moving average for hourly pattern
        window_size = min(12, len(values) // 5)  # 6 minutes window
        pattern = []
        
        for i in range(len(values) - window_size + 1):
            avg = sum(values[i:i + window_size]) / window_size
            pattern.append(avg)
        
        return pattern
    
    def _detect_daily_pattern(self, values: List[float]) -> List[float]:
        """Detect daily usage patterns."""
        # This would typically use more sophisticated time series analysis
        return values[-24:] if len(values) >= 24 else values  # Last 24 samples
    
    def predict_load(self, current_metrics: ScalingMetrics) -> Tuple[float, float]:
        """
        Predict future load and return (predicted_cpu, confidence).
        
        Returns:
            Tuple of (predicted_cpu_percent, confidence_score)
        """
        if not self.patterns['hourly']:
            return current_metrics.cpu_percent, 0.0
        
        try:
            # Simple linear extrapolation based on recent trend
            recent_values = self.patterns['hourly'][-10:]  # Last 10 samples
            
            if len(recent_values) < 3:
                return current_metrics.cpu_percent, 0.0
            
            # Calculate trend
            x = np.arange(len(recent_values))
            y = np.array(recent_values)
            
            # Linear regression
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs
            
            # Predict future value
            future_x = len(recent_values) + (self.prediction_horizon // 30)  # 30s intervals
            predicted_value = slope * future_x + intercept
            
            # Confidence based on trend consistency
            variance = np.var(recent_values)
            confidence = max(0.0, min(1.0, 1.0 - (variance / 100.0)))
            
            return max(0, min(100, predicted_value)), confidence
            
        except Exception as e:
            logger.error(f"Error in load prediction: {e}")
            return current_metrics.cpu_percent, 0.0


class AutoScaler:
    """Main auto-scaling controller."""
    
    def __init__(
        self,
        policy: ScalingPolicy,
        backend: ScalingBackend = ScalingBackend.KUBERNETES,
        custom_backend: Optional[Callable] = None
    ):
        self.policy = policy
        self.backend = backend
        self.custom_backend = custom_backend
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.predictive_scaler = PredictiveScaler(policy.prediction_horizon)
        
        # State management
        self.current_instances = policy.min_instances
        self.last_scale_action = 0
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Control loop
        self._control_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the auto-scaling system."""
        await self.metrics_collector.start()
        self._control_task = asyncio.create_task(self._control_loop())
        logger.info(f"Auto-scaler started with {self.backend.value} backend")
    
    async def stop(self):
        """Stop the auto-scaling system."""
        if self._control_task:
            self._control_task.cancel()
            try:
                await self._control_task
            except asyncio.CancelledError:
                pass
        
        await self.metrics_collector.stop()
        logger.info("Auto-scaler stopped")
    
    async def _control_loop(self):
        """Main auto-scaling control loop."""
        while True:
            try:
                # Collect current metrics
                current_metrics = await self.metrics_collector.collect_current_metrics()
                recent_metrics = self.metrics_collector.get_recent_metrics(
                    self.policy.evaluation_window
                )
                
                # Update predictive patterns
                if recent_metrics:
                    self.predictive_scaler.update_patterns(recent_metrics)
                
                # Make scaling decision
                action = await self._make_scaling_decision(current_metrics, recent_metrics)
                
                # Execute scaling action
                if action != ScalingAction.NO_ACTION:
                    success = await self._execute_scaling_action(action, current_metrics)
                    
                    # Record action
                    self.scaling_history.append({
                        'timestamp': time.time(),
                        'action': action.value,
                        'metrics': current_metrics.__dict__,
                        'success': success,
                        'instances_before': self.current_instances
                    })
                
                # Sleep until next evaluation
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaler control loop: {e}")
                await asyncio.sleep(60)
    
    async def _make_scaling_decision(
        self,
        current_metrics: ScalingMetrics,
        recent_metrics: List[ScalingMetrics]
    ) -> ScalingAction:
        """Make scaling decision based on current and historical metrics."""
        
        # Check cooldown periods
        now = time.time()
        time_since_last_action = now - self.last_scale_action
        
        # Calculate average metrics over evaluation window
        if len(recent_metrics) < self.policy.metric_samples:
            logger.debug("Not enough metric samples for scaling decision")
            return ScalingAction.NO_ACTION
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_queue_depth = sum(m.queue_depth for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.avg_response_time for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        # Check scale-out conditions
        scale_out_needed = (
            avg_cpu >= self.policy.cpu_scale_out_threshold or
            avg_memory >= self.policy.memory_scale_out_threshold or
            avg_queue_depth >= self.policy.queue_scale_out_threshold or
            avg_response_time >= self.policy.response_time_threshold or
            avg_error_rate >= self.policy.error_rate_threshold
        )
        
        # Check scale-in conditions
        scale_in_possible = (
            avg_cpu <= self.policy.cpu_scale_in_threshold and
            avg_memory <= self.policy.memory_scale_in_threshold and
            avg_queue_depth <= self.policy.queue_scale_in_threshold and
            self.current_instances > self.policy.min_instances
        )
        
        # Apply predictive scaling if enabled
        if self.policy.enable_predictive:
            predicted_cpu, confidence = self.predictive_scaler.predict_load(current_metrics)
            
            if confidence >= self.policy.confidence_threshold:
                if predicted_cpu >= self.policy.cpu_scale_out_threshold:
                    scale_out_needed = True
                    logger.info(f"Predictive scaling triggered: predicted CPU {predicted_cpu:.1f}%")
        
        # Make decision based on cooldown and conditions
        if scale_out_needed and self.current_instances < self.policy.max_instances:
            if time_since_last_action >= self.policy.scale_out_cooldown:
                return ScalingAction.SCALE_OUT
            else:
                logger.debug("Scale-out needed but in cooldown period")
        
        elif scale_in_possible:
            if time_since_last_action >= self.policy.scale_in_cooldown:
                return ScalingAction.SCALE_IN
            else:
                logger.debug("Scale-in possible but in cooldown period")
        
        return ScalingAction.NO_ACTION
    
    async def _execute_scaling_action(
        self,
        action: ScalingAction,
        metrics: ScalingMetrics
    ) -> bool:
        """Execute the scaling action using the configured backend."""
        try:
            if action == ScalingAction.SCALE_OUT:
                new_instances = min(
                    self.current_instances + self.policy.scale_out_step,
                    self.policy.max_instances
                )
            elif action == ScalingAction.SCALE_IN:
                new_instances = max(
                    self.current_instances - self.policy.scale_in_step,
                    self.policy.min_instances
                )
            else:
                return False
            
            # Execute via backend
            success = await self._scale_via_backend(new_instances)
            
            if success:
                old_instances = self.current_instances
                self.current_instances = new_instances
                self.last_scale_action = time.time()
                
                logger.info(
                    f"Scaled {action.value}: {old_instances} -> {new_instances} instances "
                    f"(CPU: {metrics.cpu_percent:.1f}%, Queue: {metrics.queue_depth})"
                )
            else:
                logger.error(f"Failed to execute scaling action: {action.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing scaling action {action.value}: {e}")
            return False
    
    async def _scale_via_backend(self, target_instances: int) -> bool:
        """Scale using the configured backend."""
        try:
            if self.backend == ScalingBackend.KUBERNETES:
                return await self._scale_kubernetes(target_instances)
            elif self.backend == ScalingBackend.DOCKER_SWARM:
                return await self._scale_docker_swarm(target_instances)
            elif self.backend == ScalingBackend.CUSTOM and self.custom_backend:
                return await self.custom_backend(target_instances)
            else:
                logger.warning(f"Scaling backend {self.backend.value} not implemented")
                return False
                
        except Exception as e:
            logger.error(f"Error scaling via {self.backend.value}: {e}")
            return False
    
    async def _scale_kubernetes(self, target_instances: int) -> bool:
        """Scale using Kubernetes API."""
        try:
            from kubernetes import client, config
            
            # Load Kubernetes config
            try:
                config.load_incluster_config()  # If running in cluster
            except:
                config.load_kube_config()  # If running locally
            
            # Get deployment name from environment
            deployment_name = os.getenv('K8S_DEPLOYMENT_NAME', 'odordiff2-api')
            namespace = os.getenv('K8S_NAMESPACE', 'default')
            
            # Update deployment replica count
            apps_v1 = client.AppsV1Api()
            
            # Get current deployment
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update replica count
            deployment.spec.replicas = target_instances
            
            # Apply update
            apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes scaling error: {e}")
            return False
    
    async def _scale_docker_swarm(self, target_instances: int) -> bool:
        """Scale using Docker Swarm API."""
        try:
            import docker
            
            client = docker.from_env()
            service_name = os.getenv('DOCKER_SERVICE_NAME', 'odordiff2-api')
            
            # Get service
            service = client.services.get(service_name)
            
            # Update replica count
            service.update(mode={'Replicated': {'Replicas': target_instances}})
            
            return True
            
        except Exception as e:
            logger.error(f"Docker Swarm scaling error: {e}")
            return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        recent_actions = [
            action for action in self.scaling_history 
            if action['timestamp'] > (time.time() - 3600)  # Last hour
        ]
        
        return {
            'policy': {
                'min_instances': self.policy.min_instances,
                'max_instances': self.policy.max_instances,
                'algorithm': self.backend.value
            },
            'current_state': {
                'instances': self.current_instances,
                'last_action_ago': time.time() - self.last_scale_action,
                'total_actions': len(self.scaling_history)
            },
            'recent_actions': recent_actions[-10:],  # Last 10 actions
            'metrics_summary': {
                'samples_collected': len(self.metrics_collector.metrics_history),
                'collection_interval': self.metrics_collector.collection_interval
            }
        }


def create_auto_scaler_from_config() -> AutoScaler:
    """Create auto-scaler from environment configuration."""
    # Scaling policy from environment
    policy = ScalingPolicy(
        cpu_scale_out_threshold=float(os.getenv('AUTOSCALE_CPU_OUT_THRESHOLD', '70')),
        memory_scale_out_threshold=float(os.getenv('AUTOSCALE_MEMORY_OUT_THRESHOLD', '80')),
        queue_scale_out_threshold=int(os.getenv('AUTOSCALE_QUEUE_OUT_THRESHOLD', '100')),
        cpu_scale_in_threshold=float(os.getenv('AUTOSCALE_CPU_IN_THRESHOLD', '30')),
        memory_scale_in_threshold=float(os.getenv('AUTOSCALE_MEMORY_IN_THRESHOLD', '40')),
        min_instances=int(os.getenv('AUTOSCALE_MIN_INSTANCES', '1')),
        max_instances=int(os.getenv('AUTOSCALE_MAX_INSTANCES', '20')),
        scale_out_cooldown=int(os.getenv('AUTOSCALE_OUT_COOLDOWN', '300')),
        scale_in_cooldown=int(os.getenv('AUTOSCALE_IN_COOLDOWN', '600')),
        enable_predictive=os.getenv('AUTOSCALE_PREDICTIVE', 'true').lower() == 'true'
    )
    
    # Backend type
    backend_str = os.getenv('AUTOSCALE_BACKEND', 'kubernetes').upper()
    try:
        backend = ScalingBackend[backend_str]
    except KeyError:
        logger.warning(f"Unknown scaling backend: {backend_str}, using KUBERNETES")
        backend = ScalingBackend.KUBERNETES
    
    return AutoScaler(policy=policy, backend=backend)