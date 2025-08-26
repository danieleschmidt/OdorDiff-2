"""
Intelligent auto-scaling system for OdorDiff-2.
"""

import time
import threading
import psutil
import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"

@dataclass
class ScalingMetrics:
    """System metrics for scaling decisions."""
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    error_rate: float
    timestamp: float

@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    cpu_scale_up_threshold: float = 0.70
    cpu_scale_down_threshold: float = 0.30
    memory_scale_up_threshold: float = 0.80
    memory_scale_down_threshold: float = 0.40
    response_time_threshold: float = 2.0
    error_rate_threshold: float = 0.10
    min_workers: int = 2
    max_workers: int = 20
    scale_up_cooldown: int = 300  # 5 minutes
    scale_down_cooldown: int = 600  # 10 minutes

class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_workers = config.min_workers
        self.metrics_history = deque(maxlen=100)
        self.last_scale_action = None
        self.last_scale_time = 0
        self.monitoring_thread = None
        self.running = False
        self.callbacks = {
            ScalingAction.SCALE_UP: [],
            ScalingAction.SCALE_DOWN: []
        }
    
    def register_scaling_callback(self, action: ScalingAction, callback: Callable[[int], None]):
        """Register callback for scaling actions."""
        self.callbacks[action].append(callback)
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring and auto-scaling."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Make scaling decision
                action = self._make_scaling_decision(metrics)
                
                if action != ScalingAction.MAINTAIN:
                    self._execute_scaling_action(action)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        try:
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent / 100.0
            
            # Mock request metrics (would come from actual monitoring system)
            request_rate = self._get_request_rate()
            response_time = self._get_avg_response_time()
            error_rate = self._get_error_rate()
            
            return ScalingMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                request_rate=request_rate,
                response_time=response_time,
                error_rate=error_rate,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return safe defaults
            return ScalingMetrics(0.0, 0.0, 0.0, 1.0, 0.0, time.time())
    
    def _get_request_rate(self) -> float:
        """Get current request rate (requests per second)."""
        # Mock implementation - would integrate with actual metrics
        return 10.0  # Default mock value
    
    def _get_avg_response_time(self) -> float:
        """Get average response time in seconds."""
        # Mock implementation - would integrate with actual metrics
        return 0.5  # Default mock value
    
    def _get_error_rate(self) -> float:
        """Get current error rate (0-1)."""
        # Mock implementation - would integrate with actual metrics
        return 0.01  # Default mock value
    
    def _make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingAction:
        """Make intelligent scaling decision based on metrics."""
        current_time = time.time()
        
        # Check cooldown periods
        if self.last_scale_time:
            time_since_last_scale = current_time - self.last_scale_time
            
            if (self.last_scale_action == ScalingAction.SCALE_UP and 
                time_since_last_scale < self.config.scale_up_cooldown):
                return ScalingAction.MAINTAIN
            
            if (self.last_scale_action == ScalingAction.SCALE_DOWN and 
                time_since_last_scale < self.config.scale_down_cooldown):
                return ScalingAction.MAINTAIN
        
        # Check for scale-up conditions
        scale_up_conditions = [
            metrics.cpu_usage > self.config.cpu_scale_up_threshold,
            metrics.memory_usage > self.config.memory_scale_up_threshold,
            metrics.response_time > self.config.response_time_threshold,
            metrics.error_rate > self.config.error_rate_threshold
        ]
        
        if any(scale_up_conditions) and self.current_workers < self.config.max_workers:
            logger.info(f"Scale up triggered: CPU={metrics.cpu_usage:.2f}, "
                       f"Memory={metrics.memory_usage:.2f}, "
                       f"ResponseTime={metrics.response_time:.2f}, "
                       f"ErrorRate={metrics.error_rate:.2f}")
            return ScalingAction.SCALE_UP
        
        # Check for scale-down conditions (only if not overloaded)
        if not any(scale_up_conditions):
            scale_down_conditions = [
                metrics.cpu_usage < self.config.cpu_scale_down_threshold,
                metrics.memory_usage < self.config.memory_scale_down_threshold,
                metrics.response_time < self.config.response_time_threshold / 2,
                metrics.error_rate < self.config.error_rate_threshold / 2
            ]
            
            if all(scale_down_conditions) and self.current_workers > self.config.min_workers:
                # Check if trend is consistently low (look at last few metrics)
                if len(self.metrics_history) >= 3:
                    recent_metrics = list(self.metrics_history)[-3:]
                    avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
                    avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
                    
                    if (avg_cpu < self.config.cpu_scale_down_threshold and 
                        avg_memory < self.config.memory_scale_down_threshold):
                        logger.info(f"Scale down triggered: AvgCPU={avg_cpu:.2f}, AvgMemory={avg_memory:.2f}")
                        return ScalingAction.SCALE_DOWN
        
        return ScalingAction.MAINTAIN
    
    def _execute_scaling_action(self, action: ScalingAction):
        """Execute scaling action."""
        if action == ScalingAction.SCALE_UP:
            new_workers = min(self.current_workers + 1, self.config.max_workers)
        elif action == ScalingAction.SCALE_DOWN:
            new_workers = max(self.current_workers - 1, self.config.min_workers)
        else:
            return
        
        if new_workers != self.current_workers:
            logger.info(f"Scaling from {self.current_workers} to {new_workers} workers")
            
            # Execute callbacks
            for callback in self.callbacks[action]:
                try:
                    callback(new_workers)
                except Exception as e:
                    logger.error(f"Scaling callback failed: {e}")
            
            self.current_workers = new_workers
            self.last_scale_action = action
            self.last_scale_time = time.time()
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        recent_metrics = list(self.metrics_history)[-1] if self.metrics_history else None
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.config.min_workers,
            'max_workers': self.config.max_workers,
            'last_scale_action': self.last_scale_action.value if self.last_scale_action else None,
            'last_scale_time': self.last_scale_time,
            'recent_metrics': {
                'cpu_usage': recent_metrics.cpu_usage if recent_metrics else None,
                'memory_usage': recent_metrics.memory_usage if recent_metrics else None,
                'response_time': recent_metrics.response_time if recent_metrics else None,
                'error_rate': recent_metrics.error_rate if recent_metrics else None
            } if recent_metrics else None,
            'monitoring_active': self.running
        }

# Global auto-scaler instance
auto_scaler = AutoScaler(ScalingConfig())
