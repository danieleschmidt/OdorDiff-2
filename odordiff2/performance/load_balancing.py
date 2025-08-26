"""
Intelligent load balancing system for OdorDiff-2.
"""

import time
import random
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    ADAPTIVE = "adaptive"

@dataclass
class WorkerNode:
    """Represents a worker node in the load balancer."""
    id: str
    weight: float = 1.0
    active_connections: int = 0
    total_requests: int = 0
    total_response_time: float = 0.0
    error_count: int = 0
    healthy: bool = True
    last_health_check: float = 0.0
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        return (self.total_response_time / self.total_requests 
                if self.total_requests > 0 else 0.0)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return (self.error_count / self.total_requests 
                if self.total_requests > 0 else 0.0)
    
    @property
    def load_score(self) -> float:
        """Calculate load score for intelligent routing."""
        base_load = self.active_connections / max(self.weight, 0.1)
        response_penalty = self.avg_response_time * 10  # 10x multiplier
        error_penalty = self.error_rate * 100  # 100x multiplier
        
        return base_load + response_penalty + error_penalty

class IntelligentLoadBalancer:
    """High-performance intelligent load balancer."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.workers: Dict[str, WorkerNode] = {}
        self.round_robin_index = 0
        self.lock = threading.RLock()
        self.health_check_interval = 30  # seconds
        self.health_check_thread = None
        self.running = False
    
    def register_worker(self, worker_id: str, weight: float = 1.0):
        """Register a new worker node."""
        with self.lock:
            self.workers[worker_id] = WorkerNode(
                id=worker_id,
                weight=weight,
                last_health_check=time.time()
            )
            logger.info(f"Registered worker: {worker_id} (weight: {weight})")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker node."""
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Unregistered worker: {worker_id}")
    
    def select_worker(self) -> Optional[str]:
        """Select optimal worker based on current strategy."""
        with self.lock:
            healthy_workers = {
                wid: worker for wid, worker in self.workers.items()
                if worker.healthy
            }
            
            if not healthy_workers:
                logger.warning("No healthy workers available")
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_select(list(healthy_workers.keys()))
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                return self._response_time_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                return self._adaptive_select(healthy_workers)
            else:
                return random.choice(list(healthy_workers.keys()))
    
    def _round_robin_select(self, worker_ids: List[str]) -> str:
        """Simple round-robin selection."""
        selected = worker_ids[self.round_robin_index % len(worker_ids)]
        self.round_robin_index += 1
        return selected
    
    def _weighted_round_robin_select(self, workers: Dict[str, WorkerNode]) -> str:
        """Weighted round-robin selection."""
        total_weight = sum(worker.weight for worker in workers.values())
        target_weight = random.uniform(0, total_weight)
        
        current_weight = 0.0
        for worker_id, worker in workers.items():
            current_weight += worker.weight
            if current_weight >= target_weight:
                return worker_id
        
        # Fallback to first worker
        return list(workers.keys())[0]
    
    def _least_connections_select(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker with least active connections."""
        return min(workers.keys(), 
                  key=lambda wid: workers[wid].active_connections / workers[wid].weight)
    
    def _response_time_select(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker with best response time."""
        return min(workers.keys(), 
                  key=lambda wid: workers[wid].avg_response_time or float('inf'))
    
    def _adaptive_select(self, workers: Dict[str, WorkerNode]) -> str:
        """Intelligent adaptive selection based on multiple factors."""
        return min(workers.keys(), key=lambda wid: workers[wid].load_score)
    
    def start_request(self, worker_id: str) -> bool:
        """Mark start of request on worker."""
        with self.lock:
            if worker_id in self.workers and self.workers[worker_id].healthy:
                self.workers[worker_id].active_connections += 1
                return True
            return False
    
    def end_request(self, worker_id: str, response_time: float, success: bool = True):
        """Mark end of request on worker."""
        with self.lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.active_connections = max(0, worker.active_connections - 1)
                worker.total_requests += 1
                worker.total_response_time += response_time
                
                if not success:
                    worker.error_count += 1
    
    def update_worker_health(self, worker_id: str, healthy: bool):
        """Update worker health status."""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].healthy = healthy
                self.workers[worker_id].last_health_check = time.time()
                logger.info(f"Worker {worker_id} health updated: {healthy}")
    
    def start_health_checks(self, health_check_func: Optional[Callable[[str], bool]] = None):
        """Start periodic health checks."""
        if self.health_check_thread and self.health_check_thread.is_alive():
            logger.warning("Health checks already running")
            return
        
        self.running = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            args=(health_check_func,),
            daemon=True
        )
        self.health_check_thread.start()
        logger.info("Health check monitoring started")
    
    def stop_health_checks(self):
        """Stop health check monitoring."""
        self.running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        logger.info("Health check monitoring stopped")
    
    def _health_check_loop(self, health_check_func: Optional[Callable[[str], bool]]):
        """Main health check loop."""
        while self.running:
            try:
                with self.lock:
                    workers_to_check = list(self.workers.keys())
                
                for worker_id in workers_to_check:
                    try:
                        if health_check_func:
                            healthy = health_check_func(worker_id)
                        else:
                            healthy = self._default_health_check(worker_id)
                        
                        self.update_worker_health(worker_id, healthy)
                        
                    except Exception as e:
                        logger.error(f"Health check failed for {worker_id}: {e}")
                        self.update_worker_health(worker_id, False)
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(self.health_check_interval)
    
    def _default_health_check(self, worker_id: str) -> bool:
        """Default health check implementation."""
        with self.lock:
            worker = self.workers.get(worker_id)
            if not worker:
                return False
            
            # Consider worker unhealthy if error rate is too high
            if worker.error_rate > 0.5:  # 50% error rate threshold
                return False
            
            # Consider worker unhealthy if response time is too high
            if worker.avg_response_time > 10.0:  # 10 second threshold
                return False
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            return {
                'strategy': self.strategy.value,
                'total_workers': len(self.workers),
                'healthy_workers': sum(1 for w in self.workers.values() if w.healthy),
                'worker_stats': {
                    wid: {
                        'weight': worker.weight,
                        'active_connections': worker.active_connections,
                        'total_requests': worker.total_requests,
                        'avg_response_time': worker.avg_response_time,
                        'error_rate': worker.error_rate,
                        'load_score': worker.load_score,
                        'healthy': worker.healthy
                    }
                    for wid, worker in self.workers.items()
                }
            }

# Global load balancer instance
load_balancer = IntelligentLoadBalancer(LoadBalancingStrategy.ADAPTIVE)
