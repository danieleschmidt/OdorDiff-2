"""
Adaptive load balancer with predictive scaling and intelligent routing.
"""

import asyncio
import time
import random
import statistics
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import threading
import weakref
import math

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Backend health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    ADAPTIVE = "adaptive"


@dataclass
class BackendMetrics:
    """Metrics for a backend server."""
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_health_check: float = 0.0
    health_check_failures: int = 0
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def get_average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    def get_error_rate(self) -> float:
        """Get error rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    def get_current_throughput(self) -> float:
        """Get current throughput (requests per second)."""
        if len(self.throughput_history) < 2:
            return 0.0
        
        # Calculate RPS from recent history
        recent_requests = sum(self.throughput_history[-10:])  # Last 10 data points
        time_window = min(10, len(self.throughput_history))
        return recent_requests / time_window if time_window > 0 else 0.0


@dataclass
class Backend:
    """Load balancer backend server."""
    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    health_status: HealthStatus = HealthStatus.HEALTHY
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: BackendMetrics = field(default_factory=BackendMetrics)
    
    @property
    def endpoint(self) -> str:
        """Get backend endpoint."""
        return f"{self.host}:{self.port}"
    
    def can_handle_request(self) -> bool:
        """Check if backend can handle new request."""
        return (
            self.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] and
            self.metrics.active_connections < self.max_connections
        )
    
    def get_load_score(self) -> float:
        """Calculate load score (lower is better)."""
        if not self.can_handle_request():
            return float('inf')
        
        # Combine multiple factors
        connection_factor = self.metrics.active_connections / max(1, self.max_connections)
        response_time_factor = self.metrics.get_average_response_time() / 1000.0  # Convert to seconds
        error_factor = self.metrics.get_error_rate() / 100.0
        cpu_factor = self.metrics.cpu_usage / 100.0
        memory_factor = self.metrics.memory_usage / 100.0
        
        # Weighted combination
        load_score = (
            connection_factor * 0.3 +
            response_time_factor * 0.25 +
            error_factor * 0.2 +
            cpu_factor * 0.15 +
            memory_factor * 0.1
        )
        
        return load_score / max(0.1, self.weight)  # Account for weight


class RequestContext:
    """Context for a load-balanced request."""
    
    def __init__(self, request_id: str, sticky_session_id: Optional[str] = None):
        self.request_id = request_id
        self.sticky_session_id = sticky_session_id
        self.start_time = time.time()
        self.backend: Optional[Backend] = None
        self.retries = 0
        self.metadata: Dict[str, Any] = {}


class Circuit:
    """Circuit breaker for backends."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half_open
    
    def can_attempt(self) -> bool:
        """Check if request can be attempted."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class AdaptiveLoadBalancer:
    """Intelligent load balancer with adaptive algorithms and predictive scaling."""
    
    def __init__(
        self,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ADAPTIVE,
        health_check_interval: float = 30.0,
        enable_sticky_sessions: bool = False,
        enable_circuit_breaker: bool = True
    ):
        self.algorithm = algorithm
        self.health_check_interval = health_check_interval
        self.enable_sticky_sessions = enable_sticky_sessions
        self.enable_circuit_breaker = enable_circuit_breaker
        
        self.backends: Dict[str, Backend] = {}
        self.backend_circuits: Dict[str, Circuit] = {}
        self.sticky_sessions: Dict[str, str] = {}  # session_id -> backend_id
        
        # Algorithm state
        self._round_robin_index = 0
        self._lock = asyncio.Lock()
        
        # Adaptive algorithm learning
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._algorithm_performance = {alg: deque(maxlen=50) for alg in LoadBalancingAlgorithm}
        self._current_algorithm = algorithm
        self._algorithm_switch_cooldown = 300.0  # 5 minutes
        self._last_algorithm_switch = 0.0
        
        # Health checking
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "algorithm_switches": 0,
            "backend_failures": 0
        }
    
    async def add_backend(self, backend: Backend):
        """Add a backend server."""
        async with self._lock:
            self.backends[backend.id] = backend
            
            if self.enable_circuit_breaker:
                self.backend_circuits[backend.id] = Circuit()
            
            logger.info(f"Added backend: {backend.id} ({backend.endpoint})")
            
            # Start health checking if this is the first backend
            if len(self.backends) == 1 and self._health_check_task is None:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def remove_backend(self, backend_id: str):
        """Remove a backend server."""
        async with self._lock:
            if backend_id in self.backends:
                # Set to draining first
                self.backends[backend_id].health_status = HealthStatus.DRAINING
                
                # Remove after grace period
                await asyncio.sleep(30)  # 30 second grace period
                
                self.backends.pop(backend_id, None)
                self.backend_circuits.pop(backend_id, None)
                
                # Remove sticky sessions
                sessions_to_remove = [
                    session_id for session_id, bid in self.sticky_sessions.items()
                    if bid == backend_id
                ]
                for session_id in sessions_to_remove:
                    self.sticky_sessions.pop(session_id, None)
                
                logger.info(f"Removed backend: {backend_id}")
    
    async def select_backend(self, context: RequestContext) -> Optional[Backend]:
        """Select the best backend for a request."""
        async with self._lock:
            available_backends = [
                backend for backend in self.backends.values()
                if backend.can_handle_request()
            ]
            
            if not available_backends:
                logger.warning("No available backends")
                return None
            
            # Check for sticky session
            if (self.enable_sticky_sessions and 
                context.sticky_session_id and 
                context.sticky_session_id in self.sticky_sessions):
                
                backend_id = self.sticky_sessions[context.sticky_session_id]
                if backend_id in self.backends and self.backends[backend_id].can_handle_request():
                    return self.backends[backend_id]
            
            # Filter by circuit breaker
            if self.enable_circuit_breaker:
                available_backends = [
                    backend for backend in available_backends
                    if self.backend_circuits[backend.id].can_attempt()
                ]
            
            if not available_backends:
                logger.warning("No backends available after circuit breaker filtering")
                return None
            
            # Select using current algorithm
            selected = await self._select_by_algorithm(available_backends, context)
            
            # Record sticky session
            if (self.enable_sticky_sessions and 
                context.sticky_session_id and 
                selected):
                self.sticky_sessions[context.sticky_session_id] = selected.id
            
            return selected
    
    async def _select_by_algorithm(self, backends: List[Backend], context: RequestContext) -> Optional[Backend]:
        """Select backend using specified algorithm."""
        if not backends:
            return None
        
        if self._current_algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            selected = backends[self._round_robin_index % len(backends)]
            self._round_robin_index += 1
            
        elif self._current_algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            selected = min(backends, key=lambda b: b.metrics.active_connections)
            
        elif self._current_algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            # Weight-based selection
            total_weight = sum(b.weight for b in backends)
            target = random.uniform(0, total_weight)
            cumulative = 0.0
            selected = backends[0]
            
            for backend in backends:
                cumulative += backend.weight
                if target <= cumulative:
                    selected = backend
                    break
                    
        elif self._current_algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            selected = min(backends, key=lambda b: b.metrics.get_average_response_time())
            
        else:  # ADAPTIVE
            selected = min(backends, key=lambda b: b.get_load_score())
        
        return selected
    
    async def record_request_start(self, context: RequestContext, backend: Backend):
        """Record the start of a request."""
        async with self._lock:
            backend.metrics.active_connections += 1
            context.backend = backend
            self.stats["total_requests"] += 1
    
    async def record_request_end(
        self,
        context: RequestContext,
        success: bool,
        response_time: float,
        error: Optional[Exception] = None
    ):
        """Record the completion of a request."""
        if not context.backend:
            return
        
        async with self._lock:
            backend = context.backend
            backend.metrics.active_connections = max(0, backend.metrics.active_connections - 1)
            backend.metrics.total_requests += 1
            backend.metrics.response_times.append(response_time)
            
            if success:
                self.stats["successful_requests"] += 1
                if self.enable_circuit_breaker:
                    self.backend_circuits[backend.id].record_success()
            else:
                backend.metrics.failed_requests += 1
                self.stats["failed_requests"] += 1
                self.stats["backend_failures"] += 1
                
                if self.enable_circuit_breaker:
                    self.backend_circuits[backend.id].record_failure()
            
            self.stats["total_response_time"] += response_time
            
            # Record performance for adaptive learning
            algorithm_name = self._current_algorithm.value
            self._performance_history[algorithm_name].append({
                "response_time": response_time,
                "success": success,
                "backend_load": backend.get_load_score(),
                "timestamp": time.time()
            })
            
            # Update throughput
            current_time = time.time()
            backend.metrics.throughput_history.append(1)  # One request completed
            
            # Consider algorithm adaptation
            await self._consider_algorithm_adaptation()
    
    async def _consider_algorithm_adaptation(self):
        """Consider switching load balancing algorithm based on performance."""
        if self.algorithm != LoadBalancingAlgorithm.ADAPTIVE:
            return
        
        current_time = time.time()
        if current_time - self._last_algorithm_switch < self._algorithm_switch_cooldown:
            return
        
        # Analyze performance of different algorithms
        best_algorithm = self._current_algorithm
        best_score = float('inf')
        
        for algorithm in LoadBalancingAlgorithm:
            if algorithm == LoadBalancingAlgorithm.ADAPTIVE:
                continue
            
            history = self._performance_history.get(algorithm.value, [])
            if len(history) < 10:  # Need sufficient data
                continue
            
            # Calculate performance score (lower is better)
            recent_history = list(history)[-20:]  # Last 20 requests
            
            avg_response_time = statistics.mean(p["response_time"] for p in recent_history)
            success_rate = sum(1 for p in recent_history if p["success"]) / len(recent_history)
            avg_load = statistics.mean(p["backend_load"] for p in recent_history)
            
            # Combined score (lower is better)
            score = avg_response_time * 0.4 + (1 - success_rate) * 1000 * 0.4 + avg_load * 0.2
            
            if score < best_score:
                best_score = score
                best_algorithm = algorithm
        
        # Switch algorithm if significant improvement
        if (best_algorithm != self._current_algorithm and 
            best_score < best_score * 0.9):  # 10% improvement threshold
            
            logger.info(f"Switching load balancing algorithm from {self._current_algorithm} to {best_algorithm}")
            self._current_algorithm = best_algorithm
            self._last_algorithm_switch = current_time
            self.stats["algorithm_switches"] += 1
    
    async def _health_check_loop(self):
        """Background health checking loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all backends."""
        async with self._lock:
            for backend in self.backends.values():
                await self._check_backend_health(backend)
    
    async def _check_backend_health(self, backend: Backend):
        """Check health of a single backend."""
        try:
            # Simulate health check (in real implementation, make HTTP request)
            start_time = time.time()
            
            # Call registered health check callbacks
            healthy = True
            for callback in self._health_check_callbacks:
                try:
                    result = await callback(backend)
                    if not result:
                        healthy = False
                        break
                except Exception:
                    healthy = False
                    break
            
            check_time = time.time() - start_time
            backend.metrics.last_health_check = time.time()
            
            # Update health status based on checks and metrics
            if healthy:
                error_rate = backend.metrics.get_error_rate()
                avg_response = backend.metrics.get_average_response_time()
                
                if error_rate > 50 or avg_response > 5000:  # 5 seconds
                    backend.health_status = HealthStatus.DEGRADED
                elif error_rate > 20 or avg_response > 2000:  # 2 seconds
                    backend.health_status = HealthStatus.DEGRADED
                else:
                    backend.health_status = HealthStatus.HEALTHY
                
                backend.metrics.health_check_failures = 0
            else:
                backend.metrics.health_check_failures += 1
                
                if backend.metrics.health_check_failures >= 3:
                    backend.health_status = HealthStatus.UNHEALTHY
                else:
                    backend.health_status = HealthStatus.DEGRADED
            
        except Exception as e:
            logger.error(f"Health check failed for {backend.id}: {e}")
            backend.metrics.health_check_failures += 1
            backend.health_status = HealthStatus.UNHEALTHY
    
    def add_health_check_callback(self, callback: Callable):
        """Add custom health check callback."""
        self._health_check_callbacks.append(callback)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        async with self._lock:
            total_requests = self.stats["total_requests"]
            
            backend_stats = {}
            for backend_id, backend in self.backends.items():
                backend_stats[backend_id] = {
                    "health_status": backend.health_status.value,
                    "active_connections": backend.metrics.active_connections,
                    "total_requests": backend.metrics.total_requests,
                    "error_rate": backend.metrics.get_error_rate(),
                    "avg_response_time": backend.metrics.get_average_response_time(),
                    "throughput": backend.metrics.get_current_throughput(),
                    "load_score": backend.get_load_score(),
                    "weight": backend.weight
                }
            
            algorithm_stats = {}
            for algorithm, history in self._performance_history.items():
                if history:
                    recent = list(history)[-20:]
                    algorithm_stats[algorithm] = {
                        "avg_response_time": statistics.mean(p["response_time"] for p in recent),
                        "success_rate": sum(1 for p in recent if p["success"]) / len(recent),
                        "sample_count": len(recent)
                    }
            
            return {
                "algorithm": self._current_algorithm.value,
                "total_backends": len(self.backends),
                "healthy_backends": sum(1 for b in self.backends.values() 
                                      if b.health_status == HealthStatus.HEALTHY),
                "total_requests": total_requests,
                "success_rate": (self.stats["successful_requests"] / total_requests * 100) if total_requests > 0 else 0,
                "avg_response_time": (self.stats["total_response_time"] / total_requests) if total_requests > 0 else 0,
                "active_sessions": len(self.sticky_sessions),
                "algorithm_switches": self.stats["algorithm_switches"],
                "backend_stats": backend_stats,
                "algorithm_performance": algorithm_stats
            }
    
    async def shutdown(self):
        """Shutdown load balancer."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Load balancer shutdown complete")


# Example usage and integration
class LoadBalancedOdorDiffusion:
    """Example of integrating load balancer with OdorDiffusion."""
    
    def __init__(self, backends: List[Dict[str, Any]]):
        self.load_balancer = AdaptiveLoadBalancer(
            algorithm=LoadBalancingAlgorithm.ADAPTIVE,
            enable_sticky_sessions=True,
            enable_circuit_breaker=True
        )
        
        # Add backends
        for backend_config in backends:
            backend = Backend(
                id=backend_config["id"],
                host=backend_config["host"],
                port=backend_config["port"],
                weight=backend_config.get("weight", 1.0),
                max_connections=backend_config.get("max_connections", 100)
            )
            asyncio.create_task(self.load_balancer.add_backend(backend))
        
        # Add health check
        self.load_balancer.add_health_check_callback(self._health_check_callback)
    
    async def _health_check_callback(self, backend: Backend) -> bool:
        """Custom health check for OdorDiffusion backends."""
        try:
            # In real implementation, make HTTP request to backend health endpoint
            # For now, simulate based on load
            return backend.get_load_score() < 1.0
        except Exception:
            return False
    
    async def generate_molecules(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate molecules using load-balanced backends."""
        context = RequestContext(
            request_id=f"req_{time.time()}",
            sticky_session_id=session_id
        )
        
        backend = await self.load_balancer.select_backend(context)
        if not backend:
            raise Exception("No available backends")
        
        start_time = time.time()
        await self.load_balancer.record_request_start(context, backend)
        
        try:
            # In real implementation, make request to selected backend
            # For simulation, just create a response
            response_time = random.uniform(0.5, 3.0)
            await asyncio.sleep(response_time / 1000)  # Simulate processing
            
            success = random.random() > 0.05  # 5% failure rate
            
            if success:
                result = {
                    "molecules": [{"smiles": "CC(C)=CCO", "confidence": 0.85}],
                    "backend_id": backend.id,
                    "processing_time": response_time
                }
            else:
                raise Exception("Generation failed")
            
            await self.load_balancer.record_request_end(context, True, response_time)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            await self.load_balancer.record_request_end(context, False, response_time, e)
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return await self.load_balancer.get_stats()


# Global load balancer for the application
app_load_balancer: Optional[AdaptiveLoadBalancer] = None


async def setup_load_balancer(backend_configs: List[Dict[str, Any]]):
    """Setup global application load balancer."""
    global app_load_balancer
    
    app_load_balancer = AdaptiveLoadBalancer(
        algorithm=LoadBalancingAlgorithm.ADAPTIVE,
        enable_sticky_sessions=True,
        enable_circuit_breaker=True
    )
    
    for config in backend_configs:
        backend = Backend(**config)
        await app_load_balancer.add_backend(backend)
    
    logger.info("Application load balancer setup complete")