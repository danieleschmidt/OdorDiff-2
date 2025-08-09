"""
Circuit breaker pattern implementation for robust external dependency handling.
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .logging import get_logger
from .error_handling import safe_execute_async, timeout

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Number of failures to open circuit
    recovery_timeout: float = 60.0       # Seconds to wait before trying half-open
    success_threshold: int = 3           # Number of successes to close circuit from half-open
    timeout_seconds: float = 30.0        # Request timeout
    expected_exceptions: tuple = (Exception,)  # Exceptions that count as failures
    half_open_max_calls: int = 5         # Max calls allowed in half-open state


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opens: int = 0
    current_consecutive_failures: int = 0
    current_consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changed_time: float = 0


class CircuitBreaker:
    """Circuit breaker for external dependencies."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.lock = threading.RLock()
        self.half_open_calls = 0
        
        # State change callbacks
        self.on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
        self.on_failure: Optional[Callable[[Exception], None]] = None
        self.on_success: Optional[Callable[[], None]] = None
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            self.metrics.total_requests += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._should_attempt_reset():
                    self._change_state(CircuitState.HALF_OPEN)
                    self.half_open_calls = 0
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    # Fail fast
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Will retry after {self._time_until_retry():.1f} seconds"
                    )
            
            # Check half-open call limit
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN but call limit reached"
                    )
                self.half_open_calls += 1
        
        # Execute the function with timeout
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                self._safe_call(func, *args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            # Handle success
            execution_time = time.time() - start_time
            await self._handle_success(execution_time)
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            timeout_error = TimeoutError(f"Circuit breaker timeout after {execution_time:.2f}s")
            await self._handle_failure(timeout_error, is_timeout=True)
            raise timeout_error
            
        except self.config.expected_exceptions as e:
            execution_time = time.time() - start_time
            await self._handle_failure(e)
            raise
    
    async def _safe_call(self, func: Callable, *args, **kwargs):
        """Safely call function (sync or async)."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def _handle_success(self, execution_time: float):
        """Handle successful execution."""
        with self.lock:
            self.metrics.successful_requests += 1
            self.metrics.current_consecutive_failures = 0
            self.metrics.current_consecutive_successes += 1
            self.metrics.last_success_time = time.time()
            
            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                if self.metrics.current_consecutive_successes >= self.config.success_threshold:
                    self._change_state(CircuitState.CLOSED)
                    logger.info(f"Circuit breaker '{self.name}' CLOSED after successful recovery")
            
            # Call success callback
            if self.on_success:
                try:
                    self.on_success()
                except Exception as e:
                    logger.error(f"Error in success callback: {e}")
        
        logger.debug(f"Circuit breaker '{self.name}' successful call in {execution_time*1000:.1f}ms")
    
    async def _handle_failure(self, exception: Exception, is_timeout: bool = False):
        """Handle failed execution."""
        with self.lock:
            self.metrics.failed_requests += 1
            if is_timeout:
                self.metrics.timeouts += 1
            
            self.metrics.current_consecutive_successes = 0
            self.metrics.current_consecutive_failures += 1
            self.metrics.last_failure_time = time.time()
            
            # State transitions
            if self.state == CircuitState.CLOSED:
                if self.metrics.current_consecutive_failures >= self.config.failure_threshold:
                    self._change_state(CircuitState.OPEN)
                    self.metrics.circuit_opens += 1
                    logger.warning(
                        f"Circuit breaker '{self.name}' OPENED after {self.config.failure_threshold} "
                        f"consecutive failures. Last error: {str(exception)}"
                    )
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state should open the circuit
                self._change_state(CircuitState.OPEN)
                self.metrics.circuit_opens += 1
                logger.warning(f"Circuit breaker '{self.name}' OPENED from HALF_OPEN due to failure: {str(exception)}")
            
            # Call failure callback
            if self.on_failure:
                try:
                    self.on_failure(exception)
                except Exception as e:
                    logger.error(f"Error in failure callback: {e}")
        
        logger.warning(f"Circuit breaker '{self.name}' failure: {str(exception)}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.metrics.last_failure_time is None:
            return True
        return (time.time() - self.metrics.last_failure_time) >= self.config.recovery_timeout
    
    def _time_until_retry(self) -> float:
        """Calculate time until next retry attempt."""
        if self.metrics.last_failure_time is None:
            return 0
        elapsed = time.time() - self.metrics.last_failure_time
        return max(0, self.config.recovery_timeout - elapsed)
    
    def _change_state(self, new_state: CircuitState):
        """Change circuit state with logging."""
        old_state = self.state
        self.state = new_state
        self.metrics.state_changed_time = time.time()
        
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self.lock:
            success_rate = (
                self.metrics.successful_requests / max(1, self.metrics.total_requests)
            ) * 100
            
            return {
                'name': self.name,
                'state': self.state.value,
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'timeouts': self.metrics.timeouts,
                'success_rate_percent': round(success_rate, 2),
                'circuit_opens': self.metrics.circuit_opens,
                'consecutive_failures': self.metrics.current_consecutive_failures,
                'consecutive_successes': self.metrics.current_consecutive_successes,
                'last_failure_time': self.metrics.last_failure_time,
                'last_success_time': self.metrics.last_success_time,
                'state_changed_time': self.metrics.state_changed_time,
                'time_until_retry': self._time_until_retry() if self.state == CircuitState.OPEN else 0,
                'half_open_calls': self.half_open_calls if self.state == CircuitState.HALF_OPEN else 0,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold,
                    'timeout_seconds': self.config.timeout_seconds,
                    'half_open_max_calls': self.config.half_open_max_calls
                }
            }
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self.lock:
            self._change_state(CircuitState.CLOSED)
            self.metrics.current_consecutive_failures = 0
            self.metrics.current_consecutive_successes = 0
            self.half_open_calls = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")
    
    def force_open(self):
        """Manually force circuit breaker to open state."""
        with self.lock:
            self._change_state(CircuitState.OPEN)
            self.metrics.last_failure_time = time.time()
            logger.warning(f"Circuit breaker '{self.name}' manually forced to OPEN")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.RLock()
        
    def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self.lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name, config)
                logger.info(f"Created new circuit breaker: {name}")
            return self.breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker."""
        with self.lock:
            if name in self.breakers:
                del self.breakers[name]
                logger.info(f"Removed circuit breaker: {name}")
                return True
            return False
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self.lock:
            return {name: breaker.get_metrics() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self.lock:
            for breaker in self.breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")
    
    def get_unhealthy_breakers(self) -> List[str]:
        """Get list of unhealthy (open or half-open) circuit breakers."""
        with self.lock:
            return [
                name for name, breaker in self.breakers.items()
                if breaker.state in [CircuitState.OPEN, CircuitState.HALF_OPEN]
            ]


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class TimeoutError(Exception):
    """Exception raised when operation times out."""
    pass


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        breaker = get_circuit_breaker_registry().get_or_create(name, config)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await breaker.call(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(breaker.call(func, *args, **kwargs))
            return sync_wrapper
    
    return decorator


# Predefined circuit breaker configurations
CONFIGS = {
    'fast_service': CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        timeout_seconds=10.0
    ),
    'slow_service': CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=120.0,
        timeout_seconds=60.0
    ),
    'critical_service': CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=300.0,
        timeout_seconds=30.0,
        success_threshold=5
    ),
    'external_api': CircuitBreakerConfig(
        failure_threshold=4,
        recovery_timeout=60.0,
        timeout_seconds=20.0,
        half_open_max_calls=3
    )
}


# Global registry instance
_global_registry: Optional[CircuitBreakerRegistry] = None

def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get global circuit breaker registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry


def get_circuit_breaker(name: str, config_name: str = None, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get circuit breaker with optional predefined config."""
    if config_name and config_name in CONFIGS:
        config = CONFIGS[config_name]
    
    return get_circuit_breaker_registry().get_or_create(name, config)


async def protected_http_request(url: str, method: str = 'GET', **kwargs):
    """Example of circuit breaker protected HTTP request."""
    import aiohttp
    
    # Get circuit breaker for HTTP requests
    breaker = get_circuit_breaker('http_requests', 'external_api')
    
    async def make_request():
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
    
    return await breaker.call(make_request)


async def protected_database_query(query_func, *args, **kwargs):
    """Example of circuit breaker protected database query."""
    breaker = get_circuit_breaker('database', 'slow_service')
    return await breaker.call(query_func, *args, **kwargs)