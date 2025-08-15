"""
Advanced error handling and resilience patterns for OdorDiff-2.
"""

import asyncio
import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
import random
import threading
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: float
    severity: ErrorSeverity
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Advanced circuit breaker pattern for fault tolerance.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Union[Exception, Tuple[Exception, ...]] = Exception,
        name: str = "circuit_breaker"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._lock = threading.RLock()
        
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    def _can_attempt_reset(self) -> bool:
        """Check if circuit breaker can attempt reset."""
        return (
            self._last_failure_time is not None and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )
    
    def _record_success(self):
        """Record successful operation."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker {self.name} reset to CLOSED")
    
    def _record_failure(self):
        """Record failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker {self.name} opened after {self._failure_count} failures"
                )
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for protecting functions with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with self._lock:
                if self._state == CircuitBreakerState.OPEN:
                    if self._can_attempt_reset():
                        self._state = CircuitBreakerState.HALF_OPEN
                        logger.info(f"Circuit breaker {self.name} attempting reset")
                    else:
                        raise CircuitBreakerError(
                            f"Circuit breaker {self.name} is OPEN"
                        )
                
                try:
                    result = func(*args, **kwargs)
                    if self._state == CircuitBreakerState.HALF_OPEN:
                        self._record_success()
                    return result
                    
                except self.expected_exception as e:
                    self._record_failure()
                    raise e
                    
        return wrapper


class RetryStrategy:
    """Advanced retry strategy with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Exception, ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if attempt <= 0:
            return 0
        
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        
        return delay
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if operation should be retried."""
        return (
            attempt < self.max_attempts and
            isinstance(exception, self.retryable_exceptions)
        )


def retry_with_strategy(strategy: RetryStrategy):
    """Decorator for retrying operations with advanced strategy."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, strategy.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not strategy.should_retry(attempt, e):
                        logger.error(
                            f"Max retries exceeded for {func.__name__}: {e}"
                        )
                        raise e
                    
                    delay = strategy.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class TimeoutManager:
    """Advanced timeout management with graceful degradation."""
    
    def __init__(self, timeout: float, fallback_func: Optional[Callable] = None):
        self.timeout = timeout
        self.fallback_func = fallback_func
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if asyncio.iscoroutinefunction(func):
                return self._async_wrapper(func, *args, **kwargs)
            else:
                return self._sync_wrapper(func, *args, **kwargs)
        
        return wrapper
    
    def _sync_wrapper(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Synchronous timeout wrapper."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(self.timeout)
        
        if thread.is_alive():
            logger.warning(f"Function {func.__name__} timed out after {self.timeout}s")
            if self.fallback_func:
                return self.fallback_func(*args, **kwargs)
            raise TimeoutError(f"Function {func.__name__} timed out")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    async def _async_wrapper(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Asynchronous timeout wrapper."""
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Async function {func.__name__} timed out after {self.timeout}s")
            if self.fallback_func:
                return await self.fallback_func(*args, **kwargs)
            raise TimeoutError(f"Async function {func.__name__} timed out")


class ResourceManager:
    """Advanced resource management with automatic cleanup."""
    
    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self._cleanup_functions: Dict[str, Callable] = {}
        self._finalizers = weakref.WeakKeyDictionary()
    
    def register_resource(
        self,
        name: str,
        resource: Any,
        cleanup_func: Optional[Callable] = None
    ):
        """Register a resource for management."""
        self._resources[name] = resource
        
        if cleanup_func:
            self._cleanup_functions[name] = cleanup_func
            # Set up automatic cleanup on garbage collection
            self._finalizers[resource] = lambda: self._cleanup_resource(name)
        
        logger.debug(f"Registered resource: {name}")
    
    def _cleanup_resource(self, name: str):
        """Clean up a specific resource."""
        if name in self._cleanup_functions:
            try:
                self._cleanup_functions[name]()
                logger.debug(f"Cleaned up resource: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up resource {name}: {e}")
            finally:
                self._cleanup_functions.pop(name, None)
                self._resources.pop(name, None)
    
    def cleanup_all(self):
        """Clean up all managed resources."""
        for name in list(self._cleanup_functions.keys()):
            self._cleanup_resource(name)
    
    def get_resource(self, name: str) -> Any:
        """Get a managed resource."""
        return self._resources.get(name)
    
    @contextmanager
    def managed_resource(self, name: str, resource: Any, cleanup_func: Callable):
        """Context manager for temporary resource management."""
        self.register_resource(name, resource, cleanup_func)
        try:
            yield resource
        finally:
            self._cleanup_resource(name)


class HealthChecker:
    """Advanced health checking with dependency tracking."""
    
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._last_results: Dict[str, Dict[str, Any]] = {}
    
    def register_check(
        self,
        name: str,
        check_func: Callable,
        dependencies: Optional[List[str]] = None
    ):
        """Register a health check function."""
        self._checks[name] = check_func
        self._dependencies[name] = dependencies or []
        logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self._checks:
            return {"status": "error", "message": f"Check {name} not found"}
        
        start_time = time.time()
        
        try:
            # Check dependencies first
            for dep in self._dependencies[name]:
                dep_result = await self.run_check(dep)
                if dep_result["status"] != "healthy":
                    return {
                        "status": "unhealthy",
                        "message": f"Dependency {dep} is unhealthy",
                        "dependency_result": dep_result,
                        "duration": time.time() - start_time
                    }
            
            # Run the actual check
            check_func = self._checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            health_result = {
                "status": "healthy" if result else "unhealthy",
                "result": result,
                "duration": time.time() - start_time,
                "timestamp": time.time()
            }
            
            self._last_results[name] = health_result
            return health_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration": time.time() - start_time,
                "timestamp": time.time()
            }
            
            self._last_results[name] = error_result
            return error_result
    
    async def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        
        for check_name in self._checks:
            results[check_name] = await self.run_check(check_name)
        
        return results
    
    def get_last_results(self) -> Dict[str, Dict[str, Any]]:
        """Get last health check results."""
        return self._last_results.copy()


# Global instances
resource_manager = ResourceManager()
health_checker = HealthChecker()


# Decorators for easy use
def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    name: str = None
):
    """Decorator for circuit breaker protection."""
    if name is None:
        name = f"breaker_{id(circuit_breaker)}"
    
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        name=name
    )
    return breaker


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Decorator for retry logic."""
    strategy = RetryStrategy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        exponential_base=exponential_base,
        jitter=jitter
    )
    return retry_with_strategy(strategy)


def timeout(seconds: float, fallback_func: Optional[Callable] = None):
    """Decorator for timeout protection."""
    return TimeoutManager(seconds, fallback_func)


# Example usage and integration patterns
class ResilientOdorDiffusion:
    """Example of applying resilience patterns to OdorDiffusion."""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self._setup_health_checks()
    
    def _setup_health_checks(self):
        """Setup health checks for the model."""
        health_checker.register_check("model_loaded", self._check_model_loaded)
        health_checker.register_check("memory_usage", self._check_memory_usage)
        health_checker.register_check("generation_capability", self._check_generation)
    
    def _check_model_loaded(self) -> bool:
        """Check if model is properly loaded."""
        return hasattr(self.base_model, 'text_encoder') and self.base_model.text_encoder is not None
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage."""
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < 90  # Consider unhealthy if > 90% memory usage
    
    def _check_generation(self) -> bool:
        """Check if generation is working."""
        try:
            # Quick test generation
            result = self.base_model._template_based_generation("test", 1)
            return len(result) > 0
        except Exception:
            return False
    
    @circuit_breaker(failure_threshold=3, recovery_timeout=30.0, name="generation")
    @retry(max_attempts=3, base_delay=0.5)
    @timeout(30.0)
    def generate_resilient(self, prompt: str, **kwargs):
        """Resilient generation with circuit breaker, retry, and timeout."""
        try:
            return self.base_model.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Generation failed for prompt '{prompt}': {e}")
            raise
    
    async def health_status(self) -> Dict[str, Any]:
        """Get complete health status."""
        return await health_checker.run_all_checks()