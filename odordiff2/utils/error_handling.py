"""
Comprehensive error handling and retry mechanisms.
"""

import asyncio
import functools
import time
import traceback
from typing import Any, Callable, Optional, Type, Union, List, Dict
from dataclasses import dataclass
from enum import Enum
import random
import inspect

from .logging import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors."""
    function_name: str
    module_name: str
    args: tuple
    kwargs: dict
    timestamp: float
    attempt: int = 1
    max_attempts: int = 1


class RetryStrategy:
    """Base retry strategy."""
    
    def should_retry(self, exception: Exception, attempt: int, max_attempts: int) -> bool:
        """Determine if operation should be retried."""
        return attempt < max_attempts
    
    def get_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate delay before retry."""
        return base_delay


class ExponentialBackoffStrategy(RetryStrategy):
    """Exponential backoff retry strategy."""
    
    def __init__(self, max_delay: float = 60.0, multiplier: float = 2.0, jitter: bool = True):
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
    
    def get_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay."""
        delay = base_delay * (self.multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter to avoid thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class LinearBackoffStrategy(RetryStrategy):
    """Linear backoff retry strategy."""
    
    def __init__(self, increment: float = 1.0, max_delay: float = 30.0):
        self.increment = increment
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate linear backoff delay."""
        delay = base_delay + (attempt - 1) * self.increment
        return min(delay, self.max_delay)


class CircuitBreakerStrategy(RetryStrategy):
    """Circuit breaker pattern for failing services."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def should_retry(self, exception: Exception, attempt: int, max_attempts: int) -> bool:
        """Circuit breaker logic."""
        current_time = time.time()
        
        if self.state == 'CLOSED':
            # Normal operation
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                self.last_failure_time = current_time
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
            return attempt < max_attempts
        
        elif self.state == 'OPEN':
            # Circuit is open, check if recovery timeout has passed
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = 'HALF_OPEN'
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False
        
        elif self.state == 'HALF_OPEN':
            # Test if service has recovered
            return attempt < max_attempts
    
    def on_success(self):
        """Reset circuit breaker on success."""
        self.failure_count = 0
        self.state = 'CLOSED'
        logger.info("Circuit breaker CLOSED - service recovered")


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_callbacks = {}
        self.ignored_exceptions = set()
        
    def register_callback(self, exception_type: Type[Exception], callback: Callable):
        """Register error callback for specific exception type."""
        self.error_callbacks[exception_type] = callback
    
    def ignore_exception(self, exception_type: Type[Exception]):
        """Add exception type to ignore list."""
        self.ignored_exceptions.add(exception_type)
    
    def handle_error(self, exception: Exception, context: ErrorContext, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Handle error with context and severity."""
        error_key = f"{context.module_name}.{context.function_name}.{type(exception).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Skip if exception type is ignored
        if type(exception) in self.ignored_exceptions:
            return
        
        # Log error with context
        self._log_error(exception, context, severity)
        
        # Execute registered callback if available
        exception_type = type(exception)
        if exception_type in self.error_callbacks:
            try:
                self.error_callbacks[exception_type](exception, context)
            except Exception as callback_error:
                logger.error(f"Error in error callback: {callback_error}")
        
        # Alert on critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._send_alert(exception, context)
    
    def _log_error(self, exception: Exception, context: ErrorContext, severity: ErrorSeverity):
        """Log error with appropriate level."""
        error_info = {
            'function': context.function_name,
            'module': context.module_name,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'attempt': context.attempt,
            'max_attempts': context.max_attempts,
            'severity': severity.value,
            'args_count': len(context.args),
            'kwargs_keys': list(context.kwargs.keys())
        }
        
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            error_info['traceback'] = traceback.format_exc()
            logger.error(f"Error in {context.function_name}: {exception}", **error_info)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Error in {context.function_name}: {exception}", **error_info)
        else:
            logger.info(f"Minor error in {context.function_name}: {exception}", **error_info)
    
    def _send_alert(self, exception: Exception, context: ErrorContext):
        """Send alert for critical errors."""
        # Placeholder for alerting system (email, Slack, etc.)
        alert_message = f"CRITICAL ERROR: {type(exception).__name__} in {context.function_name}: {exception}"
        logger.critical(alert_message)
        
        # In production, integrate with alerting systems
        # send_email_alert(alert_message)
        # send_slack_alert(alert_message)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts."""
        return dict(self.error_counts)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = None,
    exceptions: tuple = (Exception,),
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Decorator for retry with backoff strategy."""
    if strategy is None:
        strategy = ExponentialBackoffStrategy()
    
    def decorator(func: Callable) -> Callable:
        error_handler = ErrorHandler()
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    
                    except exceptions as e:
                        last_exception = e
                        context = ErrorContext(
                            function_name=func.__name__,
                            module_name=func.__module__,
                            args=args,
                            kwargs=kwargs,
                            timestamp=time.time(),
                            attempt=attempt,
                            max_attempts=max_attempts
                        )
                        
                        error_handler.handle_error(e, context, severity)
                        
                        if not strategy.should_retry(e, attempt, max_attempts):
                            break
                        
                        if attempt < max_attempts:
                            delay = strategy.get_delay(attempt, base_delay)
                            logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt}/{max_attempts})")
                            await asyncio.sleep(delay)
                
                # All retries exhausted
                raise last_exception
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    
                    except exceptions as e:
                        last_exception = e
                        context = ErrorContext(
                            function_name=func.__name__,
                            module_name=func.__module__,
                            args=args,
                            kwargs=kwargs,
                            timestamp=time.time(),
                            attempt=attempt,
                            max_attempts=max_attempts
                        )
                        
                        error_handler.handle_error(e, context, severity)
                        
                        if not strategy.should_retry(e, attempt, max_attempts):
                            break
                        
                        if attempt < max_attempts:
                            delay = strategy.get_delay(attempt, base_delay)
                            logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt}/{max_attempts})")
                            time.sleep(delay)
                
                # All retries exhausted
                raise last_exception
            
            return sync_wrapper
    
    return decorator


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Circuit breaker decorator."""
    strategy = CircuitBreakerStrategy(failure_threshold, recovery_timeout)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Check if circuit is open
                if not strategy.should_retry(None, 1, 2):
                    raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {func.__name__}")
                
                result = func(*args, **kwargs)
                strategy.on_success()
                return result
                
            except Exception as e:
                if not strategy.should_retry(e, 1, 2):
                    raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {func.__name__}")
                raise
        
        return wrapper
    return decorator


def timeout(seconds: float):
    """Timeout decorator for functions."""
    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, this is a simplified timeout
                # In production, you might want to use signal.alarm or threading
                start_time = time.time()
                
                result = func(*args, **kwargs)
                
                elapsed = time.time() - start_time
                if elapsed > seconds:
                    logger.warning(f"Function {func.__name__} took {elapsed:.2f}s (timeout: {seconds}s)")
                
                return result
            
            return sync_wrapper
    
    return decorator


def safe_execute(func: Callable, *args, default=None, log_errors: bool = True, **kwargs):
    """Safely execute function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return default


async def safe_execute_async(func: Callable, *args, default=None, log_errors: bool = True, **kwargs):
    """Safely execute async function with error handling."""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Safe async execution failed for {func.__name__}: {e}")
        return default


def graceful_shutdown(cleanup_functions: List[Callable] = None):
    """Gracefully shutdown with cleanup functions."""
    cleanup_functions = cleanup_functions or []
    
    for cleanup_func in cleanup_functions:
        try:
            if inspect.iscoroutinefunction(cleanup_func):
                asyncio.run(cleanup_func())
            else:
                cleanup_func()
            logger.info(f"Cleanup function {cleanup_func.__name__} executed successfully")
        except Exception as e:
            logger.error(f"Error in cleanup function {cleanup_func.__name__}: {e}")


class OdorDiffError(Exception):
    """Base exception for OdorDiff-2."""
    pass


class ValidationError(OdorDiffError):
    """Input validation error."""
    pass


class SafetyFilterError(OdorDiffError):
    """Safety filtering error."""
    pass


class GenerationError(OdorDiffError):
    """Molecule generation error."""
    pass


class CacheError(OdorDiffError):
    """Cache operation error."""
    pass


class SecurityError(OdorDiffError):
    """Security-related error."""
    pass


class CircuitBreakerOpenError(OdorDiffError):
    """Circuit breaker is open."""
    pass


class TimeoutError(OdorDiffError):
    """Operation timeout."""
    pass


# Global error handler instance
_global_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler