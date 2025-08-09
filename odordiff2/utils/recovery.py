"""
Automatic error recovery mechanisms and graceful degradation strategies.
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import random

from .logging import get_logger
from .error_handling import safe_execute_async, timeout, ErrorSeverity, get_error_handler
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker
from .metrics import get_metrics_collector

logger = get_logger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CACHE = "cache"
    DEGRADED = "degraded"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_FAIL = "graceful_fail"


@dataclass
class RecoveryConfig:
    """Configuration for recovery strategies."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    timeout_seconds: float = 30.0
    fallback_enabled: bool = True
    cache_fallback: bool = True
    degraded_mode_threshold: float = 0.5  # 50% failure rate
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None


@dataclass
class RecoveryMetrics:
    """Metrics for recovery operations."""
    total_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    fallback_used: int = 0
    cache_hits: int = 0
    degraded_mode_activations: int = 0
    circuit_breaker_trips: int = 0
    average_recovery_time: float = 0.0
    last_recovery_time: Optional[float] = None


class GracefulDegradation:
    """Graceful degradation manager."""
    
    def __init__(self):
        self.degraded_services: Dict[str, Dict[str, Any]] = {}
        self.service_health: Dict[str, float] = {}  # Health scores 0.0-1.0
        self.fallback_handlers: Dict[str, Callable] = {}
        self.lock = threading.RLock()
        
    def register_service(self, service_name: str, health_threshold: float = 0.5):
        """Register a service for degradation monitoring."""
        with self.lock:
            self.service_health[service_name] = 1.0
            logger.info(f"Registered service for degradation monitoring: {service_name}")
    
    def register_fallback(self, service_name: str, fallback_handler: Callable):
        """Register fallback handler for service."""
        with self.lock:
            self.fallback_handlers[service_name] = fallback_handler
            logger.info(f"Registered fallback handler for service: {service_name}")
    
    def update_health(self, service_name: str, health_score: float):
        """Update service health score."""
        with self.lock:
            previous_score = self.service_health.get(service_name, 1.0)
            self.service_health[service_name] = max(0.0, min(1.0, health_score))
            
            # Check if service should be degraded
            if health_score < 0.5 and previous_score >= 0.5:
                self._activate_degraded_mode(service_name)
            elif health_score >= 0.7 and previous_score < 0.7:
                self._deactivate_degraded_mode(service_name)
    
    def _activate_degraded_mode(self, service_name: str):
        """Activate degraded mode for service."""
        self.degraded_services[service_name] = {
            'activated_at': time.time(),
            'reason': 'low_health_score',
            'health_score': self.service_health[service_name]
        }
        logger.warning(f"Degraded mode activated for service: {service_name}")
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.increment_counter('degraded_mode_activations', tags={'service': service_name})
    
    def _deactivate_degraded_mode(self, service_name: str):
        """Deactivate degraded mode for service."""
        if service_name in self.degraded_services:
            degraded_info = self.degraded_services.pop(service_name)
            duration = time.time() - degraded_info['activated_at']
            logger.info(f"Degraded mode deactivated for service: {service_name} (duration: {duration:.1f}s)")
    
    def is_degraded(self, service_name: str) -> bool:
        """Check if service is in degraded mode."""
        return service_name in self.degraded_services
    
    def get_fallback_handler(self, service_name: str) -> Optional[Callable]:
        """Get fallback handler for service."""
        return self.fallback_handlers.get(service_name)
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all monitored services."""
        with self.lock:
            status = {}
            for service_name, health_score in self.service_health.items():
                status[service_name] = {
                    'health_score': health_score,
                    'degraded': self.is_degraded(service_name),
                    'has_fallback': service_name in self.fallback_handlers
                }
                
                if self.is_degraded(service_name):
                    status[service_name]['degraded_info'] = self.degraded_services[service_name]
            
            return status


class RecoveryManager:
    """Comprehensive recovery and degradation manager."""
    
    def __init__(self, config: RecoveryConfig = None):
        self.config = config or RecoveryConfig()
        self.metrics = RecoveryMetrics()
        self.degradation = GracefulDegradation()
        self.cache_store: Dict[str, Tuple[Any, float]] = {}  # value, expiry_time
        self.lock = threading.RLock()
        
        # Recovery handlers
        self.recovery_handlers: Dict[str, Callable] = {}
        
    def register_recovery_handler(self, service_name: str, handler: Callable):
        """Register custom recovery handler for service."""
        self.recovery_handlers[service_name] = handler
        logger.info(f"Registered recovery handler for: {service_name}")
    
    async def execute_with_recovery(
        self,
        operation: Callable,
        service_name: str,
        *args,
        fallback_result: Any = None,
        cache_key: str = None,
        **kwargs
    ) -> Any:
        """
        Execute operation with comprehensive recovery strategies.
        
        Args:
            operation: Operation to execute
            service_name: Service identifier for monitoring
            fallback_result: Result to return if all else fails
            cache_key: Key for cache fallback
            *args, **kwargs: Arguments for operation
            
        Returns:
            Operation result or fallback
        """
        start_time = time.time()
        self.metrics.total_attempts += 1
        
        # Check if service is degraded
        if self.degradation.is_degraded(service_name):
            logger.info(f"Service {service_name} is degraded, attempting fallback")
            return await self._try_fallback(service_name, fallback_result, cache_key, *args, **kwargs)
        
        # Try main operation with retries
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Use circuit breaker if configured
                if self.config.circuit_breaker_config:
                    breaker = get_circuit_breaker(f"{service_name}_recovery", 
                                                self.config.circuit_breaker_config)
                    result = await breaker.call(operation, *args, **kwargs)
                else:
                    result = await asyncio.wait_for(
                        self._safe_execute(operation, *args, **kwargs),
                        timeout=self.config.timeout_seconds
                    )
                
                # Success - update metrics and health
                self.metrics.successful_recoveries += 1
                self.degradation.update_health(service_name, 
                    min(1.0, self.degradation.service_health.get(service_name, 1.0) + 0.1))
                
                # Cache result if key provided
                if cache_key and result is not None:
                    self._cache_result(cache_key, result)
                
                execution_time = time.time() - start_time
                self.metrics.average_recovery_time = (
                    (self.metrics.average_recovery_time * (self.metrics.total_attempts - 1) + execution_time) 
                    / self.metrics.total_attempts
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for {service_name}: {str(e)}")
                
                # Update health score
                current_health = self.degradation.service_health.get(service_name, 1.0)
                new_health = max(0.0, current_health - 0.2)
                self.degradation.update_health(service_name, new_health)
                
                # Try recovery handler if available
                if attempt == 0 and service_name in self.recovery_handlers:
                    try:
                        await self.recovery_handlers[service_name](e, service_name)
                        logger.info(f"Recovery handler executed for {service_name}")
                    except Exception as recovery_error:
                        logger.error(f"Recovery handler failed: {recovery_error}")
                
                # Wait before retry (with exponential backoff and jitter)
                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.base_delay * (self.config.backoff_multiplier ** attempt),
                        self.config.max_delay
                    )
                    
                    if self.config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.info(f"Retrying {service_name} in {delay:.2f}s")
                    await asyncio.sleep(delay)
        
        # All retries failed - try fallback strategies
        logger.error(f"All retries failed for {service_name}, attempting fallback")
        self.metrics.failed_recoveries += 1
        
        return await self._try_fallback(service_name, fallback_result, cache_key, *args, **kwargs)
    
    async def _safe_execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Safely execute operation."""
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, operation, *args, **kwargs)
    
    async def _try_fallback(self, service_name: str, fallback_result: Any, 
                           cache_key: str = None, *args, **kwargs) -> Any:
        """Try fallback strategies in order of preference."""
        
        # 1. Try custom fallback handler
        fallback_handler = self.degradation.get_fallback_handler(service_name)
        if fallback_handler:
            try:
                logger.info(f"Trying custom fallback for {service_name}")
                result = await self._safe_execute(fallback_handler, *args, **kwargs)
                self.metrics.fallback_used += 1
                return result
            except Exception as e:
                logger.warning(f"Custom fallback failed for {service_name}: {e}")
        
        # 2. Try cache fallback
        if cache_key and self.config.cache_fallback:
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached result for {service_name}")
                self.metrics.cache_hits += 1
                return cached_result
        
        # 3. Return provided fallback result
        if fallback_result is not None:
            logger.info(f"Using provided fallback result for {service_name}")
            return fallback_result
        
        # 4. Graceful failure - return minimal safe result
        logger.warning(f"No fallback available for {service_name}, graceful failure")
        return self._get_graceful_failure_result(service_name)
    
    def _cache_result(self, cache_key: str, result: Any, ttl: int = 3600):
        """Cache result with TTL."""
        expiry_time = time.time() + ttl
        with self.lock:
            self.cache_store[cache_key] = (result, expiry_time)
    
    def _get_cached_result(self, cache_key: str) -> Any:
        """Get cached result if not expired."""
        with self.lock:
            if cache_key in self.cache_store:
                result, expiry_time = self.cache_store[cache_key]
                if time.time() < expiry_time:
                    return result
                else:
                    # Remove expired entry
                    del self.cache_store[cache_key]
            return None
    
    def _get_graceful_failure_result(self, service_name: str) -> Any:
        """Get safe default result for graceful failure."""
        # Return service-specific safe defaults
        defaults = {
            'molecule_generation': {
                'molecules': [],
                'processing_time': 0.0,
                'error': 'Service temporarily unavailable - please try again later',
                'cache_hit': False
            },
            'safety_assessment': {
                'assessment': {
                    'toxicity_score': 1.0,  # Assume unsafe by default
                    'skin_sensitizer': True,
                    'eco_score': 1.0,
                    'ifra_compliant': False,
                    'regulatory_flags': ['service_unavailable']
                },
                'recommendation': 'unavailable'
            },
            'molecule_search': {
                'results': [],
                'total_count': 0,
                'error': 'Search service temporarily unavailable'
            }
        }
        
        return defaults.get(service_name, {
            'error': 'Service temporarily unavailable',
            'status': 'degraded'
        })
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics."""
        success_rate = (
            self.metrics.successful_recoveries / max(1, self.metrics.total_attempts)
        ) * 100
        
        return {
            'total_attempts': self.metrics.total_attempts,
            'successful_recoveries': self.metrics.successful_recoveries,
            'failed_recoveries': self.metrics.failed_recoveries,
            'success_rate_percent': round(success_rate, 2),
            'fallback_used': self.metrics.fallback_used,
            'cache_hits': self.metrics.cache_hits,
            'degraded_mode_activations': self.metrics.degraded_mode_activations,
            'circuit_breaker_trips': self.metrics.circuit_breaker_trips,
            'average_recovery_time': round(self.metrics.average_recovery_time, 3),
            'last_recovery_time': self.metrics.last_recovery_time,
            'cache_entries': len(self.cache_store),
            'service_status': self.degradation.get_service_status()
        }
    
    def cleanup_cache(self, max_age: int = 3600):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, (result, expiry_time) in self.cache_store.items():
                if current_time >= expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache_store[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def force_degraded_mode(self, service_name: str, reason: str = "manual"):
        """Manually force service into degraded mode."""
        self.degradation.degraded_services[service_name] = {
            'activated_at': time.time(),
            'reason': reason,
            'manual': True
        }
        logger.warning(f"Manually activated degraded mode for: {service_name}")
    
    def restore_service(self, service_name: str):
        """Manually restore service from degraded mode."""
        if service_name in self.degradation.degraded_services:
            del self.degradation.degraded_services[service_name]
            self.degradation.service_health[service_name] = 1.0
            logger.info(f"Manually restored service: {service_name}")


# Global recovery manager instance
_global_recovery_manager: Optional[RecoveryManager] = None

def get_recovery_manager(config: RecoveryConfig = None) -> RecoveryManager:
    """Get global recovery manager instance."""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = RecoveryManager(config)
    return _global_recovery_manager


def recoverable(
    service_name: str,
    fallback_result: Any = None,
    cache_key_func: Callable = None,
    config: RecoveryConfig = None
):
    """Decorator to make functions recoverable with automatic fallback."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            manager = get_recovery_manager(config)
            
            # Generate cache key if function provided
            cache_key = None
            if cache_key_func:
                try:
                    cache_key = cache_key_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to generate cache key: {e}")
            
            return await manager.execute_with_recovery(
                func, service_name, *args,
                fallback_result=fallback_result,
                cache_key=cache_key,
                **kwargs
            )
        
        def sync_wrapper(*args, **kwargs):
            # Convert sync function to async for recovery manager
            async def async_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Health monitoring task
async def health_monitoring_task(manager: RecoveryManager, interval: int = 30):
    """Background task to monitor service health."""
    while True:
        try:
            # Clean up expired cache entries
            manager.cleanup_cache()
            
            # Update metrics
            metrics = get_metrics_collector()
            recovery_metrics = manager.get_recovery_metrics()
            
            for metric_name, value in recovery_metrics.items():
                if isinstance(value, (int, float)):
                    metrics.set_gauge(f'recovery_{metric_name}', value)
            
            # Log health summary
            service_status = recovery_metrics['service_status']
            degraded_services = [name for name, status in service_status.items() 
                               if status.get('degraded', False)]
            
            if degraded_services:
                logger.warning(f"Services in degraded mode: {degraded_services}")
            
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in health monitoring task: {e}")
            await asyncio.sleep(interval)


def setup_default_recovery():
    """Setup default recovery configuration."""
    config = RecoveryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
        timeout_seconds=30.0,
        fallback_enabled=True,
        cache_fallback=True,
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            timeout_seconds=30.0
        )
    )
    
    manager = get_recovery_manager(config)
    
    # Register default services
    manager.degradation.register_service('molecule_generation', 0.6)
    manager.degradation.register_service('safety_assessment', 0.5)
    manager.degradation.register_service('molecule_search', 0.7)
    manager.degradation.register_service('cache_service', 0.8)
    
    logger.info("Default recovery configuration setup completed")