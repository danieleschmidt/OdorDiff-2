"""
Advanced rate limiting system with IP-based and API key-based rate limiting.
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib
import ipaddress

from .logging import get_logger
from .error_handling import safe_execute_async

logger = get_logger(__name__)


class LimitType(Enum):
    """Rate limit types."""
    IP = "ip"
    API_KEY = "api_key" 
    USER = "user"
    ENDPOINT = "endpoint"
    GLOBAL = "global"


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int                    # Number of requests allowed
    window_seconds: int             # Time window in seconds
    burst_requests: int = None      # Burst allowance (optional)
    limit_type: LimitType = LimitType.IP
    
    def __post_init__(self):
        if self.burst_requests is None:
            self.burst_requests = self.requests


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""
    capacity: int                   # Maximum tokens
    refill_rate: float             # Tokens per second
    tokens: float = 0              # Current tokens
    last_refill: float = field(default_factory=time.time)
    requests_count: int = 0        # Total requests
    blocked_count: int = 0         # Blocked requests
    
    def refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens."""
        self.refill()
        self.requests_count += 1
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        else:
            self.blocked_count += 1
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bucket metrics."""
        self.refill()
        return {
            'capacity': self.capacity,
            'current_tokens': self.tokens,
            'refill_rate_per_second': self.refill_rate,
            'total_requests': self.requests_count,
            'blocked_requests': self.blocked_count,
            'block_rate_percent': (self.blocked_count / max(1, self.requests_count)) * 100,
            'last_refill': self.last_refill
        }


class SlidingWindowCounter:
    """Sliding window rate limit counter."""
    
    def __init__(self, window_seconds: int, max_requests: int):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests = deque()  # Store request timestamps
        self.total_requests = 0
        self.blocked_requests = 0
        
    def is_allowed(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove old requests outside window
        while self.requests and self.requests[0] < window_start:
            self.requests.popleft()
        
        self.total_requests += 1
        current_count = len(self.requests)
        
        if current_count < self.max_requests:
            self.requests.append(now)
            return True, {
                'allowed': True,
                'current_count': current_count + 1,
                'limit': self.max_requests,
                'window_seconds': self.window_seconds,
                'reset_time': window_start + self.window_seconds
            }
        else:
            self.blocked_requests += 1
            # Find when the oldest request will expire
            reset_time = self.requests[0] + self.window_seconds if self.requests else now
            
            return False, {
                'allowed': False,
                'current_count': current_count,
                'limit': self.max_requests,
                'window_seconds': self.window_seconds,
                'reset_time': reset_time,
                'retry_after': reset_time - now
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get counter metrics."""
        return {
            'window_seconds': self.window_seconds,
            'max_requests': self.max_requests,
            'current_requests': len(self.requests),
            'total_requests': self.total_requests,
            'blocked_requests': self.blocked_requests,
            'block_rate_percent': (self.blocked_requests / max(1, self.total_requests)) * 100
        }


class RateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self):
        self.limits: Dict[str, Dict[str, Union[RateLimit, RateLimitBucket, SlidingWindowCounter]]] = {}
        self.lock = threading.RLock()
        
        # IP-based tracking
        self.ip_buckets: Dict[str, RateLimitBucket] = {}
        self.ip_windows: Dict[str, SlidingWindowCounter] = {}
        
        # API key-based tracking
        self.api_key_buckets: Dict[str, RateLimitBucket] = {}
        self.api_key_windows: Dict[str, SlidingWindowCounter] = {}
        
        # Global tracking
        self.global_bucket: Optional[RateLimitBucket] = None
        self.global_window: Optional[SlidingWindowCounter] = None
        
        # Configuration
        self.default_ip_limit = RateLimit(requests=100, window_seconds=60, limit_type=LimitType.IP)
        self.default_api_key_limit = RateLimit(requests=1000, window_seconds=60, limit_type=LimitType.API_KEY)
        
        # Whitelist/Blacklist
        self.whitelisted_ips: set = set()
        self.blacklisted_ips: set = set()
        self.whitelisted_api_keys: set = set()
        
        # Custom limits per identifier
        self.custom_limits: Dict[str, RateLimit] = {}
        
        logger.info("Rate limiter initialized")
    
    def add_ip_limit(self, limit: RateLimit):
        """Add IP-based rate limit."""
        self.default_ip_limit = limit
        logger.info(f"IP rate limit set: {limit.requests} requests per {limit.window_seconds}s")
    
    def add_api_key_limit(self, limit: RateLimit):
        """Add API key-based rate limit.""" 
        self.default_api_key_limit = limit
        logger.info(f"API key rate limit set: {limit.requests} requests per {limit.window_seconds}s")
    
    def add_global_limit(self, limit: RateLimit):
        """Add global rate limit."""
        with self.lock:
            self.global_bucket = RateLimitBucket(
                capacity=limit.burst_requests,
                refill_rate=limit.requests / limit.window_seconds
            )
            self.global_window = SlidingWindowCounter(
                window_seconds=limit.window_seconds,
                max_requests=limit.requests
            )
        logger.info(f"Global rate limit set: {limit.requests} requests per {limit.window_seconds}s")
    
    def add_custom_limit(self, identifier: str, limit: RateLimit):
        """Add custom rate limit for specific identifier."""
        with self.lock:
            self.custom_limits[identifier] = limit
        logger.info(f"Custom rate limit added for {identifier}: {limit.requests} requests per {limit.window_seconds}s")
    
    def whitelist_ip(self, ip: str):
        """Add IP to whitelist."""
        try:
            # Validate IP address
            ipaddress.ip_address(ip)
            self.whitelisted_ips.add(ip)
            logger.info(f"IP {ip} added to whitelist")
        except ValueError:
            logger.error(f"Invalid IP address: {ip}")
    
    def blacklist_ip(self, ip: str):
        """Add IP to blacklist.""" 
        try:
            # Validate IP address
            ipaddress.ip_address(ip)
            self.blacklisted_ips.add(ip)
            logger.warning(f"IP {ip} added to blacklist")
        except ValueError:
            logger.error(f"Invalid IP address: {ip}")
    
    def whitelist_api_key(self, api_key: str):
        """Add API key to whitelist."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.whitelisted_api_keys.add(key_hash)
        logger.info(f"API key added to whitelist")
    
    def is_allowed(self, identifier: str, limit_type: LimitType, 
                  api_key: str = None, endpoint: str = None) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed."""
        
        # Check blacklist first
        if limit_type == LimitType.IP and identifier in self.blacklisted_ips:
            return False, {
                'allowed': False,
                'reason': 'blacklisted',
                'identifier': identifier,
                'limit_type': limit_type.value
            }
        
        # Check whitelist
        if limit_type == LimitType.IP and identifier in self.whitelisted_ips:
            return True, {
                'allowed': True,
                'reason': 'whitelisted',
                'identifier': identifier,
                'limit_type': limit_type.value
            }
        
        if api_key:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            if key_hash in self.whitelisted_api_keys:
                return True, {
                    'allowed': True,
                    'reason': 'api_key_whitelisted',
                    'limit_type': limit_type.value
                }
        
        with self.lock:
            # Check global limits first
            if self.global_window:
                global_allowed, global_info = self.global_window.is_allowed()
                if not global_allowed:
                    return False, {
                        **global_info,
                        'reason': 'global_limit_exceeded',
                        'identifier': 'global'
                    }
            
            # Get appropriate limit configuration
            if identifier in self.custom_limits:
                limit_config = self.custom_limits[identifier]
            elif limit_type == LimitType.IP:
                limit_config = self.default_ip_limit
            elif limit_type == LimitType.API_KEY:
                limit_config = self.default_api_key_limit
            else:
                # Default to IP limits
                limit_config = self.default_ip_limit
            
            # Get or create appropriate tracker
            if limit_type == LimitType.IP:
                if identifier not in self.ip_windows:
                    self.ip_windows[identifier] = SlidingWindowCounter(
                        window_seconds=limit_config.window_seconds,
                        max_requests=limit_config.requests
                    )
                tracker = self.ip_windows[identifier]
            elif limit_type == LimitType.API_KEY:
                if identifier not in self.api_key_windows:
                    self.api_key_windows[identifier] = SlidingWindowCounter(
                        window_seconds=limit_config.window_seconds,
                        max_requests=limit_config.requests
                    )
                tracker = self.api_key_windows[identifier]
            else:
                # Fallback to IP tracking
                if identifier not in self.ip_windows:
                    self.ip_windows[identifier] = SlidingWindowCounter(
                        window_seconds=limit_config.window_seconds,
                        max_requests=limit_config.requests
                    )
                tracker = self.ip_windows[identifier]
            
            # Check if request is allowed
            allowed, info = tracker.is_allowed()
            
            result = {
                **info,
                'identifier': identifier,
                'limit_type': limit_type.value,
                'endpoint': endpoint
            }
            
            if not allowed:
                result['reason'] = 'rate_limit_exceeded'
                logger.warning(
                    f"Rate limit exceeded for {limit_type.value} {identifier}: "
                    f"{info.get('current_count', 0)}/{info.get('limit', 0)} requests"
                )
            
            return allowed, result
    
    def get_rate_limit_info(self, identifier: str, limit_type: LimitType) -> Dict[str, Any]:
        """Get rate limit information for identifier."""
        with self.lock:
            info = {
                'identifier': identifier,
                'limit_type': limit_type.value,
                'whitelisted': False,
                'blacklisted': False,
                'custom_limit': identifier in self.custom_limits
            }
            
            # Check whitelist/blacklist status
            if limit_type == LimitType.IP:
                info['whitelisted'] = identifier in self.whitelisted_ips
                info['blacklisted'] = identifier in self.blacklisted_ips
                
                if identifier in self.ip_windows:
                    info['metrics'] = self.ip_windows[identifier].get_metrics()
            
            elif limit_type == LimitType.API_KEY:
                if identifier in self.api_key_windows:
                    info['metrics'] = self.api_key_windows[identifier].get_metrics()
            
            # Add limit configuration
            if identifier in self.custom_limits:
                limit_config = self.custom_limits[identifier]
            elif limit_type == LimitType.IP:
                limit_config = self.default_ip_limit
            elif limit_type == LimitType.API_KEY:
                limit_config = self.default_api_key_limit
            else:
                limit_config = self.default_ip_limit
            
            info['limit_config'] = {
                'requests': limit_config.requests,
                'window_seconds': limit_config.window_seconds,
                'burst_requests': limit_config.burst_requests
            }
            
            return info
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting metrics."""
        with self.lock:
            metrics = {
                'timestamp': time.time(),
                'total_tracked_ips': len(self.ip_windows),
                'total_tracked_api_keys': len(self.api_key_windows),
                'whitelisted_ips': len(self.whitelisted_ips),
                'blacklisted_ips': len(self.blacklisted_ips),
                'whitelisted_api_keys': len(self.whitelisted_api_keys),
                'custom_limits': len(self.custom_limits),
                'ip_metrics': {},
                'api_key_metrics': {},
                'global_metrics': None
            }
            
            # IP metrics
            for ip, window in self.ip_windows.items():
                metrics['ip_metrics'][ip] = window.get_metrics()
            
            # API key metrics (don't expose actual keys)
            for i, (key, window) in enumerate(self.api_key_windows.items()):
                metrics['api_key_metrics'][f'key_{i}'] = window.get_metrics()
            
            # Global metrics
            if self.global_window:
                metrics['global_metrics'] = self.global_window.get_metrics()
            
            return metrics
    
    def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """Clean up old tracking entries."""
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        with self.lock:
            # Clean IP windows
            ip_keys_to_remove = []
            for ip, window in self.ip_windows.items():
                if window.requests and window.requests[-1] < cutoff_time:
                    ip_keys_to_remove.append(ip)
            
            for ip in ip_keys_to_remove:
                del self.ip_windows[ip]
            
            # Clean API key windows
            api_keys_to_remove = []
            for key, window in self.api_key_windows.items():
                if window.requests and window.requests[-1] < cutoff_time:
                    api_keys_to_remove.append(key)
            
            for key in api_keys_to_remove:
                del self.api_key_windows[key]
            
            if ip_keys_to_remove or api_keys_to_remove:
                logger.info(f"Cleaned up {len(ip_keys_to_remove)} IP entries and {len(api_keys_to_remove)} API key entries")
    
    def reset_limits(self, identifier: str = None, limit_type: LimitType = None):
        """Reset rate limits for specific identifier or all."""
        with self.lock:
            if identifier and limit_type:
                if limit_type == LimitType.IP and identifier in self.ip_windows:
                    del self.ip_windows[identifier]
                elif limit_type == LimitType.API_KEY and identifier in self.api_key_windows:
                    del self.api_key_windows[identifier]
                logger.info(f"Reset rate limits for {limit_type.value} {identifier}")
            else:
                # Reset all
                self.ip_windows.clear()
                self.api_key_windows.clear()
                if self.global_window:
                    self.global_window = SlidingWindowCounter(
                        self.global_window.window_seconds,
                        self.global_window.max_requests
                    )
                logger.info("Reset all rate limits")


class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: float = None, limit_info: Dict[str, Any] = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_info = limit_info or {}


def rate_limit(identifier_func: Callable = None, limit_type: LimitType = LimitType.IP,
               limit: RateLimit = None):
    """Decorator for rate limiting functions."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                # Extract identifier
                if identifier_func:
                    identifier = identifier_func(*args, **kwargs)
                else:
                    identifier = "default"
                
                limiter = get_rate_limiter()
                allowed, info = limiter.is_allowed(identifier, limit_type)
                
                if not allowed:
                    retry_after = info.get('retry_after', 60)
                    raise RateLimitExceededError(
                        f"Rate limit exceeded for {identifier}",
                        retry_after=retry_after,
                        limit_info=info
                    )
                
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # Extract identifier
                if identifier_func:
                    identifier = identifier_func(*args, **kwargs)
                else:
                    identifier = "default"
                
                limiter = get_rate_limiter()
                allowed, info = limiter.is_allowed(identifier, limit_type)
                
                if not allowed:
                    retry_after = info.get('retry_after', 60)
                    raise RateLimitExceededError(
                        f"Rate limit exceeded for {identifier}",
                        retry_after=retry_after,
                        limit_info=info
                    )
                
                return func(*args, **kwargs)
            return sync_wrapper
    return decorator


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


def setup_default_rate_limits():
    """Setup default rate limiting configuration."""
    limiter = get_rate_limiter()
    
    # IP-based limits
    limiter.add_ip_limit(RateLimit(
        requests=100,      # 100 requests
        window_seconds=60, # per minute
        burst_requests=120 # with burst up to 120
    ))
    
    # API key-based limits
    limiter.add_api_key_limit(RateLimit(
        requests=1000,     # 1000 requests
        window_seconds=60, # per minute
        burst_requests=1200 # with burst up to 1200
    ))
    
    # Global limits (to protect against total system overload)
    limiter.add_global_limit(RateLimit(
        requests=10000,    # 10k requests
        window_seconds=60, # per minute
        burst_requests=12000 # with burst up to 12k
    ))
    
    logger.info("Default rate limits configured")


# Cleanup task for removing old entries
async def cleanup_task(interval_seconds: int = 300):
    """Background task to clean up old rate limit entries."""
    limiter = get_rate_limiter()
    
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            limiter.cleanup_old_entries()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in rate limiter cleanup task: {e}")