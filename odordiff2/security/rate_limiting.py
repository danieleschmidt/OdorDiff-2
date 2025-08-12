"""
Advanced rate limiting and request throttling for OdorDiff-2.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from collections import defaultdict, deque
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int
    window_seconds: int
    burst_allowance: int = 0
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, capacity: int, refill_rate: float, initial_tokens: Optional[int] = None):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        with self._lock:
            now = time.time()
            
            # Refill tokens based on time elapsed
            time_passed = now - self.last_refill
            new_tokens = time_passed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
            
            # Try to consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def peek(self) -> float:
        """Get current token count without consuming."""
        with self._lock:
            now = time.time()
            time_passed = now - self.last_refill
            new_tokens = time_passed * self.refill_rate
            return min(self.capacity, self.tokens + new_tokens)
    
    def time_until_tokens(self, tokens: int) -> float:
        """Calculate time until specified tokens are available."""
        current = self.peek()
        if current >= tokens:
            return 0.0
        
        needed = tokens - current
        return needed / self.refill_rate


class SlidingWindowCounter:
    """Sliding window rate limiter implementation."""
    
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests = deque()
        self._lock = threading.Lock()
    
    def allow_request(self) -> Tuple[bool, int]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (allowed, remaining_requests)
        """
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            # Remove old requests
            while self.requests and self.requests[0] <= cutoff:
                self.requests.popleft()
            
            # Check if we're at the limit
            if len(self.requests) >= self.limit:
                return False, 0
            
            # Add current request
            self.requests.append(now)
            remaining = self.limit - len(self.requests)
            
            return True, remaining
    
    def get_reset_time(self) -> float:
        """Get time when window resets."""
        if not self.requests:
            return time.time()
        return self.requests[0] + self.window_seconds


class LeakyBucket:
    """Leaky bucket rate limiter implementation."""
    
    def __init__(self, capacity: int, leak_rate: float):
        """
        Initialize leaky bucket.
        
        Args:
            capacity: Maximum bucket size
            leak_rate: Rate at which bucket leaks (requests per second)
        """
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.volume = 0
        self.last_leak = time.time()
        self._lock = threading.Lock()
    
    def allow_request(self) -> bool:
        """Check if request can be added to bucket."""
        with self._lock:
            now = time.time()
            
            # Leak based on time elapsed
            time_passed = now - self.last_leak
            leaked = time_passed * self.leak_rate
            self.volume = max(0, self.volume - leaked)
            self.last_leak = now
            
            # Check if we can add the request
            if self.volume >= self.capacity:
                return False
            
            self.volume += 1
            return True
    
    def get_volume(self) -> float:
        """Get current bucket volume."""
        with self._lock:
            now = time.time()
            time_passed = now - self.last_leak
            leaked = time_passed * self.leak_rate
            return max(0, self.volume - leaked)


class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    async def check_rate_limit(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is within rate limit."""
        pass
    
    @abstractmethod
    async def reset_limit(self, key: str) -> bool:
        """Reset rate limit for key."""
        pass


class InMemoryRateLimiter(BaseRateLimiter):
    """In-memory rate limiter implementation."""
    
    def __init__(self, default_limit: RateLimit):
        self.default_limit = default_limit
        self.limiters: Dict[str, Any] = {}
        self.custom_limits: Dict[str, RateLimit] = {}
        self._lock = threading.Lock()
    
    def set_custom_limit(self, key: str, limit: RateLimit):
        """Set custom rate limit for specific key."""
        with self._lock:
            self.custom_limits[key] = limit
    
    def _get_limiter(self, key: str) -> Any:
        """Get or create rate limiter for key."""
        if key not in self.limiters:
            limit = self.custom_limits.get(key, self.default_limit)
            
            if limit.strategy == RateLimitStrategy.TOKEN_BUCKET:
                self.limiters[key] = TokenBucket(
                    capacity=limit.requests + limit.burst_allowance,
                    refill_rate=limit.requests / limit.window_seconds
                )
            elif limit.strategy == RateLimitStrategy.SLIDING_WINDOW:
                self.limiters[key] = SlidingWindowCounter(
                    limit=limit.requests,
                    window_seconds=limit.window_seconds
                )
            elif limit.strategy == RateLimitStrategy.LEAKY_BUCKET:
                self.limiters[key] = LeakyBucket(
                    capacity=limit.requests,
                    leak_rate=limit.requests / limit.window_seconds
                )
            else:
                # Default to token bucket
                self.limiters[key] = TokenBucket(
                    capacity=limit.requests,
                    refill_rate=limit.requests / limit.window_seconds
                )
        
        return self.limiters[key]
    
    async def check_rate_limit(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check rate limit for key."""
        with self._lock:
            limiter = self._get_limiter(key)
            limit = self.custom_limits.get(key, self.default_limit)
            
            if isinstance(limiter, TokenBucket):
                allowed = limiter.consume(cost)
                remaining = int(limiter.peek())
                reset_time = time.time() + (cost - limiter.peek()) / limiter.refill_rate if not allowed else time.time() + limit.window_seconds
                retry_after = limiter.time_until_tokens(cost) if not allowed else None
                
            elif isinstance(limiter, SlidingWindowCounter):
                allowed, remaining = limiter.allow_request()
                reset_time = limiter.get_reset_time()
                retry_after = reset_time - time.time() if not allowed else None
                
            elif isinstance(limiter, LeakyBucket):
                allowed = limiter.allow_request()
                remaining = max(0, int(limiter.capacity - limiter.get_volume()))
                reset_time = time.time() + limit.window_seconds
                retry_after = 1.0 / limiter.leak_rate if not allowed else None
                
            else:
                # Fallback
                allowed = True
                remaining = limit.requests
                reset_time = time.time() + limit.window_seconds
                retry_after = None
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                metadata={
                    'strategy': limit.strategy.value,
                    'limit': limit.requests,
                    'window': limit.window_seconds,
                    'key': key
                }
            )
    
    async def reset_limit(self, key: str) -> bool:
        """Reset rate limit for key."""
        with self._lock:
            if key in self.limiters:
                del self.limiters[key]
                return True
            return False


class RedisRateLimiter(BaseRateLimiter):
    """Redis-based distributed rate limiter."""
    
    def __init__(self, redis_client, default_limit: RateLimit, key_prefix: str = "rl:"):
        self.redis = redis_client
        self.default_limit = default_limit
        self.key_prefix = key_prefix
        self.custom_limits: Dict[str, RateLimit] = {}
    
    def set_custom_limit(self, key: str, limit: RateLimit):
        """Set custom rate limit for specific key."""
        self.custom_limits[key] = limit
    
    async def check_rate_limit(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check rate limit using Redis."""
        limit = self.custom_limits.get(key, self.default_limit)
        redis_key = f"{self.key_prefix}{key}"
        
        if limit.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_check(redis_key, limit, cost)
        elif limit.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window_check(redis_key, limit, cost)
        else:
            # Default to token bucket approximation
            return await self._token_bucket_check(redis_key, limit, cost)
    
    async def _sliding_window_check(self, key: str, limit: RateLimit, cost: int) -> RateLimitResult:
        """Implement sliding window using Redis."""
        now = time.time()
        window_start = now - limit.window_seconds
        
        # Lua script for atomic sliding window check
        script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])
        local window_seconds = tonumber(ARGV[5])
        
        -- Remove old entries
        redis.call('zremrangebyscore', key, 0, window_start)
        
        -- Count current requests
        local current = redis.call('zcard', key)
        
        if current + cost > limit then
            -- Rate limited
            local oldest = redis.call('zrange', key, 0, 0, 'WITHSCORES')
            local reset_time = window_seconds
            if oldest[2] then
                reset_time = oldest[2] + window_seconds - now
            end
            return {0, limit - current, reset_time}
        else
            -- Add current request
            redis.call('zadd', key, now, now)
            redis.call('expire', key, window_seconds + 1)
            return {1, limit - current - cost, 0}
        end
        """
        
        try:
            result = await self.redis.eval(
                script, 1, key, now, window_start, limit.requests, cost, limit.window_seconds
            )
            
            allowed = bool(result[0])
            remaining = int(result[1])
            retry_after = float(result[2]) if result[2] > 0 else None
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=now + limit.window_seconds,
                retry_after=retry_after,
                metadata={'strategy': 'sliding_window', 'key': key}
            )
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to allowing request
            return RateLimitResult(
                allowed=True,
                remaining=limit.requests,
                reset_time=now + limit.window_seconds,
                metadata={'error': str(e)}
            )
    
    async def _fixed_window_check(self, key: str, limit: RateLimit, cost: int) -> RateLimitResult:
        """Implement fixed window using Redis."""
        now = time.time()
        window = int(now // limit.window_seconds)
        window_key = f"{key}:{window}"
        
        try:
            # Increment counter atomically
            current = await self.redis.incr(window_key)
            
            if current == 1:
                # First request in window, set expiration
                await self.redis.expire(window_key, limit.window_seconds)
            
            if current <= limit.requests:
                return RateLimitResult(
                    allowed=True,
                    remaining=limit.requests - current,
                    reset_time=(window + 1) * limit.window_seconds,
                    metadata={'strategy': 'fixed_window', 'window': window}
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=(window + 1) * limit.window_seconds,
                    retry_after=(window + 1) * limit.window_seconds - now,
                    metadata={'strategy': 'fixed_window', 'window': window}
                )
                
        except Exception as e:
            logger.error(f"Redis fixed window check failed: {e}")
            return RateLimitResult(
                allowed=True,
                remaining=limit.requests,
                reset_time=now + limit.window_seconds,
                metadata={'error': str(e)}
            )
    
    async def _token_bucket_check(self, key: str, limit: RateLimit, cost: int) -> RateLimitResult:
        """Approximate token bucket using Redis."""
        # This is a simplified version - real implementation would need more complex logic
        return await self._fixed_window_check(key, limit, cost)
    
    async def reset_limit(self, key: str) -> bool:
        """Reset rate limit for key."""
        try:
            pattern = f"{self.key_prefix}{key}*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            return False


class RateLimitMiddleware:
    """Middleware for applying rate limits to requests."""
    
    def __init__(self, rate_limiter: BaseRateLimiter, key_generator: Callable = None):
        """
        Initialize rate limit middleware.
        
        Args:
            rate_limiter: Rate limiter implementation
            key_generator: Function to generate rate limit keys from requests
        """
        self.rate_limiter = rate_limiter
        self.key_generator = key_generator or self._default_key_generator
        
        # Define rate limit tiers
        self.tier_limits = {
            'free': RateLimit(requests=100, window_seconds=3600),  # 100/hour
            'basic': RateLimit(requests=1000, window_seconds=3600),  # 1000/hour
            'premium': RateLimit(requests=10000, window_seconds=3600),  # 10k/hour
            'enterprise': RateLimit(requests=100000, window_seconds=3600),  # 100k/hour
        }
    
    def _default_key_generator(self, request: Any) -> str:
        """Default key generation based on client IP."""
        # This would extract IP from request in real implementation
        client_ip = getattr(request, 'client_ip', 'unknown')
        return f"ip:{client_ip}"
    
    def _extract_user_tier(self, request: Any) -> str:
        """Extract user tier from request (API key, headers, etc.)."""
        # This would check API keys, user authentication, etc.
        api_key = getattr(request, 'api_key', None)
        if api_key:
            # In real implementation, lookup user tier from API key
            return 'basic'  # Default tier
        return 'free'
    
    async def check_request(self, request: Any, endpoint: str = None) -> RateLimitResult:
        """
        Check if request passes rate limiting.
        
        Args:
            request: Request object
            endpoint: Specific endpoint for custom limits
            
        Returns:
            Rate limit result
        """
        # Generate rate limit key
        base_key = self.key_generator(request)
        
        # Add endpoint-specific keying if provided
        if endpoint:
            key = f"{base_key}:{endpoint}"
        else:
            key = base_key
        
        # Determine user tier and set appropriate limits
        user_tier = self._extract_user_tier(request)
        if isinstance(self.rate_limiter, InMemoryRateLimiter):
            tier_limit = self.tier_limits.get(user_tier, self.tier_limits['free'])
            self.rate_limiter.set_custom_limit(key, tier_limit)
        
        # Check rate limit
        result = await self.rate_limiter.check_rate_limit(key)
        
        # Log rate limit events
        if not result.allowed:
            logger.warning(f"Rate limit exceeded for {key} (tier: {user_tier})")
        
        return result
    
    def get_rate_limit_headers(self, result: RateLimitResult) -> Dict[str, str]:
        """Get HTTP headers for rate limiting info."""
        headers = {
            'X-RateLimit-Remaining': str(result.remaining),
            'X-RateLimit-Reset': str(int(result.reset_time))
        }
        
        if result.retry_after is not None:
            headers['Retry-After'] = str(int(result.retry_after))
        
        if result.metadata.get('limit'):
            headers['X-RateLimit-Limit'] = str(result.metadata['limit'])
        
        return headers


class AdaptiveRateLimiter:
    """Rate limiter that adapts based on system load and user behavior."""
    
    def __init__(self, base_limiter: BaseRateLimiter):
        self.base_limiter = base_limiter
        self.system_load_factor = 1.0
        self.user_trust_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        self.abuse_detection = AbuseDetectionSystem()
    
    async def adaptive_check(self, key: str, request_metadata: Dict[str, Any] = None) -> RateLimitResult:
        """Check rate limit with adaptive adjustments."""
        # Get base rate limit result
        result = await self.base_limiter.check_rate_limit(key)
        
        # Adjust based on system load
        if self.system_load_factor > 1.5:  # High load
            result.allowed = result.allowed and (result.remaining > 10)
            if not result.allowed:
                result.retry_after = (result.retry_after or 1.0) * 2
        
        # Adjust based on user trust score
        trust_score = self.user_trust_scores[key]
        if trust_score < 0.5:  # Low trust user
            result.allowed = result.allowed and (result.remaining > 5)
        elif trust_score > 1.5:  # High trust user
            # Allow slight burst for trusted users
            pass
        
        # Check for abuse patterns
        if request_metadata:
            abuse_score = self.abuse_detection.analyze_request(key, request_metadata)
            if abuse_score > 0.8:  # High abuse probability
                result.allowed = False
                result.retry_after = 300  # 5 minute penalty
                logger.warning(f"Potential abuse detected for {key}, score: {abuse_score}")
        
        return result
    
    def update_system_load(self, cpu_percent: float, memory_percent: float):
        """Update system load factor based on resource usage."""
        max_usage = max(cpu_percent, memory_percent)
        
        if max_usage > 90:
            self.system_load_factor = 3.0
        elif max_usage > 75:
            self.system_load_factor = 2.0
        elif max_usage > 50:
            self.system_load_factor = 1.5
        else:
            self.system_load_factor = 1.0
    
    def update_user_trust(self, key: str, positive_action: bool):
        """Update user trust score based on behavior."""
        current_score = self.user_trust_scores[key]
        
        if positive_action:
            self.user_trust_scores[key] = min(2.0, current_score * 1.1)
        else:
            self.user_trust_scores[key] = max(0.1, current_score * 0.9)


class AbuseDetectionSystem:
    """Simple abuse detection based on request patterns."""
    
    def __init__(self):
        self.request_patterns: Dict[str, List[float]] = defaultdict(list)
        self.suspicious_patterns = {
            'rapid_fire': {'window': 10, 'threshold': 50},  # >50 requests in 10s
            'identical_requests': {'threshold': 0.8},        # >80% identical requests
            'unusual_timing': {'min_interval': 0.01}         # <10ms between requests
        }
    
    def analyze_request(self, key: str, metadata: Dict[str, Any]) -> float:
        """
        Analyze request for abuse patterns.
        
        Returns:
            Abuse score from 0.0 (clean) to 1.0 (definite abuse)
        """
        now = time.time()
        self.request_patterns[key].append(now)
        
        # Keep only recent history
        cutoff = now - 300  # 5 minutes
        self.request_patterns[key] = [t for t in self.request_patterns[key] if t > cutoff]
        
        timestamps = self.request_patterns[key]
        if len(timestamps) < 3:
            return 0.0
        
        abuse_score = 0.0
        
        # Check for rapid-fire requests
        rapid_fire_window = self.suspicious_patterns['rapid_fire']['window']
        recent_requests = [t for t in timestamps if t > now - rapid_fire_window]
        if len(recent_requests) > self.suspicious_patterns['rapid_fire']['threshold']:
            abuse_score += 0.5
        
        # Check for unusual timing patterns
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval < self.suspicious_patterns['unusual_timing']['min_interval']:
            abuse_score += 0.3
        
        # Check for bot-like regularity
        if len(intervals) > 10:
            interval_variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
            if interval_variance < 0.001:  # Very regular intervals
                abuse_score += 0.2
        
        return min(1.0, abuse_score)


# Global rate limiter instances
default_rate_limit = RateLimit(requests=1000, window_seconds=3600)  # 1000/hour
global_rate_limiter = InMemoryRateLimiter(default_rate_limit)
rate_limit_middleware = RateLimitMiddleware(global_rate_limiter)


# Convenience functions
async def check_rate_limit(key: str, cost: int = 1) -> RateLimitResult:
    """Check rate limit for key."""
    return await global_rate_limiter.check_rate_limit(key, cost)


async def reset_rate_limit(key: str) -> bool:
    """Reset rate limit for key."""
    return await global_rate_limiter.reset_limit(key)


def set_custom_limit(key: str, requests: int, window_seconds: int):
    """Set custom rate limit for key."""
    custom_limit = RateLimit(requests=requests, window_seconds=window_seconds)
    global_rate_limiter.set_custom_limit(key, custom_limit)


# Example usage
if __name__ == "__main__":
    async def test_rate_limiting():
        """Test rate limiting functionality."""
        # Create test rate limiter
        test_limit = RateLimit(requests=5, window_seconds=10)
        limiter = InMemoryRateLimiter(test_limit)
        
        print("Testing rate limiting...")
        
        # Test normal requests
        for i in range(7):
            result = await limiter.check_rate_limit("test_key")
            print(f"Request {i+1}: Allowed={result.allowed}, Remaining={result.remaining}")
            
            if not result.allowed:
                print(f"Rate limited! Retry after: {result.retry_after}s")
        
        # Wait and test reset
        print("\nWaiting for reset...")
        await asyncio.sleep(3)
        
        result = await limiter.check_rate_limit("test_key")
        print(f"After wait: Allowed={result.allowed}, Remaining={result.remaining}")
    
    # Run test
    asyncio.run(test_rate_limiting())