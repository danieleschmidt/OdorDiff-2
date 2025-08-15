"""
Advanced authentication and authorization system for OdorDiff-2.
"""

import asyncio
import hashlib
import hmac
import jwt
import secrets
import time
from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import bcrypt
from functools import wraps
import threading

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    READ_MOLECULES = "read:molecules"
    GENERATE_MOLECULES = "generate:molecules"
    BATCH_GENERATE = "batch:generate"
    ADMIN_ACCESS = "admin:access"
    API_METRICS = "api:metrics"
    SAFETY_OVERRIDE = "safety:override"


class UserRole(Enum):
    """User roles with associated permissions."""
    GUEST = "guest"
    USER = "user"
    RESEARCHER = "researcher"
    ADMIN = "admin"


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    api_key: Optional[str] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    rate_limit_per_hour: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set default permissions based on role
        role_permissions = {
            UserRole.GUEST: {Permission.READ_MOLECULES},
            UserRole.USER: {Permission.READ_MOLECULES, Permission.GENERATE_MOLECULES},
            UserRole.RESEARCHER: {
                Permission.READ_MOLECULES, 
                Permission.GENERATE_MOLECULES,
                Permission.BATCH_GENERATE,
                Permission.API_METRICS
            },
            UserRole.ADMIN: {perm for perm in Permission}
        }
        
        if not self.permissions:
            self.permissions = role_permissions.get(self.role, set())


@dataclass
class AuthToken:
    """Authentication token."""
    token: str
    user_id: str
    expires_at: datetime
    permissions: Set[Permission]
    token_type: str = "access"  # access, refresh, api_key
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenManager:
    """Advanced token management with rotation and validation."""
    
    def __init__(self, secret_key: str, redis_client: Optional[redis.Redis] = None):
        self.secret_key = secret_key
        self.redis_client = redis_client
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=30)
        self.api_key_expire = timedelta(days=365)
        
        # Token blacklist
        self._blacklisted_tokens: Set[str] = set()
        self._lock = threading.RLock()
    
    def generate_access_token(self, user: User) -> AuthToken:
        """Generate access token for user."""
        expires_at = datetime.utcnow() + self.access_token_expire
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [perm.value for perm in user.permissions],
            "exp": expires_at.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "type": "access"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return AuthToken(
            token=token,
            user_id=user.user_id,
            expires_at=expires_at,
            permissions=user.permissions,
            token_type="access"
        )
    
    def generate_refresh_token(self, user: User) -> AuthToken:
        """Generate refresh token for user."""
        expires_at = datetime.utcnow() + self.refresh_token_expire
        
        payload = {
            "user_id": user.user_id,
            "exp": expires_at.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store refresh token in Redis if available
        if self.redis_client:
            self.redis_client.setex(
                f"refresh_token:{user.user_id}",
                int(self.refresh_token_expire.total_seconds()),
                token
            )
        
        return AuthToken(
            token=token,
            user_id=user.user_id,
            expires_at=expires_at,
            permissions=set(),
            token_type="refresh"
        )
    
    def generate_api_key(self, user: User) -> str:
        """Generate API key for user."""
        # Create a unique API key
        key_data = f"{user.user_id}:{user.username}:{secrets.token_urlsafe(32)}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()
        
        # Store API key mapping
        if self.redis_client:
            key_info = {
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "permissions": [perm.value for perm in user.permissions],
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + self.api_key_expire).isoformat()
            }
            
            self.redis_client.setex(
                f"api_key:{api_key}",
                int(self.api_key_expire.total_seconds()),
                json.dumps(key_info)
            )
        
        return api_key
    
    def validate_token(self, token: str) -> Optional[AuthToken]:
        """Validate and decode token."""
        try:
            # Check if token is blacklisted
            with self._lock:
                if token in self._blacklisted_tokens:
                    return None
            
            # Decode JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            if datetime.utcnow().timestamp() > payload.get("exp", 0):
                return None
            
            # Extract permissions
            permissions = {Permission(perm) for perm in payload.get("permissions", [])}
            
            return AuthToken(
                token=token,
                user_id=payload["user_id"],
                expires_at=datetime.fromtimestamp(payload["exp"]),
                permissions=permissions,
                token_type=payload.get("type", "access")
            )
            
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    def validate_api_key(self, api_key: str) -> Optional[User]:
        """Validate API key and return user."""
        if not self.redis_client:
            return None
        
        try:
            key_data = self.redis_client.get(f"api_key:{api_key}")
            if not key_data:
                return None
            
            info = json.loads(key_data)
            
            # Check expiration
            expires_at = datetime.fromisoformat(info["expires_at"])
            if datetime.utcnow() > expires_at:
                return None
            
            # Create user object
            permissions = {Permission(perm) for perm in info["permissions"]}
            
            return User(
                user_id=info["user_id"],
                username=info["username"],
                email="",  # Not stored in API key
                role=UserRole(info["role"]),
                permissions=permissions,
                api_key=api_key
            )
            
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return None
    
    def revoke_token(self, token: str):
        """Revoke a token by adding it to blacklist."""
        with self._lock:
            self._blacklisted_tokens.add(token)
        
        # Also remove from Redis if it's a refresh token
        if self.redis_client:
            try:
                decoded = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                if decoded.get("type") == "refresh":
                    self.redis_client.delete(f"refresh_token:{decoded['user_id']}")
            except jwt.InvalidTokenError:
                pass
    
    def refresh_access_token(self, refresh_token: str) -> Optional[AuthToken]:
        """Generate new access token using refresh token."""
        auth_token = self.validate_token(refresh_token)
        
        if not auth_token or auth_token.token_type != "refresh":
            return None
        
        # Verify refresh token exists in Redis
        if self.redis_client:
            stored_token = self.redis_client.get(f"refresh_token:{auth_token.user_id}")
            if not stored_token or stored_token.decode() != refresh_token:
                return None
        
        # Generate new access token (would need to fetch user from database)
        # For now, create a basic user object
        user = User(
            user_id=auth_token.user_id,
            username="",  # Would fetch from database
            email="",
            role=UserRole.USER,  # Would fetch from database
            permissions=auth_token.permissions
        )
        
        return self.generate_access_token(user)


class RateLimiter:
    """Advanced rate limiting with sliding window."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self._local_counters: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def is_allowed(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 3600,
        burst_limit: Optional[int] = None
    ) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()
        
        if self.redis_client:
            return self._redis_rate_limit(identifier, limit, window_seconds, current_time)
        else:
            return self._local_rate_limit(identifier, limit, window_seconds, current_time, burst_limit)
    
    def _redis_rate_limit(self, identifier: str, limit: int, window_seconds: int, current_time: float) -> bool:
        """Redis-based sliding window rate limiting."""
        key = f"rate_limit:{identifier}"
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, current_time - window_seconds)
        
        # Count current entries
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, window_seconds + 1)
        
        results = pipe.execute()
        request_count = results[1]
        
        return request_count < limit
    
    def _local_rate_limit(self, identifier: str, limit: int, window_seconds: int, current_time: float, burst_limit: Optional[int]) -> bool:
        """Local sliding window rate limiting."""
        with self._lock:
            if identifier not in self._local_counters:
                self._local_counters[identifier] = []
            
            # Remove old entries
            window_start = current_time - window_seconds
            self._local_counters[identifier] = [
                t for t in self._local_counters[identifier] if t > window_start
            ]
            
            # Check burst limit (last minute)
            if burst_limit:
                burst_window_start = current_time - 60
                recent_requests = sum(1 for t in self._local_counters[identifier] if t > burst_window_start)
                if recent_requests >= burst_limit:
                    return False
            
            # Check main limit
            if len(self._local_counters[identifier]) >= limit:
                return False
            
            # Add current request
            self._local_counters[identifier].append(current_time)
            return True
    
    def get_remaining(self, identifier: str, limit: int, window_seconds: int = 3600) -> int:
        """Get remaining requests for identifier."""
        current_time = time.time()
        
        if self.redis_client:
            key = f"rate_limit:{identifier}"
            # Remove old entries and count current
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, current_time - window_seconds)
            pipe.zcard(key)
            results = pipe.execute()
            current_count = results[1]
        else:
            with self._lock:
                if identifier not in self._local_counters:
                    return limit
                
                window_start = current_time - window_seconds
                current_count = sum(1 for t in self._local_counters[identifier] if t > window_start)
        
        return max(0, limit - current_count)


class AuthenticationService:
    """Complete authentication service."""
    
    def __init__(self, secret_key: str, redis_client: Optional[redis.Redis] = None):
        self.token_manager = TokenManager(secret_key, redis_client)
        self.rate_limiter = RateLimiter(redis_client)
        self.redis_client = redis_client
        
        # In-memory user store (in production, use database)
        self._users: Dict[str, User] = {}
        self._users_by_email: Dict[str, str] = {}
        self._lock = threading.RLock()
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> User:
        """Create a new user account."""
        with self._lock:
            if email in self._users_by_email:
                raise ValueError("Email already exists")
            
            user_id = secrets.token_urlsafe(16)
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                role=role
            )
            
            self._users[user_id] = user
            self._users_by_email[email] = user_id
            
            # Store password hash separately (in production, use secure database)
            if self.redis_client:
                self.redis_client.setex(
                    f"password:{user_id}",
                    86400 * 365,  # 1 year
                    password_hash
                )
            
            logger.info(f"Created user: {username} ({email})")
            return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email/password."""
        with self._lock:
            user_id = self._users_by_email.get(email)
            if not user_id:
                return None
            
            user = self._users.get(user_id)
            if not user or not user.is_active:
                return None
            
            # Verify password
            if self.redis_client:
                stored_hash = self.redis_client.get(f"password:{user_id}")
                if not stored_hash:
                    return None
                
                if not bcrypt.checkpw(password.encode(), stored_hash):
                    return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            
            return user
    
    def login(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Login user and return tokens."""
        user = self.authenticate_user(email, password)
        if not user:
            return None
        
        access_token = self.token_manager.generate_access_token(user)
        refresh_token = self.token_manager.generate_refresh_token(user)
        
        return {
            "access_token": access_token.token,
            "refresh_token": refresh_token.token,
            "token_type": "bearer",
            "expires_in": int(self.token_manager.access_token_expire.total_seconds()),
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "permissions": [perm.value for perm in user.permissions]
            }
        }
    
    def logout(self, token: str):
        """Logout user by revoking token."""
        self.token_manager.revoke_token(token)
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # This would be used with Flask/FastAPI to check request headers
                # Implementation depends on web framework
                pass
            return wrapper
        return decorator


# Example usage with FastAPI integration
def create_auth_dependency(auth_service: AuthenticationService):
    """Create FastAPI dependency for authentication."""
    from fastapi import HTTPException, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    
    security = HTTPBearer()
    
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
        """Get current authenticated user."""
        token = credentials.credentials
        
        # Try JWT token first
        auth_token = auth_service.token_manager.validate_token(token)
        if auth_token:
            user = auth_service._users.get(auth_token.user_id)
            if user and user.is_active:
                return user
        
        # Try API key
        user = auth_service.token_manager.validate_api_key(token)
        if user and user.is_active:
            return user
        
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    return get_current_user


# Global authentication service (configured at startup)
auth_service: Optional[AuthenticationService] = None


def setup_authentication(secret_key: str, redis_url: Optional[str] = None):
    """Setup global authentication service."""
    global auth_service
    
    redis_client = None
    if redis_url:
        try:
            import redis
            redis_client = redis.from_url(redis_url)
            redis_client.ping()  # Test connection
            logger.info("Connected to Redis for authentication")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
    
    auth_service = AuthenticationService(secret_key, redis_client)
    
    # Create default admin user if none exists
    try:
        admin_user = auth_service.create_user(
            username="admin",
            email="admin@odordiff.ai",
            password="change_me_in_production",
            role=UserRole.ADMIN
        )
        logger.info("Created default admin user")
    except ValueError:
        logger.info("Admin user already exists")
    
    logger.info("Authentication system initialized")