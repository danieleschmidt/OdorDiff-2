"""
Authentication and authorization system for OdorDiff-2.
"""

import hashlib
import hmac
import secrets
import time
import jwt
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import re
import base64

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for role-based access control."""
    GUEST = "guest"
    USER = "user"
    PREMIUM = "premium"
    RESEARCHER = "researcher"
    ADMIN = "admin"
    SYSTEM = "system"


class Permission(Enum):
    """System permissions."""
    READ_PUBLIC = "read:public"
    GENERATE_MOLECULES = "generate:molecules"
    GENERATE_BATCH = "generate:batch"
    ACCESS_MODELS = "access:models"
    MANAGE_USERS = "manage:users"
    ADMIN_SYSTEM = "admin:system"
    RESEARCH_ACCESS = "research:access"
    EXPORT_DATA = "export:data"


@dataclass
class APIKey:
    """API key data structure."""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit_tier: str = "basic"
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthenticationResult:
    """Result of authentication attempt."""
    success: bool
    user_id: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: List[Permission] = field(default_factory=list)
    api_key: Optional[APIKey] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class APIKeyValidator:
    """Validates and manages API keys."""
    
    def __init__(self, secret_key: str = None):
        """Initialize with secret key for HMAC validation."""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.api_keys: Dict[str, APIKey] = {}
        self.revoked_keys: set = set()
        
        # Role-permission mapping
        self.role_permissions = {
            UserRole.GUEST: [Permission.READ_PUBLIC],
            UserRole.USER: [
                Permission.READ_PUBLIC,
                Permission.GENERATE_MOLECULES
            ],
            UserRole.PREMIUM: [
                Permission.READ_PUBLIC,
                Permission.GENERATE_MOLECULES,
                Permission.GENERATE_BATCH,
                Permission.ACCESS_MODELS,
                Permission.EXPORT_DATA
            ],
            UserRole.RESEARCHER: [
                Permission.READ_PUBLIC,
                Permission.GENERATE_MOLECULES,
                Permission.GENERATE_BATCH,
                Permission.ACCESS_MODELS,
                Permission.RESEARCH_ACCESS,
                Permission.EXPORT_DATA
            ],
            UserRole.ADMIN: [
                Permission.READ_PUBLIC,
                Permission.GENERATE_MOLECULES,
                Permission.GENERATE_BATCH,
                Permission.ACCESS_MODELS,
                Permission.RESEARCH_ACCESS,
                Permission.EXPORT_DATA,
                Permission.MANAGE_USERS,
                Permission.ADMIN_SYSTEM
            ]
        }
    
    def generate_api_key(self, user_id: str, name: str, role: UserRole, 
                        expires_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Args:
            user_id: User identifier
            name: Human-readable name for the key
            role: User role
            expires_days: Optional expiration in days
            
        Returns:
            Tuple of (raw_key, api_key_object)
        """
        # Generate key components
        key_id = secrets.token_urlsafe(8)
        key_secret = secrets.token_urlsafe(32)
        raw_key = f"odr_{key_id}_{key_secret}"
        
        # Create hash for storage
        key_hash = self._hash_key(raw_key)
        
        # Set expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        # Get permissions for role
        permissions = self.role_permissions.get(role, [])
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            role=role,
            permissions=permissions,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            rate_limit_tier=self._get_rate_limit_tier(role)
        )
        
        # Store the key
        self.api_keys[key_id] = api_key
        
        logger.info(f"Generated API key for user {user_id} with role {role.value}")
        
        return raw_key, api_key
    
    def _hash_key(self, raw_key: str) -> str:
        """Create secure hash of API key."""
        return hashlib.pbkdf2_hex(
            raw_key.encode(),
            self.secret_key.encode(),
            100000  # iterations
        )
    
    def _get_rate_limit_tier(self, role: UserRole) -> str:
        """Get rate limit tier for role."""
        tier_mapping = {
            UserRole.GUEST: "free",
            UserRole.USER: "basic",
            UserRole.PREMIUM: "premium",
            UserRole.RESEARCHER: "premium",
            UserRole.ADMIN: "enterprise"
        }
        return tier_mapping.get(role, "free")
    
    async def validate_api_key(self, raw_key: str) -> AuthenticationResult:
        """
        Validate API key and return authentication result.
        
        Args:
            raw_key: Raw API key string
            
        Returns:
            Authentication result
        """
        # Parse key format
        if not raw_key or not raw_key.startswith("odr_"):
            return AuthenticationResult(
                success=False,
                error_message="Invalid API key format"
            )
        
        try:
            parts = raw_key.split("_", 2)
            if len(parts) != 3:
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid API key format"
                )
            
            key_id = parts[1]
            
            # Check if key exists
            if key_id not in self.api_keys:
                return AuthenticationResult(
                    success=False,
                    error_message="API key not found"
                )
            
            api_key = self.api_keys[key_id]
            
            # Check if key is revoked
            if key_id in self.revoked_keys or not api_key.is_active:
                return AuthenticationResult(
                    success=False,
                    error_message="API key has been revoked"
                )
            
            # Check expiration
            if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                return AuthenticationResult(
                    success=False,
                    error_message="API key has expired"
                )
            
            # Validate key hash
            key_hash = self._hash_key(raw_key)
            if not hmac.compare_digest(key_hash, api_key.key_hash):
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid API key"
                )
            
            # Update usage statistics
            api_key.last_used = datetime.utcnow()
            api_key.usage_count += 1
            
            # Successful authentication
            return AuthenticationResult(
                success=True,
                user_id=api_key.user_id,
                role=api_key.role,
                permissions=api_key.permissions,
                api_key=api_key,
                metadata={
                    'key_id': key_id,
                    'rate_limit_tier': api_key.rate_limit_tier,
                    'usage_count': api_key.usage_count
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Authentication error"
            )
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self.api_keys:
            self.revoked_keys.add(key_id)
            self.api_keys[key_id].is_active = False
            logger.info(f"Revoked API key: {key_id}")
            return True
        return False
    
    def list_user_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user."""
        return [key for key in self.api_keys.values() if key.user_id == user_id]
    
    def get_key_info(self, key_id: str) -> Optional[APIKey]:
        """Get API key information."""
        return self.api_keys.get(key_id)


class JWTManager:
    """Manages JWT tokens for session-based authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_blacklist: set = set()
    
    def create_token(self, user_id: str, role: UserRole, 
                    expires_minutes: int = 60, **claims) -> str:
        """
        Create JWT token.
        
        Args:
            user_id: User identifier
            role: User role
            expires_minutes: Token expiration in minutes
            **claims: Additional claims to include
            
        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        payload = {
            'sub': user_id,
            'role': role.value,
            'iat': now,
            'exp': now + timedelta(minutes=expires_minutes),
            'jti': secrets.token_urlsafe(16),  # JWT ID for blacklisting
            **claims
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> AuthenticationResult:
        """
        Verify JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Authentication result
        """
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is blacklisted
            jti = payload.get('jti')
            if jti in self.token_blacklist:
                return AuthenticationResult(
                    success=False,
                    error_message="Token has been revoked"
                )
            
            # Extract user information
            user_id = payload.get('sub')
            role_str = payload.get('role', 'guest')
            
            try:
                role = UserRole(role_str)
            except ValueError:
                role = UserRole.GUEST
            
            # Get permissions for role
            permissions = self._get_permissions_for_role(role)
            
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                role=role,
                permissions=permissions,
                metadata={
                    'token_type': 'jwt',
                    'jti': jti,
                    'exp': payload.get('exp'),
                    'iat': payload.get('iat')
                }
            )
            
        except jwt.ExpiredSignatureError:
            return AuthenticationResult(
                success=False,
                error_message="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            return AuthenticationResult(
                success=False,
                error_message=f"Invalid token: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error verifying JWT token: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Token verification error"
            )
    
    def revoke_token(self, token: str) -> bool:
        """Add token to blacklist."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get('jti')
            if jti:
                self.token_blacklist.add(jti)
                logger.info(f"Revoked JWT token: {jti}")
                return True
        except jwt.InvalidTokenError:
            pass
        return False
    
    def _get_permissions_for_role(self, role: UserRole) -> List[Permission]:
        """Get permissions for role."""
        role_permissions = {
            UserRole.GUEST: [Permission.READ_PUBLIC],
            UserRole.USER: [
                Permission.READ_PUBLIC,
                Permission.GENERATE_MOLECULES
            ],
            UserRole.PREMIUM: [
                Permission.READ_PUBLIC,
                Permission.GENERATE_MOLECULES,
                Permission.GENERATE_BATCH,
                Permission.ACCESS_MODELS,
                Permission.EXPORT_DATA
            ],
            UserRole.RESEARCHER: [
                Permission.READ_PUBLIC,
                Permission.GENERATE_MOLECULES,
                Permission.GENERATE_BATCH,
                Permission.ACCESS_MODELS,
                Permission.RESEARCH_ACCESS,
                Permission.EXPORT_DATA
            ],
            UserRole.ADMIN: [
                Permission.READ_PUBLIC,
                Permission.GENERATE_MOLECULES,
                Permission.GENERATE_BATCH,
                Permission.ACCESS_MODELS,
                Permission.RESEARCH_ACCESS,
                Permission.EXPORT_DATA,
                Permission.MANAGE_USERS,
                Permission.ADMIN_SYSTEM
            ]
        }
        return role_permissions.get(role, [])


class AuthenticationManager:
    """Central authentication management system."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize authentication manager."""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.api_key_validator = APIKeyValidator(self.secret_key)
        self.jwt_manager = JWTManager(self.secret_key)
        self.session_store: Dict[str, Dict[str, Any]] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin API key for initial setup."""
        admin_key, _ = self.api_key_validator.generate_api_key(
            user_id="admin",
            name="Default Admin Key",
            role=UserRole.ADMIN
        )
        logger.info(f"Default admin API key created: {admin_key}")
        # In production, this should be logged securely and not printed
    
    async def authenticate_request(self, auth_header: str) -> AuthenticationResult:
        """
        Authenticate request based on authorization header.
        
        Args:
            auth_header: Authorization header value
            
        Returns:
            Authentication result
        """
        if not auth_header:
            return AuthenticationResult(
                success=False,
                error_message="No authorization header provided"
            )
        
        # Parse authorization header
        auth_parts = auth_header.split(" ", 1)
        if len(auth_parts) != 2:
            return AuthenticationResult(
                success=False,
                error_message="Invalid authorization header format"
            )
        
        auth_type, auth_value = auth_parts
        
        # Handle different authentication types
        if auth_type.lower() == "bearer":
            # JWT token authentication
            return self.jwt_manager.verify_token(auth_value)
        elif auth_type.lower() == "apikey":
            # API key authentication
            return await self.api_key_validator.validate_api_key(auth_value)
        else:
            return AuthenticationResult(
                success=False,
                error_message="Unsupported authentication type"
            )
    
    def create_user_session(self, user_id: str, role: UserRole) -> str:
        """Create user session and return session token."""
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'role': role.value,
            'created_at': datetime.utcnow().isoformat(),
            'last_activity': datetime.utcnow().isoformat()
        }
        
        self.session_store[session_id] = session_data
        logger.info(f"Created session for user {user_id}")
        
        return session_id
    
    def validate_session(self, session_id: str) -> AuthenticationResult:
        """Validate user session."""
        if session_id not in self.session_store:
            return AuthenticationResult(
                success=False,
                error_message="Invalid session"
            )
        
        session_data = self.session_store[session_id]
        
        # Check session age (24 hours max)
        created_at = datetime.fromisoformat(session_data['created_at'])
        if datetime.utcnow() - created_at > timedelta(hours=24):
            del self.session_store[session_id]
            return AuthenticationResult(
                success=False,
                error_message="Session expired"
            )
        
        # Update last activity
        session_data['last_activity'] = datetime.utcnow().isoformat()
        
        # Get role and permissions
        role = UserRole(session_data['role'])
        permissions = self.jwt_manager._get_permissions_for_role(role)
        
        return AuthenticationResult(
            success=True,
            user_id=session_data['user_id'],
            role=role,
            permissions=permissions,
            metadata={
                'session_id': session_id,
                'session_type': 'server_side'
            }
        )
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke user session."""
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info(f"Revoked session: {session_id}")
            return True
        return False
    
    def has_permission(self, auth_result: AuthenticationResult, 
                      required_permission: Permission) -> bool:
        """Check if authentication result has required permission."""
        if not auth_result.success:
            return False
        
        return required_permission in auth_result.permissions
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission for function access."""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Extract auth_result from kwargs or context
                auth_result = kwargs.get('auth_result')
                if not auth_result or not self.has_permission(auth_result, permission):
                    raise PermissionError(f"Permission {permission.value} required")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_role(self, required_role: UserRole):
        """Decorator to require minimum role for function access."""
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.PREMIUM: 2,
            UserRole.RESEARCHER: 3,
            UserRole.ADMIN: 4,
            UserRole.SYSTEM: 5
        }
        
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                auth_result = kwargs.get('auth_result')
                if not auth_result or not auth_result.success:
                    raise PermissionError("Authentication required")
                
                user_role_level = role_hierarchy.get(auth_result.role, 0)
                required_level = role_hierarchy.get(required_role, 0)
                
                if user_role_level < required_level:
                    raise PermissionError(f"Role {required_role.value} or higher required")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Get all API keys for user."""
        return self.api_key_validator.list_user_keys(user_id)
    
    def create_api_key(self, user_id: str, name: str, role: UserRole,
                      expires_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """Create new API key for user."""
        return self.api_key_validator.generate_api_key(
            user_id, name, role, expires_days
        )
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke API key."""
        return self.api_key_validator.revoke_api_key(key_id)


# Global authentication manager
auth_manager = AuthenticationManager()


# Convenience functions
async def authenticate_request(auth_header: str) -> AuthenticationResult:
    """Authenticate request."""
    return await auth_manager.authenticate_request(auth_header)


def require_permission(permission: Permission):
    """Decorator requiring permission."""
    return auth_manager.require_permission(permission)


def require_role(role: UserRole):
    """Decorator requiring role."""
    return auth_manager.require_role(role)


# Example usage and testing
if __name__ == "__main__":
    async def test_authentication():
        """Test authentication system."""
        print("Testing authentication system...")
        
        # Create test users
        test_key, api_key = auth_manager.create_api_key(
            user_id="test_user",
            name="Test Key",
            role=UserRole.USER
        )
        
        print(f"Created API key: {test_key}")
        print(f"Key ID: {api_key.key_id}")
        print(f"Role: {api_key.role.value}")
        print(f"Permissions: {[p.value for p in api_key.permissions]}")
        
        # Test authentication
        auth_header = f"ApiKey {test_key}"
        result = await auth_manager.authenticate_request(auth_header)
        
        print(f"\nAuthentication result:")
        print(f"Success: {result.success}")
        print(f"User ID: {result.user_id}")
        print(f"Role: {result.role.value if result.role else None}")
        print(f"Permissions: {[p.value for p in result.permissions]}")
        
        # Test JWT
        jwt_token = auth_manager.jwt_manager.create_token("jwt_user", UserRole.PREMIUM)
        print(f"\nJWT Token: {jwt_token}")
        
        jwt_result = auth_manager.jwt_manager.verify_token(jwt_token)
        print(f"JWT Authentication: {jwt_result.success}")
        print(f"JWT User: {jwt_result.user_id}")
        print(f"JWT Role: {jwt_result.role.value if jwt_result.role else None}")
    
    # Run test
    asyncio.run(test_authentication())