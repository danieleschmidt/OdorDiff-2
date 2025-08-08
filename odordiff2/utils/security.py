"""
Security utilities and protection mechanisms.
"""

import hashlib
import hmac
import time
import secrets
from typing import Dict, Any, Optional, List
from functools import wraps
from datetime import datetime, timedelta
import re
import ipaddress
from collections import defaultdict
import threading

from .logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
        self.lock = threading.RLock()
        
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make request."""
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests outside time window
            client_requests[:] = [req_time for req_time in client_requests 
                                if now - req_time < self.time_window]
            
            # Check if under limit
            if len(client_requests) >= self.max_requests:
                return False
            
            # Add current request
            client_requests.append(now)
            return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get number of remaining requests for client."""
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests
            client_requests[:] = [req_time for req_time in client_requests 
                                if now - req_time < self.time_window]
            
            return max(0, self.max_requests - len(client_requests))


class SecurityValidator:
    """Security validation and sanitization utilities."""
    
    DANGEROUS_PATTERNS = [
        # SQL injection patterns
        r"(?i)(union\s+select|select\s+.*\s+from|insert\s+into|delete\s+from|update\s+.*\s+set)",
        r"(?i)(drop\s+table|alter\s+table|create\s+table)",
        r"(?i)('|\"|;|--|\*|/\*|\*/)",
        
        # XSS patterns
        r"(?i)(<script|</script|javascript:|vbscript:|on\w+\s*=)",
        r"(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()",
        
        # Command injection
        r"(?i)(;\s*rm\s|-rf\s|>\s*/dev/null|\|\s*nc\s)",
        r"(?i)(\$\(|`|\${|eval\s*\()",
        
        # Path traversal
        r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e\\)",
        
        # LDAP injection
        r"(?i)(\(\s*\||\)\s*\(|\*\s*\)|\(\s*\&)"
    ]
    
    @classmethod
    def validate_input(cls, input_text: str, max_length: int = 1000) -> bool:
        """Validate input against security threats."""
        if not isinstance(input_text, str):
            return False
        
        if len(input_text) > max_length:
            logger.warning(f"Input too long: {len(input_text)} chars")
            return False
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, input_text):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                return False
        
        return True
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        filename = re.sub(r'\.\.+', '.', filename)
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        return filename or 'unnamed'
    
    @classmethod
    def validate_smiles_security(cls, smiles: str) -> bool:
        """Security validation specifically for SMILES strings."""
        if not isinstance(smiles, str) or len(smiles) > 1000:
            return False
        
        # Allow only valid SMILES characters
        allowed_chars = set('CNOSPFClBrI[]()=#+-.0123456789@/\\')
        if not set(smiles).issubset(allowed_chars):
            logger.warning(f"Invalid characters in SMILES: {set(smiles) - allowed_chars}")
            return False
        
        # Check for suspicious patterns
        suspicious = ['eval', 'exec', 'import', '__', 'os.', 'sys.']
        if any(sus in smiles.lower() for sus in suspicious):
            logger.warning(f"Suspicious pattern in SMILES: {smiles}")
            return False
        
        return True
    
    @classmethod
    def validate_ip_address(cls, ip: str) -> bool:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @classmethod
    def is_private_ip(cls, ip: str) -> bool:
        """Check if IP is private/internal."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except ValueError:
            return False


class APIKeyManager:
    """API key management and validation."""
    
    def __init__(self):
        self.keys = {}  # In production, use secure storage
        self.key_permissions = {}
        self.key_usage = defaultdict(int)
        
    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate new API key."""
        api_key = secrets.token_urlsafe(32)
        self.keys[api_key] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'active': True
        }
        self.key_permissions[api_key] = permissions or ['generate', 'assess']
        
        logger.info(f"Generated API key for user: {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        if api_key not in self.keys:
            return None
        
        key_info = self.keys[api_key]
        if not key_info['active']:
            return None
        
        # Update usage
        self.key_usage[api_key] += 1
        
        return key_info
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        if api_key in self.keys:
            self.keys[api_key]['active'] = False
            logger.info(f"Revoked API key: {api_key[:8]}...")
            return True
        return False
    
    def has_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission."""
        return permission in self.key_permissions.get(api_key, [])


class SecurityMiddleware:
    """Security middleware for request processing."""
    
    def __init__(self, 
                 rate_limiter: RateLimiter = None,
                 api_key_manager: APIKeyManager = None,
                 enable_cors: bool = True):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.api_key_manager = api_key_manager or APIKeyManager()
        self.enable_cors = enable_cors
        self.blocked_ips = set()
        self.suspicious_activity = defaultdict(int)
        
    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive request validation."""
        client_ip = request_data.get('client_ip', 'unknown')
        user_agent = request_data.get('user_agent', '')
        
        # Check blocked IPs
        if client_ip in self.blocked_ips:
            raise SecurityError(f"IP {client_ip} is blocked")
        
        # Rate limiting
        if not self.rate_limiter.is_allowed(client_ip):
            self.suspicious_activity[client_ip] += 1
            raise SecurityError("Rate limit exceeded")
        
        # Validate user agent
        if not user_agent or len(user_agent) > 500:
            logger.warning(f"Suspicious user agent from {client_ip}: {user_agent}")
        
        # Check for bot/scanner patterns
        bot_patterns = [
            'sqlmap', 'nikto', 'nmap', 'masscan', 'zap',
            'burp', 'dirb', 'gobuster', 'wpscan'
        ]
        if any(bot in user_agent.lower() for bot in bot_patterns):
            logger.warning(f"Security scanner detected from {client_ip}")
            self.suspicious_activity[client_ip] += 10
        
        # Auto-block highly suspicious IPs
        if self.suspicious_activity[client_ip] > 50:
            self.blocked_ips.add(client_ip)
            logger.warning(f"Auto-blocked suspicious IP: {client_ip}")
            raise SecurityError("IP blocked due to suspicious activity")
        
        return {
            'client_ip': client_ip,
            'user_agent': user_agent,
            'remaining_requests': self.rate_limiter.get_remaining_requests(client_ip)
        }
    
    def block_ip(self, ip: str) -> None:
        """Manually block IP address."""
        if SecurityValidator.validate_ip_address(ip):
            self.blocked_ips.add(ip)
            logger.info(f"Manually blocked IP: {ip}")
    
    def unblock_ip(self, ip: str) -> None:
        """Unblock IP address."""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"Unblocked IP: {ip}")


class SecurityError(Exception):
    """Custom security-related exception."""
    pass


class SecureStorage:
    """Secure storage utilities."""
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> tuple:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 with high iteration count
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return hashed.hex(), salt
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash."""
        computed_hash, _ = SecureStorage.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, hashed)
    
    @staticmethod
    def encrypt_sensitive_data(data: str, key: str) -> str:
        """Encrypt sensitive data (placeholder - use proper encryption in production)."""
        # This is a placeholder - in production, use proper encryption like Fernet
        return hashlib.sha256(f"{key}:{data}".encode()).hexdigest()


def security_required(permissions: List[str] = None):
    """Decorator to require security validation for endpoints."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # In a real FastAPI implementation, this would integrate with request context
            # For now, this is a placeholder structure
            
            # Extract request context (would come from FastAPI)
            request_data = kwargs.get('request_data', {})
            
            # Validate security
            security_middleware = SecurityMiddleware()
            try:
                security_context = security_middleware.validate_request(request_data)
                kwargs['security_context'] = security_context
                
                # Check API key if required
                if permissions:
                    api_key = request_data.get('api_key')
                    if not api_key:
                        raise SecurityError("API key required")
                    
                    key_info = security_middleware.api_key_manager.validate_api_key(api_key)
                    if not key_info:
                        raise SecurityError("Invalid API key")
                    
                    # Check permissions
                    for perm in permissions:
                        if not security_middleware.api_key_manager.has_permission(api_key, perm):
                            raise SecurityError(f"Permission {perm} required")
                
                return func(*args, **kwargs)
                
            except SecurityError as e:
                logger.warning(f"Security error: {e}")
                raise
                
        return wrapper
    return decorator


def audit_log(action: str, user_id: str = None, details: Dict[str, Any] = None):
    """Log security-relevant actions."""
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'action': action,
        'user_id': user_id,
        'details': details or {}
    }
    
    logger.info(f"AUDIT: {action}", **log_entry)


# Global instances
_security_middleware = None
_api_key_manager = None

def get_security_middleware() -> SecurityMiddleware:
    """Get global security middleware instance."""
    global _security_middleware
    if _security_middleware is None:
        _security_middleware = SecurityMiddleware()
    return _security_middleware

def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager