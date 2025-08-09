"""
Enhanced security utilities and protection mechanisms with advanced hardening.
"""

import hashlib
import hmac
import time
import secrets
import base64
import json
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple
from functools import wraps
from datetime import datetime, timedelta
import re
import ipaddress
from collections import defaultdict
import threading
import asyncio
from urllib.parse import quote, unquote
import jwt

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

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


class RequestSigner:
    """Request signature verification for enhanced API security."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
    
    def sign_request(self, method: str, url: str, body: bytes = b"", timestamp: int = None) -> str:
        """Sign a request with HMAC signature."""
        if timestamp is None:
            timestamp = int(time.time())
        
        # Create canonical request string
        canonical_request = f"{method}\n{url}\n{timestamp}\n{len(body)}\n{hashlib.sha256(body).hexdigest()}"
        
        # Generate signature
        signature = hmac.new(
            self.secret_key,
            canonical_request.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}.{signature}"
    
    def verify_signature(self, signature: str, method: str, url: str, body: bytes = b"", 
                        max_age: int = 300) -> bool:
        """Verify request signature."""
        try:
            timestamp_str, provided_signature = signature.split('.', 1)
            timestamp = int(timestamp_str)
            
            # Check timestamp validity (prevent replay attacks)
            current_time = int(time.time())
            if abs(current_time - timestamp) > max_age:
                logger.warning(f"Request signature too old: {current_time - timestamp}s")
                return False
            
            # Verify signature
            expected_signature = self.sign_request(method, url, body, timestamp).split('.', 1)[1]
            return hmac.compare_digest(expected_signature, provided_signature)
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid signature format: {e}")
            return False


class JWTManager:
    """JWT token management with security best practices."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Create JWT token with expiration."""
        now = datetime.utcnow()
        payload.update({
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'jti': str(uuid.uuid4()),  # Unique token ID
            'iss': 'odordiff2-api'     # Issuer
        })
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": True, "verify_iat": True}
            )
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
        except Exception as e:
            logger.error(f"JWT verification error: {e}")
        
        return None
    
    def refresh_token(self, token: str, new_expires_in: int = 3600) -> Optional[str]:
        """Refresh JWT token if valid."""
        payload = self.verify_token(token)
        if payload:
            # Remove old timestamp claims
            payload.pop('iat', None)
            payload.pop('exp', None)
            payload.pop('jti', None)
            
            return self.create_token(payload, new_expires_in)
        return None


class AdvancedEncryption:
    """Advanced encryption utilities using industry standards."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        self.master_key = master_key or self.generate_key()
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate a new encryption key."""
        return Fernet.generate_key()
    
    def derive_key(self, password: str, salt: bytes = None) -> bytes:
        """Derive key from password using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        return base64.urlsafe_b64encode(kdf.derive(password.encode())), salt
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data with the master key."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        f = Fernet(self.master_key)
        encrypted = f.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> bytes:
        """Decrypt data with the master key."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        f = Fernet(self.master_key)
        return f.decrypt(encrypted_bytes)
    
    def encrypt_with_password(self, data: Union[str, bytes], password: str) -> str:
        """Encrypt data with password-derived key."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        key, salt = self.derive_key(password)
        f = Fernet(key)
        encrypted = f.encrypt(data)
        
        # Combine salt and encrypted data
        result = salt + encrypted
        return base64.urlsafe_b64encode(result).decode()
    
    def decrypt_with_password(self, encrypted_data: str, password: str) -> bytes:
        """Decrypt data with password-derived key."""
        combined = base64.urlsafe_b64decode(encrypted_data.encode())
        salt = combined[:16]  # First 16 bytes are salt
        encrypted = combined[16:]  # Rest is encrypted data
        
        key, _ = self.derive_key(password, salt)
        f = Fernet(key)
        return f.decrypt(encrypted)


class SecurityHeaders:
    """Security headers management for HTTP responses."""
    
    @staticmethod
    def get_security_headers(strict: bool = False) -> Dict[str, str]:
        """Get comprehensive security headers."""
        headers = {
            # Prevent clickjacking
            'X-Frame-Options': 'DENY',
            
            # Prevent MIME type sniffing
            'X-Content-Type-Options': 'nosniff',
            
            # XSS protection
            'X-XSS-Protection': '1; mode=block',
            
            # Referrer policy
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            
            # Download options (IE)
            'X-Download-Options': 'noopen',
            
            # Permitted cross-domain policies
            'X-Permitted-Cross-Domain-Policies': 'none',
            
            # Cache control for sensitive data
            'Cache-Control': 'no-cache, no-store, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0',
        }
        
        if strict:
            # Strict Transport Security (HTTPS only)
            headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
            
            # Content Security Policy
            headers['Content-Security-Policy'] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' https:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            )
            
            # Expect-CT (Certificate Transparency)
            headers['Expect-CT'] = 'max-age=86400, enforce'
        
        return headers
    
    @staticmethod
    def add_request_id_header(request_id: str) -> Dict[str, str]:
        """Add request ID header for tracking."""
        return {'X-Request-ID': request_id}


class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    # Extended dangerous patterns for enhanced security
    ADVANCED_PATTERNS = [
        # Advanced SQL injection
        r"(?i)(waitfor\s+delay|pg_sleep|benchmark\s*\(|sleep\s*\()",
        r"(?i)(information_schema|sys\.|mysql\.|pg_)",
        r"(?i)(0x[0-9a-f]+|char\s*\(|ascii\s*\(|substring\s*\()",
        
        # NoSQL injection
        r"(?i)(\$where|\$ne|\$gt|\$lt|\$regex|\$or|\$and)",
        r"(?i)(this\s*\.\s*\w+|javascript\s*:)",
        
        # Advanced XSS
        r"(?i)(data\s*:\s*text/html|data\s*:\s*application/)",
        r"(?i)(expression\s*\(|behavior\s*:|@import)",
        r"(?i)(mocha\s*:|livescript\s*:|vbscript\s*:)",
        
        # Template injection
        r"(?i)({{\s*.*\s*}}|\${.*}|\[\[.*\]\])",
        r"(?i)(__import__|exec\s*\(|eval\s*\(|compile\s*\()",
        
        # LDAP injection advanced
        r"(?i)(\)\s*\(\s*\||;\s*\w+\s*=|\*\s*\)\s*\()",
        
        # XML/XXE injection
        r"(?i)(<!entity|<!doctype|<!element|<!\[cdata\[)",
        r"(?i)(system\s+[\"']file:|system\s+[\"']http:)",
        
        # Command injection advanced
        r"(?i)(\|\s*\w+|&&\s*\w+|;\s*\w+|\$\(\w+)",
        r"(?i)(curl\s|wget\s|nc\s|netcat\s|telnet\s)",
        r"(?i)(base64\s|python\s|perl\s|ruby\s|node\s)",
        
        # Path traversal advanced
        r"(?i)(file\s*:\s*//|ftp\s*:\s*//|\\\\[.\w]+\\)",
        r"(?i)(%c0%ae%c0%ae|%c1%1c|%c0%2f|%c1%9c)",
        
        # Server-side request forgery (SSRF)
        r"(?i)(localhost|127\.0\.0\.1|0\.0\.0\.0|::1)",
        r"(?i)(169\.254\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)",
        
        # Deserialization attacks
        r"(?i)(pickle\.|marshal\.|yaml\.load|objectInputStream)"
    ]
    
    @classmethod
    def advanced_validate(cls, input_data: Union[str, bytes], 
                         context: str = "general", max_length: int = 1000) -> bool:
        """Advanced security validation with context awareness."""
        if isinstance(input_data, bytes):
            input_data = input_data.decode('utf-8', errors='ignore')
        
        if not isinstance(input_data, str):
            return False
        
        # Length check
        if len(input_data) > max_length:
            logger.warning(f"Input too long in {context}: {len(input_data)} chars")
            return False
        
        # Check base patterns
        if not SecurityValidator.validate_input(input_data, max_length):
            return False
        
        # Check advanced patterns
        for pattern in cls.ADVANCED_PATTERNS:
            if re.search(pattern, input_data):
                logger.warning(f"Advanced threat pattern detected in {context}: {pattern[:50]}")
                return False
        
        # Context-specific validation
        if context == "smiles":
            return SecurityValidator.validate_smiles_security(input_data)
        elif context == "filename":
            suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', 
                                   '.com', '.sys', '.dll', '.vbs', '.js', '.jar']
            if any(input_data.lower().endswith(ext) for ext in suspicious_extensions):
                logger.warning(f"Suspicious file extension in {context}: {input_data}")
                return False
        elif context == "url":
            # Validate URLs
            if not re.match(r'^https?://[a-zA-Z0-9.-]+(/[a-zA-Z0-9._~:/?#[\]@!$&\'()*+,;=-]*)?$', input_data):
                logger.warning(f"Invalid URL format in {context}: {input_data}")
                return False
        
        return True
    
    @classmethod
    def sanitize_json(cls, data: Any, max_depth: int = 10, max_items: int = 1000) -> Any:
        """Recursively sanitize JSON data with security checks."""
        def _sanitize_recursive(obj, depth=0):
            if depth > max_depth:
                raise SecurityError(f"JSON nesting too deep: {depth}")
            
            if isinstance(obj, dict):
                if len(obj) > max_items:
                    raise SecurityError(f"JSON object too large: {len(obj)} items")
                
                sanitized = {}
                for key, value in obj.items():
                    # Sanitize key
                    if not cls.advanced_validate(str(key), "json_key", 200):
                        continue  # Skip dangerous keys
                    
                    # Sanitize value recursively
                    sanitized[str(key)[:200]] = _sanitize_recursive(value, depth + 1)
                
                return sanitized
            
            elif isinstance(obj, (list, tuple)):
                if len(obj) > max_items:
                    raise SecurityError(f"JSON array too large: {len(obj)} items")
                
                return [_sanitize_recursive(item, depth + 1) for item in obj]
            
            elif isinstance(obj, str):
                # Validate and truncate string values
                if cls.advanced_validate(obj, "json_value", 10000):
                    return obj[:10000]  # Limit string length
                else:
                    return "[SANITIZED]"  # Replace dangerous content
            
            elif isinstance(obj, (int, float, bool)) or obj is None:
                return obj
            
            else:
                # Convert unknown types to string and validate
                str_obj = str(obj)
                if cls.advanced_validate(str_obj, "json_value", 1000):
                    return str_obj[:1000]
                else:
                    return "[SANITIZED]"
        
        return _sanitize_recursive(data)


class AuditLogger:
    """Enhanced audit logging for security events."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption = AdvancedEncryption(encryption_key) if CRYPTOGRAPHY_AVAILABLE and encryption_key else None
        self.audit_events = []
        self.lock = threading.RLock()
    
    def log_security_event(self, event_type: str, user_id: Optional[str] = None, 
                          client_ip: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
                          severity: str = "INFO") -> str:
        """Log security event with optional encryption."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        event = {
            'event_id': event_id,
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'client_ip': client_ip,
            'details': details or {},
            'severity': severity
        }
        
        with self.lock:
            # Encrypt sensitive events if encryption is available
            if self.encryption and severity in ['WARNING', 'ERROR', 'CRITICAL']:
                try:
                    encrypted_event = {
                        'event_id': event_id,
                        'timestamp': timestamp.isoformat(),
                        'encrypted_data': self.encryption.encrypt(json.dumps(event))
                    }
                    self.audit_events.append(encrypted_event)
                except Exception as e:
                    logger.error(f"Failed to encrypt audit event: {e}")
                    self.audit_events.append(event)  # Fallback to unencrypted
            else:
                self.audit_events.append(event)
            
            # Limit audit log size
            if len(self.audit_events) > 10000:
                self.audit_events = self.audit_events[-5000:]  # Keep recent half
        
        # Log to standard logger as well
        logger.log(
            level=getattr(logger, severity.lower(), logger.info),
            msg=f"AUDIT {event_type}: {user_id or 'anonymous'} from {client_ip or 'unknown'}",
            extra=event
        )
        
        return event_id
    
    def get_audit_events(self, event_type: str = None, user_id: str = None,
                        start_time: datetime = None, end_time: datetime = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve audit events with filtering."""
        with self.lock:
            filtered_events = []
            
            for event in reversed(self.audit_events):  # Most recent first
                # Decrypt if needed
                if 'encrypted_data' in event:
                    if not self.encryption:
                        continue  # Skip if can't decrypt
                    try:
                        decrypted_data = self.encryption.decrypt(event['encrypted_data'])
                        event = json.loads(decrypted_data.decode())
                    except Exception:
                        continue  # Skip if decryption fails
                
                # Apply filters
                if event_type and event.get('event_type') != event_type:
                    continue
                
                if user_id and event.get('user_id') != user_id:
                    continue
                
                event_time = datetime.fromisoformat(event['timestamp'])
                if start_time and event_time < start_time:
                    continue
                
                if end_time and event_time > end_time:
                    continue
                
                filtered_events.append(event)
                
                if len(filtered_events) >= limit:
                    break
            
            return filtered_events


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
_request_signer = None
_jwt_manager = None
_audit_logger = None

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

def get_request_signer(secret_key: str = None) -> RequestSigner:
    """Get global request signer instance."""
    global _request_signer
    if _request_signer is None:
        if not secret_key:
            secret_key = secrets.token_urlsafe(32)
        _request_signer = RequestSigner(secret_key)
    return _request_signer

def get_jwt_manager(secret_key: str = None) -> JWTManager:
    """Get global JWT manager instance."""
    global _jwt_manager
    if _jwt_manager is None:
        if not secret_key:
            secret_key = secrets.token_urlsafe(32)
        _jwt_manager = JWTManager(secret_key)
    return _jwt_manager

def get_audit_logger(encryption_key: bytes = None) -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(encryption_key)
    return _audit_logger


# Convenience security decorators
def require_api_key(permissions: List[str] = None):
    """Decorator requiring valid API key with optional permissions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract API key from request context
            api_key = kwargs.get('api_key') or (args[0].headers.get('Authorization', '').replace('Bearer ', '') if args else None)
            
            if not api_key:
                audit_logger = get_audit_logger()
                audit_logger.log_security_event('MISSING_API_KEY', severity='WARNING')
                raise SecurityError("API key required")
            
            key_manager = get_api_key_manager()
            key_info = key_manager.validate_api_key(api_key)
            
            if not key_info:
                audit_logger = get_audit_logger()
                audit_logger.log_security_event('INVALID_API_KEY', details={'key_prefix': api_key[:8]}, severity='WARNING')
                raise SecurityError("Invalid API key")
            
            # Check permissions
            if permissions:
                for perm in permissions:
                    if not key_manager.has_permission(api_key, perm):
                        audit_logger = get_audit_logger()
                        audit_logger.log_security_event(
                            'INSUFFICIENT_PERMISSIONS', 
                            user_id=key_info.get('user_id'),
                            details={'required_permission': perm},
                            severity='WARNING'
                        )
                        raise SecurityError(f"Permission '{perm}' required")
            
            # Add key info to kwargs
            kwargs['api_key_info'] = key_info
            
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def require_signature(secret_key: str = None):
    """Decorator requiring valid request signature."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(request, *args, **kwargs):
            signature = request.headers.get('X-Signature') or request.headers.get('Authorization', '').replace('Signature ', '')
            
            if not signature:
                audit_logger = get_audit_logger()
                audit_logger.log_security_event('MISSING_SIGNATURE', severity='WARNING')
                raise SecurityError("Request signature required")
            
            signer = get_request_signer(secret_key)
            body = await request.body() if hasattr(request, 'body') else b""
            
            if not signer.verify_signature(signature, request.method, str(request.url), body):
                audit_logger = get_audit_logger()
                audit_logger.log_security_event('INVALID_SIGNATURE', severity='WARNING')
                raise SecurityError("Invalid request signature")
            
            return await func(request, *args, **kwargs)
        
        def sync_wrapper(request, *args, **kwargs):
            return asyncio.run(async_wrapper(request, *args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def rate_limited(max_requests: int = 100, window_seconds: int = 3600):
    """Decorator for rate limiting endpoints."""
    limiter = RateLimiter(max_requests, window_seconds)
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(request, *args, **kwargs):
            client_ip = request.client.host if hasattr(request, 'client') else 'unknown'
            
            if not limiter.is_allowed(client_ip):
                audit_logger = get_audit_logger()
                audit_logger.log_security_event(
                    'RATE_LIMIT_EXCEEDED', 
                    client_ip=client_ip,
                    details={'limit': max_requests, 'window': window_seconds},
                    severity='WARNING'
                )
                raise SecurityError("Rate limit exceeded")
            
            return await func(request, *args, **kwargs)
        
        def sync_wrapper(request, *args, **kwargs):
            return asyncio.run(async_wrapper(request, *args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def setup_security_defaults(secret_key: str, encryption_key: bytes = None):
    """Setup default security components with provided keys."""
    # Initialize global components
    get_request_signer(secret_key)
    get_jwt_manager(secret_key)
    get_audit_logger(encryption_key)
    
    # Generate admin API key
    api_manager = get_api_key_manager()
    admin_key = api_manager.generate_api_key(
        'admin',
        permissions=['generate', 'assess', 'admin', 'batch', 'health']
    )
    
    logger.info("Security defaults initialized")
    return {
        'admin_api_key': admin_key,
        'request_signer_initialized': True,
        'jwt_manager_initialized': True,
        'audit_logger_initialized': True
    }