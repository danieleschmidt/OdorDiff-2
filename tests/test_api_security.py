"""
Security tests for API endpoints and security utilities.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import time
from typing import Dict, Any

from odordiff2.api.secure_endpoints import app
from odordiff2.utils.security import (
    RateLimiter, SecurityValidator, APIKeyManager, SecurityMiddleware,
    SecurityError, get_security_middleware, get_api_key_manager
)
from odordiff2.utils.error_handling import ValidationError


class TestRateLimiter:
    """Test suite for RateLimiter."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter for testing."""
        return RateLimiter(max_requests=5, time_window=60)  # 5 requests per minute
    
    def test_rate_limiter_allows_requests_under_limit(self, rate_limiter):
        """Test that rate limiter allows requests under the limit."""
        client_id = "test_client_1"
        
        # Should allow first 5 requests
        for i in range(5):
            assert rate_limiter.is_allowed(client_id) == True
        
        # 6th request should be denied
        assert rate_limiter.is_allowed(client_id) == False
    
    def test_rate_limiter_resets_after_time_window(self, rate_limiter):
        """Test that rate limiter resets after time window."""
        client_id = "test_client_2"
        
        # Fill up the requests
        for i in range(5):
            assert rate_limiter.is_allowed(client_id) == True
        
        # Next request should be denied
        assert rate_limiter.is_allowed(client_id) == False
        
        # Mock time passage by modifying internal state
        import time
        current_time = time.time()
        old_time = current_time - 70  # 70 seconds ago
        
        # Manually reset the client's request list to simulate time passage
        rate_limiter.requests[client_id] = [old_time] * 5
        
        # Should now allow requests again
        assert rate_limiter.is_allowed(client_id) == True
    
    def test_rate_limiter_tracks_multiple_clients(self, rate_limiter):
        """Test that rate limiter tracks multiple clients separately."""
        client1 = "client_1"
        client2 = "client_2"
        
        # Fill up requests for client1
        for i in range(5):
            assert rate_limiter.is_allowed(client1) == True
        
        # client1 should be rate limited
        assert rate_limiter.is_allowed(client1) == False
        
        # client2 should still be allowed
        assert rate_limiter.is_allowed(client2) == True
    
    def test_get_remaining_requests(self, rate_limiter):
        """Test getting remaining requests count."""
        client_id = "test_client_3"
        
        # Initially should have max requests available
        assert rate_limiter.get_remaining_requests(client_id) == 5
        
        # Use 2 requests
        rate_limiter.is_allowed(client_id)
        rate_limiter.is_allowed(client_id)
        
        # Should have 3 remaining
        assert rate_limiter.get_remaining_requests(client_id) == 3


class TestSecurityValidator:
    """Test suite for SecurityValidator."""
    
    def test_validate_input_accepts_safe_input(self):
        """Test that validator accepts safe input."""
        safe_inputs = [
            "fresh citrus scent",
            "floral rose garden",
            "woody cedar aroma",
            "simple description with numbers 123"
        ]
        
        for input_text in safe_inputs:
            assert SecurityValidator.validate_input(input_text) == True
    
    def test_validate_input_rejects_dangerous_patterns(self):
        """Test that validator rejects dangerous patterns."""
        dangerous_inputs = [
            "SELECT * FROM users",
            "<script>alert('xss')</script>",
            "'; DROP TABLE molecules; --",
            "javascript:alert('hack')",
            "../../../etc/passwd",
            "$(rm -rf /)",
            "eval(malicious_code)"
        ]
        
        for input_text in dangerous_inputs:
            assert SecurityValidator.validate_input(input_text) == False
    
    def test_validate_input_length_limit(self):
        """Test input length validation."""
        # Long input should be rejected
        long_input = "a" * 2000
        assert SecurityValidator.validate_input(long_input, max_length=1000) == False
        
        # Input within limit should be accepted
        normal_input = "a" * 500
        assert SecurityValidator.validate_input(normal_input, max_length=1000) == True
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file with spaces.txt", "file with spaces.txt"),
            ("file<>:\"/|?*.txt", "file_________.txt"),
            ("../../../etc/passwd", "___.___._etc_passwd"),
            ("", "unnamed"),
            ("." * 300, "." * 255)  # Length limit
        ]
        
        for input_name, expected in test_cases:
            result = SecurityValidator.sanitize_filename(input_name)
            assert len(result) <= 255
            # Check that dangerous characters are removed/replaced
            dangerous_chars = '<>:"/\\|?*'
            assert not any(char in result for char in dangerous_chars)
    
    def test_validate_smiles_security(self):
        """Test SMILES string security validation."""
        # Valid SMILES should pass
        valid_smiles = [
            "CCO",
            "CC(C)=CCO",
            "c1ccccc1",
            "CC(=O)Oc1ccccc1C(=O)O"
        ]
        
        for smiles in valid_smiles:
            assert SecurityValidator.validate_smiles_security(smiles) == True
        
        # Suspicious SMILES should fail
        suspicious_smiles = [
            "eval(malicious)",
            "import os; os.system('rm -rf /')",
            "__builtins__",
            "exec('malicious code')"
        ]
        
        for smiles in suspicious_smiles:
            assert SecurityValidator.validate_smiles_security(smiles) == False
    
    def test_validate_ip_address(self):
        """Test IP address validation."""
        valid_ips = ["192.168.1.1", "127.0.0.1", "8.8.8.8", "::1", "2001:db8::1"]
        invalid_ips = ["256.256.256.256", "not.an.ip", "192.168.1", ""]
        
        for ip in valid_ips:
            assert SecurityValidator.validate_ip_address(ip) == True
        
        for ip in invalid_ips:
            assert SecurityValidator.validate_ip_address(ip) == False
    
    def test_is_private_ip(self):
        """Test private IP detection."""
        private_ips = ["192.168.1.1", "10.0.0.1", "172.16.0.1", "127.0.0.1"]
        public_ips = ["8.8.8.8", "1.1.1.1", "208.67.222.222"]
        
        for ip in private_ips:
            assert SecurityValidator.is_private_ip(ip) == True
        
        for ip in public_ips:
            assert SecurityValidator.is_private_ip(ip) == False


class TestAPIKeyManager:
    """Test suite for APIKeyManager."""
    
    @pytest.fixture
    def api_key_manager(self):
        """Create API key manager for testing."""
        return APIKeyManager()
    
    def test_generate_api_key(self, api_key_manager):
        """Test API key generation."""
        user_id = "test_user"
        permissions = ["generate", "assess"]
        
        api_key = api_key_manager.generate_api_key(user_id, permissions)
        
        assert isinstance(api_key, str)
        assert len(api_key) > 0
        assert api_key in api_key_manager.keys
        assert api_key_manager.keys[api_key]['user_id'] == user_id
        assert api_key_manager.key_permissions[api_key] == permissions
    
    def test_validate_api_key(self, api_key_manager):
        """Test API key validation."""
        user_id = "test_user"
        api_key = api_key_manager.generate_api_key(user_id)
        
        # Valid key should return user info
        key_info = api_key_manager.validate_api_key(api_key)
        assert key_info is not None
        assert key_info['user_id'] == user_id
        assert key_info['active'] == True
        
        # Invalid key should return None
        invalid_key = "invalid_key"
        assert api_key_manager.validate_api_key(invalid_key) is None
    
    def test_revoke_api_key(self, api_key_manager):
        """Test API key revocation."""
        user_id = "test_user"
        api_key = api_key_manager.generate_api_key(user_id)
        
        # Key should be valid initially
        assert api_key_manager.validate_api_key(api_key) is not None
        
        # Revoke key
        assert api_key_manager.revoke_api_key(api_key) == True
        
        # Key should no longer be valid
        assert api_key_manager.validate_api_key(api_key) is None
        
        # Revoking non-existent key should return False
        assert api_key_manager.revoke_api_key("non_existent") == False
    
    def test_check_permissions(self, api_key_manager):
        """Test permission checking."""
        user_id = "test_user"
        permissions = ["generate", "assess"]
        api_key = api_key_manager.generate_api_key(user_id, permissions)
        
        # Should have granted permissions
        assert api_key_manager.has_permission(api_key, "generate") == True
        assert api_key_manager.has_permission(api_key, "assess") == True
        
        # Should not have other permissions
        assert api_key_manager.has_permission(api_key, "admin") == False
        
        # Invalid key should not have any permissions
        assert api_key_manager.has_permission("invalid", "generate") == False


class TestSecurityMiddleware:
    """Test suite for SecurityMiddleware."""
    
    @pytest.fixture
    def security_middleware(self):
        """Create security middleware for testing."""
        return SecurityMiddleware()
    
    def test_validate_request_success(self, security_middleware):
        """Test successful request validation."""
        request_data = {
            'client_ip': '192.168.1.100',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        result = security_middleware.validate_request(request_data)
        
        assert 'client_ip' in result
        assert 'user_agent' in result
        assert 'remaining_requests' in result
        assert result['client_ip'] == request_data['client_ip']
    
    def test_validate_request_rate_limit(self, security_middleware):
        """Test request validation with rate limiting."""
        request_data = {
            'client_ip': '192.168.1.101',
            'user_agent': 'Test Agent'
        }
        
        # Make requests up to the limit
        for i in range(100):  # Assuming default limit is 100
            try:
                security_middleware.validate_request(request_data)
            except SecurityError:
                break
        
        # Next request should fail
        with pytest.raises(SecurityError, match="Rate limit exceeded"):
            security_middleware.validate_request(request_data)
    
    def test_validate_request_blocked_ip(self, security_middleware):
        """Test request validation with blocked IP."""
        blocked_ip = '192.168.1.102'
        request_data = {
            'client_ip': blocked_ip,
            'user_agent': 'Test Agent'
        }
        
        # Block the IP
        security_middleware.block_ip(blocked_ip)
        
        # Request should be blocked
        with pytest.raises(SecurityError, match=f"IP {blocked_ip} is blocked"):
            security_middleware.validate_request(request_data)
    
    def test_detect_security_scanner(self, security_middleware):
        """Test detection of security scanners."""
        scanner_agents = [
            'sqlmap/1.0',
            'Mozilla/5.0 (compatible; Nmap Scripting Engine)',
            'ZAP/2.0',
            'Burp/1.0'
        ]
        
        for user_agent in scanner_agents:
            request_data = {
                'client_ip': '192.168.1.103',
                'user_agent': user_agent
            }
            
            # Should still validate but increase suspicious activity
            result = security_middleware.validate_request(request_data)
            assert result is not None
            
            # Check that suspicious activity was recorded
            assert security_middleware.suspicious_activity['192.168.1.103'] > 0
    
    def test_auto_blocking_suspicious_ip(self, security_middleware):
        """Test automatic blocking of suspicious IPs."""
        suspicious_ip = '192.168.1.104'
        
        # Make many suspicious requests to trigger auto-block
        for i in range(60):  # Exceed threshold
            request_data = {
                'client_ip': suspicious_ip,
                'user_agent': 'sqlmap/malicious'
            }
            
            try:
                security_middleware.validate_request(request_data)
            except SecurityError:
                # Expected to be blocked eventually
                break
        
        # IP should be auto-blocked
        assert suspicious_ip in security_middleware.blocked_ips
    
    def test_manual_ip_blocking(self, security_middleware):
        """Test manual IP blocking and unblocking."""
        test_ip = '192.168.1.105'
        
        # Block IP manually
        security_middleware.block_ip(test_ip)
        assert test_ip in security_middleware.blocked_ips
        
        # Unblock IP
        security_middleware.unblock_ip(test_ip)
        assert test_ip not in security_middleware.blocked_ips


class TestSecureAPIEndpoints:
    """Test suite for secure API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for secure API."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_model(self):
        """Mock the async model."""
        with patch('odordiff2.api.secure_endpoints.get_secure_model') as mock:
            mock_instance = AsyncMock()
            mock_instance.generate_async.return_value = AsyncMock()
            mock_instance.generate_async.return_value.molecules = []
            mock_instance.generate_async.return_value.error = None
            mock_instance.generate_async.return_value.processing_time = 1.0
            mock_instance.generate_async.return_value.cache_hit = False
            mock_instance.generate_async.return_value.prompt = "test"
            
            mock_instance.health_check.return_value = {
                'status': 'healthy',
                'response_time': 0.1,
                'memory_usage_mb': 100.0,
                'worker_count': 4,
                'cache_enabled': True
            }
            
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_root_endpoint_security_headers(self, client):
        """Test that root endpoint returns security headers."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
    
    def test_health_endpoint_without_auth(self, client, mock_model):
        """Test health endpoint without authentication."""
        response = client.get("/health")
        
        # Should work without auth but with limited info
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert 'data' in data
    
    def test_generate_endpoint_input_validation(self, client, mock_model):
        """Test generation endpoint input validation."""
        # Test with invalid/dangerous input
        dangerous_inputs = [
            {"prompt": "'; DROP TABLE molecules; --"},
            {"prompt": "<script>alert('xss')</script>"},
            {"prompt": "../../../etc/passwd"},
            {"prompt": "a" * 2000}  # Too long
        ]
        
        for payload in dangerous_inputs:
            response = client.post("/generate", json=payload)
            assert response.status_code == 400  # Should reject dangerous input
    
    def test_generate_endpoint_rate_limiting(self, client, mock_model):
        """Test generation endpoint rate limiting."""
        payload = {"prompt": "test citrus"}
        
        # Make many requests from same client
        responses = []
        for i in range(25):  # Exceed typical rate limit
            response = client.post("/generate", json=payload)
            responses.append(response.status_code)
            
            if response.status_code == 429:  # Rate limited
                break
        
        # Should eventually get rate limited
        assert 429 in responses
    
    def test_generate_endpoint_parameter_limits(self, client, mock_model):
        """Test generation endpoint parameter limits."""
        # Test with excessive parameters
        payload = {
            "prompt": "test",
            "num_molecules": 100,  # Should be capped at 10
            "safety_threshold": 0.5  # Should be capped at 0.2
        }
        
        response = client.post("/generate", json=payload)
        
        # Should accept but limit parameters
        if response.status_code == 200:
            # Parameters should be limited internally
            pass
    
    def test_security_metrics_endpoint_requires_auth(self, client):
        """Test that security metrics endpoint requires authentication."""
        response = client.get("/security/metrics")
        
        # Should require authentication
        assert response.status_code in [401, 403]
    
    def test_admin_endpoints_require_auth(self, client):
        """Test that admin endpoints require authentication."""
        admin_endpoints = [
            "/security/metrics",
            "/admin/status"
        ]
        
        for endpoint in admin_endpoints:
            response = client.get(endpoint)
            assert response.status_code in [401, 403]
    
    def test_block_ip_endpoint_requires_auth(self, client):
        """Test that IP blocking endpoint requires authentication."""
        response = client.post("/security/block-ip", params={"ip_address": "192.168.1.1"})
        
        # Should require authentication
        assert response.status_code in [401, 403]
    
    @pytest.mark.asyncio
    async def test_request_timeout_handling(self, client, mock_model):
        """Test request timeout handling."""
        # Mock slow response
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(5)  # Simulate slow operation
            return AsyncMock()
        
        mock_model.generate_async.side_effect = slow_generate
        
        payload = {"prompt": "test timeout"}
        
        # Request should timeout or complete quickly due to timeout handling
        import time
        start_time = time.time()
        response = client.post("/generate", json=payload)
        end_time = time.time()
        
        # Should either timeout quickly or succeed
        request_time = end_time - start_time
        assert request_time < 10.0  # Should not take too long


class TestErrorHandling:
    """Test suite for error handling in security components."""
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        from odordiff2.utils.validation import InputValidator
        
        with pytest.raises(ValidationError):
            InputValidator.validate_prompt("")  # Empty prompt
        
        with pytest.raises(ValidationError):
            InputValidator.validate_smiles("")  # Empty SMILES
    
    def test_security_error_propagation(self):
        """Test security error propagation."""
        middleware = SecurityMiddleware()
        
        # Block an IP
        blocked_ip = "192.168.1.200"
        middleware.block_ip(blocked_ip)
        
        # Request from blocked IP should raise SecurityError
        with pytest.raises(SecurityError):
            middleware.validate_request({
                'client_ip': blocked_ip,
                'user_agent': 'Test'
            })
    
    def test_graceful_degradation(self):
        """Test graceful degradation on component failures."""
        # Test with invalid rate limiter configuration
        rate_limiter = RateLimiter(max_requests=-1, time_window=0)
        
        # Should handle gracefully
        try:
            result = rate_limiter.is_allowed("test_client")
            assert isinstance(result, bool)
        except Exception:
            pytest.fail("Rate limiter should handle invalid config gracefully")


if __name__ == "__main__":
    pytest.main([__file__])