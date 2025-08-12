"""
Comprehensive robustness testing suite for OdorDiff-2.
Tests error handling, validation, security, and edge cases.
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add project root to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from odordiff2.utils.error_handling import (
    OdorDiffException, ValidationError, ModelError, ComputationError,
    handle_errors, retry_with_backoff, ErrorTracker, CircuitBreaker
)
from odordiff2.utils.validation import (
    InputSanitizer, MoleculeValidator, TextPromptValidator, 
    APIRequestValidator, ValidationResult
)
from odordiff2.monitoring.health_checks import (
    HealthMonitor, SystemResourceCheck, ModelHealthCheck,
    HealthStatus, HealthCheckResult
)
from odordiff2.security.rate_limiting import (
    TokenBucket, SlidingWindowCounter, InMemoryRateLimiter,
    RateLimit, RateLimitStrategy, RateLimitResult
)
from odordiff2.security.authentication import (
    APIKeyValidator, JWTManager, AuthenticationManager,
    UserRole, Permission, AuthenticationResult
)


class TestErrorHandling:
    """Test comprehensive error handling system."""
    
    def test_error_tracker(self):
        """Test error tracking functionality."""
        tracker = ErrorTracker()
        
        # Create test error
        error = ValidationError("Test validation error", field="test_field")
        context = {"component": "test", "operation": "test_op"}
        
        # Record error
        tracker.record_error(error, context)
        
        # Check error was recorded
        assert len(tracker.errors) == 1
        assert tracker.error_counts["VALIDATION_ERROR"] == 1
        
        # Get statistics
        stats = tracker.get_error_statistics(time_window=3600)
        assert stats["total_errors"] == 1
        assert stats["by_category"]["validation"] == 1
    
    def test_handle_errors_decorator(self):
        """Test error handling decorator."""
        @handle_errors(ValueError, default_return="handled")
        def test_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2
        
        # Test successful case
        assert test_function(5) == 10
        
        # Test error handling
        assert test_function(-1) == "handled"
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern."""
        failure_count = 0
        
        def failing_function():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception("Service failure")
            return "success"
        
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        
        # Test failures until circuit opens
        for i in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_function)
        
        # Circuit should now be open
        assert breaker.state == 'OPEN'
        
        # Test that circuit blocks calls
        with pytest.raises(OdorDiffException):
            breaker.call(failing_function)
    
    def test_retry_mechanism(self):
        """Test retry with backoff."""
        attempt_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3


class TestInputValidation:
    """Test comprehensive input validation."""
    
    def test_text_sanitization(self):
        """Test text sanitization."""
        # Test basic sanitization
        clean = InputSanitizer.sanitize_text("Hello World", max_length=50)
        assert clean == "Hello World"
        
        # Test HTML escaping
        dirty = "<script>alert('xss')</script>Hello"
        clean = InputSanitizer.sanitize_text(dirty, allow_html=False)
        assert "<script>" not in clean
        
        # Test length limiting
        long_text = "A" * 2000
        clean = InputSanitizer.sanitize_text(long_text, max_length=100)
        assert len(clean) == 100
    
    def test_smiles_validation(self):
        """Test SMILES validation."""
        validator = MoleculeValidator()
        
        # Valid SMILES
        result = validator.validate_smiles("CCO")  # Ethanol
        assert result.is_valid
        assert result.sanitized_value == "CCO"
        
        # Invalid SMILES
        result = validator.validate_smiles("InvalidSMILES")
        assert not result.is_valid
        assert "cannot be parsed" in result.errors[0]
        
        # Too large molecule
        large_smiles = "C" * 200
        result = validator.validate_smiles(large_smiles)
        assert not result.is_valid
    
    def test_prompt_validation(self):
        """Test text prompt validation."""
        validator = TextPromptValidator()
        
        # Valid prompt
        result = validator.validate_prompt("Fresh floral scent with rose")
        assert result.is_valid
        assert "rose" in result.metadata["encouraged_terms"]
        
        # Forbidden terms
        result = validator.validate_prompt("toxic poison dangerous")
        assert not result.is_valid
        assert "forbidden terms" in result.errors[0]
        
        # Too short
        result = validator.validate_prompt("Hi")
        assert not result.is_valid
    
    def test_api_request_validation(self):
        """Test API request validation."""
        validator = APIRequestValidator()
        
        # Valid generation request
        request_data = {
            "prompt": "Fresh citrus scent",
            "num_molecules": 5,
            "temperature": 1.0
        }
        
        result = validator.validate_generation_request(request_data)
        assert result.is_valid
        assert result.sanitized_value["prompt"] == "Fresh citrus scent"
        
        # Invalid request
        invalid_data = {
            "prompt": "",  # Empty prompt
            "num_molecules": -1,  # Invalid number
            "temperature": 100  # Too high
        }
        
        result = validator.validate_generation_request(invalid_data)
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_numeric_sanitization(self):
        """Test numeric input sanitization."""
        # Valid numbers
        assert InputSanitizer.sanitize_numeric("5.5") == 5.5
        assert InputSanitizer.sanitize_numeric(10) == 10.0
        
        # Range validation
        with pytest.raises(ValidationError):
            InputSanitizer.sanitize_numeric(150, min_value=0, max_value=100)
        
        # Negative validation
        with pytest.raises(ValidationError):
            InputSanitizer.sanitize_numeric(-5, allow_negative=False)
    
    def test_list_sanitization(self):
        """Test list input sanitization."""
        # Valid list
        result = InputSanitizer.sanitize_list([1, 2, 3])
        assert result == [1, 2, 3]
        
        # String to list conversion
        result = InputSanitizer.sanitize_list("a,b,c")
        assert result == ["a", "b", "c"]
        
        # Length limiting
        long_list = list(range(2000))
        result = InputSanitizer.sanitize_list(long_list, max_length=100)
        assert len(result) == 100


class TestHealthChecks:
    """Test health check system."""
    
    @pytest.mark.asyncio
    async def test_system_resource_check(self):
        """Test system resource health check."""
        check = SystemResourceCheck()
        result = await check.check()
        
        assert isinstance(result, HealthCheckResult)
        assert result.name == "system_resources"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
        assert "cpu_percent" in result.details
        assert "memory_percent" in result.details
    
    @pytest.mark.asyncio
    async def test_model_health_check(self):
        """Test model health check."""
        check = ModelHealthCheck()
        result = await check.check()
        
        assert isinstance(result, HealthCheckResult)
        assert result.name == "model_health"
        assert "models_available" in result.details
    
    @pytest.mark.asyncio
    async def test_health_monitor(self):
        """Test health monitoring system."""
        monitor = HealthMonitor()
        
        # Clear default checks for testing
        monitor.checks = []
        
        # Add test check
        test_check = SystemResourceCheck()
        monitor.add_check(test_check)
        
        # Run checks
        results = await monitor.run_all_checks()
        assert len(results) == 1
        assert "system_resources" in results
        
        # Get overall status
        status = monitor.get_overall_status()
        assert "status" in status
        assert "message" in status
    
    @pytest.mark.asyncio
    async def test_failed_health_check(self):
        """Test health check failure handling."""
        class FailingCheck:
            def __init__(self):
                self.name = "failing_check"
            
            async def check(self):
                raise Exception("Simulated failure")
        
        monitor = HealthMonitor()
        monitor.checks = [FailingCheck()]
        
        results = await monitor.run_all_checks()
        assert "failing_check" in results
        assert results["failing_check"].status == HealthStatus.CRITICAL


class TestRateLimiting:
    """Test rate limiting system."""
    
    def test_token_bucket(self):
        """Test token bucket rate limiter."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)  # 5 tokens, 1 per second
        
        # Should allow initial requests
        for i in range(5):
            assert bucket.consume() == True
        
        # Should block when empty
        assert bucket.consume() == False
        
        # Test peek function
        assert bucket.peek() < 1.0
        
        # Test time until tokens
        wait_time = bucket.time_until_tokens(1)
        assert 0 < wait_time <= 1.0
    
    def test_sliding_window(self):
        """Test sliding window rate limiter."""
        limiter = SlidingWindowCounter(limit=3, window_seconds=10)
        
        # Should allow initial requests
        for i in range(3):
            allowed, remaining = limiter.allow_request()
            assert allowed == True
            assert remaining >= 0
        
        # Should block when limit reached
        allowed, remaining = limiter.allow_request()
        assert allowed == False
        assert remaining == 0
    
    @pytest.mark.asyncio
    async def test_in_memory_rate_limiter(self):
        """Test in-memory rate limiter."""
        rate_limit = RateLimit(requests=3, window_seconds=10)
        limiter = InMemoryRateLimiter(rate_limit)
        
        # Test normal usage
        for i in range(3):
            result = await limiter.check_rate_limit("test_key")
            assert result.allowed == True
        
        # Should be rate limited
        result = await limiter.check_rate_limit("test_key")
        assert result.allowed == False
        assert result.retry_after is not None
    
    @pytest.mark.asyncio
    async def test_custom_rate_limits(self):
        """Test custom rate limits for different keys."""
        default_limit = RateLimit(requests=1, window_seconds=10)
        limiter = InMemoryRateLimiter(default_limit)
        
        # Set custom limit for premium user
        premium_limit = RateLimit(requests=10, window_seconds=10)
        limiter.set_custom_limit("premium_user", premium_limit)
        
        # Test default limit
        result = await limiter.check_rate_limit("regular_user")
        assert result.allowed == True
        result = await limiter.check_rate_limit("regular_user")
        assert result.allowed == False
        
        # Test custom limit
        for i in range(5):
            result = await limiter.check_rate_limit("premium_user")
            assert result.allowed == True


class TestAuthentication:
    """Test authentication system."""
    
    def test_api_key_generation(self):
        """Test API key generation."""
        validator = APIKeyValidator()
        
        raw_key, api_key = validator.generate_api_key(
            user_id="test_user",
            name="Test Key",
            role=UserRole.USER
        )
        
        # Check key format
        assert raw_key.startswith("odr_")
        assert len(raw_key.split("_")) == 3
        
        # Check API key object
        assert api_key.user_id == "test_user"
        assert api_key.name == "Test Key"
        assert api_key.role == UserRole.USER
        assert Permission.GENERATE_MOLECULES in api_key.permissions
    
    @pytest.mark.asyncio
    async def test_api_key_validation(self):
        """Test API key validation."""
        validator = APIKeyValidator()
        
        # Generate test key
        raw_key, api_key = validator.generate_api_key(
            user_id="test_user",
            name="Test Key",
            role=UserRole.PREMIUM
        )
        
        # Test valid key
        result = await validator.validate_api_key(raw_key)
        assert result.success == True
        assert result.user_id == "test_user"
        assert result.role == UserRole.PREMIUM
        
        # Test invalid key
        result = await validator.validate_api_key("invalid_key")
        assert result.success == False
        
        # Test revoked key
        validator.revoke_api_key(api_key.key_id)
        result = await validator.validate_api_key(raw_key)
        assert result.success == False
        assert "revoked" in result.error_message
    
    def test_jwt_tokens(self):
        """Test JWT token system."""
        jwt_manager = JWTManager("test_secret")
        
        # Create token
        token = jwt_manager.create_token("test_user", UserRole.RESEARCHER)
        assert isinstance(token, str)
        
        # Verify token
        result = jwt_manager.verify_token(token)
        assert result.success == True
        assert result.user_id == "test_user"
        assert result.role == UserRole.RESEARCHER
        assert Permission.RESEARCH_ACCESS in result.permissions
        
        # Test token revocation
        jwt_manager.revoke_token(token)
        result = jwt_manager.verify_token(token)
        assert result.success == False
    
    @pytest.mark.asyncio
    async def test_authentication_manager(self):
        """Test authentication manager."""
        auth_manager = AuthenticationManager()
        
        # Create test user
        raw_key, _ = auth_manager.create_api_key(
            user_id="test_user",
            name="Test Key", 
            role=UserRole.ADMIN
        )
        
        # Test API key authentication
        auth_header = f"ApiKey {raw_key}"
        result = await auth_manager.authenticate_request(auth_header)
        assert result.success == True
        assert result.role == UserRole.ADMIN
        
        # Test JWT authentication  
        jwt_token = auth_manager.jwt_manager.create_token("jwt_user", UserRole.USER)
        auth_header = f"Bearer {jwt_token}"
        result = await auth_manager.authenticate_request(auth_header)
        assert result.success == True
        assert result.role == UserRole.USER
    
    def test_permission_checking(self):
        """Test permission-based access control."""
        auth_manager = AuthenticationManager()
        
        # Create user with limited permissions
        user_result = AuthenticationResult(
            success=True,
            user_id="limited_user",
            role=UserRole.USER,
            permissions=[Permission.READ_PUBLIC, Permission.GENERATE_MOLECULES]
        )
        
        # Test permission checking
        assert auth_manager.has_permission(user_result, Permission.GENERATE_MOLECULES)
        assert not auth_manager.has_permission(user_result, Permission.ADMIN_SYSTEM)


class TestSecurityMeasures:
    """Test security-related functionality."""
    
    def test_input_sanitization_xss(self):
        """Test XSS prevention in input sanitization."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onload='alert(1)'",
            "<iframe src='http://evil.com'></iframe>",
            "\\x3cscript\\x3e"
        ]
        
        for malicious in malicious_inputs:
            clean = InputSanitizer.sanitize_text(malicious, allow_html=False)
            assert "<script>" not in clean
            assert "javascript:" not in clean
            assert "onload=" not in clean
            assert "<iframe" not in clean
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            '" UNION SELECT * FROM passwords --',
            "' OR 1=1 --",
            "; EXEC xp_cmdshell('dir'); --"
        ]
        
        for malicious in malicious_inputs:
            clean = InputSanitizer.sanitize_sql_input(malicious)
            assert "DROP TABLE" not in clean.upper()
            assert "UNION SELECT" not in clean.upper()
            assert "--" not in clean
    
    def test_file_path_validation(self):
        """Test file path security validation."""
        validator = APIRequestValidator()
        
        # Safe paths
        safe_paths = ["file.txt", "data/file.csv", "models/checkpoint.pt"]
        for path in safe_paths:
            try:
                result = validator.validate_file_path(path, ["txt", "csv", "pt"])
                assert result == path
            except ValidationError:
                pytest.fail(f"Safe path rejected: {path}")
        
        # Dangerous paths
        dangerous_paths = ["../../../etc/passwd", "/etc/shadow", "..\\windows\\system32"]
        for path in dangerous_paths:
            with pytest.raises(ValidationError):
                validator.validate_file_path(path)
    
    def test_api_key_security(self):
        """Test API key security features."""
        validator = APIKeyValidator()
        
        # Test key format security
        assert validator.validate_api_key("").success == False
        assert validator.validate_api_key("weak_key").success == False
        assert validator.validate_api_key("odr_invalid").success == False
        
        # Test that keys are hashed, not stored in plain text
        raw_key, api_key = validator.generate_api_key(
            user_id="test", name="test", role=UserRole.USER
        )
        assert raw_key not in str(api_key.key_hash)
        assert len(api_key.key_hash) > 32  # Should be a hash, not the original key


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        validator = MoleculeValidator()
        
        # Empty SMILES
        result = validator.validate_smiles("")
        assert not result.is_valid
        
        # Empty prompt
        prompt_validator = TextPromptValidator()
        result = prompt_validator.validate_prompt("")
        assert not result.is_valid
    
    def test_extremely_large_inputs(self):
        """Test handling of very large inputs."""
        # Large text input
        huge_text = "A" * 100000
        clean = InputSanitizer.sanitize_text(huge_text, max_length=1000)
        assert len(clean) == 1000
        
        # Large SMILES
        large_smiles = "C" * 1000
        validator = MoleculeValidator()
        result = validator.validate_smiles(large_smiles)
        assert not result.is_valid
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self):
        """Test rate limiting under concurrent load."""
        rate_limit = RateLimit(requests=10, window_seconds=10)
        limiter = InMemoryRateLimiter(rate_limit)
        
        async def make_request(key):
            return await limiter.check_rate_limit(key)
        
        # Make concurrent requests
        tasks = [make_request(f"user_{i}") for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # All should be allowed (different keys)
        assert all(r.allowed for r in results)
        
        # Test same key concurrency
        same_key_tasks = [make_request("same_user") for _ in range(15)]
        same_key_results = await asyncio.gather(*same_key_tasks)
        
        # Some should be blocked
        allowed_count = sum(1 for r in same_key_results if r.allowed)
        assert allowed_count <= 10  # Rate limit
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        validator = APIRequestValidator()
        
        # Malformed JSON-like data
        malformed_data = {
            "prompt": ["this", "should", "be", "string"],  # Wrong type
            "num_molecules": "five",  # Wrong type
            "temperature": {"value": 1.0},  # Wrong type
            "invalid_field": "should_be_ignored"
        }
        
        # Should handle gracefully
        try:
            result = validator.validate_generation_request(malformed_data)
            # Should have errors but not crash
            assert not result.is_valid
        except Exception as e:
            pytest.fail(f"Should handle malformed data gracefully: {e}")
    
    def test_unicode_handling(self):
        """Test handling of various Unicode characters."""
        unicode_inputs = [
            "Fresh floral scent 🌸",  # Emoji
            "Citrus bergamot café",  # Accented characters
            "Восточные пряности",  # Cyrillic
            "香りのテスト",  # Japanese
            "🧪⚗️🔬",  # Science emojis only
        ]
        
        for unicode_input in unicode_inputs:
            try:
                clean = InputSanitizer.sanitize_text(unicode_input, normalize_unicode=True)
                # Should not crash and should produce some output
                assert isinstance(clean, str)
                assert len(clean) >= 0
            except Exception as e:
                pytest.fail(f"Unicode handling failed for '{unicode_input}': {e}")


class TestPerformanceUnderLoad:
    """Test system performance under various load conditions."""
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance."""
        monitor = HealthMonitor()
        
        start_time = time.time()
        results = await monitor.run_all_checks()
        duration = time.time() - start_time
        
        # Health checks should complete reasonably quickly
        assert duration < 10.0  # 10 seconds max
        assert len(results) > 0
    
    def test_rate_limiter_performance(self):
        """Test rate limiter performance."""
        rate_limit = RateLimit(requests=1000, window_seconds=3600)
        limiter = InMemoryRateLimiter(rate_limit)
        
        # Time 1000 rate limit checks
        start_time = time.time()
        for i in range(1000):
            asyncio.run(limiter.check_rate_limit(f"user_{i % 100}"))
        duration = time.time() - start_time
        
        # Should handle 1000 checks in reasonable time
        assert duration < 5.0  # 5 seconds max
        assert duration / 1000 < 0.01  # Less than 10ms per check average
    
    def test_validation_performance(self):
        """Test input validation performance."""
        validator = APIRequestValidator()
        
        # Generate test data
        test_requests = [
            {
                "prompt": f"Fresh floral scent number {i}",
                "num_molecules": 5,
                "temperature": 1.0
            }
            for i in range(100)
        ]
        
        # Time validation
        start_time = time.time()
        for request in test_requests:
            validator.validate_generation_request(request)
        duration = time.time() - start_time
        
        # Should validate 100 requests quickly
        assert duration < 2.0  # 2 seconds max
        assert duration / 100 < 0.02  # Less than 20ms per validation


# Integration test fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        async def eval(self, script, num_keys, *args):
            # Simple mock implementation
            return [1, 10, 0]  # allowed, remaining, retry_after
        
        async def incr(self, key):
            self.data[key] = self.data.get(key, 0) + 1
            return self.data[key]
        
        async def expire(self, key, seconds):
            pass
        
        async def keys(self, pattern):
            return [k for k in self.data.keys() if pattern.replace("*", "") in k]
        
        async def delete(self, *keys):
            for key in keys:
                self.data.pop(key, None)
    
    return MockRedis()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])