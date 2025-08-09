#!/usr/bin/env python3
"""
Comprehensive test script for Generation 2 robustness enhancements.

This script tests all 10 robustness enhancement areas:
1. Health checks and monitoring
2. Circuit breaker patterns
3. Rate limiting systems
4. Input validation and sanitization
5. Error recovery and graceful degradation
6. Configuration management
7. Connection pooling optimization
8. Data backup and recovery
9. Security hardening
10. Enhanced observability

Usage:
    python test_robustness_enhancements.py
"""

import asyncio
import json
import time
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, List
import sys

# Test framework components
class TestResult:
    def __init__(self, name: str, passed: bool = False, message: str = "", details: Dict[str, Any] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()

class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
        print(f"\n=== Starting {self.name} ===")
    
    def finish(self):
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"=== {self.name} Complete: {passed}/{total} passed in {duration:.2f}s ===\n")
    
    def add_result(self, result: TestResult):
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}: {result.message}")
    
    def test(self, name: str, test_func, *args, **kwargs):
        """Run a test function and record the result."""
        try:
            result = test_func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            
            if isinstance(result, TestResult):
                self.add_result(result)
            elif isinstance(result, bool):
                self.add_result(TestResult(name, result, "Test completed"))
            else:
                self.add_result(TestResult(name, True, f"Returned: {result}"))
                
        except Exception as e:
            self.add_result(TestResult(name, False, f"Exception: {str(e)}"))


class RobustnessTestSuite:
    """Comprehensive test suite for all robustness enhancements."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="odordiff2_test_"))
        self.results = {}
    
    def cleanup(self):
        """Clean up test resources."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def run_all_tests(self):
        """Run all robustness enhancement tests."""
        try:
            # Test all enhancement areas
            self.test_health_monitoring()
            self.test_circuit_breakers()
            self.test_rate_limiting()
            self.test_input_validation()
            self.test_error_recovery()
            self.test_configuration_management()
            self.test_connection_pooling()
            self.test_backup_recovery()
            self.test_security_hardening()
            self.test_observability()
            
            # Generate summary report
            self.generate_test_report()
            
        finally:
            self.cleanup()
    
    def test_health_monitoring(self):
        """Test health check and monitoring systems."""
        suite = TestSuite("Health Monitoring Tests")
        suite.start()
        
        # Test 1: Import health monitoring components
        def test_health_imports():
            try:
                from odordiff2.monitoring.health import (
                    HealthChecker, HealthMonitor, SystemResourcesCheck,
                    DatabaseHealthCheck, ModelHealthCheck, ExternalDependencyCheck
                )
                return TestResult("Health imports", True, "All health monitoring components imported successfully")
            except ImportError as e:
                return TestResult("Health imports", False, f"Import failed: {e}")
        
        suite.test("Import health components", test_health_imports)
        
        # Test 2: Health checker functionality
        def test_health_checker():
            try:
                from odordiff2.monitoring.health import HealthChecker, SystemResourcesCheck
                
                checker = HealthChecker()
                checker.add_check("system", SystemResourcesCheck())
                
                # Run health check
                health_status = asyncio.run(checker.check_health())
                
                if "system" in health_status and "status" in health_status["system"]:
                    return TestResult("Health checker", True, f"Health check completed: {health_status['system']['status']}")
                else:
                    return TestResult("Health checker", False, "Health check format incorrect")
                    
            except Exception as e:
                return TestResult("Health checker", False, f"Health check failed: {e}")
        
        suite.test("Health checker functionality", test_health_checker)
        
        # Test 3: Health monitoring endpoints
        def test_health_endpoints():
            try:
                from odordiff2.monitoring.health import HealthMonitor
                
                monitor = HealthMonitor()
                
                # Test status endpoint functionality
                status = asyncio.run(monitor.get_health_status())
                
                if isinstance(status, dict) and "overall_status" in status:
                    return TestResult("Health endpoints", True, f"Health endpoint working: {status['overall_status']}")
                else:
                    return TestResult("Health endpoints", False, "Invalid health status format")
                    
            except Exception as e:
                return TestResult("Health endpoints", False, f"Health endpoint failed: {e}")
        
        suite.test("Health monitoring endpoints", test_health_endpoints)
        
        suite.finish()
        self.results["health_monitoring"] = suite.results
    
    def test_circuit_breakers(self):
        """Test circuit breaker patterns."""
        suite = TestSuite("Circuit Breaker Tests")
        suite.start()
        
        # Test 1: Circuit breaker imports
        def test_circuit_breaker_imports():
            try:
                from odordiff2.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
                return TestResult("Circuit breaker imports", True, "Circuit breaker components imported successfully")
            except ImportError as e:
                return TestResult("Circuit breaker imports", False, f"Import failed: {e}")
        
        suite.test("Import circuit breaker components", test_circuit_breaker_imports)
        
        # Test 2: Circuit breaker functionality
        def test_circuit_breaker_function():
            try:
                from odordiff2.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
                
                config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
                cb = CircuitBreaker("test_breaker", config)
                
                # Test normal operation
                def test_function():
                    return "success"
                
                result = cb.call(test_function)
                
                if result == "success" and cb.state == CircuitBreakerState.CLOSED:
                    return TestResult("Circuit breaker function", True, "Circuit breaker working correctly")
                else:
                    return TestResult("Circuit breaker function", False, f"Unexpected result: {result}, state: {cb.state}")
                    
            except Exception as e:
                return TestResult("Circuit breaker function", False, f"Circuit breaker test failed: {e}")
        
        suite.test("Circuit breaker functionality", test_circuit_breaker_function)
        
        # Test 3: Circuit breaker failure handling
        def test_circuit_breaker_failures():
            try:
                from odordiff2.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
                
                config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
                cb = CircuitBreaker("test_breaker_fail", config)
                
                # Trigger failures
                def failing_function():
                    raise Exception("Test failure")
                
                # Should fail twice and open circuit
                for _ in range(3):
                    try:
                        cb.call(failing_function)
                    except:
                        pass
                
                if cb.state == CircuitBreakerState.OPEN:
                    return TestResult("Circuit breaker failures", True, "Circuit breaker opened after failures")
                else:
                    return TestResult("Circuit breaker failures", False, f"Circuit breaker state: {cb.state}")
                    
            except Exception as e:
                return TestResult("Circuit breaker failures", False, f"Failure test failed: {e}")
        
        suite.test("Circuit breaker failure handling", test_circuit_breaker_failures)
        
        suite.finish()
        self.results["circuit_breakers"] = suite.results
    
    def test_rate_limiting(self):
        """Test rate limiting systems."""
        suite = TestSuite("Rate Limiting Tests")
        suite.start()
        
        # Test 1: Rate limiter imports
        def test_rate_limiter_imports():
            try:
                from odordiff2.utils.rate_limiting import RateLimiter, RateLimitBucket, SlidingWindowCounter
                return TestResult("Rate limiter imports", True, "Rate limiting components imported successfully")
            except ImportError as e:
                return TestResult("Rate limiter imports", False, f"Import failed: {e}")
        
        suite.test("Import rate limiting components", test_rate_limiter_imports)
        
        # Test 2: Rate limit functionality
        def test_rate_limit_function():
            try:
                from odordiff2.utils.rate_limiting import RateLimitBucket
                
                # Create rate limiter: 5 requests per 10 seconds
                bucket = RateLimitBucket(max_requests=5, window_seconds=10)
                
                # Test within limits
                allowed_count = 0
                for _ in range(7):  # Try 7 requests, should allow 5
                    if bucket.is_allowed():
                        allowed_count += 1
                
                if allowed_count == 5:
                    return TestResult("Rate limit function", True, f"Rate limiting working: {allowed_count}/5 allowed")
                else:
                    return TestResult("Rate limit function", False, f"Unexpected rate limit: {allowed_count}/5 allowed")
                    
            except Exception as e:
                return TestResult("Rate limit function", False, f"Rate limiting test failed: {e}")
        
        suite.test("Rate limiting functionality", test_rate_limit_function)
        
        # Test 3: IP-based rate limiting
        def test_ip_rate_limiting():
            try:
                from odordiff2.utils.rate_limiting import RateLimiter
                
                limiter = RateLimiter()
                
                # Test IP-based limiting
                test_ip = "192.168.1.1"
                results = []
                
                for _ in range(3):
                    result = asyncio.run(limiter.check_rate_limit(test_ip))
                    results.append(result['allowed'])
                
                if any(results):  # At least some requests should be allowed
                    return TestResult("IP rate limiting", True, f"IP rate limiting functional: {results}")
                else:
                    return TestResult("IP rate limiting", False, "All requests blocked unexpectedly")
                    
            except Exception as e:
                return TestResult("IP rate limiting", False, f"IP rate limiting test failed: {e}")
        
        suite.test("IP-based rate limiting", test_ip_rate_limiting)
        
        suite.finish()
        self.results["rate_limiting"] = suite.results
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        suite = TestSuite("Input Validation Tests")
        suite.start()
        
        # Test 1: Validation imports
        def test_validation_imports():
            try:
                from odordiff2.utils.validation import (
                    InputValidator, Sanitizer, ValidationError, validate_input
                )
                return TestResult("Validation imports", True, "Validation components imported successfully")
            except ImportError as e:
                return TestResult("Validation imports", False, f"Import failed: {e}")
        
        suite.test("Import validation components", test_validation_imports)
        
        # Test 2: Text sanitization
        def test_text_sanitization():
            try:
                from odordiff2.utils.validation import Sanitizer
                
                # Test dangerous input sanitization
                dangerous_input = "<script>alert('xss')</script>Hello"
                sanitized = Sanitizer.sanitize_text(dangerous_input)
                
                if "<script>" not in sanitized and "Hello" in sanitized:
                    return TestResult("Text sanitization", True, f"Text sanitized: {sanitized[:50]}")
                else:
                    return TestResult("Text sanitization", False, f"Sanitization failed: {sanitized}")
                    
            except Exception as e:
                return TestResult("Text sanitization", False, f"Sanitization test failed: {e}")
        
        suite.test("Text sanitization", test_text_sanitization)
        
        # Test 3: JSON schema validation
        def test_json_schema_validation():
            try:
                from odordiff2.utils.validation import InputValidator
                
                # Test generation request validation
                valid_request = {
                    "prompt": "vanilla fragrance",
                    "num_molecules": 5,
                    "safety_threshold": 0.8
                }
                
                validated = InputValidator.validate_request_data(valid_request, "generation_request")
                
                if "prompt" in validated and validated["num_molecules"] == 5:
                    return TestResult("JSON schema validation", True, "Schema validation working")
                else:
                    return TestResult("JSON schema validation", False, f"Validation result: {validated}")
                    
            except Exception as e:
                return TestResult("JSON schema validation", False, f"Schema validation failed: {e}")
        
        suite.test("JSON schema validation", test_json_schema_validation)
        
        # Test 4: Prompt validation with security filtering
        def test_prompt_validation():
            try:
                from odordiff2.utils.validation import InputValidator
                
                # Test normal prompt
                normal_prompt = "Create a nice vanilla fragrance"
                validated_normal = InputValidator.validate_prompt(normal_prompt)
                
                # Test potentially dangerous prompt
                dangerous_prompt = "javascript:alert('test')"
                try:
                    validated_dangerous = InputValidator.validate_prompt(dangerous_prompt)
                    return TestResult("Prompt validation", False, "Dangerous prompt not blocked")
                except:
                    # Should raise validation error
                    pass
                
                if validated_normal == normal_prompt:
                    return TestResult("Prompt validation", True, "Prompt validation working correctly")
                else:
                    return TestResult("Prompt validation", False, f"Normal prompt modified: {validated_normal}")
                    
            except Exception as e:
                return TestResult("Prompt validation", False, f"Prompt validation failed: {e}")
        
        suite.test("Prompt validation with security", test_prompt_validation)
        
        suite.finish()
        self.results["input_validation"] = suite.results
    
    def test_error_recovery(self):
        """Test error recovery and graceful degradation."""
        suite = TestSuite("Error Recovery Tests")
        suite.start()
        
        # Test 1: Recovery imports
        def test_recovery_imports():
            try:
                from odordiff2.utils.recovery import RecoveryManager, GracefulDegradation, RetryStrategy
                return TestResult("Recovery imports", True, "Recovery components imported successfully")
            except ImportError as e:
                return TestResult("Recovery imports", False, f"Import failed: {e}")
        
        suite.test("Import recovery components", test_recovery_imports)
        
        # Test 2: Retry mechanisms
        def test_retry_mechanisms():
            try:
                from odordiff2.utils.recovery import RecoveryManager
                from odordiff2.utils.error_handling import ExponentialBackoffStrategy
                
                manager = RecoveryManager()
                
                # Test retry with eventual success
                attempt_count = 0
                def flaky_function():
                    nonlocal attempt_count
                    attempt_count += 1
                    if attempt_count < 3:
                        raise Exception("Temporary failure")
                    return "success"
                
                result = asyncio.run(manager.execute_with_retry(
                    flaky_function,
                    max_attempts=5,
                    backoff_strategy=ExponentialBackoffStrategy()
                ))
                
                if result == "success" and attempt_count == 3:
                    return TestResult("Retry mechanisms", True, f"Retry successful after {attempt_count} attempts")
                else:
                    return TestResult("Retry mechanisms", False, f"Unexpected result: {result}, attempts: {attempt_count}")
                    
            except Exception as e:
                return TestResult("Retry mechanisms", False, f"Retry test failed: {e}")
        
        suite.test("Retry mechanisms", test_retry_mechanisms)
        
        # Test 3: Graceful degradation
        def test_graceful_degradation():
            try:
                from odordiff2.utils.recovery import GracefulDegradation
                
                degradation = GracefulDegradation()
                
                # Test service degradation
                degradation.degrade_service("test_service", "partial")
                
                status = degradation.get_service_status("test_service")
                
                if status == "partial":
                    return TestResult("Graceful degradation", True, "Service degradation working")
                else:
                    return TestResult("Graceful degradation", False, f"Unexpected status: {status}")
                    
            except Exception as e:
                return TestResult("Graceful degradation", False, f"Degradation test failed: {e}")
        
        suite.test("Graceful degradation", test_graceful_degradation)
        
        suite.finish()
        self.results["error_recovery"] = suite.results
    
    def test_configuration_management(self):
        """Test configuration management system."""
        suite = TestSuite("Configuration Management Tests")
        suite.start()
        
        # Test 1: Config imports
        def test_config_imports():
            try:
                from odordiff2.config.settings import ConfigManager, get_config, load_environment_config
                return TestResult("Config imports", True, "Configuration components imported successfully")
            except ImportError as e:
                return TestResult("Config imports", False, f"Import failed: {e}")
        
        suite.test("Import configuration components", test_config_imports)
        
        # Test 2: Environment configuration loading
        def test_environment_config():
            try:
                from odordiff2.config.settings import ConfigManager
                
                # Test config loading
                config_manager = ConfigManager()
                
                # Should load default configuration
                config = config_manager.get_config()
                
                if isinstance(config, dict) and len(config) > 0:
                    return TestResult("Environment config", True, f"Configuration loaded with {len(config)} keys")
                else:
                    return TestResult("Environment config", False, f"Invalid config: {config}")
                    
            except Exception as e:
                return TestResult("Environment config", False, f"Config loading failed: {e}")
        
        suite.test("Environment configuration", test_environment_config)
        
        # Test 3: Config validation
        def test_config_validation():
            try:
                from odordiff2.config.settings import ConfigManager
                
                config_manager = ConfigManager()
                
                # Test config validation
                test_config = {
                    "api": {
                        "host": "localhost",
                        "port": 8000
                    }
                }
                
                # Should validate without errors
                is_valid = config_manager.validate_config(test_config)
                
                if is_valid:
                    return TestResult("Config validation", True, "Configuration validation working")
                else:
                    return TestResult("Config validation", False, "Config validation failed")
                    
            except Exception as e:
                return TestResult("Config validation", False, f"Config validation test failed: {e}")
        
        suite.test("Configuration validation", test_config_validation)
        
        suite.finish()
        self.results["configuration_management"] = suite.results
    
    def test_connection_pooling(self):
        """Test database connection pooling optimization."""
        suite = TestSuite("Connection Pooling Tests")
        suite.start()
        
        # Test 1: Connection pool imports
        def test_connection_pool_imports():
            try:
                from odordiff2.data.cache import DatabaseConnectionPool, RedisConnectionPool, PersistentCache
                return TestResult("Connection pool imports", True, "Connection pool components imported successfully")
            except ImportError as e:
                return TestResult("Connection pool imports", False, f"Import failed: {e}")
        
        suite.test("Import connection pool components", test_connection_pool_imports)
        
        # Test 2: SQLite connection pool
        def test_sqlite_connection_pool():
            try:
                from odordiff2.data.cache import DatabaseConnectionPool
                
                # Create test database in temp directory
                db_path = self.temp_dir / "test.db"
                pool = DatabaseConnectionPool(str(db_path), max_connections=5)
                
                # Test connection acquisition
                with pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                
                if result and result[0] == 1:
                    return TestResult("SQLite connection pool", True, "SQLite connection pool working")
                else:
                    return TestResult("SQLite connection pool", False, f"Unexpected result: {result}")
                    
            except Exception as e:
                return TestResult("SQLite connection pool", False, f"Connection pool test failed: {e}")
        
        suite.test("SQLite connection pooling", test_sqlite_connection_pool)
        
        # Test 3: Enhanced cache with pooling
        def test_enhanced_cache():
            try:
                from odordiff2.data.cache import PersistentCache
                
                # Create cache with connection pooling
                cache_dir = self.temp_dir / "cache"
                cache = PersistentCache(cache_dir=str(cache_dir))
                
                # Test cache operations
                asyncio.run(cache.store("test_key", {"test": "data"}))
                result = asyncio.run(cache.retrieve("test_key"))
                
                if result and result.get("test") == "data":
                    return TestResult("Enhanced cache", True, "Enhanced cache with pooling working")
                else:
                    return TestResult("Enhanced cache", False, f"Cache result: {result}")
                    
            except Exception as e:
                return TestResult("Enhanced cache", False, f"Enhanced cache test failed: {e}")
        
        suite.test("Enhanced cache with pooling", test_enhanced_cache)
        
        suite.finish()
        self.results["connection_pooling"] = suite.results
    
    def test_backup_recovery(self):
        """Test data backup and recovery mechanisms."""
        suite = TestSuite("Backup Recovery Tests")
        suite.start()
        
        # Test 1: Backup imports
        def test_backup_imports():
            try:
                from odordiff2.utils.backup import BackupManager, BackupConfig, LocalBackupStorage
                return TestResult("Backup imports", True, "Backup components imported successfully")
            except ImportError as e:
                return TestResult("Backup imports", False, f"Import failed: {e}")
        
        suite.test("Import backup components", test_backup_imports)
        
        # Test 2: Local backup functionality
        def test_local_backup():
            try:
                from odordiff2.utils.backup import BackupManager, BackupConfig
                
                # Create test data to backup
                test_data_dir = self.temp_dir / "test_data"
                test_data_dir.mkdir()
                (test_data_dir / "test_file.txt").write_text("test content")
                
                # Create backup manager
                backup_dir = self.temp_dir / "backups"
                config = BackupConfig(backup_dir=str(backup_dir))
                manager = BackupManager(config)
                
                # Create backup
                backup_id = asyncio.run(manager.create_backup(str(test_data_dir), remote_storage=False))
                
                if backup_id:
                    return TestResult("Local backup", True, f"Backup created: {backup_id}")
                else:
                    return TestResult("Local backup", False, "Backup creation failed")
                    
            except Exception as e:
                return TestResult("Local backup", False, f"Backup test failed: {e}")
        
        suite.test("Local backup functionality", test_local_backup)
        
        # Test 3: Backup restoration
        def test_backup_restoration():
            try:
                from odordiff2.utils.backup import BackupManager, BackupConfig
                
                # Create test data and backup (reuse from previous test setup)
                test_data_dir = self.temp_dir / "test_data2"
                test_data_dir.mkdir()
                (test_data_dir / "restore_test.txt").write_text("restore test content")
                
                backup_dir = self.temp_dir / "backups2"
                config = BackupConfig(backup_dir=str(backup_dir))
                manager = BackupManager(config)
                
                # Create and restore backup
                backup_id = asyncio.run(manager.create_backup(str(test_data_dir), remote_storage=False))
                
                if backup_id:
                    restore_dir = self.temp_dir / "restored"
                    success = asyncio.run(manager.restore_backup(backup_id, str(restore_dir)))
                    
                    if success and (restore_dir / "test_data2" / "restore_test.txt").exists():
                        return TestResult("Backup restoration", True, "Backup restoration successful")
                    else:
                        return TestResult("Backup restoration", False, "Restoration verification failed")
                else:
                    return TestResult("Backup restoration", False, "Backup creation failed")
                    
            except Exception as e:
                return TestResult("Backup restoration", False, f"Restoration test failed: {e}")
        
        suite.test("Backup restoration", test_backup_restoration)
        
        suite.finish()
        self.results["backup_recovery"] = suite.results
    
    def test_security_hardening(self):
        """Test security hardening features."""
        suite = TestSuite("Security Hardening Tests")
        suite.start()
        
        # Test 1: Security imports
        def test_security_imports():
            try:
                from odordiff2.utils.security import (
                    SecurityManager, RequestSigner, JWTManager, AdvancedEncryption,
                    SecurityHeaders, AuditLogger
                )
                return TestResult("Security imports", True, "Security components imported successfully")
            except ImportError as e:
                return TestResult("Security imports", False, f"Import failed: {e}")
        
        suite.test("Import security components", test_security_imports)
        
        # Test 2: Request signing
        def test_request_signing():
            try:
                from odordiff2.utils.security import RequestSigner
                
                signer = RequestSigner()
                
                # Test request signing and verification
                request_data = {"test": "data"}
                signature = signer.sign_request(request_data)
                
                is_valid = signer.verify_signature(request_data, signature)
                
                if is_valid:
                    return TestResult("Request signing", True, "Request signing working correctly")
                else:
                    return TestResult("Request signing", False, "Signature verification failed")
                    
            except Exception as e:
                return TestResult("Request signing", False, f"Request signing test failed: {e}")
        
        suite.test("Request signing", test_request_signing)
        
        # Test 3: JWT token management
        def test_jwt_management():
            try:
                from odordiff2.utils.security import JWTManager
                
                jwt_manager = JWTManager()
                
                # Test token generation and validation
                payload = {"user_id": "test_user", "scope": "api_access"}
                token = jwt_manager.generate_token(payload)
                
                decoded = jwt_manager.validate_token(token)
                
                if decoded and decoded.get("user_id") == "test_user":
                    return TestResult("JWT management", True, "JWT token management working")
                else:
                    return TestResult("JWT management", False, f"Token validation failed: {decoded}")
                    
            except Exception as e:
                return TestResult("JWT management", False, f"JWT test failed: {e}")
        
        suite.test("JWT token management", test_jwt_management)
        
        # Test 4: Security headers
        def test_security_headers():
            try:
                from odordiff2.utils.security import SecurityHeaders
                
                security = SecurityHeaders()
                
                # Test security header generation
                headers = security.get_security_headers()
                
                expected_headers = ["X-Content-Type-Options", "X-Frame-Options", "X-XSS-Protection"]
                
                if all(header in headers for header in expected_headers):
                    return TestResult("Security headers", True, f"Security headers generated: {len(headers)} headers")
                else:
                    return TestResult("Security headers", False, f"Missing headers: {headers.keys()}")
                    
            except Exception as e:
                return TestResult("Security headers", False, f"Security headers test failed: {e}")
        
        suite.test("Security headers", test_security_headers)
        
        suite.finish()
        self.results["security_hardening"] = suite.results
    
    def test_observability(self):
        """Test enhanced observability features."""
        suite = TestSuite("Observability Tests")
        suite.start()
        
        # Test 1: Observability imports
        def test_observability_imports():
            try:
                from odordiff2.utils.logging import (
                    OdorDiffLogger, StructuredFormatter, DistributedTracer,
                    CorrelationContext, TraceSpan, configure_observability
                )
                return TestResult("Observability imports", True, "Observability components imported successfully")
            except ImportError as e:
                return TestResult("Observability imports", False, f"Import failed: {e}")
        
        suite.test("Import observability components", test_observability_imports)
        
        # Test 2: Structured logging
        def test_structured_logging():
            try:
                from odordiff2.utils.logging import configure_observability
                
                # Configure structured logging
                logger = configure_observability(
                    service_name="test_service",
                    log_directory=str(self.temp_dir / "logs"),
                    structured_logging=True
                )
                
                # Test structured log generation
                logger.info("Test structured log", test_field="test_value", count=42)
                
                # Check if structured log file was created
                log_files = list((self.temp_dir / "logs").glob("*.jsonl"))
                
                if log_files:
                    return TestResult("Structured logging", True, f"Structured logs created: {len(log_files)} files")
                else:
                    return TestResult("Structured logging", False, "No structured log files found")
                    
            except Exception as e:
                return TestResult("Structured logging", False, f"Structured logging test failed: {e}")
        
        suite.test("Structured logging", test_structured_logging)
        
        # Test 3: Distributed tracing
        def test_distributed_tracing():
            try:
                from odordiff2.utils.logging import DistributedTracer
                
                tracer = DistributedTracer("test_service")
                
                # Create and finish a span
                span = tracer.start_span("test_operation")
                span.add_tag("test_tag", "test_value")
                span.add_log("Test log message")
                tracer.finish_span(span)
                
                # Check tracing stats
                stats = tracer.get_trace_stats()
                
                if stats["total_completed_spans"] >= 1:
                    return TestResult("Distributed tracing", True, f"Tracing working: {stats['total_completed_spans']} spans")
                else:
                    return TestResult("Distributed tracing", False, f"No completed spans: {stats}")
                    
            except Exception as e:
                return TestResult("Distributed tracing", False, f"Tracing test failed: {e}")
        
        suite.test("Distributed tracing", test_distributed_tracing)
        
        # Test 4: Correlation context
        def test_correlation_context():
            try:
                from odordiff2.utils.logging import CorrelationContext, correlation_id_context
                
                # Test correlation context
                with CorrelationContext("test_correlation_id") as ctx:
                    current_id = correlation_id_context.get()
                    
                    if current_id == "test_correlation_id":
                        return TestResult("Correlation context", True, f"Correlation context working: {current_id}")
                    else:
                        return TestResult("Correlation context", False, f"Wrong correlation ID: {current_id}")
                    
            except Exception as e:
                return TestResult("Correlation context", False, f"Correlation context test failed: {e}")
        
        suite.test("Correlation context", test_correlation_context)
        
        suite.finish()
        self.results["observability"] = suite.results
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("ROBUSTNESS ENHANCEMENT TEST REPORT")
        print("="*80)
        
        total_tests = 0
        total_passed = 0
        
        for category, results in self.results.items():
            category_passed = sum(1 for r in results if r.passed)
            category_total = len(results)
            total_tests += category_total
            total_passed += category_passed
            
            print(f"\n{category.replace('_', ' ').title()}: {category_passed}/{category_total} passed")
            
            for result in results:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {result.name}: {result.message}")
        
        print(f"\n" + "-"*80)
        print(f"OVERALL RESULTS: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
        
        if total_passed == total_tests:
            print("🎉 ALL ROBUSTNESS ENHANCEMENTS WORKING CORRECTLY!")
        elif total_passed / total_tests >= 0.8:
            print("⚠️  Most enhancements working, some issues to address")
        else:
            print("❌ Significant issues found, review required")
        
        print("="*80)
        
        # Save detailed report to file
        report_file = Path("robustness_test_report.json")
        report_data = {
            "timestamp": time.time(),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "success_rate": total_passed / total_tests if total_tests > 0 else 0,
            "categories": {}
        }
        
        for category, results in self.results.items():
            report_data["categories"][category] = [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "timestamp": r.timestamp
                } for r in results
            ]
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Detailed report saved to: {report_file}")


def main():
    """Main test execution function."""
    print("OdorDiff-2 Generation 2 Robustness Enhancement Test Suite")
    print("Testing all 10 robustness enhancement areas...")
    
    test_suite = RobustnessTestSuite()
    
    try:
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user.")
    except Exception as e:
        print(f"\n\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    main()