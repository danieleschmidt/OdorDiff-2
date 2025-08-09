#!/usr/bin/env python3
"""
Basic functionality test for robustness enhancements.

Tests the core functionality of robustness enhancements without requiring
external dependencies like torch, rdkit, etc.
"""

import sys
import time
import tempfile
import shutil
from pathlib import Path

def test_imports():
    """Test that all robustness enhancement modules can be imported."""
    print("Testing imports...")
    
    tests = [
        ("logging", "from odordiff2.utils.logging import OdorDiffLogger, get_logger"),
        ("validation", "from odordiff2.utils.validation import Sanitizer, ValidationError"),
        ("security", "from odordiff2.utils.security import SecurityManager, RequestSigner"),
        ("circuit_breaker", "from odordiff2.utils.circuit_breaker import CircuitBreaker, CircuitBreakerState"),
        ("rate_limiting", "from odordiff2.utils.rate_limiting import RateLimitBucket, SlidingWindowCounter"),
        ("recovery", "from odordiff2.utils.recovery import RecoveryManager, GracefulDegradation"),
        ("backup", "from odordiff2.utils.backup import BackupManager, BackupConfig"),
        ("error_handling", "from odordiff2.utils.error_handling import retry_with_backoff, ExponentialBackoffStrategy"),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"  ✓ {name}: Import successful")
            passed += 1
        except ImportError as e:
            print(f"  ✗ {name}: Import failed - {e}")
        except Exception as e:
            print(f"  ✗ {name}: Unexpected error - {e}")
    
    print(f"\nImport Results: {passed}/{total} passed")
    return passed == total

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")
    
    passed = 0
    total = 0
    
    # Test 1: Basic logging
    total += 1
    try:
        sys.path.insert(0, '.')  # Add current directory to Python path
        from odordiff2.utils.logging import OdorDiffLogger
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = OdorDiffLogger("test", temp_dir)
            logger.info("Test log message")
            
            # Check if log file was created
            log_files = list(Path(temp_dir).glob("*.log"))
            if log_files:
                print("  ✓ Logging: Basic functionality working")
                passed += 1
            else:
                print("  ✗ Logging: No log files created")
                
    except Exception as e:
        print(f"  ✗ Logging: Failed - {e}")
    
    # Test 2: Text sanitization
    total += 1
    try:
        from odordiff2.utils.validation import Sanitizer
        
        dangerous_text = "<script>alert('test')</script>Hello"
        sanitized = Sanitizer.sanitize_text(dangerous_text)
        
        if "<script>" not in sanitized and "Hello" in sanitized:
            print("  ✓ Sanitization: XSS protection working")
            passed += 1
        else:
            print(f"  ✗ Sanitization: Failed - result: {sanitized}")
            
    except Exception as e:
        print(f"  ✗ Sanitization: Failed - {e}")
    
    # Test 3: Circuit breaker basic structure
    total += 1
    try:
        from odordiff2.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
        
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        if cb.state == CircuitBreakerState.CLOSED:
            print("  ✓ Circuit Breaker: Basic structure working")
            passed += 1
        else:
            print(f"  ✗ Circuit Breaker: Unexpected initial state: {cb.state}")
            
    except Exception as e:
        print(f"  ✗ Circuit Breaker: Failed - {e}")
    
    # Test 4: Rate limiting basic structure
    total += 1
    try:
        from odordiff2.utils.rate_limiting import RateLimitBucket
        
        bucket = RateLimitBucket(max_requests=5, window_seconds=60)
        
        # Should allow first request
        allowed = bucket.is_allowed()
        if allowed:
            print("  ✓ Rate Limiting: Basic functionality working")
            passed += 1
        else:
            print("  ✗ Rate Limiting: First request blocked unexpectedly")
            
    except Exception as e:
        print(f"  ✗ Rate Limiting: Failed - {e}")
    
    # Test 5: Security components
    total += 1
    try:
        from odordiff2.utils.security import RequestSigner, SecurityHeaders
        
        signer = RequestSigner()
        headers = SecurityHeaders()
        
        # Test basic functionality
        test_data = {"test": "data"}
        signature = signer.sign_request(test_data)
        security_headers = headers.get_security_headers()
        
        if signature and security_headers:
            print("  ✓ Security: Basic components working")
            passed += 1
        else:
            print("  ✗ Security: Missing signature or headers")
            
    except Exception as e:
        print(f"  ✗ Security: Failed - {e}")
    
    # Test 6: Configuration structure
    total += 1
    try:
        from odordiff2.config.settings import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        if isinstance(config, dict):
            print("  ✓ Configuration: Basic structure working")
            passed += 1
        else:
            print(f"  ✗ Configuration: Invalid config type: {type(config)}")
            
    except Exception as e:
        print(f"  ✗ Configuration: Failed - {e}")
    
    print(f"\nFunctionality Results: {passed}/{total} passed")
    return passed == total

def test_file_structure():
    """Test that all expected files exist."""
    print("\nTesting file structure...")
    
    expected_files = [
        "odordiff2/utils/logging.py",
        "odordiff2/utils/validation.py", 
        "odordiff2/utils/security.py",
        "odordiff2/utils/circuit_breaker.py",
        "odordiff2/utils/rate_limiting.py",
        "odordiff2/utils/recovery.py",
        "odordiff2/utils/backup.py",
        "odordiff2/utils/error_handling.py",
        "odordiff2/config/settings.py",
        "odordiff2/monitoring/health.py",
        "odordiff2/data/cache.py",
        "config/development.yaml",
        "config/production.yaml",
        "config/testing.yaml",
    ]
    
    passed = 0
    total = len(expected_files)
    
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}: Exists")
            passed += 1
        else:
            print(f"  ✗ {file_path}: Missing")
    
    print(f"\nFile Structure Results: {passed}/{total} files present")
    return passed == total

def main():
    """Run basic functionality tests."""
    print("OdorDiff-2 Robustness Enhancement Basic Functionality Test")
    print("="*60)
    
    # Run all tests
    import_success = test_imports()
    file_success = test_file_structure()
    func_success = test_basic_functionality()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    results = [
        ("File Structure", file_success),
        ("Import Tests", import_success), 
        ("Basic Functionality", func_success)
    ]
    
    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL BASIC TESTS PASSED!")
        print("Robustness enhancements are properly structured and functional.")
    else:
        print("⚠️  Some tests failed - see details above.")
        print("The enhancements may still work but need investigation.")
    
    print("="*60)

if __name__ == "__main__":
    main()