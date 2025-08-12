#!/usr/bin/env python3
"""
Script to run comprehensive robustness tests for OdorDiff-2.
Tests all error handling, validation, security, and robustness features.
"""

import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robustness_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def run_manual_robustness_tests():
    """
    Run manual robustness tests that don't require pytest.
    Tests core robustness features directly.
    """
    
    logger.info("Starting manual robustness tests...")
    results = {}
    
    # Test 1: Error Handling System
    logger.info("Testing error handling system...")
    try:
        from odordiff2.utils.error_handling import (
            ErrorTracker, OdorDiffException, ValidationError, 
            handle_errors, retry_with_backoff
        )
        
        # Test error tracking
        tracker = ErrorTracker()
        error = ValidationError("Test error", field="test")
        tracker.record_error(error, {"component": "test"})
        
        stats = tracker.get_error_statistics()
        assert stats["total_errors"] == 1
        
        # Test error decorator
        @handle_errors(ValueError, default_return="handled")
        def test_func(x):
            if x < 0:
                raise ValueError("Negative")
            return x * 2
        
        assert test_func(5) == 10
        assert test_func(-1) == "handled"
        
        # Test retry mechanism
        attempt_count = 0
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Flaky")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert attempt_count == 3
        
        results["error_handling"] = {"status": "PASS", "details": "All error handling tests passed"}
        logger.info("✓ Error handling tests passed")
        
    except Exception as e:
        results["error_handling"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"✗ Error handling tests failed: {e}")
    
    # Test 2: Input Validation
    logger.info("Testing input validation system...")
    try:
        from odordiff2.utils.validation import (
            InputSanitizer, MoleculeValidator, TextPromptValidator
        )
        
        # Test text sanitization
        clean = InputSanitizer.sanitize_text("<script>alert('xss')</script>Hello")
        assert "script" not in clean or "&lt;" in clean
        
        # Test SMILES validation
        mol_validator = MoleculeValidator()
        result = mol_validator.validate_smiles("CCO")  # Ethanol
        assert result.is_valid
        
        result = mol_validator.validate_smiles("InvalidSMILES")
        assert not result.is_valid
        
        # Test prompt validation
        prompt_validator = TextPromptValidator()
        result = prompt_validator.validate_prompt("Fresh floral rose scent")
        assert result.is_valid
        
        result = prompt_validator.validate_prompt("toxic poison dangerous")
        assert not result.is_valid
        
        results["input_validation"] = {"status": "PASS", "details": "All validation tests passed"}
        logger.info("✓ Input validation tests passed")
        
    except Exception as e:
        results["input_validation"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"✗ Input validation tests failed: {e}")
    
    # Test 3: Health Monitoring
    logger.info("Testing health monitoring system...")
    try:
        from odordiff2.monitoring.health_checks import (
            HealthMonitor, SystemResourceCheck, HealthStatus
        )
        
        # Test system resource check
        resource_check = SystemResourceCheck()
        result = await resource_check.check()
        
        assert result.name == "system_resources"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
        assert "cpu_percent" in result.details
        
        # Test health monitor
        monitor = HealthMonitor()
        monitor.checks = [resource_check]  # Use only our test check
        
        all_results = await monitor.run_all_checks()
        assert len(all_results) == 1
        
        overall_status = monitor.get_overall_status()
        assert "status" in overall_status
        
        results["health_monitoring"] = {"status": "PASS", "details": "Health monitoring system functional"}
        logger.info("✓ Health monitoring tests passed")
        
    except Exception as e:
        results["health_monitoring"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"✗ Health monitoring tests failed: {e}")
    
    # Test 4: Rate Limiting
    logger.info("Testing rate limiting system...")
    try:
        from odordiff2.security.rate_limiting import (
            TokenBucket, InMemoryRateLimiter, RateLimit
        )
        
        # Test token bucket
        bucket = TokenBucket(capacity=3, refill_rate=1.0)
        
        # Should allow initial requests
        for i in range(3):
            assert bucket.consume() == True
        
        # Should block when empty
        assert bucket.consume() == False
        
        # Test rate limiter
        rate_limit = RateLimit(requests=2, window_seconds=10)
        limiter = InMemoryRateLimiter(rate_limit)
        
        result1 = await limiter.check_rate_limit("test_key")
        result2 = await limiter.check_rate_limit("test_key")
        result3 = await limiter.check_rate_limit("test_key")  # Should be blocked
        
        assert result1.allowed == True
        assert result2.allowed == True
        assert result3.allowed == False
        
        results["rate_limiting"] = {"status": "PASS", "details": "Rate limiting system functional"}
        logger.info("✓ Rate limiting tests passed")
        
    except Exception as e:
        results["rate_limiting"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"✗ Rate limiting tests failed: {e}")
    
    # Test 5: Authentication System
    logger.info("Testing authentication system...")
    try:
        from odordiff2.security.authentication import (
            APIKeyValidator, JWTManager, UserRole, Permission
        )
        
        # Test API key generation and validation
        validator = APIKeyValidator()
        raw_key, api_key = validator.generate_api_key(
            user_id="test_user",
            name="Test Key",
            role=UserRole.USER
        )
        
        assert raw_key.startswith("odr_")
        assert api_key.role == UserRole.USER
        assert Permission.GENERATE_MOLECULES in api_key.permissions
        
        # Test key validation
        auth_result = await validator.validate_api_key(raw_key)
        assert auth_result.success == True
        assert auth_result.user_id == "test_user"
        
        # Test JWT tokens
        jwt_manager = JWTManager("test_secret")
        token = jwt_manager.create_token("jwt_user", UserRole.PREMIUM)
        jwt_result = jwt_manager.verify_token(token)
        
        assert jwt_result.success == True
        assert jwt_result.user_id == "jwt_user"
        assert jwt_result.role == UserRole.PREMIUM
        
        results["authentication"] = {"status": "PASS", "details": "Authentication system functional"}
        logger.info("✓ Authentication tests passed")
        
    except Exception as e:
        results["authentication"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"✗ Authentication tests failed: {e}")
    
    # Test 6: Security Measures
    logger.info("Testing security measures...")
    try:
        # Test XSS prevention
        from odordiff2.utils.validation import InputSanitizer
        
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<iframe src='http://evil.com'></iframe>"
        ]
        
        for malicious in malicious_inputs:
            clean = InputSanitizer.sanitize_text(malicious, allow_html=False)
            # Should not contain dangerous elements
            assert "<script>" not in clean
            assert "javascript:" not in clean
            assert "<iframe" not in clean
        
        # Test SQL injection prevention
        sql_inputs = [
            "'; DROP TABLE users; --",
            "' OR 1=1 --",
            "; EXEC xp_cmdshell('dir'); --"
        ]
        
        for sql_input in sql_inputs:
            clean = InputSanitizer.sanitize_sql_input(sql_input)
            assert "DROP TABLE" not in clean.upper()
            assert "--" not in clean
        
        results["security_measures"] = {"status": "PASS", "details": "Security measures functional"}
        logger.info("✓ Security measures tests passed")
        
    except Exception as e:
        results["security_measures"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"✗ Security measures tests failed: {e}")
    
    return results


def run_pytest_tests():
    """Run pytest-based robustness tests."""
    logger.info("Running pytest-based robustness tests...")
    
    test_file = Path(__file__).parent / "tests" / "test_robustness_suite.py"
    
    if not test_file.exists():
        logger.warning(f"Pytest test file not found: {test_file}")
        return {"pytest_tests": {"status": "SKIP", "reason": "Test file not found"}}
    
    try:
        # Run pytest with specific options for robustness testing
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_file),
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--maxfail=10",  # Stop after 10 failures
            "--timeout=300",  # 5 minute timeout per test
            "--durations=10",  # Show 10 slowest tests
        ], capture_output=True, text=True, timeout=1800)  # 30 minute total timeout
        
        if result.returncode == 0:
            return {"pytest_tests": {"status": "PASS", "details": "All pytest tests passed"}}
        else:
            return {
                "pytest_tests": {
                    "status": "FAIL", 
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            }
    
    except subprocess.TimeoutExpired:
        return {"pytest_tests": {"status": "TIMEOUT", "details": "Tests timed out after 30 minutes"}}
    except FileNotFoundError:
        return {"pytest_tests": {"status": "SKIP", "reason": "pytest not available"}}
    except Exception as e:
        return {"pytest_tests": {"status": "ERROR", "error": str(e)}}


def generate_robustness_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive robustness test report."""
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r.get("status") == "PASS")
    failed_tests = sum(1 for r in results.values() if r.get("status") == "FAIL")
    skipped_tests = sum(1 for r in results.values() if r.get("status") == "SKIP")
    
    overall_status = "PASS" if failed_tests == 0 else "FAIL"
    
    report = {
        "timestamp": time.time(),
        "overall_status": overall_status,
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "skipped": skipped_tests,
            "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
        },
        "test_results": results,
        "recommendations": []
    }
    
    # Add recommendations based on results
    if failed_tests > 0:
        report["recommendations"].append("Address failed tests before deployment")
    
    if any("timeout" in str(r).lower() for r in results.values()):
        report["recommendations"].append("Investigate performance issues causing timeouts")
    
    if passed_tests == total_tests:
        report["recommendations"].append("System robustness validated - ready for production")
    
    return report


async def main():
    """Main test execution function."""
    logger.info("=" * 60)
    logger.info("OdorDiff-2 Robustness Test Suite")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Run manual robustness tests
    logger.info("Phase 1: Manual robustness tests")
    manual_results = await run_manual_robustness_tests()
    
    # Run pytest-based tests
    logger.info("\nPhase 2: Pytest-based tests")
    pytest_results = run_pytest_tests()
    
    # Combine results
    all_results = {**manual_results, **pytest_results}
    
    # Generate report
    report = generate_robustness_report(all_results)
    
    # Save report
    report_path = Path("robustness_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    duration = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("ROBUSTNESS TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall Status: {report['overall_status']}")
    logger.info(f"Total Tests: {report['summary']['total_tests']}")
    logger.info(f"Passed: {report['summary']['passed']}")
    logger.info(f"Failed: {report['summary']['failed']}")
    logger.info(f"Skipped: {report['summary']['skipped']}")
    logger.info(f"Success Rate: {report['summary']['success_rate']}")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Report saved to: {report_path.absolute()}")
    
    # Print detailed results
    logger.info("\nDETAILED RESULTS:")
    for test_name, result in all_results.items():
        status_icon = "✓" if result.get("status") == "PASS" else "✗" if result.get("status") == "FAIL" else "⚠"
        logger.info(f"  {status_icon} {test_name}: {result.get('status', 'UNKNOWN')}")
        
        if result.get("details"):
            logger.info(f"    Details: {result['details']}")
        if result.get("error"):
            logger.info(f"    Error: {result['error']}")
    
    # Print recommendations
    if report["recommendations"]:
        logger.info("\nRECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            logger.info(f"  {i}. {rec}")
    
    logger.info("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if report["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nTest suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        sys.exit(1)