#!/usr/bin/env python3
"""
Comprehensive test runner with advanced features and reporting.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil


class TestRunner:
    """Advanced test runner for OdorDiff-2."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command: List[str], description: str, 
                   capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess:
        """Run command with logging and error handling."""
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(command)}", "DEBUG")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=check
            )
            
            if result.returncode == 0:
                self.log(f"✅ {description} - SUCCESS")
            else:
                self.log(f"❌ {description} - FAILED (exit code: {result.returncode})", "ERROR")
                if result.stdout:
                    self.log(f"STDOUT: {result.stdout}", "ERROR")
                if result.stderr:
                    self.log(f"STDERR: {result.stderr}", "ERROR")
            
            return result
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ {description} - FAILED with exception: {e}", "ERROR")
            raise
        except FileNotFoundError as e:
            self.log(f"❌ Command not found: {e}", "ERROR")
            raise
    
    def check_environment(self) -> bool:
        """Check if environment is properly set up."""
        self.log("Checking environment setup...")
        
        required_commands = ["python", "pip", "pytest"]
        missing_commands = []
        
        for cmd in required_commands:
            try:
                result = subprocess.run([cmd, "--version"], 
                                      capture_output=True, text=True, check=True)
                self.log(f"✅ {cmd}: {result.stdout.strip()}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_commands.append(cmd)
                self.log(f"❌ {cmd}: Not found or not working", "ERROR")
        
        if missing_commands:
            self.log(f"Missing required commands: {missing_commands}", "ERROR")
            return False
        
        # Check if project is installed
        try:
            import odordiff2
            self.log(f"✅ OdorDiff-2 package available at: {odordiff2.__file__}")
        except ImportError:
            self.log("❌ OdorDiff-2 package not installed", "ERROR")
            self.log("Please run: pip install -e .", "ERROR")
            return False
        
        return True
    
    def install_dependencies(self, test_type: str = "all") -> bool:
        """Install required dependencies for testing."""
        self.log(f"Installing dependencies for {test_type} testing...")
        
        dependency_maps = {
            "unit": "[test]",
            "integration": "[test]",
            "performance": "[test,performance]",
            "security": "[test,dev]",
            "all": "[dev,test,performance]"
        }
        
        extras = dependency_maps.get(test_type, "[test]")
        
        try:
            # Upgrade pip first
            self.run_command(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                "Upgrading pip"
            )
            
            # Install the package with extras
            self.run_command(
                [sys.executable, "-m", "pip", "install", "-e", f".{extras}"],
                f"Installing OdorDiff-2 with {extras}"
            )
            
            return True
            
        except subprocess.CalledProcessError:
            self.log("Failed to install dependencies", "ERROR")
            return False
    
    def run_code_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks."""
        self.log("Running code quality checks...")
        results = {}
        
        # Black formatting check
        try:
            result = self.run_command(
                ["black", "--check", "--diff", "odordiff2/", "tests/"],
                "Black formatting check",
                check=False
            )
            results["black"] = {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
        except FileNotFoundError:
            results["black"] = {"passed": False, "error": "Black not installed"}
        
        # isort import sorting check
        try:
            result = self.run_command(
                ["isort", "--check-only", "--diff", "odordiff2/", "tests/"],
                "isort import check",
                check=False
            )
            results["isort"] = {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
        except FileNotFoundError:
            results["isort"] = {"passed": False, "error": "isort not installed"}
        
        # flake8 linting
        try:
            result = self.run_command(
                ["flake8", "odordiff2/", "tests/", "--statistics"],
                "flake8 linting check",
                check=False
            )
            results["flake8"] = {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
        except FileNotFoundError:
            results["flake8"] = {"passed": False, "error": "flake8 not installed"}
        
        # mypy type checking
        try:
            result = self.run_command(
                ["mypy", "odordiff2/", "--show-error-codes"],
                "mypy type checking",
                check=False
            )
            results["mypy"] = {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
        except FileNotFoundError:
            results["mypy"] = {"passed": False, "error": "mypy not installed"}
        
        return results
    
    def run_security_checks(self) -> Dict[str, Any]:
        """Run security vulnerability checks."""
        self.log("Running security checks...")
        results = {}
        
        # Bandit security scan
        try:
            result = self.run_command(
                ["bandit", "-r", "odordiff2/", "-f", "json"],
                "Bandit security scan",
                check=False
            )
            results["bandit"] = {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "report": json.loads(result.stdout) if result.stdout else {}
            }
        except (FileNotFoundError, json.JSONDecodeError):
            results["bandit"] = {"passed": False, "error": "Bandit not available or invalid output"}
        
        # Safety dependency check
        try:
            result = self.run_command(
                ["safety", "check", "--json"],
                "Safety dependency check",
                check=False
            )
            results["safety"] = {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "vulnerabilities": json.loads(result.stdout) if result.stdout else []
            }
        except (FileNotFoundError, json.JSONDecodeError):
            results["safety"] = {"passed": False, "error": "Safety not available or invalid output"}
        
        return results
    
    def run_unit_tests(self, coverage_threshold: int = 85) -> Dict[str, Any]:
        """Run unit tests with coverage."""
        self.log("Running unit tests with coverage...")
        
        pytest_args = [
            "python", "-m", "pytest",
            "tests/unit/",
            "-v",
            "--tb=short",
            f"--cov=odordiff2",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            f"--cov-fail-under={coverage_threshold}",
            "--cov-branch",
            "--junitxml=junit.xml",
            "--html=report.html",
            "--self-contained-html",
            "-m", "unit and not slow"
        ]
        
        try:
            result = self.run_command(pytest_args, "Unit tests", check=False)
            
            # Parse coverage data if available
            coverage_data = {}
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr,
                "coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                "coverage_data": coverage_data
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "passed": False,
                "error": str(e),
                "output": e.stdout + e.stderr if hasattr(e, 'stdout') else ""
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        self.log("Running integration tests...")
        
        pytest_args = [
            "python", "-m", "pytest",
            "tests/integration/",
            "-v",
            "--tb=short",
            "--junitxml=junit-integration.xml",
            "--html=report-integration.html",
            "--self-contained-html",
            "-m", "integration and not slow",
            "--timeout=60"
        ]
        
        try:
            result = self.run_command(pytest_args, "Integration tests", check=False)
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "passed": False,
                "error": str(e),
                "output": e.stdout + e.stderr if hasattr(e, 'stdout') else ""
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests and benchmarks."""
        self.log("Running performance tests...")
        
        pytest_args = [
            "python", "-m", "pytest",
            "tests/performance/",
            "-v",
            "--tb=short",
            "--benchmark-only",
            "--benchmark-json=benchmark.json",
            "--benchmark-histogram=benchmark-histogram",
            "--junitxml=junit-performance.xml",
            "-m", "performance",
            "--timeout=300"
        ]
        
        try:
            result = self.run_command(pytest_args, "Performance tests", check=False)
            
            # Parse benchmark data if available
            benchmark_data = {}
            benchmark_file = self.project_root / "benchmark.json"
            if benchmark_file.exists():
                try:
                    with open(benchmark_file) as f:
                        benchmark_data = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr,
                "benchmarks": benchmark_data
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "passed": False,
                "error": str(e),
                "output": e.stdout + e.stderr if hasattr(e, 'stdout') else ""
            }
    
    def run_property_based_tests(self) -> Dict[str, Any]:
        """Run property-based tests with Hypothesis."""
        self.log("Running property-based tests...")
        
        pytest_args = [
            "python", "-m", "pytest",
            "tests/property/",
            "-v",
            "--tb=short",
            "--hypothesis-show-statistics",
            "--junitxml=junit-property.xml",
            "-m", "property",
            "--timeout=300"
        ]
        
        try:
            result = self.run_command(pytest_args, "Property-based tests", check=False)
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "passed": False,
                "error": str(e),
                "output": e.stdout + e.stderr if hasattr(e, 'stdout') else ""
            }
    
    def generate_report(self) -> None:
        """Generate comprehensive test report."""
        self.log("Generating comprehensive test report...")
        
        total_time = time.time() - self.start_time
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_time_seconds": round(total_time, 2),
            "results": self.test_results,
            "summary": {
                "total_checks": len(self.test_results),
                "passed_checks": sum(1 for r in self.test_results.values() if r.get("passed", False)),
                "failed_checks": sum(1 for r in self.test_results.values() if not r.get("passed", False))
            }
        }
        
        # Calculate success rate
        if report["summary"]["total_checks"] > 0:
            report["summary"]["success_rate"] = round(
                report["summary"]["passed_checks"] / report["summary"]["total_checks"] * 100, 1
            )
        else:
            report["summary"]["success_rate"] = 0.0
        
        # Save JSON report
        report_file = self.project_root / "test-report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable report
        markdown_report = self._generate_markdown_report(report)
        markdown_file = self.project_root / "test-report.md"
        with open(markdown_file, "w") as f:
            f.write(markdown_report)
        
        self.log(f"Reports saved:")
        self.log(f"  JSON: {report_file}")
        self.log(f"  Markdown: {markdown_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)
        print(f"Total time: {total_time:.1f}s")
        print(f"Total checks: {report['summary']['total_checks']}")
        print(f"Passed: {report['summary']['passed_checks']}")
        print(f"Failed: {report['summary']['failed_checks']}")
        print(f"Success rate: {report['summary']['success_rate']:.1f}%")
        
        if report['summary']['success_rate'] < 100:
            print("\nFAILED CHECKS:")
            for name, result in self.test_results.items():
                if not result.get("passed", False):
                    print(f"  ❌ {name}")
        else:
            print("\n✅ ALL CHECKS PASSED!")
        
        print("="*80)
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate markdown test report."""
        lines = [
            "# OdorDiff-2 Test Report",
            "",
            f"**Generated**: {report['timestamp']}",
            f"**Duration**: {report['total_time_seconds']}s",
            f"**Success Rate**: {report['summary']['success_rate']:.1f}%",
            "",
            "## Summary",
            "",
            f"- Total checks: {report['summary']['total_checks']}",
            f"- Passed: {report['summary']['passed_checks']}",
            f"- Failed: {report['summary']['failed_checks']}",
            "",
            "## Results",
            "",
            "| Check | Status | Details |",
            "|-------|--------|---------|"
        ]
        
        for name, result in report['results'].items():
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            details = result.get("error", "Success") if not result.get("passed", False) else "Success"
            lines.append(f"| {name} | {status} | {details[:100]}... |")
        
        lines.extend([
            "",
            "## Coverage",
            ""
        ])
        
        # Add coverage information if available
        unit_test_result = report['results'].get('unit_tests', {})
        if 'coverage' in unit_test_result:
            coverage = unit_test_result['coverage']
            lines.append(f"- Unit test coverage: {coverage:.1f}%")
        
        return "\n".join(lines)
    
    def run_all_tests(self, test_types: List[str], coverage_threshold: int = 85) -> bool:
        """Run all specified test types."""
        self.log(f"Starting comprehensive test run: {', '.join(test_types)}")
        
        # Check environment first
        if not self.check_environment():
            self.log("Environment check failed", "ERROR")
            return False
        
        # Run tests based on specified types
        all_passed = True
        
        if "quality" in test_types:
            self.test_results["code_quality"] = self.run_code_quality_checks()
            quality_passed = all(r.get("passed", False) for r in self.test_results["code_quality"].values())
            if not quality_passed:
                all_passed = False
        
        if "security" in test_types:
            self.test_results["security_checks"] = self.run_security_checks()
            security_passed = all(r.get("passed", False) for r in self.test_results["security_checks"].values())
            if not security_passed:
                all_passed = False
        
        if "unit" in test_types:
            self.test_results["unit_tests"] = self.run_unit_tests(coverage_threshold)
            if not self.test_results["unit_tests"].get("passed", False):
                all_passed = False
        
        if "integration" in test_types:
            self.test_results["integration_tests"] = self.run_integration_tests()
            if not self.test_results["integration_tests"].get("passed", False):
                all_passed = False
        
        if "performance" in test_types:
            self.test_results["performance_tests"] = self.run_performance_tests()
            if not self.test_results["performance_tests"].get("passed", False):
                all_passed = False
        
        if "property" in test_types:
            self.test_results["property_tests"] = self.run_property_based_tests()
            if not self.test_results["property_tests"].get("passed", False):
                all_passed = False
        
        # Generate comprehensive report
        self.generate_report()
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive test runner for OdorDiff-2")
    parser.add_argument(
        "--test-types",
        nargs="+",
        choices=["quality", "security", "unit", "integration", "performance", "property", "all"],
        default=["all"],
        help="Types of tests to run"
    )
    parser.add_argument(
        "--coverage-threshold",
        type=int,
        default=85,
        help="Minimum coverage threshold percentage"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies before running tests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Expand "all" to specific test types
    if "all" in args.test_types:
        test_types = ["quality", "security", "unit", "integration", "performance", "property"]
    else:
        test_types = args.test_types
    
    # Create test runner
    runner = TestRunner(verbose=args.verbose)
    
    # Install dependencies if requested
    if args.install_deps:
        if not runner.install_dependencies("all"):
            sys.exit(1)
    
    # Run tests
    success = runner.run_all_tests(test_types, args.coverage_threshold)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()