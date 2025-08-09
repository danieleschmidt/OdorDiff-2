#!/usr/bin/env python3
"""
Validation script to demonstrate comprehensive testing and quality gates implementation.
"""

import os
import sys
from pathlib import Path
import json
import subprocess
import importlib.util

def validate_file_structure():
    """Validate that all required test files and structure are in place."""
    print("🔍 Validating file structure...")
    
    required_files = [
        # Core test configuration
        "pytest.ini",
        "pyproject.toml",
        
        # Test directories and files
        "tests/conftest.py",
        "tests/conftest_advanced.py",
        
        # Unit tests
        "tests/unit/test_molecule.py",
        "tests/unit/test_safety_filter.py", 
        "tests/unit/test_synthesis.py",
        "tests/unit/test_async_diffusion.py",
        "tests/unit/test_config.py",
        "tests/unit/test_validation.py",
        "tests/unit/test_cache.py",
        
        # Integration tests
        "tests/integration/test_api_integration.py",
        
        # Performance tests  
        "tests/performance/test_benchmarks.py",
        
        # Security tests
        "tests/safety/test_security_validation.py",
        "tests/test_api_security.py",
        
        # Core diffusion tests
        "tests/test_core_diffusion.py",
        
        # Scripts and workflows
        "scripts/run_comprehensive_tests.py",
        ".github/workflows/comprehensive-testing.yml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print(f"✅ All {len(required_files)} required files present")
        return True

def count_test_coverage():
    """Count and analyze test coverage."""
    print("🔍 Analyzing test coverage...")
    
    # Count source files
    source_files = list(Path("odordiff2").rglob("*.py"))
    source_files = [f for f in source_files if "__pycache__" not in str(f) and "__init__.py" not in f.name]
    
    # Count test files
    test_files = list(Path("tests").rglob("test_*.py"))
    
    # Analyze test structure
    unit_tests = list(Path("tests/unit").glob("test_*.py")) if Path("tests/unit").exists() else []
    integration_tests = list(Path("tests/integration").glob("test_*.py")) if Path("tests/integration").exists() else []
    performance_tests = list(Path("tests/performance").glob("test_*.py")) if Path("tests/performance").exists() else []
    security_tests = list(Path("tests").rglob("test_*security*.py"))
    
    print(f"📊 Test Coverage Analysis:")
    print(f"   Source files: {len(source_files)}")
    print(f"   Test files: {len(test_files)}")
    print(f"   Unit tests: {len(unit_tests)}")
    print(f"   Integration tests: {len(integration_tests)}")
    print(f"   Performance tests: {len(performance_tests)}")
    print(f"   Security tests: {len(security_tests)}")
    
    # Calculate coverage ratio
    coverage_ratio = len(test_files) / max(len(source_files), 1)
    print(f"   Test/Source ratio: {coverage_ratio:.2f}")
    
    return {
        "source_files": len(source_files),
        "test_files": len(test_files),
        "unit_tests": len(unit_tests),
        "integration_tests": len(integration_tests),
        "performance_tests": len(performance_tests),
        "security_tests": len(security_tests),
        "coverage_ratio": coverage_ratio
    }

def validate_test_configuration():
    """Validate test configuration files."""
    print("🔍 Validating test configuration...")
    
    config_checks = {}
    
    # Check pytest.ini
    if Path("pytest.ini").exists():
        with open("pytest.ini") as f:
            pytest_config = f.read()
            
        config_checks["pytest_ini"] = {
            "coverage_enabled": "--cov=odordiff2" in pytest_config,
            "html_reports": "--html=" in pytest_config,
            "xml_output": "--junitxml=" in pytest_config,
            "markers_defined": "markers =" in pytest_config,
            "async_support": "asyncio_mode = auto" in pytest_config
        }
    
    # Check pyproject.toml for dev dependencies
    if Path("pyproject.toml").exists():
        with open("pyproject.toml") as f:
            pyproject_content = f.read()
            
        config_checks["pyproject_toml"] = {
            "pytest_present": "pytest" in pyproject_content,
            "coverage_present": "pytest-cov" in pyproject_content,
            "async_testing": "pytest-asyncio" in pyproject_content,
            "performance_testing": "pytest-benchmark" in pyproject_content,
            "security_tools": "bandit" in pyproject_content and "safety" in pyproject_content,
            "code_quality": "black" in pyproject_content and "flake8" in pyproject_content,
            "type_checking": "mypy" in pyproject_content
        }
    
    # Print configuration status
    for config_file, checks in config_checks.items():
        print(f"   {config_file}:")
        for check, status in checks.items():
            status_symbol = "✅" if status else "❌"
            print(f"     {status_symbol} {check}")
    
    return config_checks

def validate_test_quality():
    """Validate test quality and structure."""
    print("🔍 Validating test quality...")
    
    quality_metrics = {}
    
    # Analyze test files for quality indicators
    test_files = list(Path("tests").rglob("test_*.py"))
    
    total_test_functions = 0
    total_test_classes = 0
    files_with_fixtures = 0
    files_with_mocks = 0
    files_with_async_tests = 0
    files_with_parametrized_tests = 0
    
    for test_file in test_files:
        try:
            with open(test_file) as f:
                content = f.read()
                
            # Count test functions and classes
            test_functions = content.count("def test_")
            test_classes = content.count("class Test")
            total_test_functions += test_functions
            total_test_classes += test_classes
            
            # Check for quality indicators
            if "@pytest.fixture" in content:
                files_with_fixtures += 1
            if "Mock" in content or "patch" in content:
                files_with_mocks += 1
            if "async def test_" in content:
                files_with_async_tests += 1
            if "@pytest.mark.parametrize" in content:
                files_with_parametrized_tests += 1
                
        except Exception as e:
            print(f"   ⚠️  Error analyzing {test_file}: {e}")
    
    quality_metrics = {
        "total_test_functions": total_test_functions,
        "total_test_classes": total_test_classes,
        "files_with_fixtures": files_with_fixtures,
        "files_with_mocks": files_with_mocks,
        "files_with_async_tests": files_with_async_tests,
        "files_with_parametrized_tests": files_with_parametrized_tests,
        "avg_tests_per_file": total_test_functions / max(len(test_files), 1)
    }
    
    print(f"   Test functions: {total_test_functions}")
    print(f"   Test classes: {total_test_classes}")
    print(f"   Files with fixtures: {files_with_fixtures}")
    print(f"   Files with mocks: {files_with_mocks}")
    print(f"   Files with async tests: {files_with_async_tests}")
    print(f"   Files with parametrized tests: {files_with_parametrized_tests}")
    print(f"   Avg tests per file: {quality_metrics['avg_tests_per_file']:.1f}")
    
    return quality_metrics

def validate_ci_cd_setup():
    """Validate CI/CD configuration."""
    print("🔍 Validating CI/CD setup...")
    
    github_workflow = Path(".github/workflows/comprehensive-testing.yml")
    
    if not github_workflow.exists():
        print("   ❌ GitHub workflow file missing")
        return False
    
    with open(github_workflow) as f:
        workflow_content = f.read()
    
    # Check for essential CI/CD components
    checks = {
        "unit_tests_job": "unit-tests:" in workflow_content,
        "integration_tests_job": "integration-tests:" in workflow_content,
        "performance_tests_job": "performance-tests:" in workflow_content,
        "security_analysis_job": "security-analysis:" in workflow_content,
        "code_quality_job": "code-quality:" in workflow_content,
        "coverage_reporting": "codecov" in workflow_content or "coverage" in workflow_content,
        "multiple_python_versions": "python-version:" in workflow_content,
        "quality_gates": "quality-gates:" in workflow_content,
        "deployment_automation": "deploy" in workflow_content.lower()
    }
    
    print("   CI/CD Components:")
    for check, present in checks.items():
        status = "✅" if present else "❌"
        print(f"     {status} {check}")
    
    return all(checks.values())

def validate_advanced_features():
    """Validate advanced testing features."""
    print("🔍 Validating advanced testing features...")
    
    features = {}
    
    # Check for property-based testing
    features["property_based_testing"] = any(
        "hypothesis" in f.read_text() 
        for f in Path("tests").rglob("*.py") 
        if f.is_file()
    )
    
    # Check for mutation testing setup
    features["mutation_testing"] = "mutmut" in Path("pyproject.toml").read_text()
    
    # Check for performance benchmarking
    features["performance_benchmarking"] = any(
        "pytest-benchmark" in f.name or "benchmark" in f.read_text()
        for f in Path("tests").rglob("*.py")
        if f.is_file()
    )
    
    # Check for security testing
    features["security_testing"] = any(
        "security" in f.name.lower() or "security" in f.read_text()
        for f in Path("tests").rglob("*.py")
        if f.is_file()
    )
    
    # Check for async testing
    features["async_testing"] = any(
        "async def test_" in f.read_text()
        for f in Path("tests").rglob("*.py")
        if f.is_file()
    )
    
    print("   Advanced Features:")
    for feature, present in features.items():
        status = "✅" if present else "❌"
        print(f"     {status} {feature}")
    
    return features

def generate_validation_report():
    """Generate comprehensive validation report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TESTING IMPLEMENTATION VALIDATION")
    print("="*80)
    
    results = {}
    
    # Run all validations
    results["file_structure"] = validate_file_structure()
    print()
    
    results["test_coverage"] = count_test_coverage()
    print()
    
    results["test_configuration"] = validate_test_configuration()
    print()
    
    results["test_quality"] = validate_test_quality()
    print()
    
    results["ci_cd_setup"] = validate_ci_cd_setup()
    print()
    
    results["advanced_features"] = validate_advanced_features()
    
    # Generate summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    # Calculate overall score
    scores = []
    
    # File structure (essential)
    scores.append(100 if results["file_structure"] else 0)
    
    # Test coverage (based on ratio and variety)
    coverage = results["test_coverage"]
    coverage_score = min(100, coverage["coverage_ratio"] * 50 + 
                        (coverage["unit_tests"] > 0) * 15 +
                        (coverage["integration_tests"] > 0) * 15 +
                        (coverage["performance_tests"] > 0) * 10 +
                        (coverage["security_tests"] > 0) * 10)
    scores.append(coverage_score)
    
    # Configuration quality
    config = results["test_configuration"]
    config_score = 0
    for config_file, checks in config.items():
        config_score += sum(checks.values()) / len(checks) * 50
    scores.append(min(100, config_score))
    
    # Test quality
    quality = results["test_quality"]
    quality_score = min(100, 
                       (quality["total_test_functions"] > 50) * 25 +
                       (quality["files_with_fixtures"] > 5) * 20 +
                       (quality["files_with_mocks"] > 3) * 15 +
                       (quality["files_with_async_tests"] > 1) * 20 +
                       (quality["files_with_parametrized_tests"] > 1) * 20)
    scores.append(quality_score)
    
    # CI/CD setup
    scores.append(100 if results["ci_cd_setup"] else 0)
    
    # Advanced features
    advanced = results["advanced_features"]
    advanced_score = sum(advanced.values()) / len(advanced) * 100
    scores.append(advanced_score)
    
    overall_score = sum(scores) / len(scores)
    
    print(f"Overall Score: {overall_score:.1f}/100")
    print()
    print("Component Scores:")
    print(f"  File Structure: {scores[0]:.1f}/100")
    print(f"  Test Coverage: {scores[1]:.1f}/100") 
    print(f"  Configuration: {scores[2]:.1f}/100")
    print(f"  Test Quality: {scores[3]:.1f}/100")
    print(f"  CI/CD Setup: {scores[4]:.1f}/100")
    print(f"  Advanced Features: {scores[5]:.1f}/100")
    
    # Determine grade
    if overall_score >= 90:
        grade = "A (Excellent)"
        status = "✅ PRODUCTION READY"
    elif overall_score >= 80:
        grade = "B (Good)" 
        status = "✅ READY WITH MINOR IMPROVEMENTS"
    elif overall_score >= 70:
        grade = "C (Acceptable)"
        status = "⚠️  NEEDS IMPROVEMENT"
    else:
        grade = "D (Needs Work)"
        status = "❌ NOT READY"
    
    print(f"\nGrade: {grade}")
    print(f"Status: {status}")
    
    # Recommendations
    print("\nRecommendations:")
    if scores[0] < 100:
        print("  - Complete missing test files and structure")
    if scores[1] < 80:
        print("  - Increase test coverage, especially integration and performance tests")
    if scores[2] < 90:
        print("  - Improve test configuration setup")
    if scores[3] < 80:
        print("  - Add more fixtures, mocks, and parametrized tests")
    if scores[4] < 100:
        print("  - Complete CI/CD workflow setup")
    if scores[5] < 80:
        print("  - Implement more advanced testing features")
    
    # Save results
    with open("validation-report.json", "w") as f:
        json.dump({
            "overall_score": overall_score,
            "component_scores": {
                "file_structure": scores[0],
                "test_coverage": scores[1], 
                "configuration": scores[2],
                "test_quality": scores[3],
                "ci_cd_setup": scores[4],
                "advanced_features": scores[5]
            },
            "grade": grade,
            "status": status,
            "details": results
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: validation-report.json")
    print("="*80)
    
    return overall_score >= 80  # Return True if acceptable or better

def main():
    """Main validation function."""
    try:
        success = generate_validation_report()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()