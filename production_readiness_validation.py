#!/usr/bin/env python3
"""
Production Readiness Validation for OdorDiff-2 System

This script validates the production readiness of the entire OdorDiff-2 system
by checking code quality, security, performance, and deployment readiness.
"""

import os
import sys
import json
import subprocess
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path


class ProductionReadinessValidator:
    """Comprehensive production readiness validation."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {}
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all production readiness checks."""
        
        print("🔍 Starting Production Readiness Validation...")
        start_time = time.time()
        
        # Core validation checks
        self.results['project_structure'] = self.validate_project_structure()
        self.results['code_quality'] = self.validate_code_quality()
        self.results['security'] = self.validate_security()
        self.results['documentation'] = self.validate_documentation()
        self.results['testing'] = self.validate_testing()
        self.results['deployment'] = self.validate_deployment_readiness()
        self.results['performance'] = self.validate_performance_readiness()
        self.results['compliance'] = self.validate_compliance()
        
        # Calculate overall score
        self.results['overall'] = self.calculate_overall_score()
        
        total_time = time.time() - start_time
        self.results['metadata'] = {
            'validation_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'validator_version': '1.0.0'
        }
        
        print(f"✅ Production Readiness Validation Complete ({total_time:.2f}s)")
        return self.results
    
    def validate_project_structure(self) -> Dict[str, Any]:
        """Validate project structure and organization."""
        print("  📁 Validating project structure...")
        
        required_files = [
            'pyproject.toml',
            'requirements.txt', 
            'README.md',
            'LICENSE',
            'Dockerfile',
            'docker-compose.yml',
            'odordiff2/__init__.py'
        ]
        
        required_dirs = [
            'odordiff2/core',
            'odordiff2/api', 
            'odordiff2/safety',
            'odordiff2/models',
            'odordiff2/research',
            'tests',
            'deployment',
            'config'
        ]
        
        file_checks = {}
        for file in required_files:
            file_checks[file] = (self.project_root / file).exists()
        
        dir_checks = {}
        for directory in required_dirs:
            dir_checks[directory] = (self.project_root / directory).exists()
        
        # Check code organization
        python_files = list(self.project_root.rglob("*.py"))
        total_python_files = len(python_files)
        
        # Calculate lines of code
        total_loc = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_loc += len(f.readlines())
            except:
                pass
        
        structure_score = (
            sum(file_checks.values()) / len(required_files) * 0.4 +
            sum(dir_checks.values()) / len(required_dirs) * 0.4 +
            (1.0 if total_loc > 1000 else total_loc / 1000) * 0.2
        )
        
        return {
            'score': structure_score,
            'required_files': file_checks,
            'required_directories': dir_checks,
            'python_files_count': total_python_files,
            'total_lines_of_code': total_loc,
            'pass': structure_score >= 0.8
        }
    
    def validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality metrics."""
        print("  🎯 Validating code quality...")
        
        quality_metrics = {}
        
        # Check for docstrings in Python files
        python_files = list(self.project_root.rglob("*.py"))
        files_with_docstrings = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        files_with_docstrings += 1
            except:
                pass
        
        docstring_coverage = files_with_docstrings / max(1, len(python_files))
        
        # Check for type hints
        files_with_type_hints = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'typing' in content or '->' in content or ': str' in content:
                        files_with_type_hints += 1
            except:
                pass
        
        type_hint_coverage = files_with_type_hints / max(1, len(python_files))
        
        # Check for error handling
        files_with_error_handling = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'try:' in content and 'except' in content:
                        files_with_error_handling += 1
            except:
                pass
        
        error_handling_coverage = files_with_error_handling / max(1, len(python_files))
        
        # Check for logging
        files_with_logging = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'logging' in content or 'logger' in content:
                        files_with_logging += 1
            except:
                pass
        
        logging_coverage = files_with_logging / max(1, len(python_files))
        
        # Overall code quality score
        quality_score = (
            docstring_coverage * 0.3 +
            type_hint_coverage * 0.25 +
            error_handling_coverage * 0.25 +
            logging_coverage * 0.2
        )
        
        return {
            'score': quality_score,
            'docstring_coverage': docstring_coverage,
            'type_hint_coverage': type_hint_coverage,
            'error_handling_coverage': error_handling_coverage,
            'logging_coverage': logging_coverage,
            'python_files_analyzed': len(python_files),
            'pass': quality_score >= 0.7
        }
    
    def validate_security(self) -> Dict[str, Any]:
        """Validate security measures."""
        print("  🔒 Validating security...")
        
        security_checks = {}
        
        # Check for security-related files
        security_files = [
            'odordiff2/security/__init__.py',
            'odordiff2/security/authentication.py',
            'odordiff2/security/rate_limiting.py'
        ]
        
        security_files_present = sum(
            1 for file in security_files if (self.project_root / file).exists()
        )
        
        # Check for security imports in code
        python_files = list(self.project_root.rglob("*.py"))
        files_with_security_imports = 0
        
        security_patterns = [
            'from cryptography',
            'import secrets',
            'from passlib',
            'import hashlib',
            'from jwt',
            'rate_limit',
            'authentication',
            'validate_',
            'sanitize_'
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(pattern in content for pattern in security_patterns):
                        files_with_security_imports += 1
            except:
                pass
        
        security_implementation_coverage = files_with_security_imports / max(1, len(python_files))
        
        # Check for hardcoded secrets (basic check)
        files_with_potential_secrets = 0
        secret_patterns = ['password', 'secret', 'key', 'token', 'api_key']
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(f'{pattern}=' in content or f'{pattern}:' in content for pattern in secret_patterns):
                        files_with_potential_secrets += 1
            except:
                pass
        
        # Security score (lower is better for secrets, higher for security measures)
        security_score = (
            (security_files_present / len(security_files)) * 0.4 +
            security_implementation_coverage * 0.4 +
            max(0, 1 - files_with_potential_secrets / max(1, len(python_files))) * 0.2
        )
        
        return {
            'score': security_score,
            'security_files_present': security_files_present,
            'security_implementation_coverage': security_implementation_coverage,
            'potential_hardcoded_secrets': files_with_potential_secrets,
            'pass': security_score >= 0.7 and files_with_potential_secrets < 5
        }
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness."""
        print("  📚 Validating documentation...")
        
        # Check for documentation files
        doc_files = [
            'README.md',
            'DEPLOYMENT.md',
            'CI_WORKFLOW.md',
            'FINAL_RESEARCH_PUBLICATION_REPORT.md'
        ]
        
        doc_files_present = sum(
            1 for file in doc_files if (self.project_root / file).exists()
        )
        
        # Check README quality
        readme_quality = 0
        readme_path = self.project_root / 'README.md'
        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                    
                quality_indicators = [
                    'Installation',
                    'Usage', 
                    'Example',
                    'API',
                    'Contributing',
                    'License',
                    '```python',  # Code examples
                    '## ',  # Section headers
                    'pip install',
                    'docker'
                ]
                
                readme_quality = sum(
                    1 for indicator in quality_indicators 
                    if indicator.lower() in readme_content.lower()
                ) / len(quality_indicators)
            except:
                pass
        
        # Check for API documentation
        api_docs_present = any(
            (self.project_root / path).exists() 
            for path in ['docs/', 'api-docs/', 'openapi.json']
        )
        
        doc_score = (
            (doc_files_present / len(doc_files)) * 0.4 +
            readme_quality * 0.4 +
            (1.0 if api_docs_present else 0.0) * 0.2
        )
        
        return {
            'score': doc_score,
            'documentation_files_present': doc_files_present,
            'readme_quality': readme_quality,
            'api_docs_present': api_docs_present,
            'pass': doc_score >= 0.7
        }
    
    def validate_testing(self) -> Dict[str, Any]:
        """Validate testing infrastructure."""
        print("  🧪 Validating testing...")
        
        # Check for test files
        test_files = list(self.project_root.rglob("test_*.py"))
        test_dirs = [
            'tests/',
            'test/',
            'testing/'
        ]
        
        test_dirs_present = sum(
            1 for test_dir in test_dirs if (self.project_root / test_dir).exists()
        )
        
        # Check for testing configuration
        test_config_files = [
            'pytest.ini',
            'pyproject.toml',  # pytest config in pyproject.toml
            'tox.ini',
            'setup.cfg'
        ]
        
        test_config_present = sum(
            1 for config in test_config_files if (self.project_root / config).exists()
        )
        
        # Check for CI/CD files
        ci_files = [
            '.github/workflows/',
            '.gitlab-ci.yml',
            'Jenkinsfile',
            'CI_WORKFLOW.md'
        ]
        
        ci_present = sum(
            1 for ci_file in ci_files if (self.project_root / ci_file).exists()
        )
        
        testing_score = (
            min(1.0, len(test_files) / 10) * 0.4 +  # At least 10 test files
            min(1.0, test_dirs_present) * 0.3 +
            min(1.0, test_config_present / len(test_config_files)) * 0.2 +
            min(1.0, ci_present) * 0.1
        )
        
        return {
            'score': testing_score,
            'test_files_count': len(test_files),
            'test_directories_present': test_dirs_present,
            'test_config_present': test_config_present,
            'ci_present': ci_present,
            'pass': testing_score >= 0.6
        }
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness."""
        print("  🚀 Validating deployment readiness...")
        
        # Check for containerization
        container_files = [
            'Dockerfile',
            'docker-compose.yml',
            '.dockerignore'
        ]
        
        container_files_present = sum(
            1 for file in container_files if (self.project_root / file).exists()
        )
        
        # Check for Kubernetes deployment
        k8s_files = list(self.project_root.rglob("*.yaml")) + list(self.project_root.rglob("*.yml"))
        k8s_deployment_files = [
            f for f in k8s_files 
            if any(keyword in f.name.lower() for keyword in ['deployment', 'service', 'ingress', 'configmap'])
        ]
        
        # Check for infrastructure as code
        infrastructure_files = [
            'deployment/',
            'terraform/',
            'helm/',
            'k8s/',
            'kubernetes/'
        ]
        
        infrastructure_present = sum(
            1 for infra in infrastructure_files if (self.project_root / infra).exists()
        )
        
        # Check for environment configuration
        env_files = [
            'config/',
            '.env.example',
            'config.yaml',
            'settings.py'
        ]
        
        env_config_present = sum(
            1 for env_file in env_files if (self.project_root / env_file).exists()
        )
        
        deployment_score = (
            (container_files_present / len(container_files)) * 0.3 +
            min(1.0, len(k8s_deployment_files) / 3) * 0.3 +
            min(1.0, infrastructure_present / len(infrastructure_files)) * 0.2 +
            min(1.0, env_config_present / len(env_files)) * 0.2
        )
        
        return {
            'score': deployment_score,
            'container_files_present': container_files_present,
            'kubernetes_files_count': len(k8s_deployment_files),
            'infrastructure_dirs_present': infrastructure_present,
            'environment_config_present': env_config_present,
            'pass': deployment_score >= 0.7
        }
    
    def validate_performance_readiness(self) -> Dict[str, Any]:
        """Validate performance and monitoring readiness."""
        print("  ⚡ Validating performance readiness...")
        
        # Check for monitoring/observability
        monitoring_files = [
            'odordiff2/monitoring/',
            'prometheus.yml',
            'grafana-dashboards.json',
            'monitoring/'
        ]
        
        monitoring_present = sum(
            1 for mon_file in monitoring_files if (self.project_root / mon_file).exists()
        )
        
        # Check for performance optimization
        perf_files = [
            'odordiff2/performance/',
            'odordiff2/scaling/',
            'requirements.txt'  # Should include performance libraries
        ]
        
        perf_present = sum(
            1 for perf_file in perf_files if (self.project_root / perf_file).exists()
        )
        
        # Check for caching implementation
        cache_indicators = 0
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(cache_term in content.lower() for cache_term in ['cache', 'redis', 'memcached']):
                        cache_indicators += 1
                        break
            except:
                pass
        
        performance_score = (
            (monitoring_present / len(monitoring_files)) * 0.4 +
            (perf_present / len(perf_files)) * 0.4 +
            min(1.0, cache_indicators) * 0.2
        )
        
        return {
            'score': performance_score,
            'monitoring_components_present': monitoring_present,
            'performance_components_present': perf_present,
            'caching_implemented': cache_indicators > 0,
            'pass': performance_score >= 0.6
        }
    
    def validate_compliance(self) -> Dict[str, Any]:
        """Validate regulatory and compliance readiness."""
        print("  ⚖️ Validating compliance...")
        
        # Check for license
        license_present = (self.project_root / 'LICENSE').exists()
        
        # Check for safety/security documentation
        compliance_files = [
            'SECURITY.md',
            'PRIVACY.md', 
            'COMPLIANCE.md',
            'TERMS.md'
        ]
        
        compliance_docs = sum(
            1 for comp_file in compliance_files if (self.project_root / comp_file).exists()
        )
        
        # Check for safety implementations
        safety_modules = [
            'odordiff2/safety/',
            'odordiff2/security/'
        ]
        
        safety_implementations = sum(
            1 for safety_mod in safety_modules if (self.project_root / safety_mod).exists()
        )
        
        compliance_score = (
            (1.0 if license_present else 0.0) * 0.4 +
            (compliance_docs / len(compliance_files)) * 0.3 +
            (safety_implementations / len(safety_modules)) * 0.3
        )
        
        return {
            'score': compliance_score,
            'license_present': license_present,
            'compliance_docs_present': compliance_docs,
            'safety_implementations': safety_implementations,
            'pass': compliance_score >= 0.6 and license_present
        }
    
    def calculate_overall_score(self) -> Dict[str, Any]:
        """Calculate overall production readiness score."""
        
        category_weights = {
            'project_structure': 0.15,
            'code_quality': 0.20,
            'security': 0.15,
            'documentation': 0.10,
            'testing': 0.15,
            'deployment': 0.15,
            'performance': 0.05,
            'compliance': 0.05
        }
        
        weighted_score = sum(
            self.results[category]['score'] * weight
            for category, weight in category_weights.items()
        )
        
        # Check if all critical areas pass
        critical_categories = ['project_structure', 'code_quality', 'security', 'deployment']
        all_critical_pass = all(
            self.results[category]['pass'] for category in critical_categories
        )
        
        # Production readiness levels
        if weighted_score >= 0.9 and all_critical_pass:
            readiness_level = "PRODUCTION_READY"
        elif weighted_score >= 0.8 and all_critical_pass:
            readiness_level = "NEAR_PRODUCTION_READY"
        elif weighted_score >= 0.7:
            readiness_level = "DEVELOPMENT_READY"
        else:
            readiness_level = "NOT_READY"
        
        return {
            'overall_score': weighted_score,
            'readiness_level': readiness_level,
            'all_critical_pass': all_critical_pass,
            'category_scores': {
                category: self.results[category]['score'] 
                for category in category_weights.keys()
            },
            'recommendations': self.generate_recommendations()
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for category, result in self.results.items():
            if category in ['metadata', 'overall']:
                continue
                
            if not result.get('pass', True):
                if category == 'project_structure':
                    recommendations.append("Improve project structure by ensuring all required files and directories are present")
                elif category == 'code_quality':
                    recommendations.append("Enhance code quality with better documentation, type hints, and error handling")
                elif category == 'security':
                    recommendations.append("Strengthen security measures and remove any hardcoded secrets")
                elif category == 'documentation':
                    recommendations.append("Improve documentation coverage and quality")
                elif category == 'testing':
                    recommendations.append("Expand testing coverage and CI/CD implementation")
                elif category == 'deployment':
                    recommendations.append("Complete deployment configuration with containers and infrastructure")
                elif category == 'performance':
                    recommendations.append("Implement performance monitoring and optimization features")
                elif category == 'compliance':
                    recommendations.append("Ensure compliance documentation and safety implementations")
        
        if not recommendations:
            recommendations.append("System appears production-ready! Consider continuous monitoring and improvements.")
        
        return recommendations
    
    def save_results(self, filename: str = "production_readiness_report.json"):
        """Save validation results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to: {filename}")
    
    def print_summary(self):
        """Print validation summary."""
        if 'overall' not in self.results:
            return
            
        overall = self.results['overall']
        
        print(f"\n{'='*60}")
        print(f"🎯 PRODUCTION READINESS SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Score: {overall['overall_score']:.2f}/1.00")
        print(f"Readiness Level: {overall['readiness_level']}")
        print(f"Critical Areas Pass: {'✅' if overall['all_critical_pass'] else '❌'}")
        
        print(f"\n📊 Category Scores:")
        for category, score in overall['category_scores'].items():
            status = '✅' if self.results[category]['pass'] else '❌'
            print(f"  {status} {category.replace('_', ' ').title()}: {score:.2f}")
        
        if overall['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(overall['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print(f"{'='*60}")


def main():
    """Run production readiness validation."""
    validator = ProductionReadinessValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Print summary
    validator.print_summary()
    
    # Save results
    validator.save_results()
    
    return results


if __name__ == "__main__":
    main()