#!/usr/bin/env python3
"""
Comprehensive quality check script for OdorDiff-2.
"""

import os
import sys
import subprocess
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import time


class QualityChecker:
    """Comprehensive quality analysis system."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {}
        
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all quality checks."""
        print("🔍 Starting comprehensive quality analysis...")
        
        checks = [
            ("Code Structure", self.check_code_structure),
            ("Security Analysis", self.check_security),
            ("Performance Analysis", self.check_performance),
            ("Documentation", self.check_documentation),
            ("Test Coverage", self.analyze_test_coverage),
            ("Code Quality", self.check_code_quality),
            ("Dependencies", self.check_dependencies)
        ]
        
        for check_name, check_func in checks:
            print(f"\n📊 Running {check_name} check...")
            try:
                self.results[check_name.lower().replace(" ", "_")] = check_func()
                print(f"✅ {check_name} check completed")
            except Exception as e:
                print(f"❌ {check_name} check failed: {e}")
                self.results[check_name.lower().replace(" ", "_")] = {"error": str(e)}
        
        return self.results
    
    def check_code_structure(self) -> Dict[str, Any]:
        """Analyze code structure and organization."""
        structure = {}
        
        # Count files by type
        file_counts = {}
        total_lines = 0
        
        for file_path in self.project_root.rglob("*.py"):
            if "/__pycache__/" in str(file_path) or "/.git/" in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    
                    # Categorize by directory
                    relative_path = file_path.relative_to(self.project_root)
                    category = str(relative_path.parts[0]) if len(relative_path.parts) > 1 else 'root'
                    
                    if category not in file_counts:
                        file_counts[category] = {'files': 0, 'lines': 0}
                    
                    file_counts[category]['files'] += 1
                    file_counts[category]['lines'] += len(lines)
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        structure['file_distribution'] = file_counts
        structure['total_python_files'] = sum(cat['files'] for cat in file_counts.values())
        structure['total_lines_of_code'] = total_lines
        
        # Check for key files
        key_files = [
            'pyproject.toml', 'requirements.txt', 'README.md', 
            'LICENSE', 'Dockerfile', '.gitignore'
        ]
        
        structure['key_files_present'] = {}
        for key_file in key_files:
            structure['key_files_present'][key_file] = (self.project_root / key_file).exists()
        
        return structure
    
    def check_security(self) -> Dict[str, Any]:
        """Perform security analysis."""
        security = {
            'potential_vulnerabilities': [],
            'security_features_implemented': [],
            'recommendations': []
        }
        
        # Check for security patterns in code
        security_patterns = {
            'sql_injection_protection': [r'validate_input', r'sanitize', r'prepared_statement'],
            'xss_protection': [r'escape_html', r'sanitize_input', r'Content-Type.*nosniff'],
            'csrf_protection': [r'csrf_token', r'CSRFProtect'],
            'rate_limiting': [r'RateLimiter', r'rate_limit', r'@rate_limit'],
            'input_validation': [r'validator', r'validate_.*input', r'InputValidator'],
            'authentication': [r'auth', r'authenticate', r'APIKey', r'Bearer'],
            'encryption': [r'encrypt', r'hash', r'bcrypt', r'pbkdf2']
        }
        
        implemented_features = set()
        
        for py_file in self.project_root.rglob("*.py"):
            if "/__pycache__/" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for security features
                    for feature, patterns in security_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                implemented_features.add(feature)
                    
                    # Check for potential vulnerabilities
                    vuln_patterns = [
                        (r'eval\s*\(', 'Potential code injection via eval()'),
                        (r'exec\s*\(', 'Potential code injection via exec()'),
                        (r'shell=True', 'Potential command injection via shell=True'),
                        (r'input\s*\(["\'].*["\']\)', 'Raw input() usage'),
                        (r'pickle\.loads?', 'Unsafe pickle deserialization'),
                        (r'yaml\.load\(', 'Unsafe YAML loading')
                    ]
                    
                    for pattern, description in vuln_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            security['potential_vulnerabilities'].append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'pattern': pattern,
                                'description': description,
                                'matches': len(matches)
                            })
                            
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
        
        security['security_features_implemented'] = list(implemented_features)
        
        # Security score
        max_features = len(security_patterns)
        implemented_count = len(implemented_features)
        security['security_score'] = (implemented_count / max_features) * 100
        
        return security
    
    def check_performance(self) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        performance = {
            'async_usage': 0,
            'caching_implementation': False,
            'performance_monitoring': False,
            'optimization_features': []
        }
        
        perf_patterns = {
            'async_code': [r'async\s+def', r'await\s+', r'asyncio'],
            'caching': [r'cache', r'Cache', r'@lru_cache', r'redis', r'memcached'],
            'monitoring': [r'metric', r'monitor', r'performance', r'@time_function'],
            'optimization': [r'pool', r'batch', r'concurrent', r'threading', r'multiprocess']
        }
        
        for py_file in self.project_root.rglob("*.py"):
            if "/__pycache__/" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Count async usage
                    performance['async_usage'] += len(re.findall(r'async\s+def', content))
                    
                    # Check for performance features
                    for feature, patterns in perf_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                if feature == 'caching':
                                    performance['caching_implementation'] = True
                                elif feature == 'monitoring':
                                    performance['performance_monitoring'] = True
                                elif feature == 'optimization':
                                    performance['optimization_features'].append(pattern)
                                    
            except Exception as e:
                print(f"Error analyzing performance in {py_file}: {e}")
        
        # Remove duplicates
        performance['optimization_features'] = list(set(performance['optimization_features']))
        
        return performance
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation quality."""
        docs = {
            'readme_present': False,
            'docstring_coverage': 0,
            'api_documentation': False,
            'inline_comments': 0
        }
        
        # Check for README
        readme_files = ['README.md', 'README.rst', 'README.txt']
        for readme in readme_files:
            if (self.project_root / readme).exists():
                docs['readme_present'] = True
                break
        
        # Analyze docstring coverage
        total_functions = 0
        documented_functions = 0
        total_comments = 0
        
        for py_file in self.project_root.rglob("*.py"):
            if "/__pycache__/" in str(py_file) or "/tests/" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Count comments
                    total_comments += len(re.findall(r'#.*', content))
                    
                    # Parse AST for function analysis
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                total_functions += 1
                                # Check if function has docstring
                                if (node.body and 
                                    isinstance(node.body[0], ast.Expr) and 
                                    isinstance(node.body[0].value, ast.Constant) and 
                                    isinstance(node.body[0].value.value, str)):
                                    documented_functions += 1
                    except SyntaxError:
                        pass
                        
            except Exception as e:
                print(f"Error analyzing documentation in {py_file}: {e}")
        
        if total_functions > 0:
            docs['docstring_coverage'] = (documented_functions / total_functions) * 100
        
        docs['inline_comments'] = total_comments
        docs['total_functions'] = total_functions
        docs['documented_functions'] = documented_functions
        
        # Check for API docs
        api_doc_indicators = ['swagger', 'openapi', 'docs_url', 'redoc_url']
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for indicator in api_doc_indicators:
                        if indicator in content.lower():
                            docs['api_documentation'] = True
                            break
            except:
                continue
        
        return docs
    
    def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage (simplified)."""
        test_analysis = {
            'test_files_found': 0,
            'test_functions_count': 0,
            'test_categories': {},
            'estimated_coverage': 0
        }
        
        # Count test files
        test_patterns = ['test_*.py', '*_test.py']
        test_files = []
        
        for pattern in test_patterns:
            test_files.extend(list(self.project_root.rglob(pattern)))
        
        test_analysis['test_files_found'] = len(test_files)
        
        # Analyze test content
        total_test_functions = 0
        test_types = {}
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Count test functions
                    test_funcs = re.findall(r'def\s+test_\w+', content)
                    total_test_functions += len(test_funcs)
                    
                    # Categorize tests
                    categories = {
                        'unit': [r'test_.*unit', r'TestUnit', r'unit.*test'],
                        'integration': [r'test_.*integration', r'TestIntegration', r'integration.*test'],
                        'performance': [r'test_.*performance', r'TestPerformance', r'performance.*test'],
                        'security': [r'test_.*security', r'TestSecurity', r'security.*test']
                    }
                    
                    for category, patterns in categories.items():
                        for pattern in patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                test_types[category] = test_types.get(category, 0) + 1
                                
            except Exception as e:
                print(f"Error analyzing test file {test_file}: {e}")
        
        test_analysis['test_functions_count'] = total_test_functions
        test_analysis['test_categories'] = test_types
        
        # Estimate coverage based on test/code ratio
        total_code_functions = self.results.get('documentation', {}).get('total_functions', 1)
        if total_code_functions > 0:
            coverage_ratio = min(total_test_functions / total_code_functions, 1.0)
            test_analysis['estimated_coverage'] = coverage_ratio * 100
        
        return test_analysis
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        quality = {
            'complexity_analysis': {},
            'code_smells': [],
            'best_practices': {
                'type_hints': 0,
                'error_handling': 0,
                'logging_usage': 0
            }
        }
        
        total_files = 0
        total_complexity = 0
        
        for py_file in self.project_root.rglob("*.py"):
            if "/__pycache__/" in str(py_file) or "/tests/" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_files += 1
                    
                    # Estimate cyclomatic complexity (simplified)
                    complexity_indicators = [
                        'if ', 'elif ', 'else:', 'for ', 'while ',
                        'try:', 'except:', 'and ', 'or ', '?'
                    ]
                    
                    file_complexity = 1  # Base complexity
                    for indicator in complexity_indicators:
                        file_complexity += content.count(indicator)
                    
                    total_complexity += file_complexity
                    
                    # Check best practices
                    if '->' in content or ': int' in content or ': str' in content:
                        quality['best_practices']['type_hints'] += 1
                    
                    if 'except ' in content or 'try:' in content:
                        quality['best_practices']['error_handling'] += 1
                    
                    if 'logger.' in content or 'logging.' in content:
                        quality['best_practices']['logging_usage'] += 1
                    
                    # Check for code smells
                    smells = []
                    
                    # Long functions (>50 lines)
                    functions = re.findall(r'def\s+\w+.*?(?=\ndef|\nclass|\n\n|\Z)', content, re.DOTALL)
                    for func in functions:
                        if len(func.split('\n')) > 50:
                            smells.append('Long function detected')
                    
                    # Deep nesting (>4 levels)
                    lines = content.split('\n')
                    for line in lines:
                        indent_level = (len(line) - len(line.lstrip())) // 4
                        if indent_level > 4:
                            smells.append('Deep nesting detected')
                            break
                    
                    if smells:
                        quality['code_smells'].extend([
                            {'file': str(py_file.relative_to(self.project_root)), 'smells': smells}
                        ])
                        
            except Exception as e:
                print(f"Error analyzing code quality in {py_file}: {e}")
        
        if total_files > 0:
            quality['complexity_analysis'] = {
                'average_complexity': total_complexity / total_files,
                'total_files_analyzed': total_files
            }
        
        return quality
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        deps = {
            'requirements_files': [],
            'dependency_count': 0,
            'security_analysis': {}
        }
        
        # Check for dependency files
        dep_files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile']
        
        for dep_file in dep_files:
            file_path = self.project_root / dep_file
            if file_path.exists():
                deps['requirements_files'].append(dep_file)
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                        if dep_file == 'requirements.txt':
                            # Count dependencies in requirements.txt
                            deps['dependency_count'] += len([
                                line for line in content.split('\n') 
                                if line.strip() and not line.startswith('#')
                            ])
                        elif dep_file == 'pyproject.toml':
                            # Count dependencies in pyproject.toml
                            deps['dependency_count'] += content.count('"')  # Simplified count
                            
                except Exception as e:
                    print(f"Error reading {dep_file}: {e}")
        
        return deps
    
    def generate_report(self) -> str:
        """Generate comprehensive quality report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🎯 ODORDIFF-2 COMPREHENSIVE QUALITY ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall scoring
        scores = {}
        
        # Code Structure Score
        structure = self.results.get('code_structure', {})
        if structure:
            structure_score = min(100, structure.get('total_python_files', 0) * 2)
            scores['Code Structure'] = structure_score
            
        # Security Score
        security = self.results.get('security_analysis', {})
        if security:
            security_score = security.get('security_score', 0)
            scores['Security'] = security_score
            
        # Documentation Score
        docs = self.results.get('documentation', {})
        if docs:
            doc_score = (
                (50 if docs.get('readme_present') else 0) +
                (docs.get('docstring_coverage', 0) * 0.5)
            )
            scores['Documentation'] = min(100, doc_score)
        
        # Test Coverage Score
        tests = self.results.get('test_coverage', {})
        if tests:
            test_score = tests.get('estimated_coverage', 0)
            scores['Testing'] = test_score
        
        # Overall Score
        if scores:
            overall_score = sum(scores.values()) / len(scores)
            scores['OVERALL'] = overall_score
        
        # Report sections
        report_lines.append("📊 OVERALL QUALITY SCORES")
        report_lines.append("-" * 40)
        for category, score in scores.items():
            emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            report_lines.append(f"{emoji} {category:<20}: {score:6.1f}%")
        
        report_lines.append("")
        
        # Detailed sections
        sections = [
            ("🏗️  CODE STRUCTURE", "code_structure"),
            ("🛡️  SECURITY ANALYSIS", "security_analysis"),
            ("⚡ PERFORMANCE", "performance_analysis"),
            ("📚 DOCUMENTATION", "documentation"),
            ("🧪 TEST COVERAGE", "test_coverage"),
            ("🎨 CODE QUALITY", "code_quality"),
            ("📦 DEPENDENCIES", "dependencies")
        ]
        
        for title, key in sections:
            data = self.results.get(key, {})
            if data and 'error' not in data:
                report_lines.append(title)
                report_lines.append("-" * len(title))
                report_lines.extend(self._format_section_data(data))
                report_lines.append("")
        
        # Recommendations
        report_lines.append("💡 RECOMMENDATIONS")
        report_lines.append("-" * 20)
        recommendations = self._generate_recommendations()
        report_lines.extend(recommendations)
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("🎉 Quality analysis complete!")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _format_section_data(self, data: Dict[str, Any]) -> List[str]:
        """Format section data for report."""
        lines = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"    {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"  {key}: {len(value)} items")
                for item in value[:3]:  # Show first 3 items
                    lines.append(f"    - {item}")
                if len(value) > 3:
                    lines.append(f"    ... and {len(value) - 3} more")
            else:
                lines.append(f"  {key}: {value}")
        
        return lines
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Security recommendations
        security = self.results.get('security_analysis', {})
        if security.get('security_score', 0) < 80:
            recommendations.append("🔒 Enhance security: Implement missing security features")
        
        if security.get('potential_vulnerabilities'):
            recommendations.append("⚠️  Address potential security vulnerabilities found")
        
        # Performance recommendations
        perf = self.results.get('performance_analysis', {})
        if not perf.get('caching_implementation'):
            recommendations.append("⚡ Add caching for better performance")
        
        if not perf.get('performance_monitoring'):
            recommendations.append("📊 Implement performance monitoring")
        
        # Documentation recommendations
        docs = self.results.get('documentation', {})
        if docs.get('docstring_coverage', 0) < 70:
            recommendations.append("📖 Improve docstring coverage (target: 70%+)")
        
        # Testing recommendations
        tests = self.results.get('test_coverage', {})
        if tests.get('estimated_coverage', 0) < 85:
            recommendations.append("🧪 Increase test coverage (target: 85%+)")
        
        # Code quality recommendations
        quality = self.results.get('code_quality', {})
        complexity = quality.get('complexity_analysis', {})
        if complexity.get('average_complexity', 0) > 20:
            recommendations.append("🎨 Reduce code complexity for better maintainability")
        
        if not recommendations:
            recommendations.append("✅ Great job! All quality metrics are within acceptable ranges.")
        
        return recommendations


def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    checker = QualityChecker(project_root)
    
    try:
        results = checker.run_all_checks()
        report = checker.generate_report()
        
        print("\n" + report)
        
        # Save detailed results to JSON
        with open("quality_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: quality_analysis_results.json")
        
        # Exit with appropriate code based on overall quality
        overall_score = sum([
            results.get('security_analysis', {}).get('security_score', 0),
            results.get('test_coverage', {}).get('estimated_coverage', 0),
            results.get('documentation', {}).get('docstring_coverage', 0)
        ]) / 3
        
        if overall_score >= 80:
            print("🎉 Quality gate: PASSED (Score: {:.1f}%)".format(overall_score))
            sys.exit(0)
        elif overall_score >= 60:
            print("⚠️  Quality gate: PARTIAL (Score: {:.1f}%)".format(overall_score))
            sys.exit(1)
        else:
            print("❌ Quality gate: FAILED (Score: {:.1f}%)".format(overall_score))
            sys.exit(2)
            
    except Exception as e:
        print(f"❌ Quality analysis failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()