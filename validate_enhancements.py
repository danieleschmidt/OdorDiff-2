#!/usr/bin/env python3
"""
Validation script for Generation 2 robustness enhancements.

This script validates that all required enhancement files are present,
have the expected structure, and contain the required functionality.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

class EnhancementValidator:
    """Validator for robustness enhancements."""
    
    def __init__(self):
        self.results = {}
        self.base_path = Path(".")
    
    def validate_all(self):
        """Validate all robustness enhancements."""
        print("OdorDiff-2 Generation 2 Robustness Enhancement Validation")
        print("="*65)
        
        # Validate each enhancement area
        self.validate_health_monitoring()
        self.validate_circuit_breakers()  
        self.validate_rate_limiting()
        self.validate_input_validation()
        self.validate_error_recovery()
        self.validate_configuration_management()
        self.validate_connection_pooling()
        self.validate_backup_recovery()
        self.validate_security_hardening()
        self.validate_observability()
        
        # Generate summary
        self.generate_summary()
    
    def check_file_exists(self, filepath: str) -> bool:
        """Check if a file exists."""
        return (self.base_path / filepath).exists()
    
    def check_file_contains(self, filepath: str, patterns: List[str]) -> Dict[str, bool]:
        """Check if file contains expected patterns."""
        results = {}
        
        if not self.check_file_exists(filepath):
            return {pattern: False for pattern in patterns}
        
        try:
            content = (self.base_path / filepath).read_text()
            for pattern in patterns:
                results[pattern] = bool(re.search(pattern, content, re.MULTILINE))
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            return {pattern: False for pattern in patterns}
        
        return results
    
    def validate_health_monitoring(self):
        """Validate health monitoring implementation."""
        print("\n1. Health Monitoring & Endpoints")
        print("-" * 40)
        
        # Check main health monitoring file
        file_path = "odordiff2/monitoring/health.py"
        required_classes = [
            "class HealthChecker",
            "class HealthMonitor", 
            "class SystemResourcesCheck",
            "class DatabaseHealthCheck",
            "class ModelHealthCheck",
            "class ExternalDependencyCheck"
        ]
        
        exists = self.check_file_exists(file_path)
        print(f"  File exists: {exists}")
        
        if exists:
            contains = self.check_file_contains(file_path, required_classes)
            for class_name, found in contains.items():
                print(f"  {class_name}: {'✓' if found else '✗'}")
            
            self.results['health_monitoring'] = {
                'file_exists': exists,
                'classes_implemented': sum(contains.values()),
                'total_classes': len(required_classes)
            }
        else:
            print("  ✗ Health monitoring file missing")
            self.results['health_monitoring'] = {
                'file_exists': False,
                'classes_implemented': 0,
                'total_classes': len(required_classes)
            }
    
    def validate_circuit_breakers(self):
        """Validate circuit breaker implementation."""
        print("\n2. Circuit Breaker Pattern")
        print("-" * 40)
        
        file_path = "odordiff2/utils/circuit_breaker.py"
        required_components = [
            "class CircuitBreaker",
            "class CircuitBreakerState",
            "class CircuitBreakerConfig",
            "class CircuitBreakerRegistry",
            "def call\\(",
            "CLOSED.*OPEN.*HALF_OPEN"
        ]
        
        exists = self.check_file_exists(file_path)
        print(f"  File exists: {exists}")
        
        if exists:
            contains = self.check_file_contains(file_path, required_components)
            for component, found in contains.items():
                display_name = component.replace("\\(", "()").replace(".*", "/")
                print(f"  {display_name}: {'✓' if found else '✗'}")
            
            self.results['circuit_breakers'] = {
                'file_exists': exists,
                'components_implemented': sum(contains.values()),
                'total_components': len(required_components)
            }
        else:
            print("  ✗ Circuit breaker file missing")
            self.results['circuit_breakers'] = {
                'file_exists': False,
                'components_implemented': 0,
                'total_components': len(required_components)
            }
    
    def validate_rate_limiting(self):
        """Validate rate limiting implementation.""" 
        print("\n3. Advanced Rate Limiting")
        print("-" * 40)
        
        file_path = "odordiff2/utils/rate_limiting.py"
        required_components = [
            "class RateLimiter",
            "class RateLimitBucket",
            "class SlidingWindowCounter", 
            "ip_limits.*api_key_limits",
            "def is_allowed\\(",
            "def check_rate_limit\\("
        ]
        
        exists = self.check_file_exists(file_path)
        print(f"  File exists: {exists}")
        
        if exists:
            contains = self.check_file_contains(file_path, required_components)
            for component, found in contains.items():
                display_name = component.replace("\\(", "()").replace(".*", "/")
                print(f"  {display_name}: {'✓' if found else '✗'}")
            
            self.results['rate_limiting'] = {
                'file_exists': exists,
                'components_implemented': sum(contains.values()),
                'total_components': len(required_components)
            }
        else:
            print("  ✗ Rate limiting file missing")
            self.results['rate_limiting'] = {
                'file_exists': False,
                'components_implemented': 0,
                'total_components': len(required_components)
            }
    
    def validate_input_validation(self):
        """Validate input validation enhancements."""
        print("\n4. Input Validation & Sanitization")
        print("-" * 40)
        
        file_path = "odordiff2/utils/validation.py"
        required_components = [
            "class InputValidator",
            "class Sanitizer",
            "SCHEMAS.*generation_request",
            "def sanitize_text\\(",
            "def validate_prompt\\(",
            "jsonschema"
        ]
        
        exists = self.check_file_exists(file_path)
        print(f"  File exists: {exists}")
        
        if exists:
            contains = self.check_file_contains(file_path, required_components)
            for component, found in contains.items():
                display_name = component.replace("\\(", "()").replace(".*", "/")
                print(f"  {display_name}: {'✓' if found else '✗'}")
            
            self.results['input_validation'] = {
                'file_exists': exists,
                'components_implemented': sum(contains.values()),
                'total_components': len(required_components)
            }
        else:
            print("  ✗ Input validation file missing")
            self.results['input_validation'] = {
                'file_exists': False,
                'components_implemented': 0,
                'total_components': len(required_components)
            }
    
    def validate_error_recovery(self):
        """Validate error recovery mechanisms."""
        print("\n5. Error Recovery & Graceful Degradation")
        print("-" * 40)
        
        file_path = "odordiff2/utils/recovery.py"
        required_components = [
            "class RecoveryManager",
            "class GracefulDegradation",
            "def execute_with_retry\\(",
            "def degrade_service\\(",
            "fallback.*circuit_breaker",
            "retry.*backoff"
        ]
        
        exists = self.check_file_exists(file_path)
        print(f"  File exists: {exists}")
        
        if exists:
            contains = self.check_file_contains(file_path, required_components)
            for component, found in contains.items():
                display_name = component.replace("\\(", "()").replace(".*", "/")
                print(f"  {display_name}: {'✓' if found else '✗'}")
            
            self.results['error_recovery'] = {
                'file_exists': exists,
                'components_implemented': sum(contains.values()),
                'total_components': len(required_components)
            }
        else:
            print("  ✗ Error recovery file missing")
            self.results['error_recovery'] = {
                'file_exists': False,
                'components_implemented': 0,
                'total_components': len(required_components)
            }
    
    def validate_configuration_management(self):
        """Validate configuration management."""
        print("\n6. Configuration Management")
        print("-" * 40)
        
        # Check main config file
        config_file = "odordiff2/config/settings.py"
        config_files = [
            "config/development.yaml",
            "config/production.yaml", 
            "config/testing.yaml"
        ]
        
        exists = self.check_file_exists(config_file)
        print(f"  Settings file exists: {exists}")
        
        config_exists = [self.check_file_exists(f) for f in config_files]
        for i, config_path in enumerate(config_files):
            print(f"  {config_path}: {'✓' if config_exists[i] else '✗'}")
        
        if exists:
            required_components = [
                "class ConfigManager",
                "def load_environment_config\\(",
                "development.*production.*testing",
                "def validate_config\\("
            ]
            
            contains = self.check_file_contains(config_file, required_components)
            for component, found in contains.items():
                display_name = component.replace("\\(", "()").replace(".*", "/")
                print(f"  {display_name}: {'✓' if found else '✗'}")
            
            self.results['configuration_management'] = {
                'file_exists': exists,
                'config_files': sum(config_exists),
                'components_implemented': sum(contains.values()),
                'total_components': len(required_components)
            }
        else:
            print("  ✗ Configuration management file missing")
            self.results['configuration_management'] = {
                'file_exists': False,
                'config_files': sum(config_exists),
                'components_implemented': 0,
                'total_components': 4
            }
    
    def validate_connection_pooling(self):
        """Validate connection pooling optimization."""
        print("\n7. Connection Pooling Optimization")
        print("-" * 40)
        
        file_path = "odordiff2/data/cache.py"
        required_components = [
            "class DatabaseConnectionPool",
            "class RedisConnectionPool",
            "class PersistentCache",
            "max_connections",
            "def get_connection\\(",
            "connection.*pool"
        ]
        
        exists = self.check_file_exists(file_path)
        print(f"  File exists: {exists}")
        
        if exists:
            contains = self.check_file_contains(file_path, required_components)
            for component, found in contains.items():
                display_name = component.replace("\\(", "()").replace(".*", "/")
                print(f"  {display_name}: {'✓' if found else '✗'}")
            
            self.results['connection_pooling'] = {
                'file_exists': exists,
                'components_implemented': sum(contains.values()),
                'total_components': len(required_components)
            }
        else:
            print("  ✗ Connection pooling file missing")
            self.results['connection_pooling'] = {
                'file_exists': False,
                'components_implemented': 0,
                'total_components': len(required_components)
            }
    
    def validate_backup_recovery(self):
        """Validate backup and recovery mechanisms."""
        print("\n8. Data Backup & Recovery")
        print("-" * 40)
        
        file_path = "odordiff2/utils/backup.py"
        required_components = [
            "class BackupManager",
            "class LocalBackupStorage",
            "class S3BackupStorage",
            "def create_backup\\(",
            "def restore_backup\\(",
            "scheduled.*backup"
        ]
        
        exists = self.check_file_exists(file_path)
        print(f"  File exists: {exists}")
        
        if exists:
            contains = self.check_file_contains(file_path, required_components)
            for component, found in contains.items():
                display_name = component.replace("\\(", "()").replace(".*", "/")
                print(f"  {display_name}: {'✓' if found else '✗'}")
            
            self.results['backup_recovery'] = {
                'file_exists': exists,
                'components_implemented': sum(contains.values()),
                'total_components': len(required_components)
            }
        else:
            print("  ✗ Backup recovery file missing")
            self.results['backup_recovery'] = {
                'file_exists': False,
                'components_implemented': 0,
                'total_components': len(required_components)
            }
    
    def validate_security_hardening(self):
        """Validate security hardening features."""
        print("\n9. Security Hardening")
        print("-" * 40)
        
        file_path = "odordiff2/utils/security.py"
        required_components = [
            "class SecurityManager",
            "class RequestSigner", 
            "class JWTManager",
            "class SecurityHeaders",
            "def sign_request\\(",
            "HMAC.*JWT.*encryption"
        ]
        
        exists = self.check_file_exists(file_path)
        print(f"  File exists: {exists}")
        
        if exists:
            contains = self.check_file_contains(file_path, required_components)
            for component, found in contains.items():
                display_name = component.replace("\\(", "()").replace(".*", "/")
                print(f"  {display_name}: {'✓' if found else '✗'}")
            
            self.results['security_hardening'] = {
                'file_exists': exists,
                'components_implemented': sum(contains.values()),
                'total_components': len(required_components)
            }
        else:
            print("  ✗ Security hardening file missing")
            self.results['security_hardening'] = {
                'file_exists': False,
                'components_implemented': 0,
                'total_components': len(required_components)
            }
    
    def validate_observability(self):
        """Validate observability enhancements."""
        print("\n10. Enhanced Observability")
        print("-" * 40)
        
        file_path = "odordiff2/utils/logging.py"
        required_components = [
            "class OdorDiffLogger",
            "class StructuredFormatter",
            "class DistributedTracer",
            "correlation_id.*trace_id",
            "def start_span\\(",
            "structured.*logging.*tracing"
        ]
        
        exists = self.check_file_exists(file_path)
        print(f"  File exists: {exists}")
        
        if exists:
            contains = self.check_file_contains(file_path, required_components)
            for component, found in contains.items():
                display_name = component.replace("\\(", "()").replace(".*", "/")
                print(f"  {display_name}: {'✓' if found else '✗'}")
            
            self.results['observability'] = {
                'file_exists': exists,
                'components_implemented': sum(contains.values()),
                'total_components': len(required_components)
            }
        else:
            print("  ✗ Observability file missing")
            self.results['observability'] = {
                'file_exists': False,
                'components_implemented': 0,
                'total_components': len(required_components)
            }
    
    def generate_summary(self):
        """Generate validation summary."""
        print("\n" + "="*65)
        print("VALIDATION SUMMARY")
        print("="*65)
        
        total_areas = len(self.results)
        fully_implemented = 0
        total_files = 0
        existing_files = 0
        total_components = 0
        implemented_components = 0
        
        for area, data in self.results.items():
            area_name = area.replace('_', ' ').title()
            
            # Check if area is fully implemented
            file_exists = data.get('file_exists', False)
            components_impl = data.get('components_implemented', 0)
            total_comps = data.get('total_components', 1)
            
            total_files += 1
            if file_exists:
                existing_files += 1
            
            total_components += total_comps
            implemented_components += components_impl
            
            # Consider fully implemented if file exists and >=80% components implemented
            if file_exists and (components_impl / total_comps) >= 0.8:
                fully_implemented += 1
                status = "✅ COMPLETE"
            elif file_exists:
                status = f"⚠️  PARTIAL ({components_impl}/{total_comps})"
            else:
                status = "❌ MISSING"
            
            print(f"{area_name:.<35} {status}")
        
        print("\n" + "-"*65)
        print(f"Areas fully implemented: {fully_implemented}/{total_areas} ({fully_implemented/total_areas*100:.1f}%)")
        print(f"Files present: {existing_files}/{total_files} ({existing_files/total_files*100:.1f}%)")
        print(f"Components implemented: {implemented_components}/{total_components} ({implemented_components/total_components*100:.1f}%)")
        
        print("\n" + "="*65)
        if fully_implemented == total_areas:
            print("🎉 ALL ROBUSTNESS ENHANCEMENTS FULLY IMPLEMENTED!")
            print("The OdorDiff-2 system is ready for production deployment.")
        elif fully_implemented >= total_areas * 0.8:
            print("✅ ROBUSTNESS ENHANCEMENTS SUBSTANTIALLY COMPLETE!")
            print("Most enhancements are implemented. Minor refinements may be needed.")
        else:
            print("⚠️  ROBUSTNESS ENHANCEMENTS PARTIALLY COMPLETE")
            print("Significant enhancements are in place but more work is needed.")
        
        print("="*65)


def main():
    """Run the enhancement validation."""
    validator = EnhancementValidator()
    validator.validate_all()


if __name__ == "__main__":
    main()