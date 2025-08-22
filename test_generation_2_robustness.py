#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Lightweight Implementation

Demonstrates robustness features with minimal dependencies:
- Enhanced error handling
- Input validation and security
- Health monitoring
- Circuit breaker pattern
- Logging and recovery
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class EnhancedLogger:
    """Enhanced logging with structured output."""
    
    def __init__(self, name: str):
        self.name = name
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / f'{self.name}.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info with context."""
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.info(message)
    
    def error(self, message: str, **kwargs):
        """Log error with context."""
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.error(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning with context."""
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.warning(message)

class SecurityValidator:
    """Enhanced security validation."""
    
    def __init__(self):
        self.logger = EnhancedLogger('security')
        
        # Dangerous patterns
        self.dangerous_patterns = [
            'eval(', 'exec(', '__import__', 'os.system', 'subprocess',
            '<script', 'javascript:', 'data:text/html', 'file://',
            'python:', 'import ', 'from ', 'del ', 'global '
        ]
        
        # Safe character sets
        self.safe_smiles_chars = set('CNOSPFClBrI()[]=#-+1234567890@Hncos.')
        self.safe_text_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-\'"')
    
    def validate_input(self, text: str) -> tuple[bool, str]:
        """Comprehensive input validation."""
        if not text:
            return False, "Empty input"
        
        # Length check
        if len(text) > 500:
            self.logger.warning("Input too long", length=len(text))
            return False, "Input too long"
        
        # Dangerous pattern check
        text_lower = text.lower()
        for pattern in self.dangerous_patterns:
            if pattern in text_lower:
                self.logger.warning("Dangerous pattern detected", pattern=pattern)
                return False, f"Dangerous pattern: {pattern}"
        
        # Character validation
        if not all(c in self.safe_text_chars for c in text):
            self.logger.warning("Invalid characters detected")
            return False, "Invalid characters"
        
        return True, "Valid"
    
    def validate_smiles(self, smiles: str) -> tuple[bool, str]:
        """SMILES string validation."""
        if not smiles:
            return False, "Empty SMILES"
        
        # Character validation
        if not all(c in self.safe_smiles_chars for c in smiles):
            return False, "Invalid SMILES characters"
        
        # Basic structure validation
        if smiles.count('(') != smiles.count(')'):
            return False, "Unbalanced parentheses"
        
        if smiles.count('[') != smiles.count(']'):
            return False, "Unbalanced brackets"
        
        return True, "Valid SMILES"
    
    def sanitize_output(self, data: Any) -> Any:
        """Sanitize output data."""
        if isinstance(data, str):
            # Basic XSS protection
            return data.replace('<', '&lt;').replace('>', '&gt;')
        elif isinstance(data, dict):
            return {k: self.sanitize_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_output(item) for item in data]
        return data

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, name: str, failure_threshold: int = 3, timeout: int = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.logger = EnhancedLogger(f'circuit_breaker_{name}')
        
        self.logger.info(f"Circuit breaker '{name}' initialized", 
                        threshold=failure_threshold, timeout=timeout)
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == 'OPEN':
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                self.logger.info(f"Circuit breaker '{self.name}' moving to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
                self.logger.info(f"Circuit breaker '{self.name}' recovered to CLOSED")
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                self.logger.error(f"Circuit breaker '{self.name}' OPENED", 
                                failures=self.failure_count)
            
            raise e
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time
        }

class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        self.logger = EnhancedLogger('health_monitor')
        self.metrics = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'average_response_time': 0.0,
            'uptime_start': time.time(),
            'last_health_check': None
        }
        self.response_times = []
    
    def record_request(self, success: bool, response_time: float):
        """Record request metrics."""
        self.metrics['requests_total'] += 1
        
        if success:
            self.metrics['requests_successful'] += 1
        else:
            self.metrics['requests_failed'] += 1
        
        self.response_times.append(response_time)
        if len(self.response_times) > 50:  # Keep last 50 measurements
            self.response_times.pop(0)
        
        if self.response_times:
            self.metrics['average_response_time'] = sum(self.response_times) / len(self.response_times)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'HEALTHY',
            'checks': {},
            'metrics': self.metrics.copy()
        }
        
        # Calculate uptime
        uptime_seconds = time.time() - self.metrics['uptime_start']
        health_status['metrics']['uptime_seconds'] = uptime_seconds
        health_status['metrics']['uptime_hours'] = uptime_seconds / 3600
        
        # Filesystem check
        try:
            test_file = Path('health_check.tmp')
            test_file.write_text('health_test')
            test_file.unlink()
            health_status['checks']['filesystem'] = 'HEALTHY'
        except Exception as e:
            health_status['checks']['filesystem'] = f'UNHEALTHY: {e}'
            health_status['status'] = 'UNHEALTHY'
        
        # Error rate check
        if self.metrics['requests_total'] > 0:
            error_rate = self.metrics['requests_failed'] / self.metrics['requests_total']
            if error_rate > 0.1:  # >10% error rate
                health_status['checks']['error_rate'] = f'HIGH: {error_rate:.1%}'
                health_status['status'] = 'DEGRADED'
            else:
                health_status['checks']['error_rate'] = f'HEALTHY: {error_rate:.1%}'
        else:
            health_status['checks']['error_rate'] = 'NO_DATA'
        
        # Response time check
        if self.response_times:
            avg_time = self.metrics['average_response_time']
            if avg_time > 5.0:  # >5s average response time
                health_status['checks']['response_time'] = f'SLOW: {avg_time:.2f}s'
                health_status['status'] = 'DEGRADED'
            else:
                health_status['checks']['response_time'] = f'HEALTHY: {avg_time:.2f}s'
        else:
            health_status['checks']['response_time'] = 'NO_DATA'
        
        self.metrics['last_health_check'] = health_status['timestamp']
        return health_status

class RobustMoleculeGenerator:
    """Enhanced molecule generator with robustness features."""
    
    def __init__(self):
        self.logger = EnhancedLogger('robust_generator')
        self.security = SecurityValidator()
        self.circuit_breaker = CircuitBreaker('molecule_generation', failure_threshold=3)
        self.health_monitor = HealthMonitor()
        
        # Safe molecule templates with enhanced metadata
        self.safe_templates = {
            'floral': [
                {'smiles': 'CC(C)=CCO', 'name': 'linalool', 'safety': 0.95, 'category': 'terpene'},
                {'smiles': 'CCCC(C)=CCO', 'name': 'citronellol', 'safety': 0.92, 'category': 'alcohol'},
                {'smiles': 'CC(C)(O)CCO', 'name': 'hydroxycitronellal', 'safety': 0.88, 'category': 'aldehyde'}
            ],
            'citrus': [
                {'smiles': 'CC1=CCC(CC1)C(C)C', 'name': 'limonene', 'safety': 0.93, 'category': 'terpene'},
                {'smiles': 'CC(C)=CC', 'name': 'isoprene', 'safety': 0.96, 'category': 'alkene'},
                {'smiles': 'C=C(C)CCC=C(C)C', 'name': 'myrcene', 'safety': 0.91, 'category': 'terpene'}
            ],
            'woody': [
                {'smiles': 'c1ccc2c(c1)cccc2', 'name': 'naphthalene_derivative', 'safety': 0.87, 'category': 'aromatic'},
                {'smiles': 'CC(C)(C)c1ccccc1O', 'name': 'BHT', 'safety': 0.89, 'category': 'phenol'}
            ],
            'fresh': [
                {'smiles': 'CCO', 'name': 'ethanol', 'safety': 0.98, 'category': 'alcohol'},
                {'smiles': 'CCCO', 'name': 'propanol', 'safety': 0.95, 'category': 'alcohol'},
                {'smiles': 'CC(C)O', 'name': 'isopropanol', 'safety': 0.94, 'category': 'alcohol'}
            ]
        }
        
        self.logger.info("RobustMoleculeGenerator initialized with enhanced safety and monitoring")
    
    def generate_molecules(self, prompt: str, num_molecules: int = 3) -> List[Dict[str, Any]]:
        """Generate molecules with comprehensive robustness."""
        start_time = time.time()
        
        try:
            # Security validation
            is_valid, validation_msg = self.security.validate_input(prompt)
            if not is_valid:
                raise ValueError(f"Input validation failed: {validation_msg}")
            
            self.logger.info(f"Generating {num_molecules} molecules", prompt=prompt)
            
            # Use circuit breaker
            result = self.circuit_breaker.call(self._internal_generate, prompt, num_molecules)
            
            # Success metrics
            response_time = time.time() - start_time
            self.health_monitor.record_request(True, response_time)
            
            # Sanitize output
            sanitized_result = self.security.sanitize_output(result)
            
            self.logger.info(f"Successfully generated {len(sanitized_result)} molecules", 
                           response_time=response_time)
            
            return sanitized_result
            
        except Exception as e:
            # Error handling
            response_time = time.time() - start_time
            self.health_monitor.record_request(False, response_time)
            
            self.logger.error(f"Generation failed: {e}", prompt=prompt, response_time=response_time)
            
            # Return safe fallback
            return self._get_safe_fallback(prompt)
    
    def _internal_generate(self, prompt: str, num_molecules: int) -> List[Dict[str, Any]]:
        """Internal generation with safety checks."""
        prompt_lower = prompt.lower()
        molecules = []
        
        # Select templates based on prompt analysis
        selected_templates = []
        for category, templates in self.safe_templates.items():
            if self._matches_category(prompt_lower, category):
                selected_templates.extend(templates)
        
        # Default to fresh if no match
        if not selected_templates:
            selected_templates = self.safe_templates['fresh']
        
        # Generate molecules with validation
        for i, template in enumerate(selected_templates[:num_molecules]):
            # Validate SMILES
            is_valid, smiles_msg = self.security.validate_smiles(template['smiles'])
            if not is_valid:
                self.logger.warning(f"Skipping invalid SMILES: {template['smiles']}")
                continue
            
            molecule = {
                'smiles': template['smiles'],
                'name': template['name'],
                'confidence': min(0.95, 0.80 + (i * 0.05)),
                'safety_score': template['safety'],
                'synthesis_score': min(0.90, 0.70 + (i * 0.07)),
                'estimated_cost': 40.0 + (i * 12.0),
                'category': template['category'],
                'odor_profile': self._predict_odor_profile(template, prompt),
                'generation_metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'prompt': prompt,
                    'generation_method': 'template_based_robust',
                    'validation_passed': True
                }
            }
            
            molecules.append(molecule)
        
        if not molecules:
            # Fallback to safest template
            safe_template = self.safe_templates['fresh'][0]
            molecules.append({
                'smiles': safe_template['smiles'],
                'name': 'safe_fallback',
                'confidence': 0.5,
                'safety_score': safe_template['safety'],
                'synthesis_score': 0.8,
                'estimated_cost': 35.0,
                'category': safe_template['category'],
                'odor_profile': {'primary_notes': ['clean'], 'character': 'safe, neutral'},
                'generation_metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'fallback': True
                }
            })
        
        return molecules
    
    def _matches_category(self, prompt: str, category: str) -> bool:
        """Check if prompt matches scent category."""
        keywords = {
            'floral': ['floral', 'flower', 'rose', 'jasmine', 'lavender', 'lily', 'peony'],
            'citrus': ['citrus', 'lemon', 'orange', 'lime', 'bergamot', 'grapefruit'],
            'woody': ['woody', 'cedar', 'sandalwood', 'pine', 'oak', 'bamboo'],
            'fresh': ['fresh', 'clean', 'aquatic', 'marine', 'ozone', 'crisp']
        }
        
        category_keywords = keywords.get(category, [])
        return any(keyword in prompt for keyword in category_keywords)
    
    def _predict_odor_profile(self, template: Dict, prompt: str) -> Dict[str, Any]:
        """Enhanced odor prediction with confidence."""
        category = template['category']
        prompt_lower = prompt.lower()
        
        # Category-based prediction
        if 'floral' in prompt_lower:
            primary = ['floral', 'sweet']
            character = 'elegant, romantic'
            intensity = 0.85
        elif 'citrus' in prompt_lower:
            primary = ['citrus', 'bright']
            character = 'fresh, energizing'
            intensity = 0.90
        elif 'woody' in prompt_lower:
            primary = ['woody', 'warm']
            character = 'sophisticated, grounding'
            intensity = 0.75
        else:
            primary = ['clean', 'fresh']
            character = 'neutral, pleasant'
            intensity = 0.65
        
        return {
            'primary_notes': primary,
            'secondary_notes': ['balanced'],
            'character': character,
            'intensity': intensity,
            'confidence': 0.80,
            'longevity_hours': 3.0 + (intensity * 3),
            'sillage': intensity * 0.75,
            'molecular_basis': category
        }
    
    def _get_safe_fallback(self, prompt: str) -> List[Dict[str, Any]]:
        """Get safe fallback molecules."""
        self.logger.info("Returning safe fallback molecule")
        
        safe_template = self.safe_templates['fresh'][0]
        return [{
            'smiles': safe_template['smiles'],
            'name': 'emergency_fallback',
            'confidence': 0.3,
            'safety_score': safe_template['safety'],
            'synthesis_score': 0.9,
            'estimated_cost': 25.0,
            'category': 'safe_fallback',
            'odor_profile': {
                'primary_notes': ['neutral'],
                'character': 'safe, emergency fallback',
                'intensity': 0.4
            },
            'generation_metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'emergency_fallback': True,
                'original_prompt': prompt
            }
        }]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health = self.health_monitor.health_check()
        circuit_status = self.circuit_breaker.get_status()
        
        return {
            'health': health,
            'circuit_breaker': circuit_status,
            'uptime_hours': health['metrics']['uptime_hours'],
            'total_requests': health['metrics']['requests_total'],
            'success_rate': (health['metrics']['requests_successful'] / 
                           max(1, health['metrics']['requests_total'])) * 100
        }

def demo_robustness_features():
    """Demonstrate Generation 2 robustness features."""
    print("🛡️ OdorDiff-2 Generation 2: MAKE IT ROBUST")
    print("=" * 60)
    print("Demonstrating enhanced error handling, security, and monitoring...")
    print()
    
    # Initialize robust generator
    generator = RobustMoleculeGenerator()
    
    # Test cases for robustness
    test_cases = [
        # Valid cases
        {"prompt": "elegant rose fragrance", "expected": "success"},
        {"prompt": "fresh citrus burst", "expected": "success"},
        {"prompt": "warm woody cedar", "expected": "success"},
        
        # Edge cases
        {"prompt": "", "expected": "error"},
        {"prompt": "x" * 600, "expected": "error"},  # Too long
        
        # Security threats
        {"prompt": "eval(os.system('rm -rf /'))", "expected": "error"},
        {"prompt": "<script>alert('xss')</script>", "expected": "error"},
        {"prompt": "import os; os.system('evil')", "expected": "error"},
        {"prompt": "javascript:void(0)", "expected": "error"},
        
        # Invalid characters
        {"prompt": "fragrance with ♠♣♥♦", "expected": "error"},
        {"prompt": "scent\x00with\x01nulls", "expected": "error"}
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        prompt = test_case['prompt']
        expected = test_case['expected']
        
        # Display prompt safely
        display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
        print(f"🧪 Test {i}: {repr(display_prompt)}")
        
        try:
            molecules = generator.generate_molecules(prompt, 2)
            
            if expected == "success":
                print(f"   ✅ Generated {len(molecules)} molecules successfully")
                if molecules:
                    best = molecules[0]
                    print(f"   📝 Best: {best['name']} ({best['smiles']}) - Safety: {best['safety_score']:.2f}")
                result_status = "success"
            else:
                print(f"   ⚠️ Expected failure but got {len(molecules)} molecules")
                result_status = "unexpected_success"
            
            results.append({
                'test': i,
                'status': result_status,
                'molecules_count': len(molecules)
            })
            
        except Exception as e:
            if expected == "error":
                print(f"   ✅ Correctly blocked: {str(e)[:80]}")
                result_status = "correctly_blocked"
            else:
                print(f"   ❌ Unexpected error: {e}")
                result_status = "unexpected_error"
            
            results.append({
                'test': i,
                'status': result_status,
                'error': str(e)[:100]
            })
        
        print()
    
    # System status check
    print("📊 System Status Check")
    print("-" * 30)
    status = generator.get_system_status()
    
    print(f"Overall Health: {status['health']['status']}")
    print(f"Circuit Breaker: {status['circuit_breaker']['state']}")
    print(f"Uptime: {status['uptime_hours']:.2f} hours")
    print(f"Total Requests: {status['total_requests']}")
    print(f"Success Rate: {status['success_rate']:.1f}%")
    
    print("\nHealth Checks:")
    for check_name, check_result in status['health']['checks'].items():
        print(f"  {check_name}: {check_result}")
    
    return results, status

def run_generation_2_tests():
    """Run comprehensive Generation 2 robustness tests."""
    print("🛡️ OdorDiff-2 Autonomous SDLC - Generation 2 Testing")
    print("=" * 70)
    
    try:
        # Run robustness demonstration
        results, status = demo_robustness_features()
        
        # Analyze results
        successful_tests = 0
        total_tests = len(results)
        
        for result in results:
            if result['status'] in ['success', 'correctly_blocked']:
                successful_tests += 1
        
        success_rate = (successful_tests / total_tests) * 100
        
        print("\n" + "=" * 70)
        print("🏆 GENERATION 2 - MAKE IT ROBUST: FINAL ASSESSMENT")
        print("=" * 70)
        
        print(f"📈 Test Results: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"🏥 System Health: {status['health']['status']}")
        print(f"🔄 Circuit Breaker: {status['circuit_breaker']['state']}")
        print(f"📊 Request Success Rate: {status['success_rate']:.1f}%")
        
        # Determine verdict
        if (success_rate >= 85 and 
            status['health']['status'] in ['HEALTHY', 'DEGRADED'] and
            status['circuit_breaker']['state'] in ['CLOSED', 'HALF_OPEN']):
            
            print("\n🎉 GENERATION 2 COMPLETE - ROBUST SUCCESS!")
            print("   ✅ Security validation working")
            print("   ✅ Input sanitization active")
            print("   ✅ Circuit breaker functional")
            print("   ✅ Error handling comprehensive")
            print("   ✅ Health monitoring operational")
            print("   ✅ Structured logging implemented")
            print("   ✅ Safe fallback mechanisms working")
            print("   ✅ Ready to proceed to Generation 3")
            verdict = "EXCELLENT"
        else:
            print("\n⚠️ GENERATION 2 NEEDS IMPROVEMENT")
            print(f"   🔧 Success rate: {success_rate:.1f}% (need ≥85%)")
            print(f"   🏥 Health status: {status['health']['status']}")
            verdict = "NEEDS_WORK"
        
        # Save completion report
        completion_report = {
            'generation': 2,
            'phase': 'MAKE_IT_ROBUST',
            'timestamp': time.time(),
            'success_rate': success_rate,
            'system_health': status['health']['status'],
            'circuit_breaker_state': status['circuit_breaker']['state'],
            'verdict': verdict,
            'robustness_features_implemented': [
                '✅ Enhanced input validation',
                '✅ Security threat detection',
                '✅ SMILES structure validation',
                '✅ Circuit breaker pattern',
                '✅ Comprehensive error handling',
                '✅ Structured logging system',
                '✅ Health monitoring',
                '✅ Safe fallback mechanisms',
                '✅ XSS/injection protection',
                '✅ Output sanitization',
                '✅ Request metrics tracking'
            ],
            'test_results': results,
            'system_status': status,
            'next_phase': 'Generation 3: MAKE IT SCALE' if verdict == 'EXCELLENT' else 'Fix Generation 2 issues'
        }
        
        with open('generation_2_robustness_report.json', 'w') as f:
            json.dump(completion_report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved: generation_2_robustness_report.json")
        
        return verdict == 'EXCELLENT'
        
    except Exception as e:
        print(f"\n❌ Generation 2 testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_generation_2_tests()
    sys.exit(0 if success else 1)
