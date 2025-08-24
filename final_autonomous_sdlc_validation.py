#!/usr/bin/env python3
"""
Final Autonomous SDLC Validation
Comprehensive validation of all phases before production deployment
"""

import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.abspath('.'))

def validate_all_generations():
    """Run all generation tests to validate SDLC completion."""
    print('🔍 TERRAGON AUTONOMOUS SDLC - FINAL VALIDATION')
    print('=' * 70)
    
    generation_tests = [
        ('Generation 1 - Core Functionality', 'test_lite_gen1.py'),
        ('Generation 2 - Robustness', 'test_gen2_robustness.py'),
        ('Generation 3 - Performance', 'test_generation_3_performance.py'),
        ('Global Implementation', 'test_global_implementation.py'),
        ('Quality Gates', 'run_comprehensive_quality_gates.py')
    ]
    
    results = {}
    detailed_results = {}
    
    for test_name, test_file in generation_tests:
        print(f'\n🧪 VALIDATING: {test_name}')
        print('-' * 50)
        
        if not Path(test_file).exists():
            print(f'❌ Test file not found: {test_file}')
            results[test_name] = False
            continue
        
        try:
            result = subprocess.run(
                [sys.executable, test_file], 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            success = result.returncode == 0
            results[test_name] = success
            
            # Extract key metrics from output
            output_lines = result.stdout.split('\n')
            metrics = extract_metrics(output_lines, test_name)
            detailed_results[test_name] = {
                'success': success,
                'metrics': metrics,
                'output': result.stdout[-1000:],  # Last 1000 chars
                'errors': result.stderr[-500:] if result.stderr else None
            }
            
            status = '✅ PASSED' if success else '❌ FAILED'
            print(f'{status}: {test_name}')
            
            if metrics:
                for metric, value in metrics.items():
                    print(f'  📊 {metric}: {value}')
            
        except subprocess.TimeoutExpired:
            print(f'⏰ TIMEOUT: {test_name} (>5 minutes)')
            results[test_name] = False
            detailed_results[test_name] = {'success': False, 'error': 'timeout'}
            
        except Exception as e:
            print(f'❌ ERROR: {test_name} - {e}')
            results[test_name] = False
            detailed_results[test_name] = {'success': False, 'error': str(e)}
    
    return results, detailed_results

def extract_metrics(output_lines, test_name):
    """Extract key metrics from test output."""
    metrics = {}
    
    for line in output_lines:
        line = line.strip()
        
        # Performance metrics
        if 'molecules/second' in line:
            try:
                value = float(line.split(':')[1].strip().split()[0])
                metrics['Molecule Generation Rate'] = f'{value:.0f} mol/s'
            except:
                pass
                
        if 'properties/second' in line:
            try:
                value = float(line.split(':')[1].strip().split()[0])
                metrics['Property Calculation Rate'] = f'{value:.0f} prop/s'
            except:
                pass
        
        if 'speedup' in line:
            try:
                speedup = line.split(':')[1].strip().split('x')[0]
                metrics['Concurrent Speedup'] = f'{speedup}x'
            except:
                pass
        
        # Quality metrics
        if 'Overall:' in line and 'tests passed' in line:
            try:
                parts = line.split()
                passed = int(parts[1].split('/')[0])
                total = int(parts[1].split('/')[1])
                percentage = (passed / total) * 100
                metrics['Test Success Rate'] = f'{passed}/{total} ({percentage:.1f}%)'
            except:
                pass
        
        # Language support
        if 'Supported languages:' in line:
            try:
                count = line.count(',') + 1 if ',' in line else 0
                if count > 0:
                    metrics['Languages Supported'] = f'{count} languages'
            except:
                pass
        
        # Regional compliance
        if 'Supported regions:' in line:
            try:
                count = line.count(',') + 1 if ',' in line else 0
                if count > 0:
                    metrics['Regions Supported'] = f'{count} regions'
            except:
                pass
    
    return metrics

def calculate_overall_sdlc_score(results, detailed_results):
    """Calculate overall SDLC completion score."""
    
    # Weight different phases
    phase_weights = {
        'Generation 1 - Core Functionality': 0.20,
        'Generation 2 - Robustness': 0.25,
        'Generation 3 - Performance': 0.25,
        'Global Implementation': 0.20,
        'Quality Gates': 0.10
    }
    
    weighted_score = 0
    total_weight = 0
    
    for phase, weight in phase_weights.items():
        if phase in results:
            success = results[phase]
            weighted_score += weight if success else 0
            total_weight += weight
    
    overall_score = (weighted_score / total_weight) * 100 if total_weight > 0 else 0
    
    return overall_score

def generate_final_report(results, detailed_results, overall_score):
    """Generate comprehensive final report."""
    
    report = {
        'sdlc_completion_timestamp': datetime.now().isoformat(),
        'overall_score': overall_score,
        'deployment_readiness': determine_deployment_readiness(overall_score),
        'phase_results': results,
        'detailed_metrics': {
            phase: details.get('metrics', {}) 
            for phase, details in detailed_results.items()
        },
        'recommendations': generate_recommendations(overall_score, results),
        'system_capabilities': extract_system_capabilities(detailed_results),
        'next_steps': generate_next_steps(overall_score)
    }
    
    return report

def determine_deployment_readiness(score):
    """Determine deployment readiness based on score."""
    if score >= 90:
        return {
            'status': 'PRODUCTION_READY',
            'recommendation': 'Deploy to production immediately',
            'confidence': 'HIGH'
        }
    elif score >= 80:
        return {
            'status': 'STAGING_READY', 
            'recommendation': 'Deploy to staging, production within 30 days',
            'confidence': 'HIGH'
        }
    elif score >= 70:
        return {
            'status': 'DEVELOPMENT_READY',
            'recommendation': 'Deploy to development environment',
            'confidence': 'MEDIUM'
        }
    else:
        return {
            'status': 'NEEDS_IMPROVEMENT',
            'recommendation': 'Requires additional development',
            'confidence': 'LOW'
        }

def generate_recommendations(score, results):
    """Generate specific recommendations based on results."""
    recommendations = []
    
    failed_phases = [phase for phase, success in results.items() if not success]
    
    if not failed_phases:
        recommendations.append("🎉 All phases completed successfully - ready for deployment")
    else:
        for phase in failed_phases:
            if 'Generation 2' in phase:
                recommendations.append("🛡️ Strengthen error handling and security systems")
            elif 'Generation 3' in phase:
                recommendations.append("⚡ Optimize performance and caching systems")  
            elif 'Global' in phase:
                recommendations.append("🌍 Complete internationalization and compliance features")
            elif 'Quality' in phase:
                recommendations.append("🔍 Address failing quality gates before deployment")
    
    if score >= 80:
        recommendations.append("📊 Monitor key metrics closely in staging environment")
        recommendations.append("🔒 Complete security enhancements for production")
    
    return recommendations

def extract_system_capabilities(detailed_results):
    """Extract system capabilities from test results."""
    capabilities = {}
    
    for phase, details in detailed_results.items():
        if details.get('success') and details.get('metrics'):
            phase_key = phase.replace(' ', '_').replace('-', '_').lower()
            capabilities[phase_key] = details['metrics']
    
    return capabilities

def generate_next_steps(score):
    """Generate next steps based on score."""
    if score >= 90:
        return [
            "1. Deploy to production environment",
            "2. Set up monitoring and alerting", 
            "3. Prepare user onboarding documentation",
            "4. Plan marketing and launch activities"
        ]
    elif score >= 80:
        return [
            "1. Deploy to staging environment",
            "2. Address minor issues identified in testing",
            "3. Conduct user acceptance testing",
            "4. Prepare production deployment plan"
        ]
    elif score >= 70:
        return [
            "1. Deploy to development environment",
            "2. Fix failing test cases",
            "3. Enhance system reliability",
            "4. Re-run validation tests"
        ]
    else:
        return [
            "1. Address critical system failures",
            "2. Implement missing core functionality",
            "3. Strengthen quality assurance processes",
            "4. Re-architecture if necessary"
        ]

def main():
    """Execute final autonomous SDLC validation."""
    
    print("🤖 TERRAGON AUTONOMOUS SDLC v4.0")
    print("🎯 FINAL VALIDATION & DEPLOYMENT READINESS ASSESSMENT")
    print()
    
    # Run all validation tests
    results, detailed_results = validate_all_generations()
    
    # Calculate overall score
    overall_score = calculate_overall_sdlc_score(results, detailed_results)
    
    # Generate final report
    final_report = generate_final_report(results, detailed_results, overall_score)
    
    # Display results
    print('\n' + '=' * 70)
    print('📋 FINAL AUTONOMOUS SDLC VALIDATION RESULTS')
    print('=' * 70)
    
    # Phase summary
    passed_phases = sum(1 for success in results.values() if success)
    total_phases = len(results)
    
    print(f'📊 Phase Completion: {passed_phases}/{total_phases} phases passed')
    print(f'🎯 Overall Score: {overall_score:.1f}%')
    
    # Deployment readiness
    readiness = final_report['deployment_readiness']
    print(f'🚀 Deployment Status: {readiness["status"]}')
    print(f'📋 Recommendation: {readiness["recommendation"]}')
    print(f'🔍 Confidence: {readiness["confidence"]}')
    
    # Phase details
    print('\n📈 PHASE RESULTS:')
    for phase, success in results.items():
        status = '✅ PASS' if success else '❌ FAIL'
        print(f'  {status} {phase}')
        
        # Show key metrics if available
        metrics = detailed_results.get(phase, {}).get('metrics', {})
        for metric, value in list(metrics.items())[:2]:  # Show top 2 metrics
            print(f'      📊 {metric}: {value}')
    
    # System capabilities
    print('\n🚀 SYSTEM CAPABILITIES:')
    capabilities = final_report['system_capabilities']
    capability_count = sum(len(metrics) for metrics in capabilities.values())
    print(f'  📊 Total Capabilities Validated: {capability_count}')
    
    # Top capabilities
    all_metrics = []
    for phase_metrics in capabilities.values():
        all_metrics.extend(phase_metrics.items())
    
    for metric, value in all_metrics[:5]:  # Show top 5
        print(f'  ⚡ {metric}: {value}')
    
    # Recommendations
    print('\n💡 RECOMMENDATIONS:')
    for rec in final_report['recommendations']:
        print(f'  {rec}')
    
    # Next steps
    print('\n📋 NEXT STEPS:')
    for step in final_report['next_steps']:
        print(f'  {step}')
    
    # Save detailed report
    report_file = 'final_sdlc_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    print(f'\n💾 Detailed report saved: {report_file}')
    
    # Final status
    print('\n' + '=' * 70)
    if overall_score >= 80:
        print('🎉 AUTONOMOUS SDLC SUCCESSFULLY COMPLETED!')
        print('🚀 SYSTEM IS READY FOR DEPLOYMENT!')
    elif overall_score >= 70:
        print('✅ AUTONOMOUS SDLC SUBSTANTIALLY COMPLETED')
        print('🔧 MINOR ENHANCEMENTS NEEDED BEFORE DEPLOYMENT')
    else:
        print('⚠️  AUTONOMOUS SDLC REQUIRES ADDITIONAL WORK')
        print('🔨 SIGNIFICANT IMPROVEMENTS NEEDED')
    
    print(f'📊 Final Score: {overall_score:.1f}%')
    print('=' * 70)
    
    return overall_score >= 70  # Success threshold

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)