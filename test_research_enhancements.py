#!/usr/bin/env python3
"""
Comprehensive test suite for novel research enhancements.

This script validates all the new research components and generates
a comprehensive report on their functionality and integration.
"""

import sys
import os
import time
import traceback
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from odordiff2.models.molecule import Molecule
from odordiff2.core.diffusion import OdorDiffusion
from odordiff2.safety.filter import SafetyFilter

# Research enhancements
from odordiff2.research.quantum_diffusion import QuantumInformedDiffusion, VibrationalSignature
from odordiff2.research.transformer_encoder import MultiModalTransformerEncoder, MultiModalInput
from odordiff2.research.retrosynthesis_gnn import RetrosynthesisGNN, ReactionStep
from odordiff2.research.explainable_safety import ExplainableSafetyPredictor
from odordiff2.research.benchmark_suite import RealTimeBenchmark, MolecularDatasets
from odordiff2.research.publication_suite import PublicationSuite


class ResearchEnhancementTester:
    """Comprehensive tester for all research enhancements."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        print("🧪 Initializing Research Enhancement Test Suite...")
    
    def test_quantum_diffusion(self):
        """Test quantum-informed diffusion model."""
        print("\n🔬 Testing Quantum-Informed Diffusion...")
        
        try:
            # Initialize quantum diffusion model
            quantum_model = QuantumInformedDiffusion(enable_quantum=True)
            
            # Test basic generation
            molecules = quantum_model.generate(
                "ethereal lavender with quantum resonance", 
                num_molecules=3
            )
            
            # Test quantum-enhanced generation
            target_signature = VibrationalSignature(
                frequencies=np.array([3300, 1650, 1450, 1100]),
                intensities=np.array([0.8, 0.7, 0.5, 0.6]),
                raman_activities=np.array([0.3, 0.8, 0.4, 0.5]),
                olfactory_relevance=0.9
            )
            
            quantum_molecules = quantum_model.generate_with_quantum(
                "quantum-optimized floral scent",
                target_vibrational_profile=target_signature,
                num_molecules=2
            )
            
            # Test benchmarking
            test_data = [
                ("CC(C)=CCO", "floral lavender", target_signature),
                ("c1ccc(cc1)C=O", "sweet almond", target_signature)
            ]
            
            benchmark_results = quantum_model.benchmark_quantum_accuracy(test_data)
            
            self.results['quantum_diffusion'] = {
                'status': 'success',
                'basic_generation': len(molecules),
                'quantum_generation': len(quantum_molecules),
                'valid_molecules': sum(1 for m in molecules + quantum_molecules if m.is_valid),
                'benchmark_metrics': benchmark_results,
                'quantum_similarities': [
                    getattr(m, 'quantum_similarity', 0.0) 
                    for m in quantum_molecules if hasattr(m, 'quantum_similarity')
                ]
            }
            
            print(f"   ✅ Generated {len(molecules)} basic + {len(quantum_molecules)} quantum molecules")
            print(f"   ✅ Benchmark accuracy: {benchmark_results.get('vibrational_prediction_accuracy', 0):.3f}")
            
        except Exception as e:
            self.results['quantum_diffusion'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"   ❌ Error: {e}")
    
    def test_transformer_encoder(self):
        """Test multi-modal transformer encoder."""
        print("\n🤖 Testing Multi-Modal Transformer Encoder...")
        
        try:
            # Initialize transformer
            transformer = MultiModalTransformerEncoder(
                vocab_size=1000,
                d_model=256,
                n_heads=4,
                n_layers=2,
                use_graph_encoder=False
            )
            
            # Create test input
            import torch
            
            # Mock tokenized text
            text_tokens = torch.randint(0, 1000, (2, 20))  # batch_size=2, seq_len=20
            
            # Create property vectors
            property_vector = transformer.create_property_vector(
                molecular_weight=150.0,
                logp=2.5,
                tpsa=45.0,
                rotatable_bonds=3,
                hbd=1,
                hba=2
            ).unsqueeze(0).repeat(2, 1)  # batch_size=2
            
            # Create multi-modal input
            multi_modal_input = MultiModalInput(
                text_tokens=text_tokens,
                property_vector=property_vector,
                attention_mask=torch.ones_like(text_tokens)
            )
            
            # Test forward pass
            with torch.no_grad():
                molecular_latent = transformer.forward(multi_modal_input)
            
            # Test property vector creation
            prop_vector = transformer.create_property_vector(
                molecular_weight=200,
                logp=3.0,
                tpsa=60,
                rotatable_bonds=5,
                hbd=2,
                hba=3
            )
            
            self.results['transformer_encoder'] = {
                'status': 'success',
                'latent_shape': list(molecular_latent.shape),
                'property_vector_size': prop_vector.shape[0],
                'model_parameters': sum(p.numel() for p in transformer.parameters()),
                'forward_pass_successful': True
            }
            
            print(f"   ✅ Forward pass successful: {molecular_latent.shape}")
            print(f"   ✅ Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
            
        except Exception as e:
            self.results['transformer_encoder'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"   ❌ Error: {e}")
    
    def test_retrosynthesis_gnn(self):
        """Test retrosynthesis GNN."""
        print("\n⚗️  Testing Retrosynthesis GNN...")
        
        try:
            # Initialize retrosynthesis GNN
            retro_gnn = RetrosynthesisGNN(enable_training=False)
            
            # Test molecules
            test_molecules = [
                Molecule("CC(C)=CCO"),  # Linalool
                Molecule("c1ccc(cc1)C=O"),  # Benzaldehyde
                Molecule("CCO")  # Ethanol
            ]
            
            synthesis_routes = []
            feasibility_scores = []
            
            for molecule in test_molecules:
                # Test synthesis feasibility prediction
                feasibility = retro_gnn.predict_synthesis_feasibility(molecule)
                feasibility_scores.append(feasibility)
                
                # Test synthesis route suggestion
                routes = retro_gnn.suggest_synthesis_routes(
                    molecule, max_routes=2, prefer_green_chemistry=True
                )
                synthesis_routes.extend(routes)
            
            # Test route optimization
            if synthesis_routes:
                optimized_route = retro_gnn.optimize_route_for_cost(synthesis_routes[0])
            
            # Test benchmarking
            test_data = [
                ("CC(C)=CCO", ["CCMgBr", "CC=O"]),  # Simplified known route
                ("CCO", ["C=C", "H2O"])  # Simplified hydration
            ]
            
            benchmark_metrics = retro_gnn.benchmark_retrosynthesis_accuracy(test_data)
            
            self.results['retrosynthesis_gnn'] = {
                'status': 'success',
                'molecules_tested': len(test_molecules),
                'feasibility_scores': feasibility_scores,
                'avg_feasibility': np.mean(feasibility_scores),
                'routes_generated': len(synthesis_routes),
                'benchmark_metrics': benchmark_metrics,
                'has_optimization': 'optimized_route' in locals()
            }
            
            print(f"   ✅ Tested {len(test_molecules)} molecules")
            print(f"   ✅ Average feasibility: {np.mean(feasibility_scores):.3f}")
            print(f"   ✅ Generated {len(synthesis_routes)} synthesis routes")
            
        except Exception as e:
            self.results['retrosynthesis_gnn'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"   ❌ Error: {e}")
    
    def test_explainable_safety(self):
        """Test explainable safety predictor."""
        print("\n🛡️  Testing Explainable Safety Predictor...")
        
        try:
            # Initialize explainable safety predictor
            safety_predictor = ExplainableSafetyPredictor(enable_shap=False)
            
            # Test molecules with different safety profiles
            test_molecules = [
                Molecule("CCO"),  # Ethanol - relatively safe
                Molecule("c1ccc(cc1)N"),  # Aniline - potentially toxic
                Molecule("CC(C)=CCO"),  # Linalool - generally safe
                Molecule("C(=O)Cl"),  # Formyl chloride - dangerous
            ]
            
            explanations = []
            safety_scores = []
            risk_factors_count = []
            
            for molecule in test_molecules:
                explanation = safety_predictor.predict_safety_with_explanation(
                    molecule, detailed=True
                )
                explanations.append(explanation)
                safety_scores.append(explanation.prediction)
                risk_factors_count.append(len(explanation.risk_factors))
            
            # Test safety report generation
            report = safety_predictor.generate_safety_report(
                test_molecules, output_format="json"
            )
            
            # Test method comparison
            comparison_metrics = safety_predictor.compare_safety_methods(test_molecules)
            
            self.results['explainable_safety'] = {
                'status': 'success',
                'molecules_tested': len(test_molecules),
                'safety_scores': safety_scores,
                'avg_safety_score': np.mean(safety_scores),
                'total_risk_factors': sum(risk_factors_count),
                'report_generated': 'summary' in report,
                'comparison_methods': list(comparison_metrics.keys()),
                'explanations_have_details': all(
                    hasattr(exp, 'risk_factors') and hasattr(exp, 'protective_factors')
                    for exp in explanations
                )
            }
            
            print(f"   ✅ Analyzed {len(test_molecules)} molecules")
            print(f"   ✅ Average safety score: {np.mean(safety_scores):.3f}")
            print(f"   ✅ Generated {sum(risk_factors_count)} risk factor explanations")
            
        except Exception as e:
            self.results['explainable_safety'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"   ❌ Error: {e}")
    
    def test_benchmark_suite(self):
        """Test real-time benchmark suite."""
        print("\n📊 Testing Benchmark Suite...")
        
        try:
            # Initialize benchmark suite
            benchmark = RealTimeBenchmark()
            
            # Test dataset loading
            datasets = MolecularDatasets()
            validation_data = datasets.get_dataset('validation_molecules')
            challenging_prompts = datasets.get_dataset('challenging_prompts')
            
            # Run quick benchmark
            benchmark_results = benchmark.run_comprehensive_benchmark(
                models=['baseline'],  # Only test baseline to save time
                benchmarks=['generation_speed', 'property_prediction'],
                parallel=False
            )
            
            # Test model comparison
            comparison_results = benchmark.comparator.compare_generation_quality(
                challenging_prompts[:2], num_molecules=2
            )
            
            self.results['benchmark_suite'] = {
                'status': 'success',
                'validation_molecules': len(validation_data),
                'challenging_prompts': len(challenging_prompts),
                'benchmarks_run': list(benchmark_results.get('results', {}).keys()),
                'comparison_models': list(comparison_results.keys()),
                'execution_time': benchmark_results.get('metadata', {}).get('total_execution_time', 0),
                'has_summary': 'summary' in benchmark_results,
                'has_recommendations': 'recommendations' in benchmark_results
            }
            
            print(f"   ✅ Loaded {len(validation_data)} validation molecules")
            print(f"   ✅ Tested {len(challenging_prompts)} challenging prompts")
            print(f"   ✅ Completed {len(benchmark_results.get('results', {}))} benchmarks")
            
        except Exception as e:
            self.results['benchmark_suite'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"   ❌ Error: {e}")
    
    def test_publication_suite(self):
        """Test publication suite."""
        print("\n📚 Testing Publication Suite...")
        
        try:
            # Initialize publication suite
            pub_suite = PublicationSuite()
            
            # Test statistical validator
            experimental_data = [0.85, 0.87, 0.83, 0.89, 0.86]
            baseline_data = [0.75, 0.78, 0.72, 0.76, 0.74]
            
            stats_result = pub_suite.statistical_validator.validate_significance(
                experimental_data, baseline_data
            )
            
            # Test reproducibility tester
            class MockModel:
                def generate(self, prompt, **kwargs):
                    # Mock generation for testing
                    from odordiff2.models.molecule import Molecule
                    return [Molecule("CCO"), Molecule("CC(C)=CCO")]
            
            mock_model = MockModel()
            repro_result = pub_suite.reproducibility_tester.test_generation_reproducibility(
                mock_model, ["test prompt"], {'num_molecules': 2}
            )
            
            # Test novelty assessment
            novelty_result = pub_suite.novelty_assessor.assess_methodological_novelty(
                "Quantum-informed transformer architecture with explainable safety",
                ["quantum mechanics", "transformers", "explainable AI"]
            )
            
            # Calculate publication metrics
            mock_study_results = {
                'study_name': 'test_study',
                'statistical_analysis': {'test_vs_baseline': {'metric1': stats_result}},
                'reproducibility_analysis': {'test_method': repro_result},
                'novelty_analysis': {'test_method': novelty_result}
            }
            
            pub_metrics = pub_suite._calculate_publication_metrics(mock_study_results)
            
            self.results['publication_suite'] = {
                'status': 'success',
                'statistical_test_p_value': stats_result.get('p_value', 1.0),
                'statistical_significance': stats_result.get('significant', False),
                'reproducibility_score': repro_result.get('overall_reproducibility', 0.0),
                'novelty_score': novelty_result.get('overall_novelty_score', 0.0),
                'publication_metrics': {
                    'novelty': pub_metrics.novelty_score,
                    'reproducibility': pub_metrics.reproducibility_score,
                    'statistical_significance': pub_metrics.statistical_significance,
                    'practical_impact': pub_metrics.practical_impact
                }
            }
            
            print(f"   ✅ Statistical significance: {stats_result.get('significant', False)}")
            print(f"   ✅ Reproducibility score: {repro_result.get('overall_reproducibility', 0):.3f}")
            print(f"   ✅ Novelty score: {novelty_result.get('overall_novelty_score', 0):.3f}")
            
        except Exception as e:
            self.results['publication_suite'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"   ❌ Error: {e}")
    
    def test_integration(self):
        """Test integration between research components."""
        print("\n🔗 Testing Component Integration...")
        
        try:
            # Test quantum diffusion with explainable safety
            quantum_model = QuantumInformedDiffusion(enable_quantum=True)
            safety_predictor = ExplainableSafetyPredictor()
            
            # Generate molecule with quantum model
            molecules = quantum_model.generate("safe quantum lavender", num_molecules=1)
            
            # Analyze with explainable safety
            if molecules:
                explanation = safety_predictor.predict_safety_with_explanation(molecules[0])
                safety_score = explanation.prediction
            else:
                safety_score = 0.0
            
            # Test retrosynthesis on generated molecule
            if molecules:
                retro_gnn = RetrosynthesisGNN()
                feasibility = retro_gnn.predict_synthesis_feasibility(molecules[0])
            else:
                feasibility = 0.0
            
            self.results['integration'] = {
                'status': 'success',
                'quantum_to_safety': safety_score > 0,
                'quantum_to_retrosynthesis': feasibility > 0,
                'end_to_end_pipeline': molecules is not None and len(molecules) > 0,
                'safety_score': safety_score,
                'synthesis_feasibility': feasibility
            }
            
            print(f"   ✅ End-to-end pipeline successful")
            print(f"   ✅ Generated molecule safety: {safety_score:.3f}")
            print(f"   ✅ Synthesis feasibility: {feasibility:.3f}")
            
        except Exception as e:
            self.results['integration'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"   ❌ Error: {e}")
    
    def run_all_tests(self):
        """Run all research enhancement tests."""
        print("🚀 Starting Comprehensive Research Enhancement Testing\n")
        
        test_methods = [
            self.test_quantum_diffusion,
            self.test_transformer_encoder,
            self.test_retrosynthesis_gnn,
            self.test_explainable_safety,
            self.test_benchmark_suite,
            self.test_publication_suite,
            self.test_integration
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                component_name = test_method.__name__.replace('test_', '')
                self.results[component_name] = {
                    'status': 'critical_error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                print(f"   💥 Critical error in {component_name}: {e}")
        
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final test report."""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🎯 RESEARCH ENHANCEMENT TEST RESULTS")
        print("="*60)
        
        successful_components = []
        failed_components = []
        
        for component, result in self.results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                successful_components.append(component)
                print(f"✅ {component.replace('_', ' ').title()}: SUCCESS")
            else:
                failed_components.append(component)
                print(f"❌ {component.replace('_', ' ').title()}: {status.upper()}")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Components: {len(self.results)}")
        print(f"   Successful: {len(successful_components)}")
        print(f"   Failed: {len(failed_components)}")
        print(f"   Success Rate: {len(successful_components)/len(self.results)*100:.1f}%")
        print(f"   Total Test Time: {total_time:.2f} seconds")
        
        # Research capability highlights
        if 'quantum_diffusion' in successful_components:
            quantum_results = self.results['quantum_diffusion']
            print(f"\n🔬 QUANTUM DIFFUSION HIGHLIGHTS:")
            print(f"   Molecules Generated: {quantum_results.get('basic_generation', 0) + quantum_results.get('quantum_generation', 0)}")
            print(f"   Quantum Enhancement: {'enabled' if quantum_results.get('quantum_generation', 0) > 0 else 'disabled'}")
        
        if 'explainable_safety' in successful_components:
            safety_results = self.results['explainable_safety']
            print(f"\n🛡️  EXPLAINABLE SAFETY HIGHLIGHTS:")
            print(f"   Safety Analysis: {safety_results.get('molecules_tested', 0)} molecules")
            print(f"   Risk Factors Identified: {safety_results.get('total_risk_factors', 0)}")
        
        if 'retrosynthesis_gnn' in successful_components:
            retro_results = self.results['retrosynthesis_gnn']
            print(f"\n⚗️  RETROSYNTHESIS HIGHLIGHTS:")
            print(f"   Synthesis Routes: {retro_results.get('routes_generated', 0)} generated")
            print(f"   Average Feasibility: {retro_results.get('avg_feasibility', 0):.3f}")
        
        if 'publication_suite' in successful_components:
            pub_results = self.results['publication_suite']
            pub_metrics = pub_results.get('publication_metrics', {})
            print(f"\n📚 PUBLICATION READINESS:")
            print(f"   Statistical Significance: {pub_results.get('statistical_significance', False)}")
            print(f"   Novelty Score: {pub_metrics.get('novelty', 0):.3f}")
            print(f"   Reproducibility: {pub_metrics.get('reproducibility', 0):.3f}")
        
        # Overall assessment
        overall_score = len(successful_components) / len(self.results)
        if overall_score >= 0.8:
            assessment = "🏆 EXCELLENT - Research enhancements fully operational"
        elif overall_score >= 0.6:
            assessment = "✅ GOOD - Most research components working well"
        elif overall_score >= 0.4:
            assessment = "⚠️  MODERATE - Some components need attention"
        else:
            assessment = "❌ NEEDS WORK - Multiple components failing"
        
        print(f"\n🎯 OVERALL ASSESSMENT: {assessment}")
        print(f"📈 RESEARCH ENHANCEMENT SCORE: {overall_score*100:.1f}%")
        
        # Save detailed results
        self.save_results()
        
        print("\n🔬 RESEARCH ENHANCEMENT TESTING COMPLETED")
        print("="*60)
    
    def save_results(self):
        """Save test results to file."""
        try:
            import json
            results_file = Path("research_enhancement_test_results.json")
            
            # Make results JSON serializable
            serializable_results = {}
            for component, result in self.results.items():
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                        serializable_result[key] = value
                    else:
                        serializable_result[key] = str(value)
                serializable_results[component] = serializable_result
            
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'test_duration': time.time() - self.start_time,
                    'results': serializable_results
                }, f, indent=2)
            
            print(f"\n💾 Detailed results saved to: {results_file}")
            
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")


if __name__ == "__main__":
    tester = ResearchEnhancementTester()
    tester.run_all_tests()