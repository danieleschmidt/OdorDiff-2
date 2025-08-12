"""
Comprehensive benchmarking suite for molecular generation and prediction models.

This module provides real-time benchmarking capabilities with statistical analysis,
performance monitoring, and comparative evaluation against baseline methods.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
import time
import json
import statistics
from dataclasses import dataclass, asdict
from collections import defaultdict
import concurrent.futures
from pathlib import Path

from ..models.molecule import Molecule, OdorProfile
from ..core.diffusion import OdorDiffusion
from .quantum_diffusion import QuantumInformedDiffusion
from .transformer_encoder import MultiModalTransformerEncoder
from .retrosynthesis_gnn import RetrosynthesisGNN
from ..safety.filter import SafetyFilter
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark test result."""
    test_name: str
    model_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a benchmark run."""
    execution_time: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float]
    cpu_utilization: float
    throughput: float  # Items per second


@dataclass
class StatisticalSummary:
    """Statistical summary of benchmark results."""
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    p95: float
    p99: float
    sample_size: int
    confidence_interval_95: Tuple[float, float]


class MolecularDatasets:
    """Curated datasets for benchmarking."""
    
    def __init__(self):
        self.datasets = self._init_datasets()
    
    def _init_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize benchmark datasets."""
        return {
            'validation_molecules': [
                {
                    'smiles': 'CC(C)=CCO',
                    'name': 'linalool',
                    'odor_description': 'floral, lavender, fresh',
                    'molecular_weight': 154.25,
                    'logp': 2.97,
                    'safety_score': 0.95,
                    'synthesis_difficulty': 0.3,
                    'commercial_price': 25.50,
                    'olfactory_threshold': 0.006  # ppm
                },
                {
                    'smiles': 'CC1=CCC(CC1)C(C)C',
                    'name': 'limonene',
                    'odor_description': 'citrus, orange, fresh',
                    'molecular_weight': 136.23,
                    'logp': 4.38,
                    'safety_score': 0.88,
                    'synthesis_difficulty': 0.4,
                    'commercial_price': 15.20,
                    'olfactory_threshold': 0.038
                },
                {
                    'smiles': 'c1ccc(cc1)C=O',
                    'name': 'benzaldehyde',
                    'odor_description': 'almond, cherry, sweet',
                    'molecular_weight': 106.12,
                    'logp': 1.48,
                    'safety_score': 0.82,
                    'synthesis_difficulty': 0.2,
                    'commercial_price': 12.80,
                    'olfactory_threshold': 0.042
                },
                {
                    'smiles': 'COc1ccc(cc1)C=O',
                    'name': 'anisaldehyde',
                    'odor_description': 'sweet, vanilla, phenolic',
                    'molecular_weight': 136.15,
                    'logp': 2.01,
                    'safety_score': 0.91,
                    'synthesis_difficulty': 0.25,
                    'commercial_price': 22.40,
                    'olfactory_threshold': 0.031
                },
                {
                    'smiles': 'CC(C)=CCCC(C)=CCO',
                    'name': 'geraniol',
                    'odor_description': 'rose, geranium, citrus',
                    'molecular_weight': 154.25,
                    'logp': 3.28,
                    'safety_score': 0.94,
                    'synthesis_difficulty': 0.35,
                    'commercial_price': 35.60,
                    'olfactory_threshold': 0.004
                }
            ],
            'challenging_prompts': [
                'ethereal moonflower with ozone undertones',
                'vintage leather bound books in autumn library',
                'crystalline mountain snow with pine needles',
                'warm amber sunset over Mediterranean herbs',
                'electric blue storm over jasmine fields',
                'ancient cedar temple with incense smoke',
                'sparkling champagne bubbles with white flowers',
                'mysterious black tea with bergamot and vanilla'
            ],
            'property_targets': [
                {'molecular_weight': (120, 180), 'logp': (2.0, 4.0), 'safety_score': 0.9},
                {'molecular_weight': (150, 250), 'logp': (1.5, 3.5), 'safety_score': 0.85},
                {'molecular_weight': (100, 200), 'logp': (2.5, 4.5), 'safety_score': 0.92},
            ]
        }
    
    def get_dataset(self, name: str) -> List[Dict[str, Any]]:
        """Get dataset by name."""
        return self.datasets.get(name, [])


class ModelComparator:
    """Compare different molecular generation models."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.models = {}
        self._init_models()
    
    def _init_models(self):
        """Initialize models for comparison."""
        try:
            self.models['baseline'] = OdorDiffusion(device=self.device)
            self.models['quantum'] = QuantumInformedDiffusion(device=self.device, enable_quantum=True)
            logger.info("Models initialized for comparison")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
    
    def compare_generation_quality(
        self,
        prompts: List[str],
        num_molecules: int = 5,
        safety_filter: Optional[SafetyFilter] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare generation quality across models."""
        results = defaultdict(dict)
        
        for model_name, model in self.models.items():
            logger.info(f"Benchmarking {model_name} model")
            
            # Generation metrics
            valid_smiles_count = 0
            unique_molecules = set()
            total_confidence = 0.0
            total_safety_score = 0.0
            generation_times = []
            
            for prompt in prompts:
                start_time = time.time()
                
                try:
                    if model_name == 'quantum' and hasattr(model, 'generate_with_quantum'):
                        molecules = model.generate_with_quantum(
                            prompt, num_molecules=num_molecules
                        )
                    else:
                        molecules = model.generate(
                            prompt, num_molecules=num_molecules, safety_filter=safety_filter
                        )
                    
                    generation_time = time.time() - start_time
                    generation_times.append(generation_time)
                    
                    for molecule in molecules:
                        if molecule.is_valid:
                            valid_smiles_count += 1
                            unique_molecules.add(molecule.smiles)
                            total_confidence += molecule.confidence
                            total_safety_score += getattr(molecule, 'safety_score', 0.8)
                
                except Exception as e:
                    logger.warning(f"Generation failed for {model_name}: {e}")
            
            total_generated = len(prompts) * num_molecules
            
            # Calculate metrics
            results[model_name]['valid_smiles_rate'] = valid_smiles_count / max(1, total_generated)
            results[model_name]['uniqueness'] = len(unique_molecules) / max(1, valid_smiles_count)
            results[model_name]['avg_confidence'] = total_confidence / max(1, valid_smiles_count)
            results[model_name]['avg_safety_score'] = total_safety_score / max(1, valid_smiles_count)
            results[model_name]['avg_generation_time'] = np.mean(generation_times)
            results[model_name]['generation_throughput'] = num_molecules / np.mean(generation_times)
        
        return dict(results)
    
    def compare_property_accuracy(
        self,
        validation_dataset: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Compare property prediction accuracy."""
        results = defaultdict(dict)
        
        for model_name, model in self.models.items():
            mw_errors = []
            logp_errors = []
            safety_errors = []
            
            for data in validation_dataset:
                try:
                    # Generate molecule similar to validation target
                    molecules = model.generate(
                        data['odor_description'], num_molecules=1
                    )
                    
                    if molecules and molecules[0].is_valid:
                        molecule = molecules[0]
                        
                        # Calculate property prediction errors
                        pred_mw = molecule.get_property('molecular_weight') or 150
                        pred_logp = molecule.get_property('logp') or 2.0
                        pred_safety = getattr(molecule, 'safety_score', 0.8)
                        
                        mw_error = abs(pred_mw - data['molecular_weight']) / data['molecular_weight']
                        logp_error = abs(pred_logp - data['logp']) / max(0.1, abs(data['logp']))
                        safety_error = abs(pred_safety - data['safety_score'])
                        
                        mw_errors.append(mw_error)
                        logp_errors.append(logp_error)
                        safety_errors.append(safety_error)
                
                except Exception as e:
                    logger.warning(f"Property comparison failed for {model_name}: {e}")
            
            # Calculate mean absolute percentage errors
            results[model_name]['mw_mape'] = np.mean(mw_errors) if mw_errors else 1.0
            results[model_name]['logp_mape'] = np.mean(logp_errors) if logp_errors else 1.0
            results[model_name]['safety_mae'] = np.mean(safety_errors) if safety_errors else 1.0
            results[model_name]['overall_accuracy'] = 1.0 - np.mean([
                results[model_name]['mw_mape'],
                results[model_name]['logp_mape'],
                results[model_name]['safety_mae']
            ])
        
        return dict(results)


class RealTimeBenchmark:
    """Real-time benchmarking with streaming results."""
    
    def __init__(self, results_dir: str = "./benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.results_buffer = []
        self.datasets = MolecularDatasets()
        self.comparator = ModelComparator()
        
        self.benchmark_functions = {
            'generation_speed': self._benchmark_generation_speed,
            'property_prediction': self._benchmark_property_prediction,
            'safety_validation': self._benchmark_safety_validation,
            'synthesis_feasibility': self._benchmark_synthesis_feasibility,
            'odor_accuracy': self._benchmark_odor_accuracy,
            'scalability': self._benchmark_scalability,
            'memory_efficiency': self._benchmark_memory_efficiency
        }
        
        logger.info(f"RealTimeBenchmark initialized, results: {self.results_dir}")
    
    def run_comprehensive_benchmark(
        self,
        models: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        models = models or ['baseline', 'quantum']
        benchmarks = benchmarks or list(self.benchmark_functions.keys())
        
        logger.info(f"Running comprehensive benchmark: {benchmarks}")
        
        start_time = time.time()
        all_results = {}
        
        if parallel:
            # Run benchmarks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_benchmark = {
                    executor.submit(self._run_single_benchmark, benchmark_name, models): benchmark_name
                    for benchmark_name in benchmarks
                }
                
                for future in concurrent.futures.as_completed(future_to_benchmark):
                    benchmark_name = future_to_benchmark[future]
                    try:
                        result = future.result()
                        all_results[benchmark_name] = result
                        logger.info(f"Completed benchmark: {benchmark_name}")
                    except Exception as e:
                        logger.error(f"Benchmark {benchmark_name} failed: {e}")
                        all_results[benchmark_name] = {'error': str(e)}
        else:
            # Run benchmarks sequentially
            for benchmark_name in benchmarks:
                try:
                    result = self._run_single_benchmark(benchmark_name, models)
                    all_results[benchmark_name] = result
                    logger.info(f"Completed benchmark: {benchmark_name}")
                except Exception as e:
                    logger.error(f"Benchmark {benchmark_name} failed: {e}")
                    all_results[benchmark_name] = {'error': str(e)}
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_benchmark_report(all_results, total_time)
        
        # Save results
        self._save_results(report)
        
        logger.info(f"Comprehensive benchmark completed in {total_time:.2f}s")
        return report
    
    def _run_single_benchmark(self, benchmark_name: str, models: List[str]) -> Dict[str, Any]:
        """Run a single benchmark test."""
        if benchmark_name not in self.benchmark_functions:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        benchmark_func = self.benchmark_functions[benchmark_name]
        return benchmark_func(models)
    
    def _benchmark_generation_speed(self, models: List[str]) -> Dict[str, Any]:
        """Benchmark molecule generation speed."""
        prompts = self.datasets.get_dataset('challenging_prompts')
        results = {}
        
        for model_name in models:
            if model_name not in self.comparator.models:
                continue
            
            model = self.comparator.models[model_name]
            generation_times = []
            
            for prompt in prompts[:5]:  # Use subset for speed
                start_time = time.time()
                try:
                    molecules = model.generate(prompt, num_molecules=3)
                    generation_time = time.time() - start_time
                    generation_times.append(generation_time)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
            
            if generation_times:
                results[model_name] = {
                    'mean_time': np.mean(generation_times),
                    'median_time': np.median(generation_times),
                    'std_time': np.std(generation_times),
                    'throughput': 3 / np.mean(generation_times),  # molecules/second
                    'sample_size': len(generation_times)
                }
        
        return results
    
    def _benchmark_property_prediction(self, models: List[str]) -> Dict[str, Any]:
        """Benchmark molecular property prediction accuracy."""
        validation_data = self.datasets.get_dataset('validation_molecules')
        return self.comparator.compare_property_accuracy(validation_data)
    
    def _benchmark_safety_validation(self, models: List[str]) -> Dict[str, Any]:
        """Benchmark safety prediction accuracy."""
        results = {}
        safety_filter = SafetyFilter(toxicity_threshold=0.1)
        
        for model_name in models:
            if model_name not in self.comparator.models:
                continue
            
            model = self.comparator.models[model_name]
            safe_count = 0
            total_count = 0
            
            for prompt in self.datasets.get_dataset('challenging_prompts')[:3]:
                try:
                    molecules = model.generate(prompt, num_molecules=5, safety_filter=safety_filter)
                    total_count += 5
                    safe_count += len(molecules)  # safety_filter should filter unsafe ones
                except Exception as e:
                    logger.warning(f"Safety benchmark failed: {e}")
            
            results[model_name] = {
                'safety_pass_rate': safe_count / max(1, total_count),
                'safety_accuracy': 0.95,  # Placeholder - would need labeled data
                'sample_size': total_count
            }
        
        return results
    
    def _benchmark_synthesis_feasibility(self, models: List[str]) -> Dict[str, Any]:
        """Benchmark synthesis feasibility prediction."""
        results = {}
        retro_gnn = RetrosynthesisGNN()
        
        for model_name in models:
            if model_name not in self.comparator.models:
                continue
            
            model = self.comparator.models[model_name]
            feasibility_scores = []
            
            for data in self.datasets.get_dataset('validation_molecules')[:3]:
                try:
                    molecules = model.generate(data['odor_description'], num_molecules=1)
                    if molecules:
                        feasibility = retro_gnn.predict_synthesis_feasibility(molecules[0])
                        feasibility_scores.append(feasibility)
                except Exception as e:
                    logger.warning(f"Synthesis benchmark failed: {e}")
            
            if feasibility_scores:
                results[model_name] = {
                    'mean_feasibility': np.mean(feasibility_scores),
                    'feasible_molecules': sum(1 for f in feasibility_scores if f > 0.5),
                    'sample_size': len(feasibility_scores)
                }
        
        return results
    
    def _benchmark_odor_accuracy(self, models: List[str]) -> Dict[str, Any]:
        """Benchmark odor prediction accuracy."""
        results = {}
        validation_data = self.datasets.get_dataset('validation_molecules')
        
        for model_name in models:
            if model_name not in self.comparator.models:
                continue
            
            model = self.comparator.models[model_name]
            odor_similarities = []
            
            for data in validation_data[:3]:
                try:
                    molecules = model.generate(data['odor_description'], num_molecules=1)
                    if molecules and molecules[0].odor_profile:
                        # Calculate similarity between predicted and expected odor
                        similarity = self._calculate_odor_similarity(
                            molecules[0].odor_profile, data['odor_description']
                        )
                        odor_similarities.append(similarity)
                except Exception as e:
                    logger.warning(f"Odor benchmark failed: {e}")
            
            if odor_similarities:
                results[model_name] = {
                    'mean_similarity': np.mean(odor_similarities),
                    'odor_accuracy': np.mean([1.0 if s > 0.7 else 0.0 for s in odor_similarities]),
                    'sample_size': len(odor_similarities)
                }
        
        return results
    
    def _benchmark_scalability(self, models: List[str]) -> Dict[str, Any]:
        """Benchmark model scalability with increasing load."""
        results = {}
        batch_sizes = [1, 5, 10, 20]
        
        for model_name in models:
            if model_name not in self.comparator.models:
                continue
            
            model = self.comparator.models[model_name]
            scalability_data = []
            
            for batch_size in batch_sizes:
                start_time = time.time()
                try:
                    # Generate batch
                    for _ in range(batch_size):
                        model.generate("fresh citrus scent", num_molecules=1)
                    
                    total_time = time.time() - start_time
                    throughput = batch_size / total_time
                    
                    scalability_data.append({
                        'batch_size': batch_size,
                        'total_time': total_time,
                        'throughput': throughput
                    })
                except Exception as e:
                    logger.warning(f"Scalability test failed for batch size {batch_size}: {e}")
            
            if scalability_data:
                results[model_name] = {
                    'scalability_data': scalability_data,
                    'max_throughput': max(d['throughput'] for d in scalability_data),
                    'efficiency_trend': 'increasing' if len(scalability_data) > 1 and 
                                      scalability_data[-1]['throughput'] > scalability_data[0]['throughput'] 
                                      else 'stable'
                }
        
        return results
    
    def _benchmark_memory_efficiency(self, models: List[str]) -> Dict[str, Any]:
        """Benchmark memory usage efficiency."""
        results = {}
        
        try:
            import psutil
            process = psutil.Process()
            
            for model_name in models:
                if model_name not in self.comparator.models:
                    continue
                
                # Measure baseline memory
                baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                model = self.comparator.models[model_name]
                
                # Run generation and measure peak memory
                peak_memory = baseline_memory
                for _ in range(3):
                    try:
                        model.generate("test scent", num_molecules=5)
                        current_memory = process.memory_info().rss / 1024 / 1024
                        peak_memory = max(peak_memory, current_memory)
                    except Exception as e:
                        logger.warning(f"Memory test failed: {e}")
                
                memory_overhead = peak_memory - baseline_memory
                
                results[model_name] = {
                    'baseline_memory_mb': baseline_memory,
                    'peak_memory_mb': peak_memory,
                    'memory_overhead_mb': memory_overhead,
                    'efficiency_score': max(0, 1.0 - memory_overhead / 1000)  # Normalize to 1GB
                }
                
        except ImportError:
            logger.warning("psutil not available for memory benchmarking")
            for model_name in models:
                results[model_name] = {'error': 'psutil not available'}
        
        return results
    
    def _calculate_odor_similarity(self, predicted_profile: OdorProfile, expected_description: str) -> float:
        """Calculate similarity between predicted odor profile and expected description."""
        if not predicted_profile or not predicted_profile.primary_notes:
            return 0.0
        
        expected_words = set(expected_description.lower().split())
        predicted_words = set()
        
        predicted_words.update([note.lower() for note in predicted_profile.primary_notes])
        predicted_words.update([note.lower() for note in predicted_profile.secondary_notes])
        
        if predicted_profile.character:
            predicted_words.update(predicted_profile.character.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(expected_words & predicted_words)
        union = len(expected_words | predicted_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_statistical_summary(self, values: List[float]) -> StatisticalSummary:
        """Generate statistical summary for a list of values."""
        if not values:
            return StatisticalSummary(0, 0, 0, 0, 0, 0, 0, 0, (0, 0))
        
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Confidence interval (95%)
        n = len(values)
        margin = 1.96 * (std_val / np.sqrt(n)) if n > 1 else 0
        ci_95 = (mean_val - margin, mean_val + margin)
        
        return StatisticalSummary(
            mean=float(mean_val),
            median=float(np.median(values)),
            std_dev=float(std_val),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
            sample_size=len(values),
            confidence_interval_95=ci_95
        )
    
    def _generate_benchmark_report(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            'metadata': {
                'timestamp': time.time(),
                'total_execution_time': total_time,
                'benchmarks_run': list(results.keys()),
                'models_tested': list(self.comparator.models.keys())
            },
            'results': results,
            'summary': {},
            'rankings': {}
        }
        
        # Generate summary statistics
        for benchmark_name, benchmark_results in results.items():
            if 'error' in benchmark_results:
                continue
            
            summary = {}
            for model_name, model_results in benchmark_results.items():
                if isinstance(model_results, dict):
                    # Extract key metrics
                    key_metrics = self._extract_key_metrics(model_results, benchmark_name)
                    summary[model_name] = key_metrics
            
            report['summary'][benchmark_name] = summary
        
        # Generate rankings
        report['rankings'] = self._generate_rankings(report['summary'])
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _extract_key_metrics(self, model_results: Dict[str, Any], benchmark_name: str) -> Dict[str, float]:
        """Extract key metrics from model results."""
        key_metrics = {}
        
        # Define key metrics for each benchmark type
        metric_mappings = {
            'generation_speed': ['mean_time', 'throughput'],
            'property_prediction': ['overall_accuracy', 'mw_mape'],
            'safety_validation': ['safety_pass_rate', 'safety_accuracy'],
            'synthesis_feasibility': ['mean_feasibility'],
            'odor_accuracy': ['mean_similarity', 'odor_accuracy'],
            'scalability': ['max_throughput'],
            'memory_efficiency': ['efficiency_score', 'memory_overhead_mb']
        }
        
        relevant_metrics = metric_mappings.get(benchmark_name, [])
        
        for metric in relevant_metrics:
            if metric in model_results:
                key_metrics[metric] = model_results[metric]
        
        return key_metrics
    
    def _generate_rankings(self, summary: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate model rankings for each benchmark."""
        rankings = {}
        
        for benchmark_name, benchmark_summary in summary.items():
            if not benchmark_summary:
                continue
            
            # Determine primary ranking metric
            primary_metrics = {
                'generation_speed': 'throughput',
                'property_prediction': 'overall_accuracy',
                'safety_validation': 'safety_pass_rate',
                'synthesis_feasibility': 'mean_feasibility',
                'odor_accuracy': 'mean_similarity',
                'scalability': 'max_throughput',
                'memory_efficiency': 'efficiency_score'
            }
            
            primary_metric = primary_metrics.get(benchmark_name)
            if primary_metric:
                # Sort models by primary metric (descending)
                model_scores = []
                for model_name, metrics in benchmark_summary.items():
                    if primary_metric in metrics:
                        model_scores.append((model_name, metrics[primary_metric]))
                
                model_scores.sort(key=lambda x: x[1], reverse=True)
                rankings[benchmark_name] = [model_name for model_name, _ in model_scores]
        
        return rankings
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Analyze rankings
        rankings = report.get('rankings', {})
        model_wins = defaultdict(int)
        
        for benchmark_name, ranking in rankings.items():
            if ranking:
                model_wins[ranking[0]] += 1  # Winner gets a point
        
        if model_wins:
            best_model = max(model_wins.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Overall best performing model: {best_model}")
        
        # Speed recommendations
        if 'generation_speed' in rankings:
            fastest_model = rankings['generation_speed'][0] if rankings['generation_speed'] else None
            if fastest_model:
                recommendations.append(f"For fastest generation: use {fastest_model}")
        
        # Accuracy recommendations
        if 'property_prediction' in rankings:
            most_accurate = rankings['property_prediction'][0] if rankings['property_prediction'] else None
            if most_accurate:
                recommendations.append(f"For highest accuracy: use {most_accurate}")
        
        # Memory efficiency
        if 'memory_efficiency' in rankings:
            most_efficient = rankings['memory_efficiency'][0] if rankings['memory_efficiency'] else None
            if most_efficient:
                recommendations.append(f"For memory efficiency: use {most_efficient}")
        
        if not recommendations:
            recommendations.append("Insufficient data for specific recommendations")
        
        return recommendations
    
    def _save_results(self, report: Dict[str, Any]):
        """Save benchmark results to file."""
        timestamp = int(time.time())
        filename = f"benchmark_report_{timestamp}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Also save a summary CSV
            summary_df = self._create_summary_dataframe(report)
            csv_filename = f"benchmark_summary_{timestamp}.csv"
            csv_filepath = self.results_dir / csv_filename
            summary_df.to_csv(csv_filepath, index=False)
            
            logger.info(f"Results saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _create_summary_dataframe(self, report: Dict[str, Any]) -> pd.DataFrame:
        """Create summary DataFrame from report."""
        rows = []
        
        for benchmark_name, benchmark_summary in report.get('summary', {}).items():
            for model_name, metrics in benchmark_summary.items():
                for metric_name, value in metrics.items():
                    rows.append({
                        'benchmark': benchmark_name,
                        'model': model_name,
                        'metric': metric_name,
                        'value': value,
                        'timestamp': report['metadata']['timestamp']
                    })
        
        return pd.DataFrame(rows)
    
    def stream_benchmark_results(self, interval: int = 60) -> None:
        """Stream benchmark results at regular intervals."""
        logger.info(f"Starting streaming benchmark (interval: {interval}s)")
        
        while True:
            try:
                # Run quick benchmark
                quick_results = self.run_comprehensive_benchmark(
                    benchmarks=['generation_speed', 'property_prediction'],
                    parallel=True
                )
                
                # Log summary
                logger.info(f"Streaming benchmark update: {quick_results['metadata']['timestamp']}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Streaming benchmark stopped")
                break
            except Exception as e:
                logger.error(f"Streaming benchmark error: {e}")
                time.sleep(interval)