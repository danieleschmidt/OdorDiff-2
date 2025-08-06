"""
Performance benchmarks for OdorDiff-2 system.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from odordiff2.core.diffusion import OdorDiffusion
from odordiff2.core.async_diffusion import AsyncOdorDiffusion
from odordiff2.safety.filter import SafetyFilter
from odordiff2.models.molecule import Molecule
from odordiff2.data.cache import MoleculeCache


class TestGenerationPerformance:
    """Test performance of molecule generation."""
    
    @pytest.mark.performance
    def test_single_generation_time(self, odor_diffusion, performance_test_data):
        """Test single molecule generation performance."""
        target_time = performance_test_data['generation_targets']['single_molecule_time']
        
        start_time = time.time()
        molecules = odor_diffusion.generate("fresh citrus scent", num_molecules=1)
        elapsed = time.time() - start_time
        
        assert len(molecules) >= 1
        assert elapsed < target_time, f"Generation took {elapsed:.2f}s, target was {target_time}s"
    
    @pytest.mark.performance
    def test_batch_generation_time(self, odor_diffusion, performance_test_data):
        """Test batch generation performance."""
        target_time = performance_test_data['generation_targets']['batch_5_time']
        
        start_time = time.time()
        molecules = odor_diffusion.generate("warm vanilla fragrance", num_molecules=5)
        elapsed = time.time() - start_time
        
        assert len(molecules) >= 3  # Should generate at least some molecules
        assert elapsed < target_time, f"Batch generation took {elapsed:.2f}s, target was {target_time}s"
    
    @pytest.mark.performance
    async def test_async_generation_performance(self, async_odor_diffusion, performance_test_data):
        """Test async generation performance."""
        target_time = performance_test_data['generation_targets']['single_molecule_time']
        
        start_time = time.time()
        result = await async_odor_diffusion.generate_async("floral rose bouquet", num_molecules=1)
        elapsed = time.time() - start_time
        
        assert len(result.molecules) >= 1
        assert elapsed < target_time
        assert result.processing_time < target_time
    
    @pytest.mark.performance
    async def test_concurrent_generation(self, async_odor_diffusion):
        """Test concurrent generation performance."""
        prompts = [
            "fresh citrus scent",
            "warm vanilla fragrance", 
            "floral rose bouquet",
            "woody cedar aroma",
            "clean aquatic breeze"
        ]
        
        start_time = time.time()
        
        # Run concurrent generations
        tasks = [
            async_odor_diffusion.generate_async(prompt, num_molecules=2) 
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        # Concurrent execution should be faster than sequential
        sequential_estimate = len(prompts) * 2.0  # Rough estimate
        assert elapsed < sequential_estimate * 0.8  # At least 20% faster
        
        # All should have results
        for result in results:
            assert len(result.molecules) >= 1
    
    @pytest.mark.performance
    def test_memory_usage_during_generation(self, performance_monitor):
        """Test memory usage during generation."""
        target_memory = 500  # MB
        
        model = OdorDiffusion(device="cpu")
        
        performance_monitor.start_monitoring()
        
        # Generate multiple batches to test memory usage
        for i in range(5):
            molecules = model.generate(f"test scent {i}", num_molecules=10)
            assert len(molecules) >= 1
        
        performance_monitor.stop_monitoring()
        
        assert performance_monitor.peak_memory < target_memory, \
            f"Peak memory {performance_monitor.peak_memory:.1f}MB exceeded target {target_memory}MB"


class TestSafetyFilterPerformance:
    """Test performance of safety filtering."""
    
    @pytest.mark.performance
    def test_safety_assessment_time(self, safety_filter):
        """Test individual safety assessment performance."""
        molecule = Molecule("CC(C)=CCO")  # Linalool
        
        # Warm up
        safety_filter.assess_molecule(molecule)
        
        # Measure multiple assessments
        times = []
        for _ in range(100):
            start = time.time()
            safety_filter.assess_molecule(molecule)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
        
        assert avg_time < 0.01, f"Average assessment time {avg_time:.4f}s too high"
        assert p95_time < 0.02, f"95th percentile time {p95_time:.4f}s too high"
    
    @pytest.mark.performance
    def test_batch_filtering_performance(self, safety_filter):
        """Test batch filtering performance."""
        # Create large batch of molecules
        test_smiles = [
            "CCO", "CCCO", "CCCCO", "CC(C)O", "CC(C)CO",
            "c1ccccc1", "c1ccc(cc1)C", "c1ccc(cc1)O",
            "CC(=O)O", "CC(=O)C", "CC(C)=CCO", "c1ccc(cc1)C=O"
        ]
        
        molecules = []
        for _ in range(50):  # 600 molecules total
            for smiles in test_smiles:
                molecules.append(Molecule(smiles))
        
        start_time = time.time()
        safe_molecules, reports = safety_filter.filter_molecules(molecules)
        elapsed = time.time() - start_time
        
        molecules_per_second = len(molecules) / elapsed
        
        assert molecules_per_second > 100, f"Filtering rate {molecules_per_second:.1f} mol/s too low"
        assert len(reports) == len(molecules)
    
    @pytest.mark.performance
    def test_concurrent_safety_assessment(self, safety_filter):
        """Test concurrent safety assessments."""
        test_molecules = [
            Molecule("CCO"),
            Molecule("CC(C)=CCO"),
            Molecule("c1ccccc1"),
            Molecule("CC(=O)O")
        ] * 25  # 100 molecules total
        
        def assess_molecule(mol):
            return safety_filter.assess_molecule(mol)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(assess_molecule, mol) for mol in test_molecules]
            reports = [future.result() for future in as_completed(futures)]
        
        elapsed = time.time() - start_time
        
        assert len(reports) == len(test_molecules)
        assert elapsed < 5.0, f"Concurrent assessment took {elapsed:.2f}s, too slow"


class TestCachePerformance:
    """Test cache system performance."""
    
    @pytest.mark.performance
    def test_cache_hit_performance(self, molecule_cache, performance_test_data):
        """Test cache hit performance."""
        target_time = performance_test_data['cache_targets']['lookup_time_max']
        
        # Pre-populate cache
        test_data = {"test_key": "test_value", "complex_data": [1, 2, 3, {"nested": True}]}
        molecule_cache.generation_cache.set("test_prompt", test_data)
        
        # Measure cache hits
        times = []
        for _ in range(1000):
            start = time.time()
            result = molecule_cache.generation_cache.get("test_prompt")
            times.append(time.time() - start)
            assert result is not None
        
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]
        
        assert avg_time < target_time, f"Average cache lookup {avg_time:.4f}s > target {target_time}s"
        assert p95_time < target_time * 2
    
    @pytest.mark.performance
    def test_cache_miss_performance(self, molecule_cache):
        """Test cache miss performance."""
        # Measure cache misses
        times = []
        for i in range(100):
            start = time.time()
            result = molecule_cache.generation_cache.get(f"nonexistent_key_{i}")
            times.append(time.time() - start)
            assert result is None
        
        avg_time = statistics.mean(times)
        assert avg_time < 0.001, f"Average cache miss time {avg_time:.4f}s too high"
    
    @pytest.mark.performance
    def test_cache_eviction_performance(self, temp_dir):
        """Test cache eviction performance under load."""
        # Create cache with small size to force evictions
        cache = MoleculeCache(cache_dir=str(temp_dir / "small_cache"))
        cache.generation_cache.max_size = 100  # Small cache
        
        start_time = time.time()
        
        # Add many items to force evictions
        for i in range(500):
            cache.generation_cache.set(f"key_{i}", f"value_{i}")
        
        elapsed = time.time() - start_time
        
        # Should handle evictions efficiently
        assert elapsed < 5.0, f"Cache eviction took {elapsed:.2f}s, too slow"
        assert cache.generation_cache.get_stats()['size'] <= 100


class TestAPIPerformance:
    """Test API endpoint performance."""
    
    @pytest.mark.performance
    async def test_health_check_performance(self, mock_api_client):
        """Test health check endpoint performance."""
        times = []
        
        for _ in range(50):
            start = time.time()
            response = await mock_api_client.get("/health")
            times.append(time.time() - start)
            assert response is not None
        
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]
        
        assert avg_time < 0.1, f"Average health check time {avg_time:.3f}s too high"
        assert p95_time < 0.2
    
    @pytest.mark.performance
    async def test_concurrent_api_requests(self, mock_api_client, performance_test_data):
        """Test concurrent API request handling."""
        target_concurrent = performance_test_data['api_targets']['concurrent_requests']
        
        async def make_request():
            return await mock_api_client.post("/generate", {
                "prompt": "test scent",
                "num_molecules": 1
            })
        
        start_time = time.time()
        
        # Make concurrent requests
        tasks = [make_request() for _ in range(target_concurrent)]
        responses = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        assert len(responses) == target_concurrent
        assert all(r is not None for r in responses)
        
        # Should handle concurrent requests efficiently
        requests_per_second = target_concurrent / elapsed
        assert requests_per_second > 20, f"Request rate {requests_per_second:.1f} req/s too low"


class TestScalabilityBenchmarks:
    """Test system scalability."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_load_scalability(self, async_odor_diffusion):
        """Test system behavior under increasing load."""
        load_levels = [1, 5, 10, 20]
        response_times = []
        
        for load in load_levels:
            tasks = [
                async_odor_diffusion.generate_async(f"test scent {i}", num_molecules=1)
                for i in range(load)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start_time
            
            avg_response_time = elapsed / load
            response_times.append((load, avg_response_time))
            
            # All requests should complete successfully
            assert len(results) == load
            assert all(len(r.molecules) >= 1 for r in results)
        
        # Response time should not degrade linearly with load (due to concurrency)
        _, time_1 = response_times[0]
        _, time_20 = response_times[-1]
        
        # 20x load should not result in 20x response time
        assert time_20 < time_1 * 15, "Poor scalability detected"
    
    @pytest.mark.performance
    def test_memory_scalability(self, performance_monitor):
        """Test memory usage scalability."""
        model = OdorDiffusion(device="cpu")
        
        performance_monitor.start_monitoring()
        
        batch_sizes = [1, 5, 10, 20]
        memory_usage = []
        
        for batch_size in batch_sizes:
            # Force garbage collection
            import gc
            gc.collect()
            
            start_memory = performance_monitor.peak_memory
            
            molecules = model.generate("scalability test", num_molecules=batch_size)
            
            end_memory = performance_monitor.peak_memory
            memory_increase = end_memory - start_memory
            memory_usage.append((batch_size, memory_increase))
            
            assert len(molecules) >= 1
        
        performance_monitor.stop_monitoring()
        
        # Memory usage should scale reasonably with batch size
        _, mem_1 = memory_usage[0]
        _, mem_20 = memory_usage[-1]
        
        # Memory shouldn't increase exponentially
        if mem_1 > 0:
            memory_ratio = mem_20 / mem_1
            assert memory_ratio < 30, f"Memory usage scaled too steeply: {memory_ratio}x"


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.mark.performance
    def test_baseline_performance(self, odor_diffusion):
        """Test baseline performance metrics."""
        # This test establishes baseline performance metrics
        # In a real system, these would be compared against historical data
        
        test_prompts = [
            "fresh citrus scent",
            "warm vanilla fragrance",
            "floral rose bouquet"
        ]
        
        times = []
        for prompt in test_prompts:
            start = time.time()
            molecules = odor_diffusion.generate(prompt, num_molecules=3)
            elapsed = time.time() - start
            times.append(elapsed)
            
            assert len(molecules) >= 1
        
        avg_time = statistics.mean(times)
        
        # Store baseline metrics (in real implementation, this would be persisted)
        baseline_metrics = {
            'avg_generation_time': avg_time,
            'test_date': time.time(),
            'system_info': 'cpu_only_test'
        }
        
        # Assert reasonable baseline performance
        assert avg_time < 10.0, f"Baseline performance {avg_time:.2f}s seems too slow"
        
        print(f"Baseline performance: {avg_time:.2f}s average generation time")


@pytest.mark.performance
class TestEndToEndPerformance:
    """End-to-end performance tests."""
    
    async def test_complete_workflow_performance(self, async_odor_diffusion):
        """Test performance of complete generation workflow."""
        safety_filter = SafetyFilter()
        
        start_time = time.time()
        
        # Complete workflow: generation -> safety -> assessment
        result = await async_odor_diffusion.generate_async(
            "elegant floral fragrance with woody undertones",
            num_molecules=5,
            safety_filter=safety_filter,
            synthesizability_min=0.3
        )
        
        elapsed = time.time() - start_time
        
        assert len(result.molecules) >= 1
        assert result.error is None
        assert elapsed < 15.0, f"Complete workflow took {elapsed:.2f}s, too slow"
        
        # Verify all molecules have been properly assessed
        for mol in result.molecules:
            assert mol.safety_score > 0
            assert mol.synth_score >= 0
            assert mol.estimated_cost > 0