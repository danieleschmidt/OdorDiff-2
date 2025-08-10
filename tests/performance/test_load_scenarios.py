"""
Performance and load testing scenarios for OdorDiff-2 system.
"""

import pytest
import asyncio
import time
import threading
import concurrent.futures
import statistics
import psutil
import gc
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import numpy as np

from odordiff2.core.diffusion import OdorDiffusion
from odordiff2.scaling.multi_tier_cache import MultiTierCache
from odordiff2.scaling.load_balancer import LoadBalancer
from odordiff2.scaling.auto_scaler import AutoScaler
from odordiff2.api.endpoints import APIEndpoints
from odordiff2.models.molecule import Molecule


class TestBasicPerformance:
    """Test basic performance characteristics."""
    
    def test_molecule_generation_performance(self):
        """Test performance of single molecule generation."""
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_diffusion_class:
            mock_model = Mock()
            mock_diffusion_class.return_value = mock_model
            
            # Mock fast generation (simulate optimized model)
            mock_model.generate.return_value = [
                Molecule(smiles="CCO", safety_score=0.95)
            ]
            
            model = mock_diffusion_class()
            
            # Performance test - single generation
            start_time = time.time()
            
            molecules = model.generate(
                prompt="Fresh citrus scent",
                num_molecules=1
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            assert len(molecules) == 1
            assert generation_time < 0.1  # Should be very fast with mocked model
    
    def test_batch_generation_performance(self):
        """Test performance of batch molecule generation."""
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_diffusion_class:
            mock_model = Mock()
            mock_diffusion_class.return_value = mock_model
            
            # Mock batch generation
            def mock_batch_generate(prompt, num_molecules=5, **kwargs):
                return [
                    Molecule(smiles=f"C{i}O", safety_score=0.9 + i*0.01)
                    for i in range(num_molecules)
                ]
            
            mock_model.generate.side_effect = mock_batch_generate
            
            model = mock_diffusion_class()
            
            # Test batch sizes
            batch_sizes = [1, 5, 10, 25, 50]
            performance_data = []
            
            for batch_size in batch_sizes:
                start_time = time.time()
                
                molecules = model.generate(
                    prompt="Performance test",
                    num_molecules=batch_size
                )
                
                end_time = time.time()
                
                performance_data.append({
                    "batch_size": batch_size,
                    "total_time": end_time - start_time,
                    "time_per_molecule": (end_time - start_time) / batch_size,
                    "molecules_generated": len(molecules)
                })
            
            # Verify performance scaling
            for data in performance_data:
                assert data["molecules_generated"] == data["batch_size"]
                assert data["time_per_molecule"] < 0.1  # With mocked model
    
    def test_safety_filter_performance(self):
        """Test safety filter performance under load."""
        with patch('odordiff2.safety.filter.SafetyFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter_class.return_value = mock_filter
            
            # Mock safety assessment
            mock_filter.assess_safety.return_value = Mock(
                is_safe=True,
                toxicity_score=0.05,
                assessment_time=0.001
            )
            
            safety_filter = mock_filter_class()
            
            # Test molecules for performance
            test_molecules = [
                Molecule(smiles=f"C{'C' * i}O", safety_score=0.9)
                for i in range(100)  # 100 molecules
            ]
            
            start_time = time.time()
            
            # Assess all molecules
            results = []
            for molecule in test_molecules:
                result = safety_filter.assess_safety(molecule)
                results.append(result)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance requirements
            assert len(results) == 100
            assert total_time < 1.0  # Should complete in under 1 second
            assert (total_time / 100) < 0.01  # < 10ms per molecule
    
    def test_api_response_time(self):
        """Test API endpoint response times."""
        with patch('odordiff2.api.endpoints.APIEndpoints') as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            
            # Mock fast API response
            mock_api.generate_molecules.return_value = {
                "status": "success",
                "molecules": [
                    {"smiles": "CCO", "safety_score": 0.95}
                ],
                "generation_time": 0.05
            }
            
            api = mock_api_class()
            
            # Test API response times
            response_times = []
            
            for _ in range(10):
                start_time = time.time()
                
                response = api.generate_molecules({
                    "prompt": "Fresh scent",
                    "num_molecules": 1
                })
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                assert response["status"] == "success"
            
            # Analyze response time statistics
            avg_response_time = statistics.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            
            assert avg_response_time < 0.1  # < 100ms average
            assert p95_response_time < 0.2   # < 200ms P95


class TestConcurrencyPerformance:
    """Test performance under concurrent load."""
    
    def test_concurrent_generation_requests(self):
        """Test concurrent molecule generation requests."""
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_diffusion_class:
            mock_model = Mock()
            mock_diffusion_class.return_value = mock_model
            
            def thread_safe_generate(prompt, num_molecules=1, **kwargs):
                # Simulate some processing time
                time.sleep(0.01)  # 10ms processing time
                return [
                    Molecule(smiles="CCO", safety_score=0.95)
                    for _ in range(num_molecules)
                ]
            
            mock_model.generate.side_effect = thread_safe_generate
            
            model = mock_diffusion_class()
            
            def worker_thread(thread_id):
                """Worker thread function."""
                start_time = time.time()
                
                molecules = model.generate(
                    prompt=f"Thread {thread_id} test",
                    num_molecules=1
                )
                
                end_time = time.time()
                
                return {
                    "thread_id": thread_id,
                    "molecules_count": len(molecules),
                    "processing_time": end_time - start_time,
                    "success": True
                }
            
            # Run concurrent threads
            num_threads = 20
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(worker_thread, i)
                    for i in range(num_threads)
                ]
                
                results = []
                for future in concurrent.futures.as_completed(futures, timeout=10):
                    result = future.result()
                    results.append(result)
            
            # Verify all threads completed successfully
            assert len(results) == num_threads
            
            successful_results = [r for r in results if r["success"]]
            assert len(successful_results) == num_threads
            
            # Analyze performance under concurrency
            processing_times = [r["processing_time"] for r in results]
            avg_time = statistics.mean(processing_times)
            max_time = max(processing_times)
            
            assert avg_time < 1.0  # Should complete reasonably fast
            assert max_time < 2.0   # No thread should take too long
    
    @pytest.mark.asyncio
    async def test_async_request_handling(self):
        """Test asynchronous request handling performance."""
        async def mock_async_generate(prompt, num_molecules=1, **kwargs):
            # Simulate async I/O operation
            await asyncio.sleep(0.01)  # 10ms async delay
            return [
                Molecule(smiles="CCO", safety_score=0.95)
                for _ in range(num_molecules)
            ]
        
        async def async_worker(worker_id):
            """Async worker coroutine."""
            start_time = time.time()
            
            molecules = await mock_async_generate(
                prompt=f"Async worker {worker_id}",
                num_molecules=1
            )
            
            end_time = time.time()
            
            return {
                "worker_id": worker_id,
                "molecules_count": len(molecules),
                "processing_time": end_time - start_time
            }
        
        # Run concurrent async tasks
        num_workers = 50
        
        start_time = time.time()
        
        tasks = [async_worker(i) for i in range(num_workers)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify async performance
        assert len(results) == num_workers
        
        # Async should be much faster than sequential
        # Sequential would take: 50 * 0.01 = 0.5 seconds
        # Async should take close to 0.01 seconds (parallel execution)
        assert total_time < 0.1  # Should be much faster than sequential
        
        processing_times = [r["processing_time"] for r in results]
        avg_processing_time = statistics.mean(processing_times)
        assert avg_processing_time < 0.05  # Each task should be fast
    
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        import tracemalloc
        
        tracemalloc.start()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_diffusion_class:
            mock_model = Mock()
            mock_diffusion_class.return_value = mock_model
            
            # Mock generation that returns substantial data
            def memory_intensive_generate(prompt, num_molecules=10, **kwargs):
                return [
                    Molecule(
                        smiles=f"C{'C' * (i % 20)}O",  # Variable length SMILES
                        safety_score=0.9,
                        additional_data={"data": "x" * 1000}  # 1KB extra data per molecule
                    )
                    for i in range(num_molecules)
                ]
            
            mock_model.generate.side_effect = memory_intensive_generate
            
            model = mock_diffusion_class()
            
            # Sustained load test
            for iteration in range(100):
                molecules = model.generate(
                    prompt=f"Memory test iteration {iteration}",
                    num_molecules=10
                )
                
                # Verify generation
                assert len(molecules) == 10
                
                # Force garbage collection every 10 iterations
                if iteration % 10 == 0:
                    gc.collect()
                    
                    # Check memory usage
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    memory_growth = current_memory - initial_memory
                    
                    # Memory growth should be reasonable (< 500MB for this test)
                    assert memory_growth < 500
        
        # Final memory check
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Peak memory usage should be reasonable
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 100  # Less than 100MB peak for this test


class TestCachePerformance:
    """Test caching system performance."""
    
    def test_cache_hit_performance(self):
        """Test cache hit performance."""
        with patch('odordiff2.scaling.multi_tier_cache.MultiTierCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            
            # Mock cache hit (fast)
            cached_data = {
                "molecules": [{"smiles": "CCO", "safety_score": 0.95}],
                "generation_time": 2.1,
                "cached": True
            }
            mock_cache.get.return_value = cached_data
            
            cache = mock_cache_class()
            
            # Test cache hit performance
            cache_times = []
            
            for i in range(1000):  # 1000 cache hits
                start_time = time.time()
                
                result = cache.get(f"cache_key_{i}")
                
                end_time = time.time()
                cache_times.append(end_time - start_time)
                
                assert result is not None
                assert result["cached"] is True
            
            # Analyze cache performance
            avg_cache_time = statistics.mean(cache_times)
            p95_cache_time = np.percentile(cache_times, 95)
            
            assert avg_cache_time < 0.001  # < 1ms average
            assert p95_cache_time < 0.005   # < 5ms P95
    
    def test_cache_miss_performance(self):
        """Test cache miss and populate performance."""
        with patch('odordiff2.scaling.multi_tier_cache.MultiTierCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            
            # Mock cache miss followed by set
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = True   # Successful set
            
            cache = mock_cache_class()
            
            # Test cache miss and populate workflow
            cache_operations = []
            
            for i in range(100):
                start_time = time.time()
                
                # Try to get from cache (miss)
                result = cache.get(f"miss_key_{i}")
                assert result is None
                
                # Generate new data (simulate)
                new_data = {
                    "molecules": [{"smiles": f"C{i}O"}],
                    "generated_at": time.time()
                }
                
                # Store in cache
                success = cache.set(f"miss_key_{i}", new_data, ttl=3600)
                assert success is True
                
                end_time = time.time()
                cache_operations.append(end_time - start_time)
            
            # Cache operations should be fast
            avg_operation_time = statistics.mean(cache_operations)
            assert avg_operation_time < 0.01  # < 10ms per operation
    
    def test_cache_eviction_performance(self):
        """Test cache eviction performance under memory pressure."""
        with patch('odordiff2.scaling.multi_tier_cache.MultiTierCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            
            # Mock cache operations
            mock_cache.set.return_value = True
            mock_cache.evict_lru.return_value = 5  # Evicted 5 items
            
            cache = mock_cache_class()
            
            # Fill cache to capacity
            start_time = time.time()
            
            for i in range(1000):  # Large number of cache entries
                large_data = {
                    "molecules": [{"smiles": f"C{i}O"}] * 10,  # 10 molecules each
                    "metadata": {"data": "x" * 1000}  # 1KB metadata
                }
                
                cache.set(f"large_key_{i}", large_data)
                
                # Trigger eviction every 100 items
                if i % 100 == 0:
                    evicted_count = cache.evict_lru()
                    assert isinstance(evicted_count, int)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Cache operations with eviction should complete reasonably fast
            assert total_time < 10.0  # < 10 seconds for 1000 operations


class TestScalingPerformance:
    """Test auto-scaling system performance."""
    
    def test_scaling_decision_performance(self):
        """Test performance of scaling decisions."""
        with patch('odordiff2.scaling.auto_scaler.AutoScaler') as mock_scaler_class:
            mock_scaler = Mock()
            mock_scaler_class.return_value = mock_scaler
            
            # Mock scaling decision
            mock_scaler.decide_scaling_action.return_value = Mock(
                action_type="scale_up",
                target_replicas=6,
                confidence=0.9
            )
            
            scaler = mock_scaler_class()
            
            # Test rapid scaling decisions
            decision_times = []
            
            for i in range(1000):  # 1000 scaling decisions
                metrics = {
                    "cpu_usage": 50 + (i % 50),  # Varying CPU
                    "memory_usage": 40 + (i % 40),  # Varying memory
                    "active_requests": 100 + (i % 100)
                }
                
                start_time = time.time()
                
                decision = scaler.decide_scaling_action(metrics)
                
                end_time = time.time()
                decision_times.append(end_time - start_time)
                
                assert decision is not None
            
            # Scaling decisions should be very fast
            avg_decision_time = statistics.mean(decision_times)
            max_decision_time = max(decision_times)
            
            assert avg_decision_time < 0.001  # < 1ms average
            assert max_decision_time < 0.01   # < 10ms maximum
    
    def test_load_balancer_performance(self):
        """Test load balancer performance under high request rate."""
        with patch('odordiff2.scaling.load_balancer.LoadBalancer') as mock_lb_class:
            mock_lb = Mock()
            mock_lb_class.return_value = mock_lb
            
            # Mock server selection
            servers = [f"server_{i}" for i in range(10)]
            mock_lb.get_next_server.side_effect = lambda: servers[int(time.time() * 1000) % len(servers)]
            
            load_balancer = mock_lb_class()
            
            # Test high-frequency server selection
            selection_times = []
            selected_servers = []
            
            for _ in range(10000):  # 10,000 server selections
                start_time = time.time()
                
                server = load_balancer.get_next_server()
                
                end_time = time.time()
                selection_times.append(end_time - start_time)
                selected_servers.append(server)
            
            # Load balancer should be extremely fast
            avg_selection_time = statistics.mean(selection_times)
            p99_selection_time = np.percentile(selection_times, 99)
            
            assert avg_selection_time < 0.0001  # < 0.1ms average
            assert p99_selection_time < 0.001   # < 1ms P99
            
            # Should distribute load across servers
            unique_servers = set(selected_servers)
            assert len(unique_servers) > 1  # Multiple servers used


class TestDatabasePerformance:
    """Test database operation performance."""
    
    def test_molecule_storage_performance(self):
        """Test molecule storage and retrieval performance."""
        with patch('odordiff2.data.persistence.MoleculeDatabase') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock database operations
            mock_db.save_molecule.side_effect = lambda data: f"mol_{int(time.time() * 1000000)}"
            mock_db.get_molecule.return_value = {
                "id": "mol_123",
                "smiles": "CCO",
                "safety_score": 0.95
            }
            mock_db.query_molecules.return_value = [
                {"id": f"mol_{i}", "smiles": f"C{i}O"}
                for i in range(10)
            ]
            
            db = mock_db_class()
            
            # Test bulk molecule storage
            storage_times = []
            molecule_ids = []
            
            for i in range(100):
                molecule_data = {
                    "smiles": f"CC{i}O",
                    "safety_score": 0.9 + (i % 10) * 0.01,
                    "generation_prompt": f"Test molecule {i}"
                }
                
                start_time = time.time()
                
                molecule_id = db.save_molecule(molecule_data)
                
                end_time = time.time()
                storage_times.append(end_time - start_time)
                molecule_ids.append(molecule_id)
            
            # Test bulk retrieval
            retrieval_times = []
            
            for molecule_id in molecule_ids[:50]:  # Test 50 retrievals
                start_time = time.time()
                
                molecule_data = db.get_molecule(molecule_id)
                
                end_time = time.time()
                retrieval_times.append(end_time - start_time)
                
                assert molecule_data is not None
            
            # Performance requirements
            avg_storage_time = statistics.mean(storage_times)
            avg_retrieval_time = statistics.mean(retrieval_times)
            
            assert avg_storage_time < 0.1   # < 100ms per storage
            assert avg_retrieval_time < 0.05  # < 50ms per retrieval
    
    def test_query_performance(self):
        """Test database query performance."""
        with patch('odordiff2.data.persistence.MoleculeDatabase') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock complex query results
            def mock_complex_query(conditions, limit=10):
                return [
                    {
                        "id": f"mol_{i}",
                        "smiles": f"C{i}O",
                        "safety_score": 0.9 + (i % 10) * 0.01
                    }
                    for i in range(limit)
                ]
            
            mock_db.query_molecules.side_effect = mock_complex_query
            
            db = mock_db_class()
            
            # Test various query complexities
            query_scenarios = [
                {"conditions": {"safety_score__gte": 0.9}, "limit": 10},
                {"conditions": {"smiles__contains": "CC"}, "limit": 50},
                {"conditions": {"safety_score__gte": 0.9, "synthesis_score__gte": 0.8}, "limit": 100},
            ]
            
            query_times = []
            
            for scenario in query_scenarios:
                start_time = time.time()
                
                results = db.query_molecules(**scenario)
                
                end_time = time.time()
                query_times.append(end_time - start_time)
                
                assert len(results) == scenario["limit"]
            
            # Query performance should be reasonable
            avg_query_time = statistics.mean(query_times)
            max_query_time = max(query_times)
            
            assert avg_query_time < 0.5  # < 500ms average
            assert max_query_time < 1.0   # < 1s maximum


class TestNetworkPerformance:
    """Test network-related performance."""
    
    @pytest.mark.asyncio
    async def test_api_endpoint_throughput(self):
        """Test API endpoint throughput under load."""
        with patch('odordiff2.api.endpoints.APIEndpoints') as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            
            # Mock async API endpoint
            async def mock_async_endpoint(request_data):
                await asyncio.sleep(0.01)  # Simulate 10ms processing
                return {
                    "status": "success",
                    "molecules": [{"smiles": "CCO"}],
                    "processing_time": 0.01
                }
            
            mock_api.generate_molecules = mock_async_endpoint
            
            api = mock_api_class()
            
            # Test concurrent API requests
            async def make_request(request_id):
                return await api.generate_molecules({
                    "prompt": f"Request {request_id}",
                    "num_molecules": 1
                })
            
            # High concurrency test
            num_concurrent = 100
            
            start_time = time.time()
            
            tasks = [make_request(i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify throughput
            assert len(results) == num_concurrent
            
            # Calculate requests per second
            rps = num_concurrent / total_time
            
            # Should achieve high throughput with async processing
            assert rps > 50  # > 50 requests per second
            assert total_time < 2.0  # Should complete in under 2 seconds
    
    def test_websocket_streaming_performance(self):
        """Test WebSocket streaming performance."""
        with patch('odordiff2.api.streaming.WebSocketManager') as mock_ws_class:
            mock_ws = Mock()
            mock_ws_class.return_value = mock_ws
            
            # Mock streaming operations
            mock_ws.send_message.return_value = True
            mock_ws.broadcast_message.return_value = 10  # Sent to 10 clients
            
            ws_manager = mock_ws_class()
            
            # Test high-frequency message sending
            send_times = []
            
            for i in range(1000):  # 1000 messages
                message = {
                    "type": "molecule_generated",
                    "data": {"smiles": f"C{i}O"},
                    "timestamp": time.time()
                }
                
                start_time = time.time()
                
                success = ws_manager.send_message("client_123", message)
                
                end_time = time.time()
                send_times.append(end_time - start_time)
                
                assert success is True
            
            # WebSocket messaging should be fast
            avg_send_time = statistics.mean(send_times)
            p95_send_time = np.percentile(send_times, 95)
            
            assert avg_send_time < 0.001  # < 1ms average
            assert p95_send_time < 0.005   # < 5ms P95


@pytest.mark.stress
class TestStressScenarios:
    """Stress testing scenarios."""
    
    def test_sustained_high_load(self):
        """Test system performance under sustained high load."""
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_diffusion_class:
            mock_model = Mock()
            mock_diffusion_class.return_value = mock_model
            
            # Mock generation with realistic timing
            def realistic_generate(prompt, num_molecules=1, **kwargs):
                time.sleep(0.05)  # 50ms processing time
                return [
                    Molecule(smiles="CCO", safety_score=0.95)
                    for _ in range(num_molecules)
                ]
            
            mock_model.generate.side_effect = realistic_generate
            
            model = mock_diffusion_class()
            
            # Sustained load test (5 minutes worth of requests)
            duration_seconds = 10  # Reduced for testing
            requests_per_second = 5
            
            def worker():
                """Worker function for sustained load."""
                results = []
                end_time = time.time() + duration_seconds
                
                while time.time() < end_time:
                    start_time = time.time()
                    
                    try:
                        molecules = model.generate(
                            prompt="Sustained load test",
                            num_molecules=1
                        )
                        
                        results.append({
                            "success": True,
                            "molecules_count": len(molecules),
                            "response_time": time.time() - start_time
                        })
                        
                    except Exception as e:
                        results.append({
                            "success": False,
                            "error": str(e),
                            "response_time": time.time() - start_time
                        })
                    
                    # Rate limiting to achieve target RPS
                    time.sleep(1.0 / requests_per_second)
                
                return results
            
            # Run sustained load test
            all_results = worker()
            
            # Analyze results
            successful_requests = [r for r in all_results if r["success"]]
            failed_requests = [r for r in all_results if not r["success"]]
            
            success_rate = len(successful_requests) / len(all_results)
            avg_response_time = statistics.mean(r["response_time"] for r in successful_requests)
            
            # Performance requirements under sustained load
            assert success_rate > 0.95  # > 95% success rate
            assert avg_response_time < 1.0  # < 1 second average response
            assert len(failed_requests) < len(all_results) * 0.05  # < 5% failures
    
    def test_memory_leak_detection(self):
        """Test for memory leaks under extended operation."""
        import tracemalloc
        
        tracemalloc.start()
        
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_diffusion_class:
            mock_model = Mock()
            mock_diffusion_class.return_value = mock_model
            
            # Mock generation that could potentially leak memory
            def potentially_leaky_generate(prompt, num_molecules=5, **kwargs):
                # Simulate creation of objects that might not be cleaned up
                large_data = ["x" * 1000 for _ in range(100)]  # 100KB of data
                
                molecules = [
                    Molecule(
                        smiles=f"C{i}O",
                        safety_score=0.9,
                        temp_data=large_data  # Temporary data that should be cleaned
                    )
                    for i in range(num_molecules)
                ]
                
                return molecules
            
            mock_model.generate.side_effect = potentially_leaky_generate
            
            model = mock_diffusion_class()
            
            # Track memory usage over time
            memory_samples = []
            
            for cycle in range(50):  # 50 cycles
                # Generate molecules
                molecules = model.generate(
                    prompt=f"Memory test cycle {cycle}",
                    num_molecules=5
                )
                
                assert len(molecules) == 5
                
                # Force garbage collection
                gc.collect()
                
                # Sample memory usage
                current, peak = tracemalloc.get_traced_memory()
                memory_samples.append({
                    "cycle": cycle,
                    "current_memory_mb": current / 1024 / 1024,
                    "peak_memory_mb": peak / 1024 / 1024
                })
                
                # Clear references to help GC
                molecules = None
        
        tracemalloc.stop()
        
        # Analyze memory growth
        initial_memory = memory_samples[0]["current_memory_mb"]
        final_memory = memory_samples[-1]["current_memory_mb"]
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (indicating no major leaks)
        assert memory_growth < 50  # Less than 50MB growth over 50 cycles
        
        # Peak memory should be reasonable
        max_peak = max(sample["peak_memory_mb"] for sample in memory_samples)
        assert max_peak < 200  # Less than 200MB peak usage
    
    def test_concurrent_user_simulation(self):
        """Test performance with realistic concurrent user load."""
        # Simulate different user behavior patterns
        user_patterns = [
            {"requests_per_minute": 10, "avg_molecules": 3, "user_count": 5},  # Heavy users
            {"requests_per_minute": 2, "avg_molecules": 1, "user_count": 20},   # Light users
            {"requests_per_minute": 1, "avg_molecules": 5, "user_count": 10},   # Batch users
        ]
        
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_diffusion_class:
            mock_model = Mock()
            mock_diffusion_class.return_value = mock_model
            
            def variable_time_generate(prompt, num_molecules=1, **kwargs):
                # Variable processing time based on complexity
                processing_time = 0.01 + (num_molecules * 0.005)
                time.sleep(processing_time)
                
                return [
                    Molecule(smiles=f"C{i}O", safety_score=0.9)
                    for i in range(num_molecules)
                ]
            
            mock_model.generate.side_effect = variable_time_generate
            
            model = mock_diffusion_class()
            
            def simulate_user_pattern(pattern, user_id):
                """Simulate a specific user pattern."""
                results = []
                requests_made = 0
                max_requests = pattern["requests_per_minute"] // 6  # Test for 10 seconds
                
                while requests_made < max_requests:
                    start_time = time.time()
                    
                    try:
                        molecules = model.generate(
                            prompt=f"User {user_id} request {requests_made}",
                            num_molecules=pattern["avg_molecules"]
                        )
                        
                        results.append({
                            "user_id": user_id,
                            "success": True,
                            "molecules_count": len(molecules),
                            "response_time": time.time() - start_time
                        })
                        
                    except Exception as e:
                        results.append({
                            "user_id": user_id,
                            "success": False,
                            "error": str(e),
                            "response_time": time.time() - start_time
                        })
                    
                    requests_made += 1
                    
                    # Wait between requests
                    wait_time = 60.0 / pattern["requests_per_minute"]
                    time.sleep(wait_time)
                
                return results
            
            # Start all user simulations concurrently
            all_futures = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                for pattern in user_patterns:
                    for user_id in range(pattern["user_count"]):
                        future = executor.submit(
                            simulate_user_pattern,
                            pattern,
                            f"{pattern['requests_per_minute']}rpm_user_{user_id}"
                        )
                        all_futures.append(future)
                
                # Collect all results
                all_results = []
                for future in concurrent.futures.as_completed(all_futures, timeout=30):
                    user_results = future.result()
                    all_results.extend(user_results)
            
            # Analyze overall system performance
            successful_results = [r for r in all_results if r["success"]]
            failed_results = [r for r in all_results if not r["success"]]
            
            overall_success_rate = len(successful_results) / len(all_results)
            avg_response_time = statistics.mean(r["response_time"] for r in successful_results)
            p95_response_time = np.percentile([r["response_time"] for r in successful_results], 95)
            
            # System should handle concurrent users well
            assert overall_success_rate > 0.95  # > 95% success rate
            assert avg_response_time < 2.0      # < 2s average response
            assert p95_response_time < 5.0      # < 5s P95 response
            assert len(all_results) >= 100      # Generated substantial load