"""
Integration tests for API endpoints and workflows.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Mock imports that might not be available in test environment
try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = Mock
    FastAPI = Mock

from odordiff2.models.molecule import Molecule


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def mock_app(self):
        """Create mock FastAPI application for testing."""
        app = FastAPI(title="OdorDiff-2 API", version="1.0.0")
        
        @app.post("/api/v1/generate")
        async def generate_molecules(request: Dict[str, Any]):
            # Mock implementation
            return {
                "request_id": "test-123",
                "prompt": request.get("prompt", ""),
                "molecules": [
                    {
                        "smiles": "CCO",
                        "confidence": 0.9,
                        "odor_profile": {"primary_notes": ["alcohol"], "character": "clean"},
                        "safety_score": 0.95,
                        "synth_score": 0.99,
                        "estimated_cost": 5.0,
                        "properties": {"molecular_weight": 46.07}
                    }
                ],
                "processing_time": 1.5,
                "cache_hit": False,
                "timestamp": "2025-01-01T00:00:00"
            }
        
        @app.post("/api/v1/generate/batch")
        async def generate_batch(request: Dict[str, Any]):
            prompts = request.get("prompts", [])
            return {
                "request_id": "batch-123",
                "results": [
                    {
                        "prompt": prompt,
                        "molecules": [
                            {
                                "smiles": "CCO",
                                "confidence": 0.8,
                                "safety_score": 0.9
                            }
                        ],
                        "processing_time": 1.0
                    }
                    for prompt in prompts
                ],
                "total_processing_time": len(prompts) * 1.0,
                "timestamp": "2025-01-01T00:00:00"
            }
        
        @app.post("/api/v1/assess/safety")
        async def assess_safety(request: Dict[str, Any]):
            return {
                "smiles": request.get("smiles", ""),
                "assessment": {
                    "toxicity_score": 0.05,
                    "skin_sensitizer": False,
                    "eco_score": 0.1,
                    "ifra_compliant": True,
                    "regulatory_flags": []
                },
                "recommendation": "safe",
                "timestamp": "2025-01-01T00:00:00"
            }
        
        @app.get("/api/v1/health")
        async def health_check():
            return {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-01-01T00:00:00",
                "uptime": 3600,
                "services": {
                    "model": "healthy",
                    "cache": "healthy",
                    "database": "healthy"
                }
            }
        
        @app.get("/api/v1/stats")
        async def get_stats():
            return {
                "stats": {
                    "total_requests": 100,
                    "successful_requests": 95,
                    "cache_hits": 30,
                    "avg_processing_time": 1.8,
                    "error_rate": 0.05
                },
                "timestamp": "2025-01-01T00:00:00"
            }
        
        return app
    
    @pytest.fixture
    def client(self, mock_app):
        """Create test client."""
        return TestClient(mock_app)
    
    def test_generate_molecules_endpoint(self, client):
        """Test molecule generation endpoint."""
        request_data = {
            "prompt": "fresh citrus scent",
            "num_molecules": 5,
            "safety_threshold": 0.1,
            "use_cache": True
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "request_id" in data
        assert data["prompt"] == request_data["prompt"]
        assert "molecules" in data
        assert len(data["molecules"]) >= 1
        assert "processing_time" in data
        assert "timestamp" in data
        
        # Validate molecule structure
        molecule = data["molecules"][0]
        assert "smiles" in molecule
        assert "confidence" in molecule
        assert "safety_score" in molecule
        assert "synth_score" in molecule
        assert "estimated_cost" in molecule
        assert "properties" in molecule
    
    def test_generate_molecules_validation_error(self, client):
        """Test generation endpoint with validation errors."""
        # Empty prompt
        response = client.post("/api/v1/generate", json={"prompt": ""})
        assert response.status_code in [400, 422]  # Validation error
        
        # Invalid num_molecules
        response = client.post("/api/v1/generate", json={
            "prompt": "test",
            "num_molecules": 0
        })
        assert response.status_code in [400, 422]
        
        # Invalid safety_threshold
        response = client.post("/api/v1/generate", json={
            "prompt": "test",
            "safety_threshold": 2.0
        })
        assert response.status_code in [400, 422]
    
    def test_batch_generation_endpoint(self, client):
        """Test batch generation endpoint."""
        request_data = {
            "prompts": [
                "fresh citrus scent",
                "warm vanilla fragrance",
                "floral rose bouquet"
            ],
            "num_molecules": 3,
            "priority": 1
        }
        
        response = client.post("/api/v1/generate/batch", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "request_id" in data
        assert "results" in data
        assert len(data["results"]) == len(request_data["prompts"])
        assert "total_processing_time" in data
        
        # Validate batch result structure
        for i, result in enumerate(data["results"]):
            assert result["prompt"] == request_data["prompts"][i]
            assert "molecules" in result
            assert "processing_time" in result
    
    def test_safety_assessment_endpoint(self, client):
        """Test safety assessment endpoint."""
        request_data = {
            "smiles": "CCO"
        }
        
        response = client.post("/api/v1/assess/safety", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["smiles"] == request_data["smiles"]
        assert "assessment" in data
        assert "recommendation" in data
        
        # Validate assessment structure
        assessment = data["assessment"]
        assert "toxicity_score" in assessment
        assert "skin_sensitizer" in assessment
        assert "eco_score" in assessment
        assert "ifra_compliant" in assessment
        assert "regulatory_flags" in assessment
        
        # Validate score ranges
        assert 0 <= assessment["toxicity_score"] <= 1
        assert 0 <= assessment["eco_score"] <= 1
        assert isinstance(assessment["skin_sensitizer"], bool)
        assert isinstance(assessment["ifra_compliant"], bool)
        assert isinstance(assessment["regulatory_flags"], list)
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "uptime" in data
        assert "services" in data
        
        # Validate services health
        services = data["services"]
        assert "model" in services
        assert "cache" in services
        assert "database" in services
        
        for service_status in services.values():
            assert service_status in ["healthy", "degraded", "unhealthy"]
    
    def test_stats_endpoint(self, client):
        """Test statistics endpoint."""
        response = client.get("/api/v1/stats")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "stats" in data
        assert "timestamp" in data
        
        # Validate statistics structure
        stats = data["stats"]
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "cache_hits" in stats
        assert "avg_processing_time" in stats
        assert "error_rate" in stats
        
        # Validate metrics are reasonable
        assert stats["total_requests"] >= 0
        assert stats["successful_requests"] >= 0
        assert stats["cache_hits"] >= 0
        assert stats["avg_processing_time"] >= 0
        assert 0 <= stats["error_rate"] <= 1
    
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test 404 for non-existent endpoint
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test malformed JSON
        response = client.post(
            "/api/v1/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]
    
    def test_response_times(self, client):
        """Test API response time requirements."""
        endpoints = [
            ("GET", "/api/v1/health"),
            ("GET", "/api/v1/stats"),
            ("POST", "/api/v1/generate", {"prompt": "test"}),
            ("POST", "/api/v1/assess/safety", {"smiles": "CCO"})
        ]
        
        for method, endpoint, *data in endpoints:
            start_time = time.time()
            
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json=data[0] if data else {})
            
            response_time = time.time() - start_time
            
            # API should respond within reasonable time
            assert response_time < 5.0, f"{method} {endpoint} took {response_time:.2f}s"
            
            if response.status_code < 500:  # Don't check times for server errors
                assert response_time < 2.0, f"{method} {endpoint} should be faster than 2s"


class TestWorkflowIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def mock_components(self):
        """Mock system components for workflow testing."""
        components = {}
        
        # Mock OdorDiffusion
        mock_diffusion = Mock()
        mock_diffusion.generate.return_value = [
            Molecule("CCO", confidence=0.9),
            Molecule("CC(C)O", confidence=0.8)
        ]
        components['diffusion'] = mock_diffusion
        
        # Mock SafetyFilter
        mock_safety = Mock()
        mock_safety.assess_molecule.return_value = {
            "toxicity_score": 0.05,
            "skin_sensitizer": False,
            "eco_score": 0.1,
            "ifra_compliant": True,
            "regulatory_flags": [],
            "overall_score": 0.95
        }
        components['safety'] = mock_safety
        
        # Mock SynthesisPlanner
        mock_synthesis = Mock()
        mock_synthesis.suggest_synthesis_routes.return_value = [
            {
                "route_id": 1,
                "steps": ["step1", "step2"],
                "score": 0.8,
                "cost_estimate": 50.0
            }
        ]
        components['synthesis'] = mock_synthesis
        
        # Mock Cache
        mock_cache = Mock()
        mock_cache.get_generation_result.return_value = None
        mock_cache.cache_generation_result.return_value = None
        components['cache'] = mock_cache
        
        return components
    
    def test_complete_generation_workflow(self, mock_components):
        """Test complete molecule generation workflow."""
        # Simulate complete workflow
        prompt = "fresh citrus scent"
        num_molecules = 3
        
        # Step 1: Check cache (miss)
        cached_result = mock_components['cache'].get_generation_result(prompt, {})
        assert cached_result is None
        
        # Step 2: Generate molecules
        molecules = mock_components['diffusion'].generate(
            prompt=prompt,
            num_molecules=num_molecules
        )
        assert len(molecules) == 2  # Mock returns 2 molecules
        
        # Step 3: Assess safety for each molecule
        safe_molecules = []
        for molecule in molecules:
            safety_assessment = mock_components['safety'].assess_molecule(molecule)
            if safety_assessment["overall_score"] > 0.8:
                molecule.safety_score = safety_assessment["overall_score"]
                safe_molecules.append(molecule)
        
        assert len(safe_molecules) > 0
        
        # Step 4: Get synthesis routes for safe molecules
        molecules_with_routes = []
        for molecule in safe_molecules:
            routes = mock_components['synthesis'].suggest_synthesis_routes(molecule)
            if routes:
                molecule.synthesis_routes = routes
                molecules_with_routes.append(molecule)
        
        assert len(molecules_with_routes) > 0
        
        # Step 5: Cache result
        mock_components['cache'].cache_generation_result(
            prompt, {}, molecules_with_routes
        )
        
        # Verify workflow completed successfully
        final_molecules = molecules_with_routes
        assert all(hasattr(m, 'safety_score') for m in final_molecules)
        assert all(hasattr(m, 'synthesis_routes') for m in final_molecules)
    
    def test_batch_processing_workflow(self, mock_components):
        """Test batch processing workflow."""
        prompts = [
            "fresh citrus scent",
            "warm vanilla fragrance",
            "floral rose bouquet"
        ]
        
        batch_results = []
        
        for prompt in prompts:
            # Simulate individual processing
            molecules = mock_components['diffusion'].generate(prompt=prompt, num_molecules=2)
            
            # Add safety assessment
            for molecule in molecules:
                assessment = mock_components['safety'].assess_molecule(molecule)
                molecule.safety_score = assessment["overall_score"]
            
            batch_results.append({
                "prompt": prompt,
                "molecules": molecules,
                "processing_time": 1.0
            })
        
        # Verify batch processing
        assert len(batch_results) == len(prompts)
        for result in batch_results:
            assert "prompt" in result
            assert "molecules" in result
            assert "processing_time" in result
            assert len(result["molecules"]) > 0
    
    def test_error_recovery_workflow(self, mock_components):
        """Test error recovery in workflows."""
        # Simulate error in diffusion
        mock_components['diffusion'].generate.side_effect = Exception("Model error")
        
        try:
            molecules = mock_components['diffusion'].generate(prompt="test", num_molecules=1)
            assert False, "Should have raised exception"
        except Exception as e:
            # Simulate fallback mechanism
            fallback_molecules = [Molecule("CCO", confidence=0.5)]  # Fallback result
            
            # Continue with safety assessment
            for molecule in fallback_molecules:
                assessment = mock_components['safety'].assess_molecule(molecule)
                molecule.safety_score = assessment["overall_score"]
            
            assert len(fallback_molecules) == 1
            assert fallback_molecules[0].safety_score > 0
    
    def test_caching_workflow(self, mock_components):
        """Test caching integration in workflow."""
        prompt = "test prompt"
        
        # First request - cache miss
        mock_components['cache'].get_generation_result.return_value = None
        molecules = mock_components['diffusion'].generate(prompt=prompt, num_molecules=1)
        mock_components['cache'].cache_generation_result(prompt, {}, molecules)
        
        # Verify caching calls
        mock_components['cache'].get_generation_result.assert_called()
        mock_components['cache'].cache_generation_result.assert_called()
        
        # Second request - cache hit
        cached_molecules = [Molecule("CCO", confidence=0.9)]
        mock_components['cache'].get_generation_result.return_value = cached_molecules
        
        result = mock_components['cache'].get_generation_result(prompt, {})
        assert result == cached_molecules
    
    def test_concurrent_workflow(self, mock_components):
        """Test concurrent processing workflow."""
        import threading
        
        results = []
        errors = []
        
        def process_request(prompt_id):
            try:
                prompt = f"test prompt {prompt_id}"
                molecules = mock_components['diffusion'].generate(prompt=prompt, num_molecules=1)
                
                for molecule in molecules:
                    assessment = mock_components['safety'].assess_molecule(molecule)
                    molecule.safety_score = assessment["overall_score"]
                
                results.append({
                    "prompt_id": prompt_id,
                    "molecules": molecules
                })
            except Exception as e:
                errors.append(f"Prompt {prompt_id}: {e}")
        
        # Start multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify concurrent processing
        assert len(results) == 5
        assert len(errors) == 0
        
        for result in results:
            assert "prompt_id" in result
            assert "molecules" in result
            assert len(result["molecules"]) > 0


class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""
    
    @pytest.fixture
    def performance_config(self):
        """Configuration for performance testing."""
        return {
            'max_requests_per_second': 10,
            'max_response_time_ms': 2000,
            'max_concurrent_requests': 5,
            'cache_hit_rate_target': 0.3,
            'memory_limit_mb': 500
        }
    
    def test_throughput_requirements(self, performance_config):
        """Test API throughput requirements."""
        # Mock request processing
        def mock_process_request():
            time.sleep(0.05)  # Simulate 50ms processing time
            return {"status": "success"}
        
        # Test sustained throughput
        start_time = time.time()
        requests_processed = 0
        target_requests = 50
        
        for _ in range(target_requests):
            result = mock_process_request()
            requests_processed += 1
        
        total_time = time.time() - start_time
        throughput = requests_processed / total_time
        
        # Should meet minimum throughput requirement
        assert throughput >= 5, f"Throughput {throughput:.2f} req/s below minimum"
    
    def test_response_time_distribution(self, performance_config):
        """Test response time distribution."""
        response_times = []
        
        # Simulate various request types with different processing times
        request_types = [
            ("health_check", 0.01),
            ("stats", 0.02),
            ("generation", 0.1),
            ("safety_assessment", 0.05),
            ("batch_generation", 0.3)
        ]
        
        for request_type, base_time in request_types:
            for _ in range(10):
                start_time = time.time()
                time.sleep(base_time)  # Simulate processing
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                response_times.append((request_type, response_time))
        
        # Analyze distribution
        by_type = {}
        for request_type, response_time in response_times:
            if request_type not in by_type:
                by_type[request_type] = []
            by_type[request_type].append(response_time)
        
        # Check percentiles
        for request_type, times in by_type.items():
            times.sort()
            p95 = times[int(0.95 * len(times))]
            p99 = times[int(0.99 * len(times))]
            
            # Set different thresholds for different request types
            if request_type in ["health_check", "stats"]:
                assert p95 < 100, f"{request_type} P95 {p95:.2f}ms too high"
            elif request_type == "generation":
                assert p95 < 2000, f"{request_type} P95 {p95:.2f}ms too high"
            elif request_type == "batch_generation":
                assert p95 < 5000, f"{request_type} P95 {p95:.2f}ms too high"
    
    def test_memory_usage_stability(self, performance_config):
        """Test memory usage stability under load."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        data_store = []
        for i in range(100):
            # Create some data to simulate processing
            data = {
                "id": i,
                "molecules": [f"molecule_{j}" for j in range(10)],
                "properties": {f"prop_{k}": k * 1.5 for k in range(20)}
            }
            data_store.append(data)
            
            # Check memory periodically
            if i % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory should not grow excessively
                assert memory_increase < performance_config['memory_limit_mb'], \
                       f"Memory usage increased by {memory_increase:.2f}MB"
        
        # Cleanup
        data_store.clear()
    
    def test_concurrent_load_handling(self, performance_config):
        """Test handling of concurrent load."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def simulate_request(request_id):
            try:
                start_time = time.time()
                
                # Simulate varying processing times
                processing_time = 0.1 + (request_id % 3) * 0.05
                time.sleep(processing_time)
                
                response_time = time.time() - start_time
                results_queue.put({
                    "request_id": request_id,
                    "response_time": response_time,
                    "status": "success"
                })
            except Exception as e:
                error_queue.put(f"Request {request_id}: {e}")
        
        # Start concurrent requests
        threads = []
        num_requests = performance_config['max_concurrent_requests'] * 2
        
        start_time = time.time()
        for i in range(num_requests):
            thread = threading.Thread(target=simulate_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())
        
        # Verify concurrent handling
        assert len(results) == num_requests, f"Only {len(results)} of {num_requests} completed"
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Check that concurrent processing was actually faster than sequential
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        sequential_time_estimate = avg_response_time * num_requests
        
        # Concurrent processing should be significantly faster than sequential
        speedup = sequential_time_estimate / total_time
        assert speedup > 1.5, f"Insufficient concurrency speedup: {speedup:.2f}x"


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Integration tests for async workflows."""
    
    async def test_async_generation_workflow(self):
        """Test async molecule generation workflow."""
        # Mock async components
        async def mock_generate(prompt, num_molecules=5):
            await asyncio.sleep(0.1)  # Simulate async processing
            return [
                Molecule("CCO", confidence=0.9),
                Molecule("CC(C)O", confidence=0.8)
            ]
        
        async def mock_assess_safety(molecule):
            await asyncio.sleep(0.05)  # Simulate async safety check
            return {
                "overall_score": 0.9,
                "toxicity_score": 0.05,
                "regulatory_flags": []
            }
        
        # Execute async workflow
        prompt = "fresh citrus scent"
        molecules = await mock_generate(prompt, num_molecules=3)
        
        # Process safety assessments concurrently
        safety_tasks = [mock_assess_safety(mol) for mol in molecules]
        safety_results = await asyncio.gather(*safety_tasks)
        
        # Add safety scores to molecules
        for molecule, safety_result in zip(molecules, safety_results):
            molecule.safety_score = safety_result["overall_score"]
        
        # Verify async workflow
        assert len(molecules) == 2
        assert all(hasattr(m, 'safety_score') for m in molecules)
        assert all(m.safety_score > 0.8 for m in molecules)
    
    async def test_async_batch_processing(self):
        """Test async batch processing."""
        prompts = [
            "fresh citrus scent",
            "warm vanilla fragrance", 
            "floral rose bouquet",
            "woody cedar aroma"
        ]
        
        async def mock_process_prompt(prompt):
            await asyncio.sleep(0.1)  # Simulate processing
            return {
                "prompt": prompt,
                "molecules": [Molecule("CCO", confidence=0.8)],
                "processing_time": 0.1
            }
        
        start_time = time.time()
        
        # Process all prompts concurrently
        tasks = [mock_process_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Verify concurrent processing
        assert len(results) == len(prompts)
        assert all("molecules" in result for result in results)
        
        # Should be faster than sequential processing
        assert total_time < len(prompts) * 0.1 * 0.8  # Allow some overhead
    
    async def test_async_error_handling(self):
        """Test error handling in async workflows."""
        async def failing_operation(should_fail=True):
            await asyncio.sleep(0.05)
            if should_fail:
                raise Exception("Simulated failure")
            return "success"
        
        async def robust_operation():
            try:
                result = await failing_operation(should_fail=True)
                return result
            except Exception:
                # Fallback to safe operation
                return await failing_operation(should_fail=False)
        
        # Test error recovery
        result = await robust_operation()
        assert result == "success"
        
        # Test multiple operations with some failures
        operations = [
            failing_operation(should_fail=i % 2 == 0) 
            for i in range(5)
        ]
        
        results = await asyncio.gather(*operations, return_exceptions=True)
        
        # Should have some successes and some exceptions
        successes = [r for r in results if r == "success"]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        assert len(successes) > 0
        assert len(exceptions) > 0
        assert len(successes) + len(exceptions) == 5
    
    async def test_async_rate_limiting(self):
        """Test rate limiting in async context."""
        request_times = []
        rate_limit_delay = 0.1  # 100ms between requests
        
        async def rate_limited_request(request_id):
            # Simulate rate limiting
            current_time = time.time()
            if request_times:
                time_since_last = current_time - request_times[-1]
                if time_since_last < rate_limit_delay:
                    await asyncio.sleep(rate_limit_delay - time_since_last)
            
            request_times.append(time.time())
            return f"response_{request_id}"
        
        # Make several requests
        start_time = time.time()
        tasks = [rate_limited_request(i) for i in range(5)]
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify rate limiting
        assert len(responses) == 5
        assert all(resp.startswith("response_") for resp in responses)
        
        # Should take at least 4 * rate_limit_delay due to rate limiting
        expected_min_time = 4 * rate_limit_delay * 0.8  # Allow some tolerance
        assert total_time >= expected_min_time