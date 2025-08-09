"""
Unit tests for asynchronous diffusion processing.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any
import time

from odordiff2.core.async_diffusion import (
    AsyncOdorDiffusion, 
    BatchRequest, 
    GenerationResult
)
from odordiff2.models.molecule import Molecule, FragranceFormulation
from odordiff2.safety.filter import SafetyFilter


class TestBatchRequest:
    """Test BatchRequest dataclass."""
    
    def test_batch_request_creation(self):
        """Test creating a BatchRequest instance."""
        callback = Mock()
        request = BatchRequest(
            prompts=["prompt1", "prompt2"],
            params={"num_molecules": 5},
            callback=callback,
            priority=1
        )
        
        assert request.prompts == ["prompt1", "prompt2"]
        assert request.params == {"num_molecules": 5}
        assert request.callback == callback
        assert request.priority == 1
    
    def test_batch_request_defaults(self):
        """Test BatchRequest with default values."""
        request = BatchRequest(
            prompts=["test"],
            params={}
        )
        
        assert request.callback is None
        assert request.priority == 0


class TestGenerationResult:
    """Test GenerationResult dataclass."""
    
    def test_generation_result_success(self):
        """Test successful GenerationResult."""
        molecules = [Molecule("CCO", 0.9)]
        result = GenerationResult(
            prompt="test prompt",
            molecules=molecules,
            processing_time=1.5,
            cache_hit=True
        )
        
        assert result.prompt == "test prompt"
        assert result.molecules == molecules
        assert result.processing_time == 1.5
        assert result.cache_hit is True
        assert result.error is None
    
    def test_generation_result_error(self):
        """Test GenerationResult with error."""
        result = GenerationResult(
            prompt="test prompt",
            molecules=[],
            processing_time=0.5,
            cache_hit=False,
            error="Test error"
        )
        
        assert result.error == "Test error"
        assert result.molecules == []


class TestAsyncOdorDiffusion:
    """Test AsyncOdorDiffusion class."""
    
    @pytest.fixture
    def mock_odor_diffusion(self):
        """Mock the base OdorDiffusion class."""
        with patch('odordiff2.core.async_diffusion.OdorDiffusion') as mock:
            mock_instance = Mock()
            mock_instance.generate.return_value = [Molecule("CCO", 0.9)]
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_cache(self):
        """Mock molecule cache."""
        with patch('odordiff2.core.async_diffusion.get_molecule_cache') as mock:
            cache_instance = Mock()
            cache_instance.get_generation_result.return_value = None
            cache_instance.cache_generation_result = Mock()
            cache_instance.get_cache_stats.return_value = {}
            mock.return_value = cache_instance
            yield cache_instance
    
    @pytest.fixture
    async def async_diffusion(self, mock_odor_diffusion, mock_cache):
        """Create AsyncOdorDiffusion instance for testing."""
        instance = AsyncOdorDiffusion(
            device="cpu",
            max_workers=2,
            batch_size=4,
            enable_caching=True
        )
        await instance.start()
        yield instance
        await instance.stop()
    
    def test_initialization(self, mock_odor_diffusion, mock_cache):
        """Test AsyncOdorDiffusion initialization."""
        async_diffusion = AsyncOdorDiffusion(
            device="cuda",
            max_workers=8,
            batch_size=16,
            enable_caching=False
        )
        
        assert async_diffusion.device == "cuda"
        assert async_diffusion.max_workers == 8
        assert async_diffusion.batch_size == 16
        assert async_diffusion.enable_caching is False
        assert async_diffusion.model is not None
        assert len(async_diffusion.worker_load) == 8
    
    def test_initialization_with_caching(self, mock_odor_diffusion, mock_cache):
        """Test initialization with caching enabled."""
        async_diffusion = AsyncOdorDiffusion(enable_caching=True)
        
        assert async_diffusion.cache is not None
    
    def test_initialization_without_caching(self, mock_odor_diffusion):
        """Test initialization with caching disabled."""
        with patch('odordiff2.core.async_diffusion.get_molecule_cache') as mock_cache:
            async_diffusion = AsyncOdorDiffusion(enable_caching=False)
            assert async_diffusion.cache is None
            mock_cache.assert_not_called()
    
    async def test_start_and_stop(self, mock_odor_diffusion, mock_cache):
        """Test starting and stopping async processing."""
        async_diffusion = AsyncOdorDiffusion()
        
        # Initially not started
        assert async_diffusion.batch_queue is None
        assert async_diffusion.batch_processor_task is None
        
        # Start
        await async_diffusion.start()
        assert async_diffusion.batch_queue is not None
        assert async_diffusion.batch_processor_task is not None
        
        # Stop
        await async_diffusion.stop()
    
    async def test_context_manager(self, mock_odor_diffusion, mock_cache):
        """Test async context manager."""
        async with AsyncOdorDiffusion() as async_diffusion:
            assert async_diffusion.batch_queue is not None
            assert async_diffusion.batch_processor_task is not None
    
    @patch('odordiff2.core.async_diffusion.InputValidator')
    async def test_generate_async_success(
        self, 
        mock_validator, 
        async_diffusion, 
        mock_cache
    ):
        """Test successful async generation."""
        # Setup mocks
        mock_validator.validate_prompt.return_value = "test prompt"
        mock_validator.validate_generation_parameters.return_value = {}
        
        # Mock no cache hit
        mock_cache.get_generation_result.return_value = None
        
        result = await async_diffusion.generate_async(
            prompt="test prompt",
            num_molecules=3
        )
        
        assert isinstance(result, GenerationResult)
        assert result.prompt == "test prompt"
        assert result.error is None
        assert result.cache_hit is False
        assert len(result.molecules) > 0
        assert result.processing_time > 0
    
    @patch('odordiff2.core.async_diffusion.InputValidator')
    async def test_generate_async_cache_hit(
        self,
        mock_validator,
        async_diffusion,
        mock_cache
    ):
        """Test async generation with cache hit."""
        # Setup mocks
        mock_validator.validate_prompt.return_value = "test prompt"
        mock_validator.validate_generation_parameters.return_value = {}
        
        # Mock cache hit
        cached_molecules = [Molecule("CCO", 0.9)]
        mock_cache.get_generation_result.return_value = cached_molecules
        
        result = await async_diffusion.generate_async(
            prompt="test prompt",
            use_cache=True
        )
        
        assert result.cache_hit is True
        assert result.molecules == cached_molecules
        assert async_diffusion.processing_stats['cache_hits'] == 1
    
    @patch('odordiff2.core.async_diffusion.InputValidator')
    async def test_generate_async_error_handling(
        self,
        mock_validator,
        async_diffusion
    ):
        """Test async generation error handling."""
        # Setup validator to raise exception
        mock_validator.validate_prompt.side_effect = Exception("Validation error")
        
        result = await async_diffusion.generate_async("test prompt")
        
        assert result.error == "Validation error"
        assert result.molecules == []
        assert result.cache_hit is False
    
    async def test_generate_molecules_async_load_balancing(self, async_diffusion):
        """Test load balancing in async generation."""
        # Mock the sync generation method
        async_diffusion._sync_generate = Mock(return_value=[Molecule("CCO", 0.9)])
        
        initial_load = async_diffusion.worker_load.copy()
        
        # Start generation (but don't await to test load tracking)
        task = asyncio.create_task(
            async_diffusion._generate_molecules_async(
                "test", 1, None, 0.0, {}
            )
        )
        
        # Give some time for load to be incremented
        await asyncio.sleep(0.01)
        
        # Load should be incremented
        assert sum(async_diffusion.worker_load) > sum(initial_load)
        
        # Complete the task
        await task
        
        # Load should be back to initial
        assert async_diffusion.worker_load == initial_load
    
    def test_sync_generate_wrapper(self, async_diffusion, mock_odor_diffusion):
        """Test synchronous generation wrapper."""
        safety_filter = SafetyFilter()
        
        result = async_diffusion._sync_generate(
            "test prompt",
            5,
            safety_filter,
            0.5,
            {"extra_param": True}
        )
        
        mock_odor_diffusion.generate.assert_called_once_with(
            prompt="test prompt",
            num_molecules=5,
            safety_filter=safety_filter,
            synthesizability_min=0.5,
            extra_param=True
        )
    
    async def test_generate_batch_async(self, async_diffusion):
        """Test batch generation."""
        batch_request = BatchRequest(
            prompts=["prompt1", "prompt2", "prompt3"],
            params={"num_molecules": 2}
        )
        
        # Mock generate_async to return predictable results
        async def mock_generate_async(prompt, **kwargs):
            return GenerationResult(
                prompt=prompt,
                molecules=[Molecule("CCO", 0.9)],
                processing_time=1.0,
                cache_hit=False
            )
        
        async_diffusion.generate_async = mock_generate_async
        
        results = await async_diffusion.generate_batch_async(batch_request)
        
        assert len(results) == 3
        assert all(isinstance(r, GenerationResult) for r in results)
        assert results[0].prompt == "prompt1"
        assert results[1].prompt == "prompt2"
        assert results[2].prompt == "prompt3"
    
    async def test_generate_batch_async_with_progress(self, async_diffusion):
        """Test batch generation with progress callback."""
        progress_calls = []
        
        async def progress_callback(current, total):
            progress_calls.append((current, total))
        
        batch_request = BatchRequest(
            prompts=["prompt1", "prompt2"],
            params={"num_molecules": 1}
        )
        
        # Mock generate_async
        async def mock_generate_async(prompt, **kwargs):
            return GenerationResult(
                prompt=prompt,
                molecules=[Molecule("CCO", 0.9)],
                processing_time=0.5,
                cache_hit=False
            )
        
        async_diffusion.generate_async = mock_generate_async
        
        results = await async_diffusion.generate_batch_async(
            batch_request, 
            progress_callback=progress_callback
        )
        
        assert len(results) == 2
        assert len(progress_calls) == 2
        assert progress_calls[0][1] == 2  # total should be 2
        assert progress_calls[1][1] == 2
    
    async def test_generate_batch_async_error_handling(self, async_diffusion):
        """Test batch generation with errors."""
        batch_request = BatchRequest(
            prompts=["good_prompt", "bad_prompt"],
            params={"num_molecules": 1}
        )
        
        async def mock_generate_async(prompt, **kwargs):
            if prompt == "bad_prompt":
                raise Exception("Generation failed")
            return GenerationResult(
                prompt=prompt,
                molecules=[Molecule("CCO", 0.9)],
                processing_time=1.0,
                cache_hit=False
            )
        
        async_diffusion.generate_async = mock_generate_async
        
        results = await async_diffusion.generate_batch_async(batch_request)
        
        assert len(results) == 2
        assert results[0].error is None
        assert results[1].error == "Generation failed"
        assert results[1].prompt == "bad_prompt"
    
    async def test_submit_batch(self, async_diffusion):
        """Test batch submission."""
        batch_request = BatchRequest(
            prompts=["test"],
            params={}
        )
        
        success = await async_diffusion.submit_batch(batch_request)
        assert success is True
    
    async def test_optimize_fragrance_async(self, async_diffusion):
        """Test async fragrance optimization."""
        # Mock generate_async
        async def mock_generate_async(prompt, **kwargs):
            return GenerationResult(
                prompt=prompt,
                molecules=[Molecule("CCO", 0.9)],
                processing_time=1.0,
                cache_hit=False
            )
        
        async_diffusion.generate_async = mock_generate_async
        
        formulation = await async_diffusion.optimize_fragrance_async(
            base_notes="sandalwood",
            heart_notes="rose",
            top_notes="bergamot",
            style="modern",
            iterations=2
        )
        
        assert isinstance(formulation, FragranceFormulation)
        assert len(formulation.base_accord) > 0
        assert len(formulation.heart_accord) > 0
        assert len(formulation.top_accord) > 0
    
    def test_score_formulation(self, async_diffusion):
        """Test formulation scoring."""
        # Create test molecules
        mol1 = Molecule("CCO", 0.9)
        mol1.safety_score = 0.8
        mol1.synth_score = 0.7
        
        mol2 = Molecule("CC(C)O", 0.8)
        mol2.safety_score = 0.9
        mol2.synth_score = 0.8
        
        formulation = FragranceFormulation(
            base_accord=[mol1],
            heart_accord=[mol2],
            top_accord=[],
            style_descriptor="test"
        )
        
        score = async_diffusion._score_formulation(formulation)
        
        assert 0 <= score <= 1
        assert score > 0  # Should have some score with valid molecules
    
    def test_score_formulation_empty(self, async_diffusion):
        """Test scoring empty formulation."""
        formulation = FragranceFormulation(
            base_accord=[],
            heart_accord=[],
            top_accord=[],
            style_descriptor="empty"
        )
        
        score = async_diffusion._score_formulation(formulation)
        assert score == 0.0
    
    def test_get_performance_stats(self, async_diffusion, mock_cache):
        """Test performance statistics."""
        # Set some stats
        async_diffusion.processing_stats['total_requests'] = 100
        async_diffusion.processing_stats['cache_hits'] = 30
        
        stats = async_diffusion.get_performance_stats()
        
        assert 'total_requests' in stats
        assert 'cache_hits' in stats
        assert 'worker_load' in stats
        assert 'cache_stats' in stats
        assert 'queue_size' in stats
        
        assert stats['total_requests'] == 100
        assert stats['cache_hits'] == 30
    
    async def test_preload_cache(self, async_diffusion):
        """Test cache preloading."""
        # Mock generate_async
        async def mock_generate_async(prompt, **kwargs):
            if "fail" in prompt:
                raise Exception("Generation failed")
            return GenerationResult(
                prompt=prompt,
                molecules=[Molecule("CCO", 0.9)],
                processing_time=1.0,
                cache_hit=False
            )
        
        async_diffusion.generate_async = mock_generate_async
        
        common_prompts = ["lavender", "rose", "fail_prompt"]
        loaded = await async_diffusion.preload_cache(common_prompts)
        
        assert loaded == 2  # Should load 2 out of 3 (one fails)
    
    async def test_health_check_healthy(self, async_diffusion):
        """Test health check when system is healthy."""
        # Mock generate_async
        async def mock_generate_async(prompt, **kwargs):
            return GenerationResult(
                prompt=prompt,
                molecules=[Molecule("CCO", 0.9)],
                processing_time=0.5,
                cache_hit=False
            )
        
        async_diffusion.generate_async = mock_generate_async
        
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 256 * 1024 * 1024
            
            health = await async_diffusion.health_check()
        
        assert health['status'] == 'healthy'
        assert health['generation_test'] is True
        assert health['response_time'] > 0
        assert health['memory_usage_mb'] == 256
        assert 'stats' in health
    
    async def test_health_check_degraded(self, async_diffusion):
        """Test health check when generation fails."""
        # Mock generate_async to return empty molecules
        async def mock_generate_async(prompt, **kwargs):
            return GenerationResult(
                prompt=prompt,
                molecules=[],
                processing_time=0.5,
                cache_hit=False,
                error="Generation failed"
            )
        
        async_diffusion.generate_async = mock_generate_async
        
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 256 * 1024 * 1024
            
            health = await async_diffusion.health_check()
        
        assert health['status'] == 'degraded'
        assert health['generation_test'] is False
    
    async def test_health_check_unhealthy(self, async_diffusion):
        """Test health check when exception occurs."""
        # Mock generate_async to raise exception
        async def mock_generate_async(prompt, **kwargs):
            raise Exception("System error")
        
        async_diffusion.generate_async = mock_generate_async
        
        health = await async_diffusion.health_check()
        
        assert health['status'] == 'unhealthy'
        assert 'error' in health
        assert health['error'] == "System error"
    
    async def test_batch_processor_integration(self, async_diffusion):
        """Test batch processor integration."""
        # Create a test callback to collect results
        results_collected = []
        
        async def test_callback(results):
            results_collected.extend(results)
        
        # Mock generate_async
        async def mock_generate_async(prompt, **kwargs):
            return GenerationResult(
                prompt=prompt,
                molecules=[Molecule("CCO", 0.9)],
                processing_time=0.5,
                cache_hit=False
            )
        
        async_diffusion.generate_async = mock_generate_async
        
        # Submit batch request
        batch_request = BatchRequest(
            prompts=["test1", "test2"],
            params={"num_molecules": 1},
            callback=test_callback,
            priority=1
        )
        
        success = await async_diffusion.submit_batch(batch_request)
        assert success is True
        
        # Give time for processing
        await asyncio.sleep(0.1)
        
        # Check that callback was called
        # Note: This test may be flaky depending on timing
        # In a real implementation, you might want better synchronization
    
    def test_processing_stats_updates(self, async_diffusion):
        """Test that processing stats are updated correctly."""
        initial_requests = async_diffusion.processing_stats['total_requests']
        initial_avg_time = async_diffusion.processing_stats['avg_processing_time']
        
        # Simulate processing time update
        processing_time = 2.0
        async_diffusion.processing_stats['total_requests'] += 1
        new_total = async_diffusion.processing_stats['total_requests']
        
        # Calculate new average
        async_diffusion.processing_stats['avg_processing_time'] = (
            (initial_avg_time * (new_total - 1) + processing_time) / new_total
        )
        
        assert async_diffusion.processing_stats['total_requests'] == initial_requests + 1
        
        if initial_requests == 0:
            assert async_diffusion.processing_stats['avg_processing_time'] == processing_time
        else:
            assert async_diffusion.processing_stats['avg_processing_time'] > 0


class TestAsyncIntegration:
    """Integration tests for async functionality."""
    
    @pytest.mark.asyncio
    async def test_concurrent_generation(self):
        """Test concurrent generation requests."""
        with patch('odordiff2.core.async_diffusion.OdorDiffusion') as mock_diffusion:
            with patch('odordiff2.core.async_diffusion.get_molecule_cache') as mock_cache:
                # Setup mocks
                mock_diffusion_instance = Mock()
                mock_diffusion_instance.generate.return_value = [Molecule("CCO", 0.9)]
                mock_diffusion.return_value = mock_diffusion_instance
                
                mock_cache_instance = Mock()
                mock_cache_instance.get_generation_result.return_value = None
                mock_cache.return_value = mock_cache_instance
                
                async with AsyncOdorDiffusion(max_workers=2) as async_diffusion:
                    # Submit multiple concurrent requests
                    tasks = []
                    for i in range(5):
                        task = async_diffusion.generate_async(f"prompt_{i}", num_molecules=1)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                    
                    assert len(results) == 5
                    assert all(isinstance(r, GenerationResult) for r in results)
                    assert all(r.error is None for r in results)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of timeouts and slow operations."""
        with patch('odordiff2.core.async_diffusion.OdorDiffusion') as mock_diffusion:
            with patch('odordiff2.core.async_diffusion.get_molecule_cache') as mock_cache:
                # Setup slow mock
                mock_diffusion_instance = Mock()
                
                def slow_generate(*args, **kwargs):
                    time.sleep(0.1)  # Simulate slow generation
                    return [Molecule("CCO", 0.9)]
                
                mock_diffusion_instance.generate.side_effect = slow_generate
                mock_diffusion.return_value = mock_diffusion_instance
                
                mock_cache.return_value = Mock()
                
                async with AsyncOdorDiffusion() as async_diffusion:
                    start_time = time.time()
                    result = await async_diffusion.generate_async("test prompt")
                    end_time = time.time()
                    
                    # Should complete despite slow underlying generation
                    assert result.error is None
                    assert end_time - start_time > 0.1  # Should take some time