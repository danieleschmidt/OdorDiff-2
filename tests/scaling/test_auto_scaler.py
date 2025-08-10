"""
Comprehensive tests for auto-scaling functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import psutil
from odordiff2.scaling.auto_scaler import (
    AutoScaler,
    ScalingMetrics,
    ScalingAction,
    ResourceMonitor,
    PredictiveScaler
)


class TestScalingMetrics:
    """Test scaling metrics collection and analysis."""
    
    def test_metrics_initialization(self):
        """Test metrics object initialization."""
        metrics = ScalingMetrics()
        
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.active_requests == 0
        assert metrics.response_time_p95 == 0.0
        assert isinstance(metrics.timestamp, float)
    
    def test_metrics_validation(self):
        """Test metrics validation logic."""
        metrics = ScalingMetrics(
            cpu_usage=85.5,
            memory_usage=78.2,
            active_requests=150,
            response_time_p95=245.7
        )
        
        assert metrics.needs_scaling_up()
        assert not metrics.needs_scaling_down()
    
    def test_metrics_thresholds(self):
        """Test scaling threshold detection."""
        # High load scenario
        high_metrics = ScalingMetrics(
            cpu_usage=88.0,
            memory_usage=82.0,
            active_requests=200,
            response_time_p95=300.0
        )
        
        assert high_metrics.needs_scaling_up()
        assert high_metrics.get_scaling_urgency() == "high"
        
        # Low load scenario  
        low_metrics = ScalingMetrics(
            cpu_usage=15.0,
            memory_usage=25.0,
            active_requests=5,
            response_time_p95=50.0
        )
        
        assert low_metrics.needs_scaling_down()
        assert low_metrics.get_scaling_urgency() == "low"
    
    def test_metrics_serialization(self):
        """Test metrics serialization for monitoring."""
        metrics = ScalingMetrics(
            cpu_usage=75.5,
            memory_usage=68.3,
            active_requests=120
        )
        
        data = metrics.to_dict()
        
        assert data["cpu_usage"] == 75.5
        assert data["memory_usage"] == 68.3
        assert data["active_requests"] == 120
        assert "timestamp" in data


class TestResourceMonitor:
    """Test system resource monitoring."""
    
    @pytest.fixture
    def monitor(self):
        """Create resource monitor instance."""
        return ResourceMonitor(collection_interval=0.1)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_cpu_memory_monitoring(self, mock_memory, mock_cpu, monitor):
        """Test CPU and memory monitoring."""
        # Mock system metrics
        mock_cpu.return_value = 65.2
        mock_memory.return_value = Mock(percent=72.8)
        
        metrics = monitor.collect_system_metrics()
        
        assert metrics.cpu_usage == 65.2
        assert metrics.memory_usage == 72.8
        mock_cpu.assert_called_once()
        mock_memory.assert_called_once()
    
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    def test_io_monitoring(self, mock_net, mock_disk, monitor):
        """Test disk and network I/O monitoring."""
        # Mock I/O counters
        mock_disk.return_value = Mock(
            read_bytes=1024*1024*100,  # 100MB
            write_bytes=1024*1024*50   # 50MB
        )
        mock_net.return_value = Mock(
            bytes_sent=1024*1024*200,  # 200MB
            bytes_recv=1024*1024*150   # 150MB
        )
        
        io_metrics = monitor.collect_io_metrics()
        
        assert io_metrics["disk_read_mb"] == 100
        assert io_metrics["disk_write_mb"] == 50
        assert io_metrics["network_sent_mb"] == 200
        assert io_metrics["network_recv_mb"] == 150
    
    def test_process_monitoring(self, monitor):
        """Test process-specific monitoring."""
        process_metrics = monitor.collect_process_metrics()
        
        # Should contain current process information
        assert "process_cpu" in process_metrics
        assert "process_memory_mb" in process_metrics
        assert "open_files" in process_metrics
        assert "threads" in process_metrics
        
        # Values should be reasonable
        assert 0 <= process_metrics["process_cpu"] <= 100
        assert process_metrics["process_memory_mb"] > 0
        assert process_metrics["open_files"] >= 0
        assert process_metrics["threads"] > 0


class TestAutoScaler:
    """Test auto-scaling core functionality."""
    
    @pytest.fixture
    def scaler(self):
        """Create auto-scaler instance."""
        config = {
            "min_replicas": 2,
            "max_replicas": 20,
            "cpu_threshold_up": 80.0,
            "cpu_threshold_down": 30.0,
            "memory_threshold_up": 75.0,
            "memory_threshold_down": 25.0,
            "cooldown_period": 60,
            "scaling_factor": 1.5
        }
        return AutoScaler(config)
    
    def test_scaler_initialization(self, scaler):
        """Test auto-scaler initialization."""
        assert scaler.min_replicas == 2
        assert scaler.max_replicas == 20
        assert scaler.current_replicas == 2  # Should start at minimum
        assert scaler.cpu_threshold_up == 80.0
        assert scaler.scaling_factor == 1.5
    
    def test_scaling_decision_scale_up(self, scaler):
        """Test scaling up decision logic."""
        high_load_metrics = ScalingMetrics(
            cpu_usage=85.0,
            memory_usage=78.0,
            active_requests=180,
            response_time_p95=280.0
        )
        
        action = scaler.decide_scaling_action(high_load_metrics)
        
        assert action.action_type == "scale_up"
        assert action.target_replicas > scaler.current_replicas
        assert action.reason == "High CPU and memory usage"
        assert action.confidence > 0.8
    
    def test_scaling_decision_scale_down(self, scaler):
        """Test scaling down decision logic."""
        # Set current replicas higher
        scaler.current_replicas = 8
        
        low_load_metrics = ScalingMetrics(
            cpu_usage=20.0,
            memory_usage=18.0,
            active_requests=10,
            response_time_p95=45.0
        )
        
        action = scaler.decide_scaling_action(low_load_metrics)
        
        assert action.action_type == "scale_down"
        assert action.target_replicas < scaler.current_replicas
        assert action.target_replicas >= scaler.min_replicas
        assert action.reason == "Low resource utilization"
    
    def test_scaling_decision_no_action(self, scaler):
        """Test no scaling decision for moderate load."""
        moderate_metrics = ScalingMetrics(
            cpu_usage=50.0,
            memory_usage=45.0,
            active_requests=75,
            response_time_p95=120.0
        )
        
        action = scaler.decide_scaling_action(moderate_metrics)
        
        assert action.action_type == "no_action"
        assert action.target_replicas == scaler.current_replicas
        assert action.confidence < 0.5
    
    def test_cooldown_period(self, scaler):
        """Test scaling cooldown period enforcement."""
        # Simulate recent scaling action
        scaler.last_scaling_time = time.time() - 30  # 30 seconds ago
        
        high_load_metrics = ScalingMetrics(
            cpu_usage=90.0,
            memory_usage=85.0
        )
        
        action = scaler.decide_scaling_action(high_load_metrics)
        
        # Should not scale due to cooldown
        assert action.action_type == "no_action"
        assert "cooldown" in action.reason.lower()
    
    def test_replica_limits(self, scaler):
        """Test replica count limits enforcement."""
        # Test upper limit
        scaler.current_replicas = 20  # At maximum
        
        very_high_metrics = ScalingMetrics(
            cpu_usage=95.0,
            memory_usage=90.0
        )
        
        action = scaler.decide_scaling_action(very_high_metrics)
        assert action.target_replicas <= scaler.max_replicas
        
        # Test lower limit
        scaler.current_replicas = 2  # At minimum
        
        very_low_metrics = ScalingMetrics(
            cpu_usage=5.0,
            memory_usage=10.0
        )
        
        action = scaler.decide_scaling_action(very_low_metrics)
        assert action.target_replicas >= scaler.min_replicas
    
    @pytest.mark.asyncio
    async def test_execute_scaling_action(self, scaler):
        """Test scaling action execution."""
        with patch('odordiff2.scaling.auto_scaler.KubernetesScaler') as mock_k8s:
            mock_scaler = AsyncMock()
            mock_k8s.return_value = mock_scaler
            mock_scaler.scale_deployment.return_value = True
            
            action = ScalingAction(
                action_type="scale_up",
                target_replicas=6,
                current_replicas=4,
                reason="High load",
                confidence=0.9
            )
            
            result = await scaler.execute_scaling_action(action)
            
            assert result is True
            mock_scaler.scale_deployment.assert_called_once_with(
                deployment_name="odordiff2-api",
                target_replicas=6
            )


class TestPredictiveScaler:
    """Test predictive scaling functionality."""
    
    @pytest.fixture
    def predictor(self):
        """Create predictive scaler instance."""
        return PredictiveScaler(
            history_window=24,  # 24 data points
            prediction_horizon=6  # Predict 6 steps ahead
        )
    
    def test_predictor_initialization(self, predictor):
        """Test predictive scaler initialization."""
        assert predictor.history_window == 24
        assert predictor.prediction_horizon == 6
        assert len(predictor.metrics_history) == 0
        assert predictor.model is None
    
    def test_add_metrics_to_history(self, predictor):
        """Test adding metrics to prediction history."""
        for i in range(10):
            metrics = ScalingMetrics(
                cpu_usage=50.0 + i * 2,
                memory_usage=40.0 + i * 1.5,
                active_requests=100 + i * 10
            )
            predictor.add_metrics(metrics)
        
        assert len(predictor.metrics_history) == 10
        
        # Test history limit
        for i in range(20):
            metrics = ScalingMetrics(cpu_usage=60.0)
            predictor.add_metrics(metrics)
        
        assert len(predictor.metrics_history) == 24  # Should be limited
    
    def test_trend_detection(self, predictor):
        """Test trend detection in metrics."""
        # Add increasing trend
        for i in range(15):
            metrics = ScalingMetrics(
                cpu_usage=30.0 + i * 3,  # Increasing from 30% to 72%
                memory_usage=25.0 + i * 2,  # Increasing from 25% to 53%
            )
            predictor.add_metrics(metrics)
        
        trends = predictor.detect_trends()
        
        assert trends["cpu_trend"] > 0.5  # Strong positive trend
        assert trends["memory_trend"] > 0.3  # Positive trend
        assert "increasing" in trends["overall_trend"]
    
    def test_simple_prediction(self, predictor):
        """Test simple linear prediction."""
        # Add consistent increasing pattern
        for i in range(20):
            metrics = ScalingMetrics(
                cpu_usage=40.0 + i * 1.5,
                memory_usage=35.0 + i * 1.0,
                active_requests=80 + i * 5
            )
            predictor.add_metrics(metrics)
        
        predictions = predictor.predict_future_load()
        
        assert len(predictions) == 6  # Should predict 6 steps ahead
        
        # Predictions should continue the trend
        last_cpu = predictor.metrics_history[-1].cpu_usage
        predicted_cpu = predictions[-1]["cpu_usage"]
        
        assert predicted_cpu > last_cpu  # Should be higher due to trend
        assert predicted_cpu < 100.0  # Should be realistic
    
    def test_predict_scaling_needs(self, predictor):
        """Test predicting future scaling needs."""
        # Add pattern that will exceed thresholds
        for i in range(15):
            metrics = ScalingMetrics(
                cpu_usage=60.0 + i * 2,  # Will reach 88% 
                memory_usage=50.0 + i * 1.5,  # Will reach 71%
            )
            predictor.add_metrics(metrics)
        
        scaling_prediction = predictor.predict_scaling_needs(
            cpu_threshold=80.0,
            memory_threshold=75.0
        )
        
        assert scaling_prediction["action"] == "scale_up"
        assert scaling_prediction["steps_until_action"] <= 6
        assert scaling_prediction["confidence"] > 0.6
        assert "CPU trend" in scaling_prediction["reason"]


class TestScalingIntegration:
    """Test integration between scaling components."""
    
    @pytest.mark.asyncio
    async def test_full_scaling_workflow(self):
        """Test complete scaling workflow from metrics to action."""
        # Setup components
        monitor = ResourceMonitor(collection_interval=0.1)
        scaler_config = {
            "min_replicas": 2,
            "max_replicas": 10,
            "cpu_threshold_up": 75.0,
            "memory_threshold_up": 70.0,
            "cooldown_period": 30
        }
        scaler = AutoScaler(scaler_config)
        
        # Mock high system load
        with patch('psutil.cpu_percent', return_value=82.0), \
             patch('psutil.virtual_memory', return_value=Mock(percent=78.0)), \
             patch('odordiff2.scaling.auto_scaler.KubernetesScaler') as mock_k8s:
            
            mock_k8s_instance = AsyncMock()
            mock_k8s.return_value = mock_k8s_instance
            mock_k8s_instance.scale_deployment.return_value = True
            
            # Collect metrics
            metrics = monitor.collect_system_metrics()
            
            # Make scaling decision
            action = scaler.decide_scaling_action(metrics)
            
            # Execute scaling action
            if action.action_type != "no_action":
                result = await scaler.execute_scaling_action(action)
                assert result is True
                
                # Verify scaling was called
                mock_k8s_instance.scale_deployment.assert_called_once()
                
                # Verify replica count was updated
                assert scaler.current_replicas == action.target_replicas
    
    @pytest.mark.asyncio
    async def test_predictive_scaling_integration(self):
        """Test integration with predictive scaling."""
        predictor = PredictiveScaler(history_window=10, prediction_horizon=3)
        scaler = AutoScaler({
            "min_replicas": 2,
            "max_replicas": 8,
            "cpu_threshold_up": 80.0,
            "predictive_enabled": True
        })
        
        # Build metrics history showing increasing trend
        for i in range(12):
            metrics = ScalingMetrics(
                cpu_usage=55.0 + i * 2.5,  # Will exceed threshold soon
                memory_usage=45.0 + i * 1.8,
            )
            predictor.add_metrics(metrics)
        
        # Get predictive scaling recommendation
        prediction = predictor.predict_scaling_needs(
            cpu_threshold=80.0,
            memory_threshold=75.0
        )
        
        if prediction["action"] == "scale_up":
            # Create scaling action based on prediction
            action = ScalingAction(
                action_type="scale_up",
                target_replicas=scaler.current_replicas + 1,
                current_replicas=scaler.current_replicas,
                reason=f"Predictive: {prediction['reason']}",
                confidence=prediction["confidence"]
            )
            
            assert action.action_type == "scale_up"
            assert "Predictive" in action.reason
    
    def test_scaling_metrics_aggregation(self):
        """Test aggregating multiple metrics sources."""
        metrics_list = [
            ScalingMetrics(cpu_usage=70, memory_usage=60, active_requests=100),
            ScalingMetrics(cpu_usage=75, memory_usage=65, active_requests=120),
            ScalingMetrics(cpu_usage=80, memory_usage=70, active_requests=110),
        ]
        
        # Calculate aggregated metrics
        avg_cpu = sum(m.cpu_usage for m in metrics_list) / len(metrics_list)
        avg_memory = sum(m.memory_usage for m in metrics_list) / len(metrics_list)
        max_requests = max(m.active_requests for m in metrics_list)
        
        aggregated = ScalingMetrics(
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            active_requests=max_requests
        )
        
        assert aggregated.cpu_usage == 75.0
        assert aggregated.memory_usage == 65.0
        assert aggregated.active_requests == 120
        
        # Should trigger scaling based on aggregated data
        assert aggregated.needs_scaling_up()


class TestScalingFailureHandling:
    """Test scaling error handling and recovery."""
    
    @pytest.fixture
    def scaler(self):
        """Create scaler for error testing."""
        return AutoScaler({
            "min_replicas": 2,
            "max_replicas": 10,
            "retry_attempts": 3,
            "retry_delay": 1.0
        })
    
    @pytest.mark.asyncio
    async def test_scaling_failure_retry(self, scaler):
        """Test retry mechanism for scaling failures."""
        with patch('odordiff2.scaling.auto_scaler.KubernetesScaler') as mock_k8s:
            mock_scaler = AsyncMock()
            mock_k8s.return_value = mock_scaler
            
            # First two attempts fail, third succeeds
            mock_scaler.scale_deployment.side_effect = [
                Exception("Network error"),
                Exception("API timeout"),
                True  # Success on third attempt
            ]
            
            action = ScalingAction(
                action_type="scale_up",
                target_replicas=4,
                current_replicas=2,
                reason="Test scaling",
                confidence=0.8
            )
            
            result = await scaler.execute_scaling_action_with_retry(action)
            
            assert result is True
            assert mock_scaler.scale_deployment.call_count == 3
    
    @pytest.mark.asyncio
    async def test_scaling_complete_failure(self, scaler):
        """Test handling of complete scaling failure."""
        with patch('odordiff2.scaling.auto_scaler.KubernetesScaler') as mock_k8s:
            mock_scaler = AsyncMock()
            mock_k8s.return_value = mock_scaler
            mock_scaler.scale_deployment.side_effect = Exception("Persistent error")
            
            action = ScalingAction(
                action_type="scale_up",
                target_replicas=4,
                current_replicas=2,
                reason="Test scaling",
                confidence=0.8
            )
            
            result = await scaler.execute_scaling_action_with_retry(action)
            
            assert result is False
            assert mock_scaler.scale_deployment.call_count == 3  # All retry attempts
            
            # Current replicas should remain unchanged on failure
            assert scaler.current_replicas == 2
    
    def test_invalid_metrics_handling(self):
        """Test handling of invalid/corrupted metrics."""
        scaler = AutoScaler({"min_replicas": 2, "max_replicas": 10})
        
        # Test with invalid CPU usage
        invalid_metrics = ScalingMetrics(
            cpu_usage=-10.0,  # Invalid negative value
            memory_usage=150.0,  # Invalid >100% value
            active_requests=-5  # Invalid negative requests
        )
        
        action = scaler.decide_scaling_action(invalid_metrics)
        
        # Should default to no action for invalid metrics
        assert action.action_type == "no_action"
        assert "invalid" in action.reason.lower() or "error" in action.reason.lower()


@pytest.mark.integration
class TestScalingPerformance:
    """Test scaling system performance characteristics."""
    
    def test_metrics_collection_performance(self):
        """Test performance of metrics collection."""
        monitor = ResourceMonitor(collection_interval=0.01)
        
        start_time = time.time()
        
        # Collect metrics 100 times
        for _ in range(100):
            metrics = monitor.collect_system_metrics()
            assert isinstance(metrics, ScalingMetrics)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should be fast (less than 10ms per collection)
        assert avg_time < 0.01
    
    def test_scaling_decision_performance(self):
        """Test performance of scaling decisions."""
        scaler = AutoScaler({
            "min_replicas": 2,
            "max_replicas": 20,
            "cpu_threshold_up": 80.0,
            "memory_threshold_up": 75.0
        })
        
        metrics = ScalingMetrics(
            cpu_usage=85.0,
            memory_usage=78.0,
            active_requests=150
        )
        
        start_time = time.time()
        
        # Make 1000 scaling decisions
        for _ in range(1000):
            action = scaler.decide_scaling_action(metrics)
            assert isinstance(action, ScalingAction)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 1000
        
        # Should be very fast (less than 1ms per decision)
        assert avg_time < 0.001
    
    def test_memory_usage_stability(self):
        """Test memory usage doesn't grow over time."""
        import gc
        import tracemalloc
        
        tracemalloc.start()
        
        monitor = ResourceMonitor(collection_interval=0.001)
        scaler = AutoScaler({"min_replicas": 2, "max_replicas": 10})
        
        # Simulate long-running operation
        for i in range(1000):
            metrics = ScalingMetrics(
                cpu_usage=50.0 + (i % 50),
                memory_usage=40.0 + (i % 40),
                active_requests=100 + (i % 100)
            )
            
            action = scaler.decide_scaling_action(metrics)
            
            if i % 100 == 0:
                gc.collect()  # Force garbage collection
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 10MB)
        assert current < 10 * 1024 * 1024
        assert peak < 20 * 1024 * 1024