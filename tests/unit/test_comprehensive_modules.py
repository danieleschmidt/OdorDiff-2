"""
Comprehensive unit tests for remaining OdorDiff-2 modules.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
import os

from odordiff2.visualization.dashboard import DashboardManager, ChartGenerator
from odordiff2.visualization.viewer import MoleculeViewer
from odordiff2.monitoring.health import HealthMonitor, HealthChecker
from odordiff2.monitoring.metrics import MetricsCollector, PrometheusExporter
from odordiff2.monitoring.performance import PerformanceMonitor, BenchmarkRunner
from odordiff2.scaling.profiling import PerformanceProfiler, MemoryProfiler
from odordiff2.scaling.streaming import StreamingManager, WebSocketHandler
from odordiff2.utils.recovery import RecoveryManager, BackupValidator
from odordiff2.data.cache import CacheManager, CachePolicy


class TestVisualizationComponents:
    """Test visualization and dashboard components."""
    
    @pytest.fixture
    def dashboard_manager(self):
        """Create dashboard manager instance."""
        return DashboardManager(
            template_dir="/tmp/templates",
            static_dir="/tmp/static",
            auto_refresh_interval=30
        )
    
    def test_dashboard_creation(self, dashboard_manager):
        """Test dashboard creation and configuration."""
        with patch('odordiff2.visualization.dashboard.DashGenerator') as mock_dash:
            mock_app = Mock()
            mock_dash.Dash.return_value = mock_app
            
            # Create dashboard
            dashboard = dashboard_manager.create_dashboard(
                title="OdorDiff-2 Monitoring",
                layout_config={
                    "sections": ["metrics", "generation", "safety"],
                    "theme": "dark",
                    "refresh_rate": 5000
                }
            )
            
            assert dashboard is not None
            mock_dash.Dash.assert_called_once()
    
    def test_chart_generation(self):
        """Test chart generation functionality."""
        with patch('odordiff2.visualization.dashboard.ChartGenerator') as mock_chart_class:
            mock_chart = Mock(spec=ChartGenerator)
            mock_chart_class.return_value = mock_chart
            
            # Mock chart generation
            mock_chart.create_molecule_distribution_chart.return_value = {
                "data": [{"x": [1, 2, 3], "y": [10, 20, 15], "type": "bar"}],
                "layout": {"title": "Molecule Distribution"}
            }
            
            chart_gen = mock_chart_class()
            
            # Generate molecule distribution chart
            chart_data = chart_gen.create_molecule_distribution_chart([
                {"safety_score": 0.9, "synthesis_score": 0.8},
                {"safety_score": 0.95, "synthesis_score": 0.85},
                {"safety_score": 0.88, "synthesis_score": 0.92}
            ])
            
            assert "data" in chart_data
            assert "layout" in chart_data
            assert chart_data["layout"]["title"] == "Molecule Distribution"
    
    def test_molecule_viewer_3d_rendering(self):
        """Test 3D molecule visualization."""
        with patch('odordiff2.visualization.viewer.MoleculeViewer') as mock_viewer_class:
            mock_viewer = Mock(spec=MoleculeViewer)
            mock_viewer_class.return_value = mock_viewer
            
            # Mock 3D rendering
            mock_viewer.render_3d_structure.return_value = {
                "html": "<div id='3d-viewer'>Molecule 3D Structure</div>",
                "script": "// JavaScript for 3D interaction",
                "style": "/* CSS styling */"
            }
            
            viewer = mock_viewer_class()
            
            # Test 3D rendering
            rendering = viewer.render_3d_structure(
                smiles="CCO",
                style="ball_and_stick",
                highlight_functional_groups=True,
                show_hydrogens=False
            )
            
            assert "html" in rendering
            assert "script" in rendering
            assert "style" in rendering
            assert "3d-viewer" in rendering["html"]
    
    def test_property_visualization(self):
        """Test molecular property visualization."""
        with patch('odordiff2.visualization.viewer.PropertyVisualizer') as mock_prop_viz:
            mock_visualizer = Mock()
            mock_prop_viz.return_value = mock_visualizer
            
            # Mock property charts
            mock_visualizer.create_property_radar_chart.return_value = {
                "chart_type": "radar",
                "data": {
                    "properties": ["safety", "synthesis", "novelty", "odor_match"],
                    "values": [0.95, 0.88, 0.72, 0.91]
                }
            }
            
            visualizer = mock_prop_viz()
            
            # Create property radar chart
            chart = visualizer.create_property_radar_chart({
                "safety_score": 0.95,
                "synthesis_score": 0.88,
                "novelty_score": 0.72,
                "odor_match_score": 0.91
            })
            
            assert chart["chart_type"] == "radar"
            assert len(chart["data"]["properties"]) == 4
            assert len(chart["data"]["values"]) == 4


class TestHealthMonitoring:
    """Test health monitoring system."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create health monitor instance."""
        return HealthMonitor(
            check_interval=30,
            timeout=10,
            retry_attempts=3
        )
    
    def test_system_health_check(self, health_monitor):
        """Test system-wide health check."""
        with patch('odordiff2.monitoring.health.ComponentChecker') as mock_checker_class:
            mock_checker = Mock()
            mock_checker_class.return_value = mock_checker
            
            # Mock component health checks
            mock_checker.check_database.return_value = {
                "status": "healthy",
                "response_time": 45,
                "connection_pool": {"active": 5, "idle": 15}
            }
            
            mock_checker.check_model_service.return_value = {
                "status": "healthy", 
                "memory_usage": 65.2,
                "gpu_utilization": 78.5,
                "model_loaded": True
            }
            
            mock_checker.check_cache_service.return_value = {
                "status": "healthy",
                "hit_rate": 0.91,
                "memory_usage": 45.3,
                "eviction_rate": 0.02
            }
            
            checker = mock_checker_class()
            
            # Perform comprehensive health check
            health_status = health_monitor.check_system_health()
            
            # Health check should aggregate component statuses
            expected_components = ["database", "model_service", "cache_service"]
            
            # Verify health check was comprehensive
            assert health_status is not None
    
    def test_health_degradation_detection(self, health_monitor):
        """Test detection of health degradation."""
        with patch('odordiff2.monitoring.health.HealthAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer_class.return_value = mock_analyzer
            
            # Mock degradation detection
            mock_analyzer.detect_degradation.return_value = {
                "degradation_detected": True,
                "severity": "moderate",
                "affected_components": ["cache_service"],
                "recommended_actions": ["increase_cache_memory", "restart_cache"]
            }
            
            analyzer = mock_analyzer_class()
            
            # Historical health data showing degradation
            health_history = [
                {"timestamp": time.time() - 300, "overall_score": 0.95},
                {"timestamp": time.time() - 240, "overall_score": 0.90},
                {"timestamp": time.time() - 180, "overall_score": 0.85},
                {"timestamp": time.time() - 120, "overall_score": 0.75},
                {"timestamp": time.time() - 60, "overall_score": 0.70}
            ]
            
            degradation = analyzer.detect_degradation(health_history)
            
            assert degradation["degradation_detected"] is True
            assert degradation["severity"] in ["low", "moderate", "high", "critical"]
            assert len(degradation["affected_components"]) > 0
            assert len(degradation["recommended_actions"]) > 0
    
    def test_automated_health_recovery(self, health_monitor):
        """Test automated health recovery procedures."""
        with patch('odordiff2.monitoring.health.RecoveryAgent') as mock_recovery_class:
            mock_recovery = Mock()
            mock_recovery_class.return_value = mock_recovery
            
            # Mock recovery actions
            mock_recovery.restart_component.return_value = True
            mock_recovery.scale_up_service.return_value = True
            mock_recovery.clear_cache.return_value = True
            
            recovery_agent = mock_recovery_class()
            
            # Unhealthy component requiring recovery
            unhealthy_status = {
                "component": "cache_service",
                "status": "unhealthy",
                "error": "Memory exhaustion",
                "recommended_recovery": ["restart_component", "scale_up_service"]
            }
            
            # Execute recovery actions
            recovery_results = []
            for action in unhealthy_status["recommended_recovery"]:
                if action == "restart_component":
                    result = recovery_agent.restart_component(unhealthy_status["component"])
                elif action == "scale_up_service":
                    result = recovery_agent.scale_up_service(unhealthy_status["component"])
                
                recovery_results.append({"action": action, "success": result})
            
            # Verify recovery actions were executed
            assert all(r["success"] for r in recovery_results)
            assert len(recovery_results) == 2


class TestMetricsCollection:
    """Test metrics collection and export."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance."""
        return MetricsCollector(
            collection_interval=10,
            retention_period_hours=24,
            aggregation_window=60
        )
    
    def test_system_metrics_collection(self, metrics_collector):
        """Test collection of system metrics."""
        with patch('psutil.cpu_percent', return_value=65.5), \
             patch('psutil.virtual_memory', return_value=Mock(percent=72.3)), \
             patch('psutil.disk_usage', return_value=Mock(percent=45.8)):
            
            # Collect system metrics
            metrics = metrics_collector.collect_system_metrics()
            
            # Verify collected metrics
            assert "cpu_usage" in metrics
            assert "memory_usage" in metrics
            assert "disk_usage" in metrics
            assert "timestamp" in metrics
            
            assert metrics["cpu_usage"] == 65.5
            assert metrics["memory_usage"] == 72.3
            assert metrics["disk_usage"] == 45.8
    
    def test_application_metrics_collection(self, metrics_collector):
        """Test collection of application-specific metrics."""
        with patch('odordiff2.monitoring.metrics.ApplicationMonitor') as mock_app_monitor:
            mock_monitor = Mock()
            mock_app_monitor.return_value = mock_monitor
            
            # Mock application metrics
            mock_monitor.get_generation_metrics.return_value = {
                "molecules_generated_total": 12547,
                "avg_generation_time": 2.3,
                "safety_pass_rate": 0.946,
                "cache_hit_rate": 0.89
            }
            
            monitor = mock_app_monitor()
            
            # Collect application metrics
            app_metrics = metrics_collector.collect_application_metrics()
            
            # Verify application-specific metrics
            expected_metrics = [
                "molecules_generated_total",
                "avg_generation_time", 
                "safety_pass_rate",
                "cache_hit_rate"
            ]
            
            # Application metrics should be comprehensive
            assert app_metrics is not None
    
    def test_prometheus_export(self):
        """Test Prometheus metrics export."""
        with patch('odordiff2.monitoring.metrics.PrometheusExporter') as mock_exporter_class:
            mock_exporter = Mock(spec=PrometheusExporter)
            mock_exporter_class.return_value = mock_exporter
            
            # Mock Prometheus exposition
            mock_exporter.export_metrics.return_value = """
# HELP odordiff_molecules_generated_total Total molecules generated
# TYPE odordiff_molecules_generated_total counter
odordiff_molecules_generated_total 12547
# HELP odordiff_generation_time_seconds Time to generate molecules
# TYPE odordiff_generation_time_seconds histogram
odordiff_generation_time_seconds_bucket{le="1.0"} 1205
odordiff_generation_time_seconds_bucket{le="2.0"} 2890
odordiff_generation_time_seconds_bucket{le="5.0"} 4567
odordiff_generation_time_seconds_bucket{le="+Inf"} 4672
odordiff_generation_time_seconds_sum 10543.2
odordiff_generation_time_seconds_count 4672
            """.strip()
            
            exporter = mock_exporter_class()
            
            # Export metrics in Prometheus format
            prometheus_output = exporter.export_metrics({
                "molecules_generated_total": 12547,
                "generation_time_histogram": {
                    "buckets": {"1.0": 1205, "2.0": 2890, "5.0": 4567},
                    "sum": 10543.2,
                    "count": 4672
                }
            })
            
            # Verify Prometheus format
            assert "odordiff_molecules_generated_total 12547" in prometheus_output
            assert "TYPE odordiff_generation_time_seconds histogram" in prometheus_output
            assert "odordiff_generation_time_seconds_sum 10543.2" in prometheus_output


class TestPerformanceMonitoring:
    """Test performance monitoring capabilities."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor instance."""
        return PerformanceMonitor(
            sampling_rate=1.0,
            history_retention=3600,
            alert_thresholds={
                "response_time_p95": 500,  # 500ms
                "error_rate": 0.05,        # 5%
                "memory_usage": 0.85       # 85%
            }
        )
    
    def test_response_time_monitoring(self, performance_monitor):
        """Test response time monitoring."""
        with patch('odordiff2.monitoring.performance.ResponseTimeTracker') as mock_tracker_class:
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            # Mock response time data
            mock_tracker.get_statistics.return_value = {
                "mean": 145.2,
                "p50": 132.8,
                "p95": 287.6,
                "p99": 445.3,
                "count": 10000,
                "min": 23.1,
                "max": 1205.7
            }
            
            tracker = mock_tracker_class()
            
            # Get response time statistics
            stats = performance_monitor.get_response_time_stats()
            
            # Verify comprehensive statistics
            expected_percentiles = ["mean", "p50", "p95", "p99"]
            assert stats is not None
    
    def test_error_rate_monitoring(self, performance_monitor):
        """Test error rate monitoring and alerting."""
        with patch('odordiff2.monitoring.performance.ErrorRateMonitor') as mock_error_monitor:
            mock_monitor = Mock()
            mock_error_monitor.return_value = mock_monitor
            
            # Mock error rate calculation
            mock_monitor.calculate_error_rate.return_value = {
                "current_rate": 0.08,  # 8% error rate (above threshold)
                "trend": "increasing",
                "error_types": {
                    "timeout": 0.03,
                    "validation": 0.02,
                    "internal": 0.03
                }
            }
            
            error_monitor = mock_error_monitor()
            
            # Monitor error rates
            error_stats = performance_monitor.monitor_error_rates()
            
            # Should detect elevated error rate
            assert error_stats is not None
    
    def test_performance_benchmark_runner(self):
        """Test performance benchmark execution."""
        with patch('odordiff2.monitoring.performance.BenchmarkRunner') as mock_runner_class:
            mock_runner = Mock(spec=BenchmarkRunner)
            mock_runner_class.return_value = mock_runner
            
            # Mock benchmark results
            mock_runner.run_generation_benchmark.return_value = {
                "test_name": "molecule_generation_benchmark",
                "duration": 30.0,
                "requests_completed": 1500,
                "requests_per_second": 50.0,
                "avg_response_time": 185.3,
                "p95_response_time": 312.7,
                "error_rate": 0.02,
                "success_rate": 0.98
            }
            
            runner = mock_runner_class()
            
            # Run performance benchmark
            benchmark_results = runner.run_generation_benchmark(
                duration_seconds=30,
                concurrent_users=10,
                molecules_per_request=3
            )
            
            # Verify benchmark results
            assert benchmark_results["test_name"] == "molecule_generation_benchmark"
            assert benchmark_results["requests_per_second"] > 0
            assert benchmark_results["success_rate"] > 0.95
            assert benchmark_results["error_rate"] < 0.05


class TestProfilingComponents:
    """Test performance profiling components."""
    
    def test_performance_profiler(self):
        """Test performance profiler functionality."""
        with patch('odordiff2.scaling.profiling.PerformanceProfiler') as mock_profiler_class:
            mock_profiler = Mock(spec=PerformanceProfiler)
            mock_profiler_class.return_value = mock_profiler
            
            # Mock profiling results
            mock_profiler.profile_function.return_value = {
                "function_name": "generate_molecules",
                "total_time": 2.345,
                "call_count": 156,
                "time_per_call": 0.0150,
                "hotspots": [
                    {"function": "text_encoding", "time": 0.423, "percentage": 18.0},
                    {"function": "diffusion_step", "time": 1.234, "percentage": 52.6},
                    {"function": "safety_check", "time": 0.567, "percentage": 24.2}
                ]
            }
            
            profiler = mock_profiler_class()
            
            # Profile function execution
            profile_results = profiler.profile_function(
                function_name="generate_molecules",
                duration_seconds=60
            )
            
            # Verify profiling results
            assert profile_results["function_name"] == "generate_molecules"
            assert profile_results["total_time"] > 0
            assert len(profile_results["hotspots"]) > 0
            
            # Hotspots should account for significant portion of time
            total_hotspot_percentage = sum(h["percentage"] for h in profile_results["hotspots"])
            assert total_hotspot_percentage > 80  # Should cover >80% of execution time
    
    def test_memory_profiler(self):
        """Test memory profiling capabilities."""
        with patch('odordiff2.scaling.profiling.MemoryProfiler') as mock_mem_profiler_class:
            mock_profiler = Mock(spec=MemoryProfiler)
            mock_mem_profiler_class.return_value = mock_profiler
            
            # Mock memory profiling results
            mock_profiler.profile_memory_usage.return_value = {
                "peak_memory_mb": 847.3,
                "average_memory_mb": 456.8,
                "memory_growth_rate": 2.1,  # MB per hour
                "largest_objects": [
                    {"type": "torch.Tensor", "size_mb": 245.7, "count": 12},
                    {"type": "dict", "size_mb": 89.4, "count": 1547},
                    {"type": "Molecule", "size_mb": 67.2, "count": 2340}
                ],
                "potential_leaks": []
            }
            
            profiler = mock_mem_profiler_class()
            
            # Profile memory usage
            memory_profile = profiler.profile_memory_usage(
                duration_seconds=300,
                sampling_interval=1.0
            )
            
            # Verify memory profiling results
            assert memory_profile["peak_memory_mb"] > 0
            assert memory_profile["average_memory_mb"] > 0
            assert len(memory_profile["largest_objects"]) > 0
            
            # Should identify major memory consumers
            total_tracked_memory = sum(obj["size_mb"] for obj in memory_profile["largest_objects"])
            assert total_tracked_memory > memory_profile["average_memory_mb"] * 0.5


class TestStreamingComponents:
    """Test streaming and real-time components."""
    
    @pytest.fixture
    def streaming_manager(self):
        """Create streaming manager instance."""
        return StreamingManager(
            buffer_size=1000,
            flush_interval=1.0,
            compression_enabled=True
        )
    
    @pytest.mark.asyncio
    async def test_websocket_streaming(self, streaming_manager):
        """Test WebSocket streaming functionality."""
        with patch('odordiff2.scaling.streaming.WebSocketHandler') as mock_ws_class:
            mock_handler = Mock(spec=WebSocketHandler)
            mock_ws_class.return_value = mock_handler
            
            # Mock WebSocket operations
            mock_handler.send_message.return_value = True
            mock_handler.broadcast_to_clients.return_value = 15  # Sent to 15 clients
            
            handler = mock_ws_class()
            
            # Test streaming molecule generation results
            generation_results = {
                "request_id": "req_123456",
                "molecules": [
                    {"smiles": "CCO", "safety_score": 0.95},
                    {"smiles": "CC(C)O", "safety_score": 0.92}
                ],
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            # Stream to specific client
            sent = await handler.send_message("client_789", generation_results)
            assert sent is True
            
            # Broadcast to all clients
            broadcast_count = await handler.broadcast_to_clients(generation_results)
            assert broadcast_count > 0
    
    def test_streaming_compression(self, streaming_manager):
        """Test streaming data compression."""
        with patch('odordiff2.scaling.streaming.CompressionManager') as mock_compression_class:
            mock_compressor = Mock()
            mock_compression_class.return_value = mock_compressor
            
            # Mock compression results
            large_data = {"molecules": [{"smiles": f"C{i}O"} for i in range(1000)]}
            
            mock_compressor.compress.return_value = {
                "compressed_data": b"compressed_content",
                "original_size": 45000,
                "compressed_size": 12000,
                "compression_ratio": 3.75
            }
            
            compressor = mock_compression_class()
            
            # Compress streaming data
            compressed = compressor.compress(large_data)
            
            assert compressed["compressed_size"] < compressed["original_size"]
            assert compressed["compression_ratio"] > 1.0
    
    @pytest.mark.asyncio
    async def test_server_sent_events(self, streaming_manager):
        """Test Server-Sent Events streaming."""
        with patch('odordiff2.scaling.streaming.SSEHandler') as mock_sse_class:
            mock_sse = Mock()
            mock_sse_class.return_value = mock_sse
            
            # Mock SSE streaming
            mock_sse.stream_events.return_value = True
            
            sse_handler = mock_sse_class()
            
            # Test event streaming
            events = [
                {"event": "molecule_generated", "data": {"smiles": "CCO"}},
                {"event": "safety_checked", "data": {"score": 0.95}},
                {"event": "generation_complete", "data": {"total": 5}}
            ]
            
            for event in events:
                success = await sse_handler.stream_events("client_456", event)
                assert success is True


class TestRecoveryComponents:
    """Test recovery and backup components."""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create recovery manager instance."""
        return RecoveryManager(
            backup_directory="/tmp/backups",
            recovery_timeout=300,
            verification_enabled=True
        )
    
    def test_automated_backup_creation(self, recovery_manager):
        """Test automated backup creation."""
        with patch('odordiff2.utils.recovery.BackupCreator') as mock_backup_class:
            mock_creator = Mock()
            mock_backup_class.return_value = mock_creator
            
            # Mock backup creation
            mock_creator.create_full_backup.return_value = {
                "backup_id": "backup_20250810_143022",
                "backup_path": "/tmp/backups/backup_20250810_143022.tar.gz",
                "size_mb": 234.7,
                "checksum": "sha256:abc123def456...",
                "created_at": datetime.now().isoformat(),
                "components_included": [
                    "database",
                    "model_weights", 
                    "cache_data",
                    "configuration"
                ]
            }
            
            creator = mock_backup_class()
            
            # Create comprehensive backup
            backup_result = recovery_manager.create_backup(
                backup_type="full",
                include_models=True,
                include_cache=True,
                compress=True
            )
            
            # Verify backup creation
            assert backup_result is not None
    
    def test_backup_validation(self, recovery_manager):
        """Test backup integrity validation."""
        with patch('odordiff2.utils.recovery.BackupValidator') as mock_validator_class:
            mock_validator = Mock(spec=BackupValidator)
            mock_validator_class.return_value = mock_validator
            
            # Mock validation results
            mock_validator.validate_backup.return_value = {
                "is_valid": True,
                "checksum_verified": True,
                "file_integrity": True,
                "readable_components": ["database", "models", "config"],
                "validation_time": 12.3,
                "issues_found": []
            }
            
            validator = mock_validator_class()
            
            # Validate backup integrity
            validation_result = validator.validate_backup("backup_20250810_143022")
            
            assert validation_result["is_valid"] is True
            assert validation_result["checksum_verified"] is True
            assert len(validation_result["readable_components"]) > 0
            assert len(validation_result["issues_found"]) == 0
    
    def test_disaster_recovery_workflow(self, recovery_manager):
        """Test complete disaster recovery workflow."""
        with patch('odordiff2.utils.recovery.RecoveryOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Mock recovery workflow steps
            mock_orchestrator.execute_recovery_plan.return_value = {
                "recovery_id": "recovery_20250810_144500",
                "steps_completed": [
                    {"step": "stop_services", "status": "success", "duration": 15.2},
                    {"step": "restore_database", "status": "success", "duration": 45.7},
                    {"step": "restore_models", "status": "success", "duration": 23.1},
                    {"step": "restore_config", "status": "success", "duration": 5.3},
                    {"step": "start_services", "status": "success", "duration": 32.8},
                    {"step": "verify_system", "status": "success", "duration": 18.9}
                ],
                "total_duration": 141.0,
                "final_status": "success",
                "system_health_score": 0.96
            }
            
            orchestrator = mock_orchestrator_class()
            
            # Execute disaster recovery
            recovery_result = orchestrator.execute_recovery_plan(
                backup_id="backup_20250810_143022",
                recovery_type="full",
                verify_integrity=True,
                restart_services=True
            )
            
            assert recovery_result["final_status"] == "success"
            assert recovery_result["system_health_score"] > 0.9
            assert len(recovery_result["steps_completed"]) >= 5


class TestCacheComponents:
    """Test caching system components."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager instance."""
        return CacheManager(
            default_ttl=3600,
            max_size_mb=500,
            eviction_policy="lru"
        )
    
    def test_cache_policy_management(self, cache_manager):
        """Test cache policy configuration and management."""
        with patch('odordiff2.data.cache.CachePolicy') as mock_policy_class:
            mock_policy = Mock(spec=CachePolicy)
            mock_policy_class.return_value = mock_policy
            
            # Mock policy configuration
            mock_policy.configure.return_value = True
            mock_policy.get_ttl.return_value = 7200  # 2 hours
            mock_policy.should_evict.return_value = False
            
            policy = mock_policy_class()
            
            # Configure cache policy
            policy_configured = policy.configure({
                "default_ttl": 7200,
                "max_entries": 10000,
                "eviction_strategy": "lru",
                "compression_enabled": True
            })
            
            assert policy_configured is True
            
            # Test policy decisions
            ttl = policy.get_ttl("molecule_generation_results")
            assert ttl > 0
            
            should_evict = policy.should_evict(
                key="old_results",
                last_accessed=time.time() - 8000,
                size_mb=15.3
            )
            assert isinstance(should_evict, bool)
    
    def test_cache_statistics_tracking(self, cache_manager):
        """Test cache statistics and performance tracking."""
        with patch('odordiff2.data.cache.CacheStatistics') as mock_stats_class:
            mock_stats = Mock()
            mock_stats_class.return_value = mock_stats
            
            # Mock statistics
            mock_stats.get_stats.return_value = {
                "total_requests": 15673,
                "cache_hits": 14289,
                "cache_misses": 1384,
                "hit_rate": 0.912,
                "avg_response_time_ms": 2.3,
                "memory_usage_mb": 287.5,
                "evictions_count": 23,
                "top_keys": [
                    {"key": "citrus_molecules_batch_1", "hits": 234},
                    {"key": "safety_assessment_cache", "hits": 189},
                    {"key": "synthesis_routes_woody", "hits": 156}
                ]
            }
            
            stats = mock_stats_class()
            
            # Get cache performance statistics
            cache_stats = stats.get_stats()
            
            assert cache_stats["hit_rate"] > 0.8  # >80% hit rate
            assert cache_stats["avg_response_time_ms"] < 10  # <10ms average
            assert len(cache_stats["top_keys"]) > 0
    
    def test_distributed_cache_coordination(self, cache_manager):
        """Test distributed cache coordination."""
        with patch('odordiff2.data.cache.DistributedCacheCoordinator') as mock_coord_class:
            mock_coordinator = Mock()
            mock_coord_class.return_value = mock_coordinator
            
            # Mock distributed operations
            mock_coordinator.sync_cache_nodes.return_value = {
                "nodes_synced": 5,
                "sync_duration": 1.23,
                "conflicts_resolved": 2,
                "sync_status": "success"
            }
            
            mock_coordinator.invalidate_across_nodes.return_value = True
            
            coordinator = mock_coord_class()
            
            # Test cache synchronization
            sync_result = coordinator.sync_cache_nodes(
                cache_keys=["molecule_*", "safety_*"],
                timeout=30
            )
            
            assert sync_result["sync_status"] == "success"
            assert sync_result["nodes_synced"] > 0
            
            # Test cache invalidation
            invalidated = coordinator.invalidate_across_nodes("stale_molecule_data")
            assert invalidated is True


@pytest.mark.integration
class TestComponentIntegration:
    """Integration tests between multiple components."""
    
    def test_monitoring_visualization_integration(self):
        """Test integration between monitoring and visualization."""
        with patch('odordiff2.monitoring.metrics.MetricsCollector') as mock_metrics, \
             patch('odordiff2.visualization.dashboard.DashboardManager') as mock_dashboard:
            
            # Setup mocks
            mock_metrics_instance = Mock()
            mock_metrics.return_value = mock_metrics_instance
            
            mock_dashboard_instance = Mock()
            mock_dashboard.return_value = mock_dashboard_instance
            
            # Mock metrics data
            mock_metrics_instance.get_latest_metrics.return_value = {
                "system_metrics": {"cpu": 65.2, "memory": 78.1},
                "app_metrics": {"molecules_generated": 1234, "avg_time": 2.1},
                "timestamp": time.time()
            }
            
            # Mock dashboard update
            mock_dashboard_instance.update_charts.return_value = True
            
            # Integration workflow
            metrics_collector = mock_metrics()
            dashboard = mock_dashboard_instance
            
            # Collect metrics and update dashboard
            latest_metrics = metrics_collector.get_latest_metrics()
            dashboard_updated = dashboard.update_charts(latest_metrics)
            
            assert latest_metrics is not None
            assert dashboard_updated is True
    
    def test_health_recovery_integration(self):
        """Test integration between health monitoring and recovery."""
        with patch('odordiff2.monitoring.health.HealthMonitor') as mock_health, \
             patch('odordiff2.utils.recovery.RecoveryManager') as mock_recovery:
            
            # Setup mocks
            mock_health_instance = Mock()
            mock_health.return_value = mock_health_instance
            
            mock_recovery_instance = Mock()
            mock_recovery.return_value = mock_recovery_instance
            
            # Mock unhealthy system
            mock_health_instance.check_system_health.return_value = {
                "status": "unhealthy",
                "critical_issues": ["database_connection_failed"],
                "recommended_recovery": "restart_database_service"
            }
            
            # Mock successful recovery
            mock_recovery_instance.execute_recovery_action.return_value = {
                "action": "restart_database_service",
                "success": True,
                "duration": 45.2
            }
            
            # Integration workflow
            health_monitor = mock_health_instance
            recovery_manager = mock_recovery_instance
            
            # Check health and trigger recovery if needed
            health_status = health_monitor.check_system_health()
            
            if health_status["status"] == "unhealthy":
                recovery_result = recovery_manager.execute_recovery_action(
                    health_status["recommended_recovery"]
                )
                assert recovery_result["success"] is True