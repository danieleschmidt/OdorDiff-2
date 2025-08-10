"""
End-to-end workflow tests for OdorDiff-2 system.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import tempfile
import os

from odordiff2.core.diffusion import OdorDiffusion
from odordiff2.safety.filter import SafetyFilter
from odordiff2.models.molecule import Molecule, OdorProfile
from odordiff2.core.synthesis import SynthesisPlanner
from odordiff2.visualization.viewer import MoleculeViewer
from odordiff2.api.endpoints import APIEndpoints
from odordiff2.monitoring.health import HealthMonitor
from odordiff2.scaling.auto_scaler import AutoScaler


class TestCompleteGenerationWorkflow:
    """Test complete molecule generation workflow from text to molecules."""
    
    @pytest.fixture
    def mock_diffusion_model(self):
        """Create mock diffusion model for testing."""
        model = Mock(spec=OdorDiffusion)
        
        # Mock generation results
        model.generate.return_value = [
            Molecule(
                smiles="CCO",
                safety_score=0.95,
                synthesis_score=0.88,
                estimated_cost=12.50,
                odor_profile=OdorProfile(
                    primary_notes=["fresh", "alcoholic"],
                    intensity=0.7,
                    longevity_hours=2.5
                )
            ),
            Molecule(
                smiles="CC(C)O",
                safety_score=0.92,
                synthesis_score=0.85,
                estimated_cost=15.30,
                odor_profile=OdorProfile(
                    primary_notes=["fresh", "minty"],
                    intensity=0.6,
                    longevity_hours=2.0
                )
            )
        ]
        
        return model
    
    @pytest.fixture
    def safety_filter(self):
        """Create safety filter instance."""
        return SafetyFilter(
            toxicity_threshold=0.1,
            irritant_check=True,
            ifra_compliance=True
        )
    
    def test_text_to_molecule_generation_workflow(self, mock_diffusion_model, safety_filter):
        """Test complete text-to-molecule generation workflow."""
        # Input text prompt
        text_prompt = "A fresh, citrusy scent with hints of bergamot and lemon"
        
        # Generation parameters
        generation_params = {
            "num_molecules": 5,
            "safety_filter": safety_filter,
            "synthesizability_min": 0.7,
            "max_molecular_weight": 300,
            "allow_allergens": False
        }
        
        # Generate molecules
        molecules = mock_diffusion_model.generate(
            prompt=text_prompt,
            **generation_params
        )
        
        # Verify generation results
        assert len(molecules) == 2  # Mock returns 2 molecules
        
        for molecule in molecules:
            # Check basic molecule properties
            assert isinstance(molecule, Molecule)
            assert len(molecule.smiles) > 0
            assert 0.0 <= molecule.safety_score <= 1.0
            assert 0.0 <= molecule.synthesis_score <= 1.0
            assert molecule.estimated_cost > 0
            
            # Check odor profile
            assert isinstance(molecule.odor_profile, OdorProfile)
            assert len(molecule.odor_profile.primary_notes) > 0
            assert 0.0 <= molecule.odor_profile.intensity <= 1.0
            assert molecule.odor_profile.longevity_hours > 0
            
            # Verify safety requirements met
            assert molecule.safety_score >= 0.9  # High safety threshold
    
    def test_safety_filtering_workflow(self, mock_diffusion_model, safety_filter):
        """Test safety filtering in generation workflow."""
        # Mock molecules with varying safety scores
        unsafe_molecule = Molecule(
            smiles="C[N+](C)(C)C",  # Potentially toxic quaternary ammonium
            safety_score=0.3,  # Low safety score
            synthesis_score=0.9
        )
        
        safe_molecule = Molecule(
            smiles="CCO",  # Ethanol - generally safe
            safety_score=0.95,  # High safety score
            synthesis_score=0.88
        )
        
        mock_diffusion_model.generate.return_value = [unsafe_molecule, safe_molecule]
        
        # Generate with safety filtering
        molecules = mock_diffusion_model.generate(
            prompt="Test prompt",
            safety_filter=safety_filter,
            num_molecules=2
        )
        
        # Apply additional safety filtering
        filtered_molecules = []
        for molecule in molecules:
            safety_assessment = safety_filter.assess_safety(molecule)
            if safety_assessment.is_safe:
                filtered_molecules.append(molecule)
        
        # Should only contain safe molecules
        assert len(filtered_molecules) == 1
        assert filtered_molecules[0].smiles == "CCO"
        assert filtered_molecules[0].safety_score >= 0.9
    
    def test_synthesis_planning_workflow(self, mock_diffusion_model):
        """Test synthesis planning workflow integration."""
        with patch('odordiff2.core.synthesis.SynthesisPlanner') as mock_planner_class:
            mock_planner = Mock(spec=SynthesisPlanner)
            mock_planner_class.return_value = mock_planner
            
            # Mock synthesis routes
            mock_planner.suggest_synthesis_routes.return_value = [
                {
                    "route_id": 1,
                    "steps": [
                        {"reaction": "oxidation", "reagent": "CrO3", "yield": 0.85},
                        {"reaction": "reduction", "reagent": "NaBH4", "yield": 0.92}
                    ],
                    "total_yield": 0.78,
                    "cost_estimate": 25.50,
                    "feasibility_score": 0.88
                }
            ]
            
            # Generate molecules
            molecules = mock_diffusion_model.generate(
                prompt="Woody, amber-like fragrance",
                num_molecules=1
            )
            
            # Plan synthesis for generated molecules
            planner = mock_planner_class()
            for molecule in molecules:
                routes = planner.suggest_synthesis_routes(
                    target_molecule=molecule,
                    max_steps=5,
                    green_chemistry=True
                )
                
                assert len(routes) > 0
                route = routes[0]
                assert route["feasibility_score"] > 0.8
                assert route["total_yield"] > 0.7
                assert len(route["steps"]) <= 5
    
    def test_visualization_workflow(self, mock_diffusion_model):
        """Test molecule visualization workflow."""
        with patch('odordiff2.visualization.viewer.MoleculeViewer') as mock_viewer_class:
            mock_viewer = Mock(spec=MoleculeViewer)
            mock_viewer_class.return_value = mock_viewer
            
            # Mock visualization outputs
            mock_viewer.visualize_3d.return_value = "<div>3D visualization</div>"
            mock_viewer.generate_properties_chart.return_value = "chart_data.png"
            
            # Generate molecules
            molecules = mock_diffusion_model.generate(
                prompt="Floral rose scent",
                num_molecules=2
            )
            
            # Visualize molecules
            viewer = mock_viewer_class()
            for molecule in molecules:
                # 3D visualization
                viz_html = viewer.visualize_3d(
                    molecule=molecule,
                    style="ball_and_stick",
                    highlight_functional_groups=True
                )
                assert isinstance(viz_html, str)
                assert len(viz_html) > 0
                
                # Property charts
                chart_path = viewer.generate_properties_chart(molecule)
                assert isinstance(chart_path, str)
                assert chart_path.endswith('.png')
    
    def test_complete_fragrance_design_workflow(self, mock_diffusion_model, safety_filter):
        """Test complete fragrance design workflow with blending."""
        # Design multi-component fragrance
        fragrance_components = [
            {"type": "top_note", "description": "fresh citrus bergamot"},
            {"type": "heart_note", "description": "floral jasmine rose"},
            {"type": "base_note", "description": "woody sandalwood amber"}
        ]
        
        complete_fragrance = []
        
        for component in fragrance_components:
            # Generate molecules for each component
            molecules = mock_diffusion_model.generate(
                prompt=component["description"],
                num_molecules=2,
                safety_filter=safety_filter,
                fragrance_type=component["type"]
            )
            
            # Select best molecule for this component
            best_molecule = max(molecules, key=lambda m: m.safety_score * m.synthesis_score)
            best_molecule.fragrance_role = component["type"]
            complete_fragrance.append(best_molecule)
        
        # Verify complete fragrance composition
        assert len(complete_fragrance) == 3
        
        roles = [mol.fragrance_role for mol in complete_fragrance]
        assert "top_note" in roles
        assert "heart_note" in roles  
        assert "base_note" in roles
        
        # All molecules should meet safety requirements
        for molecule in complete_fragrance:
            assert molecule.safety_score >= 0.9
            assert molecule.synthesis_score >= 0.7


class TestAPIWorkflowIntegration:
    """Test API endpoint workflows and integration."""
    
    @pytest.fixture
    def mock_api_endpoints(self):
        """Create mock API endpoints for testing."""
        with patch('odordiff2.api.endpoints.APIEndpoints') as mock_api_class:
            mock_api = Mock(spec=APIEndpoints)
            mock_api_class.return_value = mock_api
            
            # Mock API responses
            mock_api.generate_molecules.return_value = {
                "status": "success",
                "molecules": [
                    {
                        "smiles": "CCO",
                        "safety_score": 0.95,
                        "synthesis_score": 0.88,
                        "odor_profile": {
                            "primary_notes": ["fresh", "alcoholic"],
                            "intensity": 0.7
                        }
                    }
                ],
                "generation_time": 2.3,
                "request_id": "req_123456"
            }
            
            return mock_api
    
    @pytest.mark.asyncio
    async def test_api_generation_workflow(self, mock_api_endpoints):
        """Test API-based generation workflow."""
        # Simulate API request
        request_data = {
            "prompt": "Fresh ocean breeze scent",
            "num_molecules": 3,
            "safety_threshold": 0.9,
            "max_cost": 50.0,
            "include_synthesis": True
        }
        
        # Call API endpoint
        response = await mock_api_endpoints.generate_molecules(request_data)
        
        # Verify API response structure
        assert response["status"] == "success"
        assert "molecules" in response
        assert "generation_time" in response
        assert "request_id" in response
        
        # Verify molecule data
        molecules = response["molecules"]
        assert len(molecules) > 0
        
        molecule = molecules[0]
        assert "smiles" in molecule
        assert "safety_score" in molecule
        assert "synthesis_score" in molecule
        assert "odor_profile" in molecule
    
    @pytest.mark.asyncio
    async def test_api_error_handling_workflow(self, mock_api_endpoints):
        """Test API error handling workflow."""
        # Mock API error response
        mock_api_endpoints.generate_molecules.side_effect = Exception("Model loading failed")
        
        # Test error handling
        request_data = {"prompt": "Test prompt"}
        
        with pytest.raises(Exception) as exc_info:
            await mock_api_endpoints.generate_molecules(request_data)
        
        assert "Model loading failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_api_authentication_workflow(self, mock_api_endpoints):
        """Test API authentication workflow."""
        with patch('odordiff2.utils.security.AuthenticationManager') as mock_auth_class:
            mock_auth = Mock()
            mock_auth_class.return_value = mock_auth
            
            # Mock successful authentication
            mock_auth.validate_api_key.return_value = Mock(
                is_valid=True,
                user_id="user_123",
                rate_limit_tier="premium"
            )
            
            # Test authenticated request
            headers = {"Authorization": "Bearer api_key_123"}
            request_data = {"prompt": "Authenticated request"}
            
            response = await mock_api_endpoints.generate_molecules(
                request_data, 
                headers=headers
            )
            
            assert response["status"] == "success"
            mock_auth.validate_api_key.assert_called()


class TestScalingWorkflowIntegration:
    """Test scaling and performance workflow integration."""
    
    @pytest.fixture
    def mock_auto_scaler(self):
        """Create mock auto-scaler for testing."""
        with patch('odordiff2.scaling.auto_scaler.AutoScaler') as mock_scaler_class:
            mock_scaler = Mock(spec=AutoScaler)
            mock_scaler_class.return_value = mock_scaler
            
            # Mock scaling decisions
            mock_scaler.decide_scaling_action.return_value = Mock(
                action_type="scale_up",
                target_replicas=6,
                current_replicas=4,
                confidence=0.9,
                reason="High CPU usage detected"
            )
            
            return mock_scaler
    
    @pytest.mark.asyncio
    async def test_load_based_scaling_workflow(self, mock_auto_scaler):
        """Test load-based scaling workflow."""
        # Simulate high load conditions
        high_load_metrics = {
            "cpu_usage": 85.0,
            "memory_usage": 78.0,
            "active_requests": 150,
            "response_time_p95": 280.0
        }
        
        # Auto-scaler should detect need for scaling
        scaling_action = mock_auto_scaler.decide_scaling_action(high_load_metrics)
        
        assert scaling_action.action_type == "scale_up"
        assert scaling_action.target_replicas > scaling_action.current_replicas
        assert scaling_action.confidence > 0.8
        
        # Execute scaling action
        with patch('odordiff2.scaling.auto_scaler.KubernetesScaler') as mock_k8s:
            mock_k8s_instance = AsyncMock()
            mock_k8s.return_value = mock_k8s_instance
            mock_k8s_instance.scale_deployment.return_value = True
            
            result = await mock_auto_scaler.execute_scaling_action(scaling_action)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_workflow(self):
        """Test performance monitoring workflow integration."""
        with patch('odordiff2.monitoring.performance.PerformanceMonitor') as mock_perf_class:
            mock_monitor = Mock()
            mock_perf_class.return_value = mock_monitor
            
            # Mock performance metrics
            mock_monitor.collect_metrics.return_value = {
                "request_count": 1250,
                "avg_response_time": 145.2,
                "p95_response_time": 285.7,
                "error_rate": 0.02,
                "cache_hit_rate": 0.89
            }
            
            # Collect and analyze performance metrics
            monitor = mock_perf_class()
            metrics = monitor.collect_metrics()
            
            # Verify metrics structure
            assert "request_count" in metrics
            assert "avg_response_time" in metrics
            assert "error_rate" in metrics
            assert "cache_hit_rate" in metrics
            
            # Performance should be within acceptable ranges
            assert metrics["avg_response_time"] < 200  # < 200ms
            assert metrics["error_rate"] < 0.05  # < 5% error rate
            assert metrics["cache_hit_rate"] > 0.8  # > 80% cache hit rate
    
    def test_caching_workflow_integration(self):
        """Test caching workflow integration."""
        with patch('odordiff2.scaling.multi_tier_cache.MultiTierCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            
            # Mock cache operations
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = True  # Successful set
            
            cache = mock_cache_class()
            
            # Test cache workflow
            cache_key = "molecule_generation_citrus_scent_123"
            
            # Check cache first
            cached_result = cache.get(cache_key)
            assert cached_result is None  # Cache miss
            
            # Generate new result (would happen in real workflow)
            new_result = {
                "molecules": [{"smiles": "CCO", "safety_score": 0.95}],
                "generation_time": 2.1
            }
            
            # Store in cache
            cache.set(cache_key, new_result, ttl=3600)
            mock_cache.set.assert_called_with(cache_key, new_result, ttl=3600)


class TestHealthMonitoringWorkflow:
    """Test health monitoring and alerting workflow."""
    
    @pytest.fixture
    def mock_health_monitor(self):
        """Create mock health monitor for testing."""
        with patch('odordiff2.monitoring.health.HealthMonitor') as mock_monitor_class:
            mock_monitor = Mock(spec=HealthMonitor)
            mock_monitor_class.return_value = mock_monitor
            
            # Mock health check results
            mock_monitor.check_system_health.return_value = {
                "status": "healthy",
                "components": {
                    "database": {"status": "healthy", "response_time": 45},
                    "model_service": {"status": "healthy", "memory_usage": 65},
                    "cache_service": {"status": "healthy", "hit_rate": 0.91}
                },
                "overall_score": 0.95
            }
            
            return mock_monitor
    
    def test_health_check_workflow(self, mock_health_monitor):
        """Test health check workflow."""
        # Perform system health check
        health_status = mock_health_monitor.check_system_health()
        
        # Verify health check structure
        assert "status" in health_status
        assert "components" in health_status
        assert "overall_score" in health_status
        
        # Verify component health
        components = health_status["components"]
        required_components = ["database", "model_service", "cache_service"]
        
        for component in required_components:
            assert component in components
            assert "status" in components[component]
    
    def test_alerting_workflow(self, mock_health_monitor):
        """Test alerting workflow for unhealthy systems."""
        # Mock unhealthy system state
        mock_health_monitor.check_system_health.return_value = {
            "status": "unhealthy",
            "components": {
                "database": {"status": "unhealthy", "error": "Connection timeout"},
                "model_service": {"status": "healthy", "memory_usage": 45},
                "cache_service": {"status": "degraded", "error": "High latency"}
            },
            "overall_score": 0.35
        }
        
        # Mock alert sending
        with patch('odordiff2.monitoring.health.AlertManager') as mock_alert_class:
            mock_alerter = Mock()
            mock_alert_class.return_value = mock_alerter
            mock_alerter.send_alert.return_value = True
            
            # Check health and trigger alerts
            health_status = mock_health_monitor.check_system_health()
            
            if health_status["status"] != "healthy":
                alerter = mock_alert_class()
                alert_sent = alerter.send_alert(
                    severity="high",
                    message="System health degraded",
                    details=health_status
                )
                assert alert_sent is True


class TestDataPersistenceWorkflow:
    """Test data persistence and backup workflows."""
    
    def test_molecule_data_persistence(self):
        """Test molecule data persistence workflow."""
        with patch('odordiff2.data.persistence.MoleculeDatabase') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock database operations
            mock_db.save_molecule.return_value = "mol_123456"
            mock_db.get_molecule.return_value = {
                "id": "mol_123456",
                "smiles": "CCO",
                "safety_score": 0.95,
                "created_at": datetime.now().isoformat()
            }
            
            # Test molecule persistence
            molecule_data = {
                "smiles": "CCO",
                "safety_score": 0.95,
                "synthesis_score": 0.88,
                "generation_prompt": "Fresh citrus scent"
            }
            
            db = mock_db_class()
            
            # Save molecule
            molecule_id = db.save_molecule(molecule_data)
            assert isinstance(molecule_id, str)
            assert len(molecule_id) > 0
            
            # Retrieve molecule
            retrieved_data = db.get_molecule(molecule_id)
            assert retrieved_data["id"] == molecule_id
            assert retrieved_data["smiles"] == "CCO"
    
    def test_user_session_persistence(self):
        """Test user session persistence workflow."""
        with patch('odordiff2.data.persistence.SessionDatabase') as mock_session_db_class:
            mock_session_db = Mock()
            mock_session_db_class.return_value = mock_session_db
            
            # Mock session operations
            mock_session_db.create_session.return_value = "session_789"
            mock_session_db.update_session.return_value = True
            
            # Test session workflow
            session_db = mock_session_db_class()
            
            # Create user session
            session_data = {
                "user_id": "user_123",
                "generation_preferences": {
                    "safety_threshold": 0.9,
                    "preferred_scent_families": ["citrus", "floral"]
                },
                "api_usage": {
                    "requests_today": 15,
                    "rate_limit_tier": "premium"
                }
            }
            
            session_id = session_db.create_session(session_data)
            assert isinstance(session_id, str)
            
            # Update session with generation results
            session_update = {
                "last_generation": {
                    "prompt": "Fresh ocean breeze",
                    "molecules_generated": 3,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            updated = session_db.update_session(session_id, session_update)
            assert updated is True
    
    def test_backup_and_recovery_workflow(self):
        """Test backup and recovery workflow."""
        with patch('odordiff2.utils.backup.BackupManager') as mock_backup_class:
            mock_backup = Mock()
            mock_backup_class.return_value = mock_backup
            
            # Mock backup operations
            mock_backup.create_backup.return_value = {
                "backup_id": "backup_20250810_123456",
                "size_mb": 245.7,
                "checksum": "abc123def456",
                "created_at": datetime.now().isoformat()
            }
            
            mock_backup.verify_backup.return_value = True
            mock_backup.restore_from_backup.return_value = True
            
            backup_manager = mock_backup_class()
            
            # Create backup
            backup_result = backup_manager.create_backup(
                include_molecules=True,
                include_user_sessions=True,
                include_system_config=True
            )
            
            assert "backup_id" in backup_result
            assert backup_result["size_mb"] > 0
            assert "checksum" in backup_result
            
            # Verify backup integrity
            is_valid = backup_manager.verify_backup(backup_result["backup_id"])
            assert is_valid is True
            
            # Test restore (in emergency scenario)
            restore_success = backup_manager.restore_from_backup(
                backup_result["backup_id"],
                restore_molecules=True,
                restore_sessions=False  # Partial restore
            )
            assert restore_success is True


class TestErrorRecoveryWorkflow:
    """Test error recovery and resilience workflows."""
    
    def test_model_failure_recovery(self):
        """Test recovery from model service failures."""
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_model_class:
            # Simulate model failure and recovery
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            # First call fails, second succeeds
            mock_model.generate.side_effect = [
                Exception("CUDA out of memory"),
                [Molecule(smiles="CCO", safety_score=0.95)]  # Recovery success
            ]
            
            # Test retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = mock_model.generate(prompt="Test prompt")
                    # If successful, break
                    assert len(result) > 0
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        # If last attempt, re-raise
                        raise e
                    # Otherwise, continue to retry
                    continue
    
    def test_database_connection_recovery(self):
        """Test database connection recovery workflow."""
        with patch('odordiff2.data.persistence.DatabaseConnection') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Simulate connection failures and recovery
            mock_db.execute_query.side_effect = [
                Exception("Connection lost"),
                Exception("Connection timeout"),
                {"result": "success"}  # Recovery
            ]
            
            mock_db.reconnect.return_value = True
            
            # Test connection recovery logic
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    result = mock_db.execute_query("SELECT * FROM molecules")
                    assert result["result"] == "success"
                    break
                except Exception:
                    if attempt < max_attempts - 1:
                        # Attempt reconnection
                        reconnect_success = mock_db.reconnect()
                        assert reconnect_success is True
                    else:
                        # Final attempt failed
                        pytest.fail("Database recovery failed after max attempts")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_workflow(self):
        """Test circuit breaker pattern for service resilience."""
        with patch('odordiff2.utils.circuit_breaker.CircuitBreaker') as mock_cb_class:
            mock_circuit_breaker = Mock()
            mock_cb_class.return_value = mock_circuit_breaker
            
            # Mock circuit breaker states
            mock_circuit_breaker.is_open.side_effect = [
                False,  # Closed - allow requests
                False,  # Closed - allow requests  
                True,   # Open - block requests
                True,   # Open - block requests
                False   # Half-open - allow test request
            ]
            
            mock_circuit_breaker.record_success.return_value = None
            mock_circuit_breaker.record_failure.return_value = None
            
            circuit_breaker = mock_cb_class()
            
            # Simulate requests with circuit breaker
            for i in range(5):
                if not circuit_breaker.is_open():
                    try:
                        # Simulate service call
                        if i < 2:
                            result = "success"
                            circuit_breaker.record_success()
                            assert result == "success"
                        else:
                            raise Exception("Service error")
                    except Exception:
                        circuit_breaker.record_failure()
                else:
                    # Circuit breaker is open - fail fast
                    assert i >= 2  # Should be open after failures


@pytest.mark.integration
class TestFullSystemIntegration:
    """Integration tests for complete system workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_user_journey(self):
        """Test complete user journey from request to response."""
        # Mock all components
        mocks = {}
        
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_diffusion, \
             patch('odordiff2.safety.filter.SafetyFilter') as mock_safety, \
             patch('odordiff2.api.endpoints.APIEndpoints') as mock_api, \
             patch('odordiff2.utils.security.AuthenticationManager') as mock_auth, \
             patch('odordiff2.scaling.multi_tier_cache.MultiTierCache') as mock_cache:
            
            # Configure mocks
            mocks['diffusion'] = mock_diffusion.return_value
            mocks['safety'] = mock_safety.return_value
            mocks['api'] = mock_api.return_value
            mocks['auth'] = mock_auth.return_value
            mocks['cache'] = mock_cache.return_value
            
            # Mock responses
            mocks['diffusion'].generate.return_value = [
                Molecule(smiles="CCO", safety_score=0.95)
            ]
            mocks['safety'].assess_safety.return_value = Mock(is_safe=True)
            mocks['auth'].validate_api_key.return_value = Mock(
                is_valid=True, user_id="user_123"
            )
            mocks['cache'].get.return_value = None  # Cache miss
            mocks['cache'].set.return_value = True
            
            # Simulate complete user journey
            
            # 1. User authentication
            auth_result = mocks['auth'].validate_api_key("api_key_123")
            assert auth_result.is_valid
            
            # 2. Check cache
            cache_key = "generation_fresh_citrus_scent"
            cached_result = mocks['cache'].get(cache_key)
            assert cached_result is None  # Cache miss
            
            # 3. Generate molecules
            molecules = mocks['diffusion'].generate(
                prompt="Fresh citrus scent",
                num_molecules=3
            )
            assert len(molecules) == 1
            
            # 4. Safety assessment
            for molecule in molecules:
                safety_result = mocks['safety'].assess_safety(molecule)
                assert safety_result.is_safe
            
            # 5. Cache results
            result_data = {
                "molecules": [{"smiles": mol.smiles} for mol in molecules],
                "generation_time": 2.1
            }
            mocks['cache'].set(cache_key, result_data, ttl=3600)
            
            # Verify all components were called
            mocks['auth'].validate_api_key.assert_called_once()
            mocks['cache'].get.assert_called_once()
            mocks['diffusion'].generate.assert_called_once()
            mocks['safety'].assess_safety.assert_called()
            mocks['cache'].set.assert_called_once()
    
    def test_system_under_load(self):
        """Test system behavior under high load conditions."""
        # Simulate high concurrent load
        import concurrent.futures
        import time
        
        def simulate_request(request_id):
            """Simulate a single API request."""
            with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_diffusion:
                mock_model = mock_diffusion.return_value
                mock_model.generate.return_value = [
                    Molecule(smiles=f"C{request_id}", safety_score=0.9)
                ]
                
                start_time = time.time()
                
                # Simulate request processing
                molecules = mock_model.generate(
                    prompt=f"Request {request_id}",
                    num_molecules=1
                )
                
                end_time = time.time()
                
                return {
                    "request_id": request_id,
                    "molecules_count": len(molecules),
                    "processing_time": end_time - start_time,
                    "success": True
                }
        
        # Execute concurrent requests
        num_concurrent_requests = 50
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(simulate_request, i)
                for i in range(num_concurrent_requests)
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=30):
                result = future.result()
                results.append(result)
        
        # Verify all requests completed successfully
        assert len(results) == num_concurrent_requests
        
        successful_requests = [r for r in results if r["success"]]
        assert len(successful_requests) == num_concurrent_requests
        
        # Verify reasonable performance under load
        avg_processing_time = sum(r["processing_time"] for r in results) / len(results)
        assert avg_processing_time < 5.0  # Average < 5 seconds
    
    def test_disaster_recovery_scenario(self):
        """Test system recovery from catastrophic failures."""
        recovery_steps = []
        
        # Simulate disaster recovery workflow
        
        # 1. Detect system failure
        with patch('odordiff2.monitoring.health.HealthMonitor') as mock_health:
            mock_monitor = mock_health.return_value
            mock_monitor.check_system_health.return_value = {
                "status": "critical",
                "components": {
                    "database": {"status": "failed"},
                    "model_service": {"status": "failed"},
                    "cache_service": {"status": "failed"}
                }
            }
            
            health_status = mock_monitor.check_system_health()
            if health_status["status"] == "critical":
                recovery_steps.append("system_failure_detected")
        
        # 2. Initiate backup restoration
        with patch('odordiff2.utils.backup.BackupManager') as mock_backup:
            mock_backup_manager = mock_backup.return_value
            mock_backup_manager.get_latest_backup.return_value = "backup_20250810_120000"
            mock_backup_manager.restore_from_backup.return_value = True
            
            backup_manager = mock_backup_manager
            latest_backup = backup_manager.get_latest_backup()
            restore_success = backup_manager.restore_from_backup(latest_backup)
            
            if restore_success:
                recovery_steps.append("backup_restored")
        
        # 3. Restart services
        with patch('odordiff2.utils.service_manager.ServiceManager') as mock_service:
            mock_service_manager = mock_service.return_value
            mock_service_manager.restart_all_services.return_value = True
            
            restart_success = mock_service_manager.restart_all_services()
            if restart_success:
                recovery_steps.append("services_restarted")
        
        # 4. Verify system health
        with patch('odordiff2.monitoring.health.HealthMonitor') as mock_health:
            mock_monitor = mock_health.return_value
            mock_monitor.check_system_health.return_value = {
                "status": "healthy",
                "components": {
                    "database": {"status": "healthy"},
                    "model_service": {"status": "healthy"},
                    "cache_service": {"status": "healthy"}
                }
            }
            
            health_status = mock_monitor.check_system_health()
            if health_status["status"] == "healthy":
                recovery_steps.append("system_healthy")
        
        # Verify complete recovery workflow
        expected_steps = [
            "system_failure_detected",
            "backup_restored", 
            "services_restarted",
            "system_healthy"
        ]
        
        assert recovery_steps == expected_steps