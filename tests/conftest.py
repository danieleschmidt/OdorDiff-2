"""
Pytest configuration and fixtures for OdorDiff-2 tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Generator

from odordiff2.core.diffusion import OdorDiffusion
from odordiff2.core.async_diffusion import AsyncOdorDiffusion
from odordiff2.safety.filter import SafetyFilter
from odordiff2.models.molecule import Molecule
from odordiff2.data.cache import MoleculeCache


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_smiles() -> List[str]:
    """Sample SMILES strings for testing."""
    return [
        "CC(C)=CCO",  # Linalool - floral
        "CC(C)=CCCC(C)=CCO",  # Geraniol - rose
        "CC1=CCC(CC1)C(C)C",  # Limonene - citrus
        "c1ccc(cc1)C=O",  # Benzaldehyde - almond
        "COc1ccccc1",  # Anisole - sweet
        "CCO",  # Ethanol - simple alcohol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1"  # Benzene - aromatic
    ]


@pytest.fixture
def sample_molecules(sample_smiles) -> List[Molecule]:
    """Create sample molecule objects."""
    molecules = []
    for smiles in sample_smiles:
        mol = Molecule(smiles, confidence=0.8)
        mol.safety_score = 0.9
        mol.synth_score = 0.7
        mol.estimated_cost = 50.0
        molecules.append(mol)
    return molecules


@pytest.fixture
def safety_filter() -> SafetyFilter:
    """Create safety filter instance."""
    return SafetyFilter(
        toxicity_threshold=0.1,
        irritant_check=True,
        eco_threshold=0.3,
        ifra_compliance=True
    )


@pytest.fixture
def odor_diffusion() -> OdorDiffusion:
    """Create OdorDiffusion model instance."""
    return OdorDiffusion(device="cpu")


@pytest.fixture
async def async_odor_diffusion() -> AsyncOdorDiffusion:
    """Create AsyncOdorDiffusion model instance."""
    model = AsyncOdorDiffusion(
        device="cpu",
        max_workers=2,
        batch_size=4,
        enable_caching=True
    )
    await model.start()
    yield model
    await model.stop()


@pytest.fixture
def molecule_cache(temp_dir) -> MoleculeCache:
    """Create molecule cache instance with temporary directory."""
    cache = MoleculeCache(cache_dir=str(temp_dir / "cache"))
    return cache


@pytest.fixture
def test_prompts() -> List[str]:
    """Sample prompts for testing."""
    return [
        "fresh citrus scent",
        "warm vanilla fragrance",
        "floral rose bouquet",
        "woody cedar aroma",
        "clean aquatic breeze"
    ]


@pytest.fixture
def performance_test_data():
    """Data for performance testing."""
    return {
        'generation_targets': {
            'single_molecule_time': 2.0,  # seconds
            'batch_5_time': 8.0,
            'memory_limit_mb': 500
        },
        'api_targets': {
            'response_time_95th': 3.0,  # seconds
            'concurrent_requests': 10
        },
        'cache_targets': {
            'hit_rate_min': 0.7,  # 70%
            'lookup_time_max': 0.01  # 10ms
        }
    }


@pytest.fixture
def safety_test_molecules():
    """Molecules with known safety characteristics for testing."""
    return [
        # Safe molecules
        {"smiles": "CCO", "expected_safe": True, "name": "ethanol"},
        {"smiles": "CC(C)=CCO", "expected_safe": True, "name": "linalool"},
        {"smiles": "c1ccc(cc1)CO", "expected_safe": True, "name": "benzyl_alcohol"},
        
        # Potentially unsafe molecules (simplified test cases)
        {"smiles": "c1ccc(cc1)C=O", "expected_safe": True, "name": "benzaldehyde"},  # Generally safe in fragrance use
        {"smiles": "CCCl", "expected_safe": False, "name": "chloroethane"},  # Contains halogen
        {"smiles": "[As]c1ccccc1", "expected_safe": False, "name": "phenyl_arsenic"},  # Contains toxic element
    ]


class MockAPIClient:
    """Mock API client for testing."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.calls = []
    
    async def post(self, endpoint: str, data: dict):
        """Mock POST request."""
        self.calls.append(("POST", endpoint, data))
        
        # Simulate different responses based on endpoint
        if endpoint == "/generate":
            return {
                "request_id": "test-123",
                "prompt": data.get("prompt", "test"),
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
        
        elif endpoint == "/assess/safety":
            return {
                "smiles": data.get("smiles", "CCO"),
                "assessment": {
                    "toxicity_score": 0.05,
                    "skin_sensitizer": False,
                    "eco_score": 0.1,
                    "ifra_compliant": True,
                    "regulatory_flags": []
                },
                "recommendation": "safe"
            }
        
        return {"status": "success"}
    
    async def get(self, endpoint: str):
        """Mock GET request."""
        self.calls.append(("GET", endpoint, None))
        
        if endpoint == "/health":
            return {
                "status": "healthy",
                "response_time": 0.1,
                "memory_usage_mb": 256.0,
                "worker_count": 4,
                "cache_enabled": True,
                "stats": {}
            }
        
        elif endpoint == "/stats":
            return {
                "stats": {
                    "total_requests": 100,
                    "cache_hits": 70,
                    "avg_processing_time": 1.8
                }
            }
        
        return {"data": "test"}


@pytest.fixture
def mock_api_client():
    """Create mock API client for testing."""
    return MockAPIClient()


# Utility functions for tests

def assert_valid_molecule(molecule: Molecule):
    """Assert that a molecule object is valid and has expected properties."""
    assert molecule is not None
    assert isinstance(molecule.smiles, str)
    assert len(molecule.smiles) > 0
    assert 0 <= molecule.confidence <= 1
    assert 0 <= molecule.safety_score <= 1
    assert 0 <= molecule.synth_score <= 1
    assert molecule.estimated_cost >= 0


def assert_valid_safety_report(report):
    """Assert that a safety report has expected structure."""
    assert hasattr(report, 'toxicity')
    assert hasattr(report, 'skin_sensitizer')
    assert hasattr(report, 'eco_score')
    assert hasattr(report, 'ifra_compliant')
    assert hasattr(report, 'regulatory_flags')
    
    assert 0 <= report.toxicity <= 1
    assert 0 <= report.eco_score <= 1
    assert isinstance(report.skin_sensitizer, bool)
    assert isinstance(report.ifra_compliant, bool)
    assert isinstance(report.regulatory_flags, list)


def create_test_fragrance_data():
    """Create test data for fragrance formulations."""
    return {
        "base_notes": "sandalwood, amber, musk",
        "heart_notes": "jasmine, rose, ylang-ylang",
        "top_notes": "bergamot, lemon, green apple",
        "style": "modern, ethereal, long-lasting",
        "constraints": {
            'molecular_weight': (150, 350),
            'logP': (1.5, 4.5),
            'allergenic': False,
            'biodegradable': True
        }
    }


# Performance testing utilities

@pytest.fixture
def performance_monitor():
    """Simple performance monitoring utility."""
    import time
    import psutil
    import threading
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.peak_memory = 0
            self.monitoring = False
            self._monitor_thread = None
        
        def start_monitoring(self):
            self.start_time = time.time()
            self.monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_memory)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
        
        def stop_monitoring(self):
            self.end_time = time.time()
            self.monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)
        
        def _monitor_memory(self):
            process = psutil.Process()
            while self.monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.peak_memory = max(self.peak_memory, memory_mb)
                    time.sleep(0.1)
                except:
                    break
        
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return PerformanceMonitor()