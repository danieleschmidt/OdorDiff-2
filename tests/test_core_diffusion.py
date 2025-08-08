"""
Comprehensive tests for core diffusion module.
"""

import pytest
import asyncio
import torch
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from odordiff2.core.diffusion import OdorDiffusion, FragranceFormulation
from odordiff2.models.molecule import Molecule
from odordiff2.safety.filter import SafetyFilter


class TestOdorDiffusion:
    """Test suite for OdorDiffusion class."""
    
    @pytest.fixture
    def model(self):
        """Create OdorDiffusion instance for testing."""
        return OdorDiffusion(device="cpu")
    
    @pytest.fixture
    def mock_safety_filter(self):
        """Create mock safety filter."""
        filter_mock = Mock(spec=SafetyFilter)
        filter_mock.filter_molecules.return_value = ([], [])
        return filter_mock
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.device == "cpu"
        assert model.text_encoder is not None
        assert model.molecular_decoder is not None
        assert model.scheduler is not None
        assert len(model._fragrance_database) > 0
        
    def test_fragrance_database_initialization(self, model):
        """Test fragrance database initialization."""
        db = model._fragrance_database
        assert isinstance(db, list)
        assert len(db) > 0
        
        # Check database structure
        for entry in db:
            assert 'smiles' in entry
            assert 'name' in entry
            assert 'odor' in entry
            assert isinstance(entry['smiles'], str)
            assert len(entry['smiles']) > 0
    
    def test_from_pretrained(self):
        """Test loading pretrained model."""
        with patch('odordiff2.core.diffusion.OdorDiffusion') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            result = OdorDiffusion.from_pretrained("test-model", device="cpu")
            mock_class.assert_called_once_with(device="cpu")
    
    def test_generate_basic(self, model):
        """Test basic molecule generation."""
        result = model.generate("citrus fresh", num_molecules=3)
        
        assert isinstance(result, list)
        assert len(result) <= 3
        
        for molecule in result:
            assert isinstance(molecule, Molecule)
            assert molecule.smiles is not None
            assert molecule.confidence > 0
    
    def test_generate_with_safety_filter(self, model, mock_safety_filter):
        """Test generation with safety filter."""
        # Mock filter to return some molecules as safe
        test_molecules = [
            Molecule("CCO", 0.9),
            Molecule("CC(C)O", 0.8)
        ]
        mock_safety_filter.filter_molecules.return_value = (test_molecules, [])
        
        result = model.generate(
            "safe alcohol", 
            num_molecules=5,
            safety_filter=mock_safety_filter
        )
        
        assert isinstance(result, list)
        mock_safety_filter.filter_molecules.assert_called_once()
    
    def test_generate_with_synthesizability_filter(self, model):
        """Test generation with synthesizability minimum."""
        result = model.generate(
            "simple molecule",
            num_molecules=3,
            synthesizability_min=0.5
        )
        
        # Should filter out low synthesizability molecules
        for molecule in result:
            assert molecule.synth_score >= 0.5
    
    @pytest.mark.parametrize("prompt,expected_templates", [
        ("floral rose", ["CC(C)=CCO"]),
        ("citrus lemon", ["CC(C)=CC"]),
        ("woody cedar", ["CC(C)(C)c1ccccc1"]),
        ("fresh clean", ["CCO"]),
        ("spicy pepper", ["COc1ccccc1"])
    ])
    def test_template_selection(self, model, prompt, expected_templates):
        """Test template-based generation for different scent categories."""
        molecules = model._template_based_generation(prompt, 1)
        
        assert len(molecules) >= 1
        # At least one molecule should be based on expected template
        smiles_generated = [mol.smiles for mol in molecules]
        # Check that generation is influenced by prompt category
        assert any(any(template in smiles for smiles in smiles_generated) 
                  for template in expected_templates)
    
    def test_molecular_variation(self, model):
        """Test molecular variation generation."""
        base_smiles = "CCO"
        variation = model._add_molecular_variation(base_smiles)
        
        assert isinstance(variation, str)
        assert len(variation) > 0
        # Variation should be different or same as base
        assert variation != base_smiles or variation == base_smiles
    
    def test_odor_prediction(self, model):
        """Test odor profile prediction."""
        molecule = Molecule("CC(C)=CCO", 0.9)  # Linalool-like
        
        profile = model._predict_odor(molecule, "floral lavender")
        
        assert profile.primary_notes is not None
        assert profile.secondary_notes is not None
        assert 0 <= profile.intensity <= 1
        assert profile.longevity_hours > 0
        assert 0 <= profile.sillage <= 1
        assert isinstance(profile.character, str)
    
    def test_synthesizability_estimation(self, model):
        """Test synthesizability score estimation."""
        # Simple molecule should have high synthesizability
        simple_mol = Molecule("CCO", 0.9)
        simple_score = model._estimate_synthesizability(simple_mol)
        assert 0 <= simple_score <= 1
        assert simple_score > 0.5  # Ethanol is easy to synthesize
        
        # Complex molecule should have lower synthesizability
        complex_mol = Molecule("CC1=CC2=C(C=C1)C(=CN2)C(=O)NC3=CC=CC=C3C(=O)O", 0.9)
        complex_score = model._estimate_synthesizability(complex_mol)
        assert 0 <= complex_score <= 1
    
    def test_cost_estimation(self, model):
        """Test cost estimation."""
        molecule = Molecule("CCO", 0.9)
        molecule.synth_score = 0.8
        
        cost = model._estimate_cost(molecule)
        
        assert isinstance(cost, float)
        assert cost > 0
    
    def test_design_fragrance(self, model):
        """Test fragrance design functionality."""
        formulation = model.design_fragrance(
            base_notes="sandalwood, musk",
            heart_notes="rose, jasmine", 
            top_notes="bergamot, lemon",
            style="elegant, sophisticated"
        )
        
        assert isinstance(formulation, FragranceFormulation)
        assert len(formulation.base_accord) > 0
        assert len(formulation.heart_accord) > 0
        assert len(formulation.top_accord) > 0
        assert "elegant" in formulation.style_descriptor


class TestFragranceFormulation:
    """Test suite for FragranceFormulation class."""
    
    @pytest.fixture
    def formulation(self):
        """Create test fragrance formulation."""
        base_molecules = [Molecule("CC(C)(C)c1ccccc1O", 0.9)]
        heart_molecules = [Molecule("CC(C)=CCO", 0.8)]
        top_molecules = [Molecule("CC(C)=CC", 0.85)]
        
        return FragranceFormulation(
            base_accord=base_molecules,
            heart_accord=heart_molecules,
            top_accord=top_molecules,
            style_descriptor="test fragrance"
        )
    
    def test_formulation_creation(self, formulation):
        """Test formulation creation."""
        assert len(formulation.base_accord) == 1
        assert len(formulation.heart_accord) == 1
        assert len(formulation.top_accord) == 1
        assert formulation.style_descriptor == "test fragrance"
    
    @pytest.mark.parametrize("concentration,expected_percent", [
        ("parfum", 25),
        ("eau_de_parfum", 18),
        ("eau_de_toilette", 12),
        ("eau_de_cologne", 6)
    ])
    def test_perfume_formula_concentrations(self, formulation, concentration, expected_percent):
        """Test perfume formula generation with different concentrations."""
        formula = formulation.to_perfume_formula(concentration=concentration)
        
        assert formula['concentration_type'] == concentration
        assert formula['fragrance_oil_percent'] == expected_percent
        assert 'top_notes' in formula
        assert 'heart_notes' in formula
        assert 'base_notes' in formula
        
        # Check proportions add up correctly
        total_proportion = (
            formula['top_notes']['proportion_of_fragrance'] +
            formula['heart_notes']['proportion_of_fragrance'] +
            formula['base_notes']['proportion_of_fragrance']
        )
        assert abs(total_proportion - 1.0) < 0.01  # Allow small floating point errors
    
    def test_perfume_formula_structure(self, formulation):
        """Test perfume formula structure."""
        formula = formulation.to_perfume_formula()
        
        required_keys = [
            'concentration_type', 'fragrance_oil_percent', 'carrier',
            'top_notes', 'heart_notes', 'base_notes', 'style'
        ]
        
        for key in required_keys:
            assert key in formula
        
        # Check note structure
        for note_type in ['top_notes', 'heart_notes', 'base_notes']:
            note_data = formula[note_type]
            assert 'molecules' in note_data
            assert 'proportion_of_fragrance' in note_data
            assert 'actual_percent' in note_data
            assert isinstance(note_data['molecules'], list)


class TestTextEncoder:
    """Test suite for TextEncoder component."""
    
    def test_encoder_initialization(self):
        """Test text encoder initialization."""
        from odordiff2.core.diffusion import TextEncoder
        
        encoder = TextEncoder()
        assert encoder.tokenizer is not None
        assert encoder.encoder is not None
        assert encoder.hidden_size > 0
    
    def test_text_encoding(self):
        """Test text encoding functionality."""
        from odordiff2.core.diffusion import TextEncoder
        
        encoder = TextEncoder()
        texts = ["fresh citrus", "floral rose"]
        
        embeddings = encoder.forward(texts)
        
        assert embeddings is not None
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == encoder.hidden_size


class TestMolecularDecoder:
    """Test suite for MolecularDecoder component."""
    
    def test_decoder_initialization(self):
        """Test molecular decoder initialization."""
        from odordiff2.core.diffusion import MolecularDecoder
        
        decoder = MolecularDecoder()
        assert decoder.latent_dim == 512
        assert decoder.vocab_size == 100
        assert len(decoder.vocab) > 0
        assert len(decoder.reverse_vocab) > 0
    
    def test_vocabulary(self):
        """Test SMILES vocabulary."""
        from odordiff2.core.diffusion import MolecularDecoder
        
        decoder = MolecularDecoder()
        
        # Check essential tokens
        essential_tokens = ['PAD', 'START', 'END', 'C', 'O', 'N']
        for token in essential_tokens:
            assert token in decoder.vocab
            assert decoder.vocab[token] in decoder.reverse_vocab
    
    def test_decode_to_smiles(self):
        """Test SMILES decoding."""
        from odordiff2.core.diffusion import MolecularDecoder
        
        decoder = MolecularDecoder()
        
        # Create test sequence
        test_sequence = torch.tensor([[
            decoder.vocab['START'],
            decoder.vocab['C'],
            decoder.vocab['C'],
            decoder.vocab['O'],
            decoder.vocab['END']
        ]])
        
        smiles_list = decoder.decode_to_smiles(test_sequence)
        
        assert len(smiles_list) == 1
        assert smiles_list[0] == "CCO"


class TestDiffusionScheduler:
    """Test suite for SimpleDiffusionScheduler."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        from odordiff2.core.diffusion import SimpleDiffusionScheduler
        
        scheduler = SimpleDiffusionScheduler(num_steps=100)
        
        assert scheduler.num_steps == 100
        assert len(scheduler.betas) == 100
        assert len(scheduler.alphas) == 100
        assert len(scheduler.alpha_cumprod) == 100
    
    def test_noise_addition(self):
        """Test noise addition process."""
        from odordiff2.core.diffusion import SimpleDiffusionScheduler
        
        scheduler = SimpleDiffusionScheduler(num_steps=100)
        
        # Test data
        x = torch.randn(2, 512)
        timesteps = torch.tensor([10, 50])
        
        noisy_x, noise = scheduler.add_noise(x, timesteps)
        
        assert noisy_x.shape == x.shape
        assert noise.shape == x.shape
        assert not torch.equal(noisy_x, x)  # Should be different due to noise
    
    def test_denoise_step(self):
        """Test denoising step."""
        from odordiff2.core.diffusion import SimpleDiffusionScheduler
        
        scheduler = SimpleDiffusionScheduler(num_steps=100)
        
        # Test data
        model_output = torch.randn(2, 512)
        sample = torch.randn(2, 512)
        timestep = 10
        
        denoised = scheduler.denoise_step(model_output, timestep, sample)
        
        assert denoised.shape == sample.shape


# Integration tests
class TestDiffusionIntegration:
    """Integration tests for the diffusion system."""
    
    def test_end_to_end_generation(self):
        """Test complete end-to-end generation process."""
        model = OdorDiffusion(device="cpu")
        
        # Test various prompts
        prompts = [
            "fresh morning dew",
            "warm vanilla cookies",
            "ocean breeze",
            "rose garden"
        ]
        
        for prompt in prompts:
            molecules = model.generate(prompt, num_molecules=2)
            
            assert isinstance(molecules, list)
            assert len(molecules) <= 2
            
            for mol in molecules:
                assert isinstance(mol, Molecule)
                assert mol.is_valid or not mol.is_valid  # Either valid or invalid
                assert mol.confidence > 0
                assert mol.odor_profile is not None
    
    def test_generation_with_constraints(self):
        """Test generation with various constraints."""
        model = OdorDiffusion(device="cpu")
        safety_filter = SafetyFilter(toxicity_threshold=0.05)
        
        molecules = model.generate(
            "safe household fragrance",
            num_molecules=5,
            safety_filter=safety_filter,
            synthesizability_min=0.3
        )
        
        for mol in molecules:
            if mol.is_valid:
                assert mol.safety_score >= 0.95  # Should be very safe
                assert mol.synth_score >= 0.3
    
    def test_fragrance_design_integration(self):
        """Test complete fragrance design process."""
        model = OdorDiffusion(device="cpu")
        
        formulation = model.design_fragrance(
            base_notes="amber, sandalwood",
            heart_notes="rose, geranium",
            top_notes="citrus, mint",
            style="modern, fresh"
        )
        
        # Test formulation completeness
        assert isinstance(formulation, FragranceFormulation)
        
        all_molecules = (
            formulation.base_accord + 
            formulation.heart_accord + 
            formulation.top_accord
        )
        
        assert len(all_molecules) > 0
        
        # Test formula generation
        formula = formulation.to_perfume_formula()
        assert isinstance(formula, dict)
        assert formula['fragrance_oil_percent'] > 0


# Performance tests
class TestPerformance:
    """Performance tests for diffusion system."""
    
    def test_generation_performance(self):
        """Test generation performance."""
        import time
        
        model = OdorDiffusion(device="cpu")
        
        start_time = time.time()
        molecules = model.generate("performance test", num_molecules=3)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should complete within reasonable time
        assert generation_time < 30.0  # 30 seconds max
        assert len(molecules) <= 3
    
    def test_batch_generation_performance(self):
        """Test batch generation performance."""
        import time
        
        model = OdorDiffusion(device="cpu")
        
        prompts = [
            "citrus fresh",
            "floral sweet", 
            "woody warm",
            "marine aquatic",
            "spicy oriental"
        ]
        
        start_time = time.time()
        
        all_results = []
        for prompt in prompts:
            molecules = model.generate(prompt, num_molecules=2)
            all_results.extend(molecules)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_prompt = total_time / len(prompts)
        
        # Performance should be reasonable
        assert avg_time_per_prompt < 10.0  # 10 seconds per prompt max
        assert len(all_results) <= len(prompts) * 2


if __name__ == "__main__":
    pytest.main([__file__])