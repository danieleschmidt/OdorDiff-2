"""
Unit tests for Molecule class and related functionality.
"""

import pytest
import numpy as np
from rdkit import Chem

from odordiff2.models.molecule import Molecule, OdorProfile, SynthesisRoute, SafetyReport


class TestMolecule:
    """Test Molecule class functionality."""
    
    def test_molecule_creation(self):
        """Test basic molecule creation."""
        mol = Molecule("CCO", confidence=0.9)
        
        assert mol.smiles == "CCO"
        assert mol.confidence == 0.9
        assert mol.is_valid
        assert mol.mol is not None
    
    def test_invalid_molecule(self):
        """Test invalid SMILES handling."""
        mol = Molecule("invalid_smiles")
        
        assert not mol.is_valid
        assert mol.mol is None
    
    def test_molecular_properties(self):
        """Test molecular property calculation."""
        mol = Molecule("CCO")  # Ethanol
        
        mw = mol.get_property('molecular_weight')
        logp = mol.get_property('logP')
        
        assert mw is not None
        assert 40 < mw < 50  # Ethanol MW ≈ 46
        assert logp is not None
        assert -2 < logp < 1  # Ethanol LogP ≈ -0.31
    
    def test_molecular_fingerprint(self):
        """Test molecular fingerprint generation."""
        mol = Molecule("CCO")
        fp = mol.get_molecular_fingerprint()
        
        assert fp is not None
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 2048  # Standard Morgan fingerprint size
    
    def test_similarity_calculation(self):
        """Test molecular similarity calculation."""
        mol1 = Molecule("CCO")  # Ethanol
        mol2 = Molecule("CCCO")  # Propanol
        mol3 = Molecule("c1ccccc1")  # Benzene
        
        # Similar alcohols should have higher similarity
        sim_similar = mol1.calculate_similarity(mol2)
        sim_different = mol1.calculate_similarity(mol3)
        
        assert 0 <= sim_similar <= 1
        assert 0 <= sim_different <= 1
        assert sim_similar > sim_different
    
    def test_molecule_serialization(self):
        """Test molecule to/from dict conversion."""
        mol = Molecule("CCO", confidence=0.8)
        mol.safety_score = 0.9
        mol.synth_score = 0.7
        mol.estimated_cost = 25.5
        mol.odor_profile.primary_notes = ["clean", "alcohol"]
        mol.odor_profile.character = "fresh"
        
        # Convert to dict
        mol_dict = mol.to_dict()
        
        assert mol_dict['smiles'] == "CCO"
        assert mol_dict['confidence'] == 0.8
        assert mol_dict['safety_score'] == 0.9
        assert mol_dict['odor_profile']['primary_notes'] == ["clean", "alcohol"]
        
        # Convert back from dict
        mol2 = Molecule.from_dict(mol_dict)
        
        assert mol2.smiles == mol.smiles
        assert mol2.confidence == mol.confidence
        assert mol2.safety_score == mol.safety_score
        assert mol2.odor_profile.primary_notes == mol.odor_profile.primary_notes
    
    def test_visualization_generation(self, temp_dir):
        """Test 3D visualization generation."""
        mol = Molecule("CCO")
        viz_file = temp_dir / "test_viz.html"
        
        mol.visualize_3d(str(viz_file))
        
        assert viz_file.exists()
        content = viz_file.read_text()
        assert "CCO" in content
        assert "html" in content.lower()


class TestOdorProfile:
    """Test OdorProfile class."""
    
    def test_odor_profile_creation(self):
        """Test odor profile creation."""
        profile = OdorProfile(
            primary_notes=["floral", "sweet"],
            secondary_notes=["powdery", "creamy"],
            intensity=0.8,
            longevity_hours=6.0,
            sillage=0.7,
            character="elegant feminine"
        )
        
        assert profile.primary_notes == ["floral", "sweet"]
        assert profile.intensity == 0.8
        assert profile.longevity_hours == 6.0
    
    def test_odor_profile_string_representation(self):
        """Test string representation of odor profile."""
        profile = OdorProfile(
            primary_notes=["citrus", "fresh", "zesty"],
            intensity=0.9
        )
        
        profile_str = str(profile)
        assert "citrus, fresh, zesty" in profile_str
        assert "0.90" in profile_str


class TestSynthesisRoute:
    """Test SynthesisRoute class."""
    
    def test_synthesis_route_creation(self):
        """Test synthesis route creation."""
        steps = [
            {
                'reaction_type': 'Oxidation',
                'reactants': ['CCO'],
                'products': ['CC=O'],
                'reagents': ['PCC'],
                'conditions': {'temperature': 25},
                'yield_estimate': 0.8,
                'difficulty': 2,
                'cost_factor': 1.5
            }
        ]
        
        route = SynthesisRoute(
            steps=steps,
            score=0.85,
            total_yield=0.8,
            cost_estimate=45.0
        )
        
        assert len(route.steps) == 1
        assert route.score == 0.85
        assert route.total_yield == 0.8
        assert route.cost_estimate == 45.0


class TestSafetyReport:
    """Test SafetyReport class."""
    
    def test_safety_report_creation(self):
        """Test safety report creation."""
        report = SafetyReport(
            toxicity=0.05,
            skin_sensitizer=False,
            eco_score=0.2,
            ifra_compliant=True,
            regulatory_flags=[
                {"region": "EU", "status": "APPROVED"}
            ]
        )
        
        assert report.toxicity == 0.05
        assert not report.skin_sensitizer
        assert report.ifra_compliant
        assert len(report.regulatory_flags) == 1


class TestMoleculeEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_smiles(self):
        """Test handling of empty SMILES."""
        mol = Molecule("")
        assert not mol.is_valid
    
    def test_none_smiles(self):
        """Test handling of None SMILES."""
        with pytest.raises(TypeError):
            Molecule(None)
    
    def test_invalid_confidence(self):
        """Test handling of invalid confidence values."""
        mol = Molecule("CCO", confidence=1.5)
        assert mol.confidence == 1.5  # Should accept but may be clamped elsewhere
    
    def test_property_caching(self):
        """Test that molecular properties are cached."""
        mol = Molecule("CCO")
        
        # First access
        mw1 = mol.get_property('molecular_weight')
        # Second access (should use cache)
        mw2 = mol.get_property('molecular_weight')
        
        assert mw1 == mw2
        assert 'molecular_weight' in mol._properties
    
    def test_similarity_with_invalid_molecule(self):
        """Test similarity calculation with invalid molecule."""
        mol1 = Molecule("CCO")
        mol2 = Molecule("invalid_smiles")
        
        similarity = mol1.calculate_similarity(mol2)
        assert similarity == 0.0
    
    def test_fingerprint_with_invalid_molecule(self):
        """Test fingerprint generation with invalid molecule."""
        mol = Molecule("invalid_smiles")
        fp = mol.get_molecular_fingerprint()
        
        assert fp is None


class TestMoleculeIntegration:
    """Integration tests for molecule functionality."""
    
    def test_molecule_with_all_features(self):
        """Test molecule with all features populated."""
        mol = Molecule("CC(C)=CCO", confidence=0.95)  # Linalool
        
        # Set all attributes
        mol.safety_score = 0.88
        mol.synth_score = 0.75
        mol.estimated_cost = 85.0
        
        mol.odor_profile.primary_notes = ["floral", "lavender"]
        mol.odor_profile.secondary_notes = ["woody", "fresh"]
        mol.odor_profile.intensity = 0.7
        mol.odor_profile.longevity_hours = 4.5
        mol.odor_profile.sillage = 0.6
        mol.odor_profile.character = "elegant, calming"
        
        # Test all functionality
        assert mol.is_valid
        assert mol.get_property('molecular_weight') is not None
        assert mol.get_molecular_fingerprint() is not None
        
        # Test serialization round-trip
        mol_dict = mol.to_dict()
        mol2 = Molecule.from_dict(mol_dict)
        
        assert mol2.smiles == mol.smiles
        assert mol2.odor_profile.primary_notes == mol.odor_profile.primary_notes
        assert mol2.safety_score == mol.safety_score
    
    def test_molecule_comparison(self):
        """Test comparison of multiple molecules."""
        molecules = [
            Molecule("CCO"),      # Ethanol
            Molecule("CCCO"),     # Propanol  
            Molecule("CCCCO"),    # Butanol
            Molecule("c1ccccc1")  # Benzene
        ]
        
        # All should be valid
        for mol in molecules:
            assert mol.is_valid
        
        # Test pairwise similarities
        alcohols = molecules[:3]
        benzene = molecules[3]
        
        # Alcohols should be more similar to each other than to benzene
        alcohol_similarities = []
        benzene_similarities = []
        
        for i in range(len(alcohols)):
            for j in range(i + 1, len(alcohols)):
                sim = alcohols[i].calculate_similarity(alcohols[j])
                alcohol_similarities.append(sim)
            
            sim = alcohols[i].calculate_similarity(benzene)
            benzene_similarities.append(sim)
        
        avg_alcohol_sim = np.mean(alcohol_similarities)
        avg_benzene_sim = np.mean(benzene_similarities)
        
        assert avg_alcohol_sim > avg_benzene_sim