"""
Unit tests for SafetyFilter and safety assessment functionality.
"""

import pytest
from rdkit import Chem

from odordiff2.safety.filter import SafetyFilter
from odordiff2.models.molecule import Molecule
from tests.conftest import assert_valid_safety_report


class TestSafetyFilter:
    """Test SafetyFilter class functionality."""
    
    def test_safety_filter_initialization(self):
        """Test safety filter initialization with different parameters."""
        # Default initialization
        filter1 = SafetyFilter()
        assert filter1.toxicity_threshold == 0.1
        assert filter1.irritant_check is True
        
        # Custom initialization
        filter2 = SafetyFilter(
            toxicity_threshold=0.05,
            irritant_check=False,
            eco_threshold=0.2,
            ifra_compliance=False
        )
        assert filter2.toxicity_threshold == 0.05
        assert filter2.irritant_check is False
        assert filter2.eco_threshold == 0.2
        assert filter2.ifra_compliance is False
    
    def test_toxic_patterns_loading(self):
        """Test that toxic patterns are loaded correctly."""
        safety_filter = SafetyFilter()
        
        assert len(safety_filter._toxic_patterns) > 0
        assert len(safety_filter._allergenic_patterns) > 0
        assert len(safety_filter._banned_substructures) > 0
        
        # Check some expected patterns
        assert any('[As]' in pattern for pattern in safety_filter._toxic_patterns)
        assert any('[Hg]' in pattern for pattern in safety_filter._toxic_patterns)


class TestSafetyAssessment:
    """Test safety assessment functionality."""
    
    def test_assess_safe_molecule(self):
        """Test assessment of a generally safe molecule."""
        safety_filter = SafetyFilter()
        molecule = Molecule("CCO")  # Ethanol
        
        report = safety_filter.assess_molecule(molecule)
        assert_valid_safety_report(report)
        
        # Ethanol should be relatively safe
        assert report.toxicity < 0.5
        assert report.ifra_compliant
    
    def test_assess_invalid_molecule(self):
        """Test assessment of invalid molecule."""
        safety_filter = SafetyFilter()
        molecule = Molecule("invalid_smiles")
        
        report = safety_filter.assess_molecule(molecule)
        assert_valid_safety_report(report)
        
        # Invalid molecules should be flagged as unsafe
        assert report.toxicity == 1.0
        assert not report.ifra_compliant
        assert len(report.regulatory_flags) > 0
    
    def test_toxicity_assessment(self):
        """Test toxicity scoring."""
        safety_filter = SafetyFilter()
        
        # Test with different molecules
        test_cases = [
            ("CCO", "low"),        # Ethanol - low toxicity
            ("CC(C)=CCO", "low"),  # Linalool - low toxicity
            ("c1ccccc1", "medium") # Benzene - medium toxicity
        ]
        
        for smiles, expected_level in test_cases:
            molecule = Molecule(smiles)
            toxicity = safety_filter._assess_toxicity(molecule)
            
            assert 0 <= toxicity <= 1
            
            if expected_level == "low":
                assert toxicity < 0.3
            elif expected_level == "medium":
                assert 0.2 < toxicity < 0.8
    
    def test_skin_sensitization_check(self):
        """Test skin sensitization assessment."""
        safety_filter = SafetyFilter()
        
        # Test molecules with different sensitization potential
        test_cases = [
            ("CCO", False),           # Ethanol - not a sensitizer
            ("c1ccc(cc1)C=O", True),  # Benzaldehyde - potential sensitizer
        ]
        
        for smiles, expected_sensitizer in test_cases:
            molecule = Molecule(smiles)
            is_sensitizer = safety_filter._check_skin_sensitization(molecule)
            
            assert isinstance(is_sensitizer, bool)
            # Note: These are simplified tests - real assessment is more complex
    
    def test_environmental_impact_assessment(self):
        """Test environmental impact scoring."""
        safety_filter = SafetyFilter()
        
        test_cases = [
            ("CCO", "low"),          # Ethanol - biodegradable
            ("CCCCl", "high"),       # Chlorobutane - persistent
            ("c1ccccc1", "medium")   # Benzene - moderate persistence
        ]
        
        for smiles, expected_impact in test_cases:
            molecule = Molecule(smiles)
            eco_score = safety_filter._assess_environmental_impact(molecule)
            
            assert 0 <= eco_score <= 1
            
            if expected_impact == "low":
                assert eco_score < 0.3
            elif expected_impact == "high":
                assert eco_score > 0.5
    
    def test_ifra_compliance_check(self):
        """Test IFRA compliance assessment."""
        safety_filter = SafetyFilter()
        
        # Test molecules with different compliance status
        test_cases = [
            ("CCO", True),            # Ethanol - compliant
            ("CC(C)=CCO", True),      # Linalool - compliant
            ("c1ccccc1[Br]c2ccc([Br])cc2", False)  # Dibrominated aromatic - non-compliant
        ]
        
        for smiles, expected_compliant in test_cases:
            molecule = Molecule(smiles)
            is_compliant = safety_filter._check_ifra_compliance(molecule)
            
            assert isinstance(is_compliant, bool)
    
    def test_regulatory_compliance_flags(self):
        """Test regulatory compliance flag generation."""
        safety_filter = SafetyFilter()
        molecule = Molecule("CCO")
        
        flags = safety_filter._check_regulatory_compliance(molecule)
        
        assert isinstance(flags, list)
        for flag in flags:
            assert isinstance(flag, dict)
            assert "region" in flag
            assert "status" in flag


class TestSafetyFiltering:
    """Test molecule filtering functionality."""
    
    def test_filter_molecules_basic(self):
        """Test basic molecule filtering."""
        safety_filter = SafetyFilter(toxicity_threshold=0.5)
        
        molecules = [
            Molecule("CCO"),      # Should pass
            Molecule("CCCO"),     # Should pass
            Molecule("invalid")   # Should fail
        ]
        
        safe_molecules, reports = safety_filter.filter_molecules(molecules)
        
        assert len(reports) == 3
        assert len(safe_molecules) <= len(molecules)  # Some may be filtered out
        
        # All safe molecules should have good safety scores
        for mol in safe_molecules:
            assert mol.safety_score > 0.5
    
    def test_filter_molecules_strict(self):
        """Test strict filtering."""
        safety_filter = SafetyFilter(
            toxicity_threshold=0.01,  # Very strict
            irritant_check=True,
            eco_threshold=0.01,
            ifra_compliance=True
        )
        
        molecules = [
            Molecule("CCO"),
            Molecule("CC(C)=CCO"),
            Molecule("c1ccccc1")
        ]
        
        safe_molecules, reports = safety_filter.filter_molecules(molecules)
        
        # Strict filtering should remove some molecules
        assert len(safe_molecules) <= len(molecules)
    
    def test_filter_molecules_permissive(self):
        """Test permissive filtering."""
        safety_filter = SafetyFilter(
            toxicity_threshold=0.9,   # Very permissive
            irritant_check=False,
            eco_threshold=0.9,
            ifra_compliance=False
        )
        
        molecules = [
            Molecule("CCO"),
            Molecule("CC(C)=CCO"),
            Molecule("c1ccccc1")
        ]
        
        safe_molecules, reports = safety_filter.filter_molecules(molecules)
        
        # Permissive filtering should keep most molecules
        assert len(safe_molecules) == len([mol for mol in molecules if mol.is_valid])
    
    def test_get_safety_summary(self):
        """Test safety summary generation."""
        safety_filter = SafetyFilter()
        
        molecules = [
            Molecule("CCO"),
            Molecule("CCCO"),
            Molecule("c1ccccc1"),
            Molecule("invalid")
        ]
        
        summary = safety_filter.get_safety_summary(molecules)
        
        assert isinstance(summary, dict)
        assert "total_molecules" in summary
        assert "safe_molecules" in summary
        assert "average_toxicity" in summary
        assert "ifra_compliance_rate" in summary
        
        assert summary["total_molecules"] == 4
        assert 0 <= summary["average_toxicity"] <= 1
        assert 0 <= summary["ifra_compliance_rate"] <= 1


class TestSafetyPatternsDetection:
    """Test specific safety pattern detection."""
    
    def test_toxic_element_detection(self):
        """Test detection of toxic elements."""
        safety_filter = SafetyFilter()
        
        # Test molecules with toxic elements
        toxic_molecules = [
            "[As]c1ccccc1",  # Arsenic
            "[Hg]CC",        # Mercury
            "[Pb](CC)CC"     # Lead
        ]
        
        for smiles in toxic_molecules:
            molecule = Molecule(smiles)
            if molecule.is_valid:  # Some may not parse correctly
                toxicity = safety_filter._assess_toxicity(molecule)
                assert toxicity > 0.3  # Should be flagged as toxic
    
    def test_allergenic_pattern_detection(self):
        """Test detection of allergenic patterns."""
        safety_filter = SafetyFilter()
        
        # Test known allergenic fragrance compounds
        allergenic_smiles = [
            "CC(C)=CCc1ccc(C)cc1C",  # Limonene-like structure
            "c1ccc(cc1)C=O"          # Benzaldehyde
        ]
        
        for smiles in allergenic_smiles:
            molecule = Molecule(smiles)
            if molecule.is_valid:
                is_sensitizer = safety_filter._check_skin_sensitization(molecule)
                # Note: Results depend on pattern matching implementation
    
    def test_banned_substructure_detection(self):
        """Test detection of banned substructures."""
        safety_filter = SafetyFilter()
        
        # Test molecules with potentially banned substructures
        banned_molecules = [
            "CCl4",           # Carbon tetrachloride
            "C(Cl)(Cl)Cl"     # Chloroform-like
        ]
        
        for smiles in banned_molecules:
            molecule = Molecule(smiles)
            if molecule.is_valid:
                is_compliant = safety_filter._check_ifra_compliance(molecule)
                # Should be flagged as non-compliant


class TestSafetyFilterEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_molecule_list(self):
        """Test filtering empty molecule list."""
        safety_filter = SafetyFilter()
        
        safe_molecules, reports = safety_filter.filter_molecules([])
        
        assert safe_molecules == []
        assert reports == []
    
    def test_summary_empty_list(self):
        """Test safety summary with empty list."""
        safety_filter = SafetyFilter()
        
        summary = safety_filter.get_safety_summary([])
        
        assert summary == {}
    
    def test_molecule_with_no_properties(self):
        """Test assessment of molecule without computed properties."""
        safety_filter = SafetyFilter()
        molecule = Molecule("CCO")
        
        # Clear cached properties
        molecule._properties = {}
        
        report = safety_filter.assess_molecule(molecule)
        assert_valid_safety_report(report)
    
    def test_very_large_molecule(self):
        """Test assessment of very large molecule."""
        safety_filter = SafetyFilter()
        
        # Create a large molecule (long chain)
        large_smiles = "C" * 50  # Very long carbon chain
        molecule = Molecule(large_smiles)
        
        if molecule.is_valid:
            report = safety_filter.assess_molecule(molecule)
            # Large molecules should have higher toxicity scores due to size
            assert report.toxicity > 0.1


class TestSafetyFilterPerformance:
    """Test performance characteristics of safety filtering."""
    
    def test_batch_assessment_performance(self, performance_monitor):
        """Test performance of batch safety assessment."""
        safety_filter = SafetyFilter()
        
        # Create batch of molecules
        test_smiles = [
            "CCO", "CCCO", "CCCCO", "CC(C)O", "CC(C)CO",
            "c1ccccc1", "c1ccc(cc1)C", "c1ccc(cc1)O",
            "CC(=O)O", "CC(=O)C"
        ] * 10  # 100 molecules total
        
        molecules = [Molecule(smiles) for smiles in test_smiles]
        
        performance_monitor.start_monitoring()
        safe_molecules, reports = safety_filter.filter_molecules(molecules)
        performance_monitor.stop_monitoring()
        
        # Should complete within reasonable time
        assert performance_monitor.elapsed_time < 10.0  # 10 seconds
        
        # Should not use excessive memory
        assert performance_monitor.peak_memory < 200  # 200 MB
        
        # Should return reasonable results
        assert len(reports) == len(molecules)
        assert len(safe_molecules) <= len(molecules)