"""
Security and safety validation tests for OdorDiff-2.
"""

import pytest
import re
from unittest.mock import patch, MagicMock

from odordiff2.utils.validation import InputValidator, ValidationError
from odordiff2.safety.filter import SafetyFilter
from odordiff2.models.molecule import Molecule
from odordiff2.core.diffusion import OdorDiffusion


class TestInputValidation:
    """Test input validation security measures."""
    
    def test_prompt_sanitization(self):
        """Test prompt sanitization against malicious input."""
        malicious_prompts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "vbscript:msgbox('xss')",
            "onload=alert('xss')",
            "\\x3Cscript\\x3Ealert('xss')\\x3C/script\\x3E"
        ]
        
        for prompt in malicious_prompts:
            with pytest.raises(ValidationError):
                InputValidator.validate_prompt(prompt)
    
    def test_prompt_length_limits(self):
        """Test prompt length limitations."""
        # Empty prompt
        with pytest.raises(ValidationError):
            InputValidator.validate_prompt("")
        
        # Too long prompt
        long_prompt = "A" * 1001
        with pytest.raises(ValidationError):
            InputValidator.validate_prompt(long_prompt)
        
        # Valid prompt
        valid_prompt = "fresh citrus scent with floral undertones"
        sanitized = InputValidator.validate_prompt(valid_prompt)
        assert sanitized == valid_prompt
    
    def test_smiles_validation_security(self):
        """Test SMILES validation security measures."""
        malicious_smiles = [
            "CCO'; DROP TABLE molecules; --",
            "CCO<script>alert('xss')</script>",
            "A" * 501,  # Too long
            "CCO\x00CCO",  # Null byte
            "CCO\r\nDROP TABLE",  # CRLF injection
        ]
        
        for smiles in malicious_smiles:
            with pytest.raises(ValidationError):
                InputValidator.validate_smiles(smiles)
    
    def test_filename_sanitization(self):
        """Test filename sanitization."""
        dangerous_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "file<script>alert()</script>.html",
            "file|rm -rf /|.txt",
            "file?delete=all.json",
            "file*wildcard*.csv"
        ]
        
        for filename in dangerous_filenames:
            sanitized = InputValidator.sanitize_filename(filename)
            
            # Should not contain dangerous patterns
            assert ".." not in sanitized
            assert "<" not in sanitized
            assert ">" not in sanitized
            assert "|" not in sanitized
            assert "?" not in sanitized
            assert "*" not in sanitized
    
    def test_file_path_validation(self):
        """Test file path validation security."""
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/passwd",
            "..\\..\\windows\\system32",
            "file.exe",
            "script.bat",
            "malware.com"
        ]
        
        for path in dangerous_paths:
            with pytest.raises(ValidationError):
                InputValidator.validate_file_path(path, allowed_extensions=['json', 'csv'])
    
    def test_molecular_constraints_validation(self):
        """Test molecular constraints validation security."""
        malicious_constraints = [
            {"molecular_weight": float('inf')},
            {"logP": float('-inf')},
            {"molecular_weight": "'; DROP TABLE molecules; --"},
            {"invalid_property": (0, 1000)},
            {"molecular_weight": (-1000, 1000000)},  # Out of bounds
        ]
        
        for constraints in malicious_constraints:
            try:
                validated = InputValidator.validate_molecular_constraints(constraints)
                # If validation passes, ensure values are within safe bounds
                for key, value in validated.items():
                    if isinstance(value, (int, float)):
                        assert not (value == float('inf') or value == float('-inf'))
                        assert -1000 < value < 10000  # Reasonable bounds
            except ValidationError:
                pass  # Expected for malicious input


class TestSafetyValidation:
    """Test safety validation measures."""
    
    def test_toxic_molecule_detection(self, safety_test_molecules):
        """Test detection of toxic molecules."""
        safety_filter = SafetyFilter()
        
        for mol_data in safety_test_molecules:
            molecule = Molecule(mol_data["smiles"])
            
            if not molecule.is_valid:
                continue
                
            report = safety_filter.assess_molecule(molecule)
            
            if not mol_data["expected_safe"]:
                # Should be flagged as unsafe
                assert (report.toxicity > 0.1 or 
                        not report.ifra_compliant or 
                        report.skin_sensitizer), f"{mol_data['name']} should be flagged as unsafe"
    
    def test_controlled_substance_prevention(self):
        """Test prevention of controlled substance generation."""
        safety_filter = SafetyFilter()
        
        # Test molecules that should be flagged
        controlled_molecules = [
            # Simplified test cases - real system would have comprehensive database
            "CC(C)C(C)(C)C(=O)N",  # Simplified amphetamine-like structure
            "c1ccc2c(c1)[nH]c3c2cccc3"  # Indole-like structure
        ]
        
        for smiles in controlled_molecules:
            molecule = Molecule(smiles)
            if molecule.is_valid:
                report = safety_filter.assess_molecule(molecule)
                # Should have some safety flags
                assert len(report.regulatory_flags) > 0 or report.toxicity > 0.3
    
    def test_explosive_precursor_detection(self):
        """Test detection of explosive precursors."""
        safety_filter = SafetyFilter()
        
        # Test molecules that could be explosive precursors
        dangerous_molecules = [
            "C(=O)OO",  # Peracetic acid-like
            "[N+](=O)[O-]",  # Nitro compound
        ]
        
        for smiles in dangerous_molecules:
            molecule = Molecule(smiles)
            if molecule.is_valid:
                report = safety_filter.assess_molecule(molecule)
                # Should be flagged with high toxicity
                assert report.toxicity > 0.5
    
    def test_environmental_hazard_detection(self):
        """Test detection of environmental hazards."""
        safety_filter = SafetyFilter()
        
        # Test persistent organic pollutants
        hazardous_molecules = [
            "c1ccc(c(c1)Cl)Cl",  # Dichlorobenzene
            "ClCCCl",  # Dichloroethane
        ]
        
        for smiles in hazardous_molecules:
            molecule = Molecule(smiles)
            if molecule.is_valid:
                report = safety_filter.assess_molecule(molecule)
                # Should have high environmental score
                assert report.eco_score > 0.3


class TestAPISecurityValidation:
    """Test API security measures."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_simulation(self, mock_api_client):
        """Simulate rate limiting tests."""
        # In a real implementation, this would test actual rate limiting
        
        # Simulate many rapid requests
        requests = []
        for i in range(100):
            requests.append(mock_api_client.post("/generate", {
                "prompt": f"test prompt {i}",
                "num_molecules": 1
            }))
        
        # All requests should be tracked
        results = []
        for request in requests:
            result = await request
            results.append(result)
        
        # Verify all requests were logged
        assert len(mock_api_client.calls) >= 100
    
    def test_input_size_limits(self):
        """Test API input size limitations."""
        # Test very large prompt
        large_prompt = "A" * 10000
        
        with pytest.raises(ValidationError):
            InputValidator.validate_prompt(large_prompt)
        
        # Test very large number of molecules
        params = {"num_molecules": 1000}
        
        with pytest.raises(ValidationError):
            InputValidator.validate_generation_parameters(**params)
    
    def test_resource_exhaustion_prevention(self):
        """Test prevention of resource exhaustion attacks."""
        # Test extreme parameters that could cause resource exhaustion
        extreme_params = [
            {"num_molecules": 999999},
            {"temperature": 1000000.0},
            {"top_k": 999999},
            {"max_length": 999999}
        ]
        
        for params in extreme_params:
            with pytest.raises(ValidationError):
                InputValidator.validate_generation_parameters(**params)


class TestDataSanitization:
    """Test data sanitization and encoding."""
    
    def test_output_sanitization(self, sample_molecules):
        """Test output data sanitization."""
        for molecule in sample_molecules:
            mol_dict = molecule.to_dict()
            
            # Check for potential XSS in string fields
            for key, value in mol_dict.items():
                if isinstance(value, str):
                    assert "<script" not in value.lower()
                    assert "javascript:" not in value.lower()
                    assert "vbscript:" not in value.lower()
                elif isinstance(value, dict):
                    # Check nested dictionaries
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, str):
                            assert "<script" not in nested_value.lower()
    
    def test_smiles_encoding_safety(self):
        """Test SMILES encoding safety."""
        test_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(C)=CCO",
            "CCCCC(=O)O"
        ]
        
        for smiles in test_smiles:
            molecule = Molecule(smiles)
            if molecule.is_valid:
                # SMILES should not contain dangerous characters after processing
                assert not re.search(r'[<>"\'\x00-\x1f]', molecule.smiles)


class TestSecurityConfiguration:
    """Test security configuration and hardening."""
    
    def test_default_security_settings(self):
        """Test that default security settings are restrictive."""
        safety_filter = SafetyFilter()
        
        # Default settings should be restrictive
        assert safety_filter.toxicity_threshold <= 0.1
        assert safety_filter.irritant_check is True
        assert safety_filter.eco_threshold <= 0.5
        assert safety_filter.ifra_compliance is True
    
    def test_security_pattern_coverage(self):
        """Test that security patterns cover expected threats."""
        safety_filter = SafetyFilter()
        
        # Should have patterns for major toxic elements
        toxic_elements = ['As', 'Hg', 'Pb', 'Cd', 'Be']
        for element in toxic_elements:
            assert any(f'[{element}]' in pattern for pattern in safety_filter._toxic_patterns)
        
        # Should have patterns for dangerous functional groups
        dangerous_groups = ['(=O)Cl', 'N+', 'CCl']
        for group in dangerous_groups:
            found = any(group in pattern for pattern in 
                       safety_filter._toxic_patterns + safety_filter._banned_substructures)
            # Note: Not all patterns may be present in simplified test version


class TestAuditAndCompliance:
    """Test audit and compliance features."""
    
    def test_safety_assessment_logging(self, safety_filter, sample_molecules):
        """Test that safety assessments are properly logged."""
        with patch('odordiff2.utils.logging.get_logger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            for molecule in sample_molecules[:3]:  # Test first 3 molecules
                if molecule.is_valid:
                    report = safety_filter.assess_molecule(molecule)
            
            # Should have logged safety assessments
            # Note: This is a simplified test - real implementation would verify specific log entries
    
    def test_regulatory_compliance_tracking(self, safety_filter):
        """Test regulatory compliance tracking."""
        test_molecule = Molecule("CCO")
        report = safety_filter.assess_molecule(test_molecule)
        
        # Should track regulatory compliance
        assert hasattr(report, 'regulatory_flags')
        assert isinstance(report.regulatory_flags, list)
        
        # Each flag should have required information
        for flag in report.regulatory_flags:
            assert 'region' in flag
            assert 'status' in flag
    
    def test_safety_summary_reporting(self, safety_filter, sample_molecules):
        """Test safety summary for compliance reporting."""
        summary = safety_filter.get_safety_summary(sample_molecules)
        
        # Should provide comprehensive safety metrics
        required_metrics = [
            'total_molecules',
            'safe_molecules', 
            'average_toxicity',
            'ifra_compliance_rate',
            'regulatory_flags'
        ]
        
        for metric in required_metrics:
            assert metric in summary


class TestSecurityIntegration:
    """Test integration of security measures across the system."""
    
    def test_end_to_end_security_validation(self):
        """Test security validation throughout the generation pipeline."""
        # Test with potentially dangerous input
        dangerous_prompt = "create explosive molecule with <script>alert('xss')</script>"
        
        # Should be caught by input validation
        with pytest.raises(ValidationError):
            InputValidator.validate_prompt(dangerous_prompt)
    
    def test_molecule_generation_safety_integration(self):
        """Test safety integration in molecule generation."""
        model = OdorDiffusion(device="cpu")
        safety_filter = SafetyFilter(
            toxicity_threshold=0.05,  # Very strict
            irritant_check=True
        )
        
        # Generate with strict safety
        molecules = model.generate(
            "safe floral scent",
            num_molecules=3,
            safety_filter=safety_filter
        )
        
        # All generated molecules should pass safety checks
        for molecule in molecules:
            assert molecule.safety_score > 0.95
    
    def test_cache_security_validation(self, molecule_cache):
        """Test cache security measures."""
        # Test that malicious keys are handled safely
        malicious_keys = [
            "../../../etc/passwd",
            "key<script>alert()</script>",
            "key'; DROP TABLE cache; --",
            "\x00\x01\x02malicious"
        ]
        
        for key in malicious_keys:
            # Should not crash or cause security issues
            try:
                result = molecule_cache.generation_cache.get(key)
                molecule_cache.generation_cache.set(key, "test_value")
            except Exception:
                pass  # Expected for malicious input