"""
Unit tests for validation and sanitization utilities.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from odordiff2.utils.validation import (
    ValidationError, SanitizationError, Sanitizer, InputValidator,
    ValidationMiddleware, SCHEMAS, validate_input, sanitize_user_input,
    validate_generation_request, validate_batch_request, is_safe_filename,
    is_valid_ip
)


class TestValidationError:
    """Test ValidationError class."""
    
    def test_basic_error(self):
        """Test basic ValidationError creation."""
        error = ValidationError("Test error")
        
        assert str(error) == "Test error"
        assert error.field is None
        assert error.value is None
    
    def test_error_with_field_and_value(self):
        """Test ValidationError with field and value."""
        error = ValidationError("Invalid value", field="test_field", value="test_value")
        
        assert str(error) == "Invalid value"
        assert error.field == "test_field"
        assert error.value == "test_value"


class TestSanitizationError:
    """Test SanitizationError class."""
    
    def test_sanitization_error(self):
        """Test SanitizationError creation."""
        error = SanitizationError("Sanitization failed")
        
        assert str(error) == "Sanitization failed"


class TestSanitizer:
    """Test Sanitizer class methods."""
    
    def test_sanitize_text_basic(self):
        """Test basic text sanitization."""
        text = "  Hello, World!  "
        result = Sanitizer.sanitize_text(text)
        
        assert result == "Hello, World!"
    
    def test_sanitize_text_max_length(self):
        """Test text length truncation."""
        text = "a" * 1500
        result = Sanitizer.sanitize_text(text, max_length=100)
        
        assert len(result) == 100
        assert result == "a" * 100
    
    def test_sanitize_text_invalid_input(self):
        """Test sanitization with invalid input type."""
        with pytest.raises(SanitizationError, match="Input must be a string"):
            Sanitizer.sanitize_text(123)
    
    def test_sanitize_text_control_characters(self):
        """Test removal of control characters."""
        text = "Hello\x00\x01World\x1f"
        result = Sanitizer.sanitize_text(text)
        
        assert result == "HelloWorld"
    
    def test_sanitize_text_preserve_whitespace(self):
        """Test preservation of normal whitespace."""
        text = "Hello\n\r\tWorld"
        result = Sanitizer.sanitize_text(text)
        
        assert result == "Hello\n\r\tWorld"
    
    def test_sanitize_text_html_escape(self):
        """Test HTML escaping."""
        text = "<script>alert('xss')</script>"
        result = Sanitizer.sanitize_text(text)
        
        assert "&lt;" in result
        assert "&gt;" in result
        assert "<script>" not in result
    
    def test_sanitize_text_allow_html(self):
        """Test allowing HTML when specified."""
        text = "<b>Bold text</b>"
        result = Sanitizer.sanitize_text(text, allow_html=True)
        
        # With allow_html=True and dangerous pattern removal, 
        # safe HTML should be preserved
        assert "Bold text" in result
    
    def test_sanitize_text_dangerous_patterns(self):
        """Test removal of dangerous patterns."""
        dangerous_texts = [
            "javascript:alert('xss')",
            "vbscript:msgbox('test')",
            "data:text/html,<script>",
            "<script>alert(1)</script>",
            "<iframe src='evil'></iframe>",
            "onclick=alert(1)",
            "\\x41\\x42",
            "\\u0041\\u0042"
        ]
        
        for text in dangerous_texts:
            result = Sanitizer.sanitize_text(text)
            # Should remove or neutralize dangerous content
            assert "javascript:" not in result.lower()
            assert "vbscript:" not in result.lower()
            assert "<script" not in result.lower()
            assert "<iframe" not in result.lower()
    
    def test_sanitize_text_url_decode(self):
        """Test URL decoding."""
        text = "Hello%20World%21"
        result = Sanitizer.sanitize_text(text)
        
        assert result == "Hello World!"
    
    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        filename = "test_file.txt"
        result = Sanitizer.sanitize_filename(filename)
        
        assert result == "test_file.txt"
    
    def test_sanitize_filename_dangerous_chars(self):
        """Test removal of dangerous characters."""
        filename = "test<>:\"/\\|?*file.txt"
        result = Sanitizer.sanitize_filename(filename)
        
        assert result == "test_________file.txt"
    
    def test_sanitize_filename_directory_traversal(self):
        """Test prevention of directory traversal."""
        filename = "../../../etc/passwd"
        result = Sanitizer.sanitize_filename(filename)
        
        assert result == "_./._./._/etc/passwd"
    
    def test_sanitize_filename_reserved_names(self):
        """Test handling of reserved Windows names."""
        reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
        
        for name in reserved_names:
            result = Sanitizer.sanitize_filename(name)
            assert not result.upper().startswith(name.upper())
    
    def test_sanitize_filename_long_filename(self):
        """Test truncation of long filenames."""
        filename = "a" * 250 + ".txt"
        result = Sanitizer.sanitize_filename(filename)
        
        assert len(result) <= 200
        assert result.endswith(".txt")
    
    def test_sanitize_filename_empty(self):
        """Test handling of empty filename."""
        result = Sanitizer.sanitize_filename("")
        assert result == "unnamed"
        
        result = Sanitizer.sanitize_filename("   ")
        assert result == "unnamed"
    
    def test_sanitize_filename_invalid_type(self):
        """Test filename sanitization with invalid type."""
        with pytest.raises(SanitizationError, match="Filename must be a string"):
            Sanitizer.sanitize_filename(123)
    
    def test_sanitize_json_basic(self):
        """Test basic JSON sanitization."""
        data = {
            "string": "Hello World",
            "number": 42,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3]
        }
        
        result = Sanitizer.sanitize_json(data)
        
        assert isinstance(result, dict)
        assert result["string"] == "Hello World"
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["null"] is None
        assert result["array"] == [1, 2, 3]
    
    def test_sanitize_json_nested(self):
        """Test nested JSON sanitization."""
        data = {
            "level1": {
                "level2": {
                    "value": "test"
                }
            }
        }
        
        result = Sanitizer.sanitize_json(data)
        
        assert result["level1"]["level2"]["value"] == "test"
    
    def test_sanitize_json_max_depth(self):
        """Test max depth enforcement."""
        # Create deeply nested structure
        data = {}
        current = data
        for i in range(15):
            current["next"] = {}
            current = current["next"]
        current["value"] = "deep"
        
        with pytest.raises(SanitizationError, match="Maximum nesting depth"):
            Sanitizer.sanitize_json(data, max_depth=10)
    
    def test_sanitize_json_max_items_dict(self):
        """Test max items enforcement for dictionaries."""
        data = {f"key_{i}": i for i in range(1500)}
        
        with pytest.raises(SanitizationError, match="Dictionary too large"):
            Sanitizer.sanitize_json(data, max_items=1000)
    
    def test_sanitize_json_max_items_list(self):
        """Test max items enforcement for lists."""
        data = list(range(1500))
        
        with pytest.raises(SanitizationError, match="Array too large"):
            Sanitizer.sanitize_json(data, max_items=1000)
    
    def test_sanitize_json_string_sanitization(self):
        """Test that strings in JSON are sanitized."""
        data = {
            "clean": "normal text",
            "dirty": "<script>alert('xss')</script>",
            "with_controls": "text\x00with\x01controls"
        }
        
        result = Sanitizer.sanitize_json(data)
        
        assert result["clean"] == "normal text"
        assert "<script>" not in result["dirty"]
        assert "\x00" not in result["with_controls"]
        assert "\x01" not in result["with_controls"]
    
    def test_sanitize_json_key_sanitization(self):
        """Test that dictionary keys are sanitized."""
        data = {
            "<script>key</script>": "value",
            "normal_key": "value2"
        }
        
        result = Sanitizer.sanitize_json(data)
        
        # Keys should be sanitized
        keys = list(result.keys())
        assert not any("<script>" in key for key in keys)
        assert len(keys) == 2
    
    def test_validate_ip_address_valid_ipv4(self):
        """Test valid IPv4 addresses."""
        valid_ips = [
            "127.0.0.1",
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "8.8.8.8"
        ]
        
        for ip in valid_ips:
            assert Sanitizer.validate_ip_address(ip) is True
    
    def test_validate_ip_address_valid_ipv6(self):
        """Test valid IPv6 addresses."""
        valid_ips = [
            "::1",
            "2001:db8::1",
            "fe80::1",
            "::ffff:192.168.1.1"
        ]
        
        for ip in valid_ips:
            assert Sanitizer.validate_ip_address(ip) is True
    
    def test_validate_ip_address_invalid(self):
        """Test invalid IP addresses."""
        invalid_ips = [
            "256.256.256.256",
            "192.168.1",
            "not.an.ip.address",
            "",
            "192.168.1.1.1",
            "::g"
        ]
        
        for ip in invalid_ips:
            assert Sanitizer.validate_ip_address(ip) is False
    
    def test_sanitize_sql_input_basic(self):
        """Test basic SQL sanitization."""
        text = "normal text"
        result = Sanitizer.sanitize_sql_input(text)
        
        assert result == "normal text"
    
    def test_sanitize_sql_input_injection_patterns(self):
        """Test removal of SQL injection patterns."""
        dangerous_inputs = [
            "'; DROP TABLE users; --",
            '"; DELETE FROM products; --',
            "'; UNION SELECT * FROM passwords; --",
            "admin'; UPDATE users SET role='admin'; --",
            "EXEC xp_cmdshell('dir')",
            "'; EXEC sp_executesql N'malicious'; --"
        ]
        
        for dangerous_input in dangerous_inputs:
            result = Sanitizer.sanitize_sql_input(dangerous_input)
            
            # Should remove dangerous patterns
            assert "DROP TABLE" not in result.upper()
            assert "DELETE FROM" not in result.upper()
            assert "UNION SELECT" not in result.upper()
            assert "EXEC " not in result.upper()
            assert "xp_" not in result.lower()
            assert "sp_" not in result.lower()
    
    def test_sanitize_sql_input_non_string(self):
        """Test SQL sanitization with non-string input."""
        result = Sanitizer.sanitize_sql_input(123)
        assert result == "123"
        
        result = Sanitizer.sanitize_sql_input(None)
        assert result == "None"


class TestInputValidator:
    """Test InputValidator class methods."""
    
    def test_validate_with_schema_available(self):
        """Test schema validation when jsonschema is available."""
        if not hasattr(InputValidator, 'validate_with_schema'):
            pytest.skip("Schema validation not available")
        
        data = {
            "prompt": "test prompt",
            "num_molecules": 5,
            "safety_threshold": 0.1
        }
        
        with patch('odordiff2.utils.validation.JSONSCHEMA_AVAILABLE', True):
            with patch('odordiff2.utils.validation.validate') as mock_validate:
                mock_validate.return_value = None
                
                result = InputValidator.validate_with_schema(data, 'generation_request')
                
                assert result == data
                mock_validate.assert_called_once()
    
    def test_validate_with_schema_not_available(self):
        """Test schema validation when jsonschema is not available."""
        data = {"test": "data"}
        
        with patch('odordiff2.utils.validation.JSONSCHEMA_AVAILABLE', False):
            result = InputValidator.validate_with_schema(data, 'generation_request')
            
            assert result == data
    
    def test_validate_with_schema_unknown_schema(self):
        """Test validation with unknown schema."""
        data = {"test": "data"}
        
        with pytest.raises(ValidationError, match="Unknown schema"):
            InputValidator.validate_with_schema(data, 'unknown_schema')
    
    def test_validate_prompt_basic(self):
        """Test basic prompt validation."""
        prompt = "Generate a floral scent"
        result = InputValidator.validate_prompt(prompt)
        
        assert result == "Generate a floral scent"
    
    def test_validate_prompt_invalid_type(self):
        """Test prompt validation with invalid type."""
        with pytest.raises(ValidationError, match="Prompt must be a string"):
            InputValidator.validate_prompt(123)
    
    def test_validate_prompt_empty(self):
        """Test prompt validation with empty string."""
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            InputValidator.validate_prompt("")
        
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            InputValidator.validate_prompt("   ")
    
    def test_validate_prompt_too_long(self):
        """Test prompt validation with excessive length."""
        long_prompt = "a" * 1001
        
        with pytest.raises(ValidationError, match="Prompt too long"):
            InputValidator.validate_prompt(long_prompt)
    
    def test_validate_prompt_dangerous_content(self):
        """Test prompt validation with dangerous content."""
        dangerous_prompts = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "vbscript:msgbox(1)",
            "data:text/html,<script>",
            "onclick=alert(1)",
            "\\x41\\x42",
            "eval(malicious_code)",
            "import os",
            "__import__('os')"
        ]
        
        for prompt in dangerous_prompts:
            with pytest.raises(ValidationError, match="potentially harmful content"):
                InputValidator.validate_prompt(prompt)
    
    def test_validate_prompt_flagged_terms(self):
        """Test prompt validation with flagged terms."""
        flagged_prompts = [
            "Create a toxin for research",
            "Generate poison molecules",
            "Make explosive compounds"
        ]
        
        # Flagged terms should be logged but allowed
        for prompt in flagged_prompts:
            result = InputValidator.validate_prompt(prompt)
            assert isinstance(result, str)
    
    def test_validate_prompt_unusual_characters(self):
        """Test prompt validation with unusual characters."""
        prompt = "Generate scent with émojis 😀🌸"
        result = InputValidator.validate_prompt(prompt)
        
        # Should remove unusual characters but keep basic text
        assert "Generate scent with" in result
    
    def test_validate_smiles_basic(self):
        """Test basic SMILES validation."""
        smiles = "CCO"  # ethanol
        result = InputValidator.validate_smiles(smiles)
        
        assert result == "CCO"  # Should return canonical form
    
    def test_validate_smiles_invalid_type(self):
        """Test SMILES validation with invalid type."""
        with pytest.raises(ValidationError, match="SMILES must be a string"):
            InputValidator.validate_smiles(123)
    
    def test_validate_smiles_empty(self):
        """Test SMILES validation with empty string."""
        with pytest.raises(ValidationError, match="SMILES cannot be empty"):
            InputValidator.validate_smiles("")
    
    def test_validate_smiles_too_long(self):
        """Test SMILES validation with excessive length."""
        long_smiles = "C" * 501
        
        with pytest.raises(ValidationError, match="SMILES too long"):
            InputValidator.validate_smiles(long_smiles)
    
    def test_validate_smiles_invalid_characters(self):
        """Test SMILES validation with invalid characters."""
        invalid_smiles = "CC@#$%^&*()O"
        
        with pytest.raises(ValidationError, match="SMILES contains invalid characters"):
            InputValidator.validate_smiles(invalid_smiles)
    
    @patch('odordiff2.utils.validation.Chem.MolFromSmiles')
    @patch('odordiff2.utils.validation.Chem.MolToSmiles')
    def test_validate_smiles_rdkit_validation(self, mock_mol_to_smiles, mock_mol_from_smiles):
        """Test SMILES validation with RDKit."""
        mock_mol_from_smiles.return_value = Mock()
        mock_mol_to_smiles.return_value = "CCO"
        
        result = InputValidator.validate_smiles("CCO")
        
        assert result == "CCO"
        mock_mol_from_smiles.assert_called_once_with("CCO")
        mock_mol_to_smiles.assert_called_once()
    
    @patch('odordiff2.utils.validation.Chem.MolFromSmiles')
    def test_validate_smiles_rdkit_invalid_strict(self, mock_mol_from_smiles):
        """Test SMILES validation with invalid SMILES in strict mode."""
        mock_mol_from_smiles.return_value = None
        
        with pytest.raises(ValidationError, match="Invalid SMILES"):
            InputValidator.validate_smiles("INVALID", strict=True)
    
    @patch('odordiff2.utils.validation.Chem.MolFromSmiles')
    def test_validate_smiles_rdkit_invalid_non_strict(self, mock_mol_from_smiles):
        """Test SMILES validation with invalid SMILES in non-strict mode."""
        mock_mol_from_smiles.return_value = None
        
        result = InputValidator.validate_smiles("INVALID", strict=False)
        assert result == "INVALID"
    
    def test_validate_molecular_constraints_basic(self):
        """Test basic molecular constraints validation."""
        constraints = {
            "molecular_weight": 250.5,
            "logP": 2.5,
            "allergenic": False
        }
        
        result = InputValidator.validate_molecular_constraints(constraints)
        
        assert result["molecular_weight"] == 250.5
        assert result["logP"] == 2.5
        assert result["allergenic"] is False
    
    def test_validate_molecular_constraints_invalid_type(self):
        """Test constraints validation with invalid type."""
        with pytest.raises(ValidationError, match="Constraints must be a dictionary"):
            InputValidator.validate_molecular_constraints("not a dict")
    
    def test_validate_molecular_constraints_range_values(self):
        """Test constraints validation with range values."""
        constraints = {
            "molecular_weight": (150, 300),
            "logP": [1.0, 4.0]
        }
        
        result = InputValidator.validate_molecular_constraints(constraints)
        
        assert result["molecular_weight"] == (150, 300)
        assert result["logP"] == [1.0, 4.0]
    
    def test_validate_molecular_constraints_invalid_range(self):
        """Test constraints validation with invalid ranges."""
        # Min > Max
        constraints = {"molecular_weight": (300, 150)}
        
        with pytest.raises(ValidationError, match="Invalid range"):
            InputValidator.validate_molecular_constraints(constraints)
        
        # Out of bounds
        constraints = {"molecular_weight": (10, 2000)}
        
        with pytest.raises(ValidationError, match="outside valid bounds"):
            InputValidator.validate_molecular_constraints(constraints)
    
    def test_validate_molecular_constraints_boolean_property(self):
        """Test constraints validation with boolean properties."""
        constraints = {"allergenic": True, "biodegradable": False}
        
        result = InputValidator.validate_molecular_constraints(constraints)
        
        assert result["allergenic"] is True
        assert result["biodegradable"] is False
    
    def test_validate_molecular_constraints_invalid_boolean(self):
        """Test constraints validation with invalid boolean."""
        constraints = {"allergenic": "yes"}
        
        with pytest.raises(ValidationError, match="must be boolean"):
            InputValidator.validate_molecular_constraints(constraints)
    
    def test_validate_molecular_constraints_unknown_property(self):
        """Test constraints validation with unknown property."""
        constraints = {
            "molecular_weight": 200,
            "unknown_property": 123
        }
        
        result = InputValidator.validate_molecular_constraints(constraints)
        
        # Should ignore unknown properties
        assert "molecular_weight" in result
        assert "unknown_property" not in result
    
    def test_validate_generation_parameters_basic(self):
        """Test basic generation parameters validation."""
        result = InputValidator.validate_generation_parameters(
            num_molecules=10,
            temperature=1.5,
            top_k=100,
            max_length=200
        )
        
        assert result["num_molecules"] == 10
        assert result["temperature"] == 1.5
        assert result["top_k"] == 100
        assert result["max_length"] == 200
    
    def test_validate_generation_parameters_invalid_num_molecules(self):
        """Test generation parameters with invalid num_molecules."""
        with pytest.raises(ValidationError, match="num_molecules must be positive integer"):
            InputValidator.validate_generation_parameters(num_molecules=0)
        
        with pytest.raises(ValidationError, match="num_molecules too large"):
            InputValidator.validate_generation_parameters(num_molecules=200)
    
    def test_validate_generation_parameters_invalid_temperature(self):
        """Test generation parameters with invalid temperature."""
        with pytest.raises(ValidationError, match="temperature must be positive number"):
            InputValidator.validate_generation_parameters(temperature=0)
        
        with pytest.raises(ValidationError, match="temperature too high"):
            InputValidator.validate_generation_parameters(temperature=15)
    
    def test_validate_generation_parameters_invalid_top_k(self):
        """Test generation parameters with invalid top_k."""
        with pytest.raises(ValidationError, match="top_k must be positive integer"):
            InputValidator.validate_generation_parameters(top_k=0)
        
        with pytest.raises(ValidationError, match="top_k too large"):
            InputValidator.validate_generation_parameters(top_k=1500)
    
    def test_validate_generation_parameters_invalid_max_length(self):
        """Test generation parameters with invalid max_length."""
        with pytest.raises(ValidationError, match="max_length must be at least 10"):
            InputValidator.validate_generation_parameters(max_length=5)
        
        with pytest.raises(ValidationError, match="max_length too large"):
            InputValidator.validate_generation_parameters(max_length=600)
    
    def test_validate_safety_threshold_valid(self):
        """Test valid safety threshold validation."""
        assert InputValidator.validate_safety_threshold(0.5) == 0.5
        assert InputValidator.validate_safety_threshold(0) == 0.0
        assert InputValidator.validate_safety_threshold(1) == 1.0
        assert InputValidator.validate_safety_threshold(1.0) == 1.0
    
    def test_validate_safety_threshold_invalid_type(self):
        """Test safety threshold validation with invalid type."""
        with pytest.raises(ValidationError, match="Safety threshold must be numeric"):
            InputValidator.validate_safety_threshold("0.5")
    
    def test_validate_safety_threshold_out_of_range(self):
        """Test safety threshold validation with out-of-range values."""
        with pytest.raises(ValidationError, match="Safety threshold must be between 0 and 1"):
            InputValidator.validate_safety_threshold(-0.1)
        
        with pytest.raises(ValidationError, match="Safety threshold must be between 0 and 1"):
            InputValidator.validate_safety_threshold(1.1)
    
    def test_validate_file_path_basic(self):
        """Test basic file path validation."""
        path = "data/molecules.csv"
        result = InputValidator.validate_file_path(path)
        
        assert result == "data/molecules.csv"
    
    def test_validate_file_path_invalid_type(self):
        """Test file path validation with invalid type."""
        with pytest.raises(ValidationError, match="File path must be a string"):
            InputValidator.validate_file_path(123)
    
    def test_validate_file_path_directory_traversal(self):
        """Test file path validation with directory traversal."""
        with pytest.raises(ValidationError, match="directory traversal detected"):
            InputValidator.validate_file_path("../../../etc/passwd")
        
        with pytest.raises(ValidationError, match="directory traversal detected"):
            InputValidator.validate_file_path("/etc/passwd")
    
    def test_validate_file_path_allowed_extensions(self):
        """Test file path validation with allowed extensions."""
        # Valid extension
        result = InputValidator.validate_file_path("data.csv", ["csv", "json"])
        assert result == "data.csv"
        
        # Invalid extension
        with pytest.raises(ValidationError, match="File extension not allowed"):
            InputValidator.validate_file_path("data.txt", ["csv", "json"])
    
    def test_validate_api_key_valid(self):
        """Test valid API key validation."""
        valid_keys = [
            "a" * 32,
            "abcdefghijklmnopqrstuvwxyz123456",
            "ABC123_-" + "x" * 24,
        ]
        
        for key in valid_keys:
            assert InputValidator.validate_api_key(key) is True
    
    def test_validate_api_key_invalid(self):
        """Test invalid API key validation."""
        invalid_keys = [
            "short",
            123,
            None,
            "key with spaces",
            "test",
            "demo",
            "example",
            "contains@symbols#",
        ]
        
        for key in invalid_keys:
            assert InputValidator.validate_api_key(key) is False
    
    def test_validate_pagination_params_valid(self):
        """Test valid pagination parameters."""
        offset, limit = InputValidator.validate_pagination_params(10, 50)
        
        assert offset == 10
        assert limit == 50
    
    def test_validate_pagination_params_defaults(self):
        """Test pagination parameters with defaults."""
        offset, limit = InputValidator.validate_pagination_params()
        
        assert offset == 0
        assert limit == 50
    
    def test_validate_pagination_params_invalid_values(self):
        """Test pagination parameters with invalid values."""
        # Negative offset
        offset, limit = InputValidator.validate_pagination_params(-5, 25)
        assert offset == 0
        assert limit == 25
        
        # Zero or negative limit
        offset, limit = InputValidator.validate_pagination_params(0, 0)
        assert offset == 0
        assert limit == 50
        
        # Limit too high
        offset, limit = InputValidator.validate_pagination_params(0, 5000, max_limit=1000)
        assert offset == 0
        assert limit == 1000


class TestValidationMiddleware:
    """Test ValidationMiddleware class."""
    
    def test_middleware_initialization_default(self):
        """Test middleware initialization with defaults."""
        middleware = ValidationMiddleware()
        
        assert '/generate' in middleware.schema_mappings
        assert '/generate/batch' in middleware.schema_mappings
    
    def test_middleware_initialization_custom(self):
        """Test middleware initialization with custom mappings."""
        custom_mappings = {'/custom': 'custom_schema'}
        middleware = ValidationMiddleware(custom_mappings)
        
        assert middleware.schema_mappings == custom_mappings
    
    @patch.object(InputValidator, 'validate_request_data')
    def test_validate_request_with_schema(self, mock_validate):
        """Test request validation with mapped schema."""
        mock_validate.return_value = {"validated": True}
        middleware = ValidationMiddleware()
        
        data = {"prompt": "test"}
        result = middleware.validate_request('/generate', data)
        
        assert result == {"validated": True}
        mock_validate.assert_called_once_with(data, 'generation_request')
    
    @patch.object(Sanitizer, 'sanitize_json')
    def test_validate_request_without_schema(self, mock_sanitize):
        """Test request validation without mapped schema."""
        mock_sanitize.return_value = {"sanitized": True}
        middleware = ValidationMiddleware()
        
        data = {"test": "data"}
        result = middleware.validate_request('/unknown', data)
        
        assert result == {"sanitized": True}
        mock_sanitize.assert_called_once_with(data)


class TestValidateInputDecorator:
    """Test validate_input decorator."""
    
    def test_decorator_basic(self):
        """Test basic decorator functionality."""
        @validate_input()
        def test_function(value):
            return value * 2
        
        result = test_function(5)
        assert result == 10
    
    @patch.object(InputValidator, 'validate_prompt')
    def test_decorator_prompt_validation(self, mock_validate):
        """Test decorator with prompt validation."""
        mock_validate.return_value = "validated prompt"
        
        @validate_input()
        def test_function(prompt):
            return prompt.upper()
        
        result = test_function(prompt="test prompt")
        
        assert result == "VALIDATED PROMPT"
        mock_validate.assert_called_once_with("test prompt")
    
    @patch.object(InputValidator, 'validate_smiles')
    def test_decorator_smiles_validation(self, mock_validate):
        """Test decorator with SMILES validation."""
        mock_validate.return_value = "CCO"
        
        @validate_input()
        def test_function(smiles):
            return smiles
        
        result = test_function(smiles="CCO")
        
        assert result == "CCO"
        mock_validate.assert_called_once_with("CCO")
    
    def test_decorator_validation_error_strict(self):
        """Test decorator with validation error in strict mode."""
        @validate_input(strict=True)
        def test_function(prompt):
            return prompt
        
        with pytest.raises(ValidationError):
            test_function(prompt="")  # Empty prompt should fail
    
    def test_decorator_validation_error_non_strict(self):
        """Test decorator with validation error in non-strict mode."""
        @validate_input(strict=False)
        def test_function(prompt):
            return prompt
        
        # Should not raise exception in non-strict mode
        result = test_function(prompt="")
        assert result == ""


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch.object(Sanitizer, 'sanitize_json')
    def test_sanitize_user_input(self, mock_sanitize):
        """Test sanitize_user_input convenience function."""
        mock_sanitize.return_value = {"sanitized": True}
        
        result = sanitize_user_input({"test": "data"})
        
        assert result == {"sanitized": True}
        mock_sanitize.assert_called_once_with({"test": "data"})
    
    @patch.object(InputValidator, 'validate_request_data')
    def test_validate_generation_request(self, mock_validate):
        """Test validate_generation_request convenience function."""
        mock_validate.return_value = {"validated": True}
        
        data = {"prompt": "test"}
        result = validate_generation_request(data)
        
        assert result == {"validated": True}
        mock_validate.assert_called_once_with(data, 'generation_request')
    
    @patch.object(InputValidator, 'validate_request_data')
    def test_validate_batch_request(self, mock_validate):
        """Test validate_batch_request convenience function."""
        mock_validate.return_value = {"validated": True}
        
        data = {"prompts": ["test1", "test2"]}
        result = validate_batch_request(data)
        
        assert result == {"validated": True}
        mock_validate.assert_called_once_with(data, 'batch_request')
    
    def test_is_safe_filename_safe(self):
        """Test is_safe_filename with safe filename."""
        assert is_safe_filename("test_file.txt") is True
        assert is_safe_filename("document.pdf") is True
    
    def test_is_safe_filename_unsafe(self):
        """Test is_safe_filename with unsafe filename."""
        assert is_safe_filename("../../../etc/passwd") is False
        assert is_safe_filename("con.txt") is False
        assert is_safe_filename("file<>name.txt") is False
    
    def test_is_valid_ip_valid(self):
        """Test is_valid_ip with valid addresses."""
        assert is_valid_ip("127.0.0.1") is True
        assert is_valid_ip("192.168.1.1") is True
        assert is_valid_ip("::1") is True
    
    def test_is_valid_ip_invalid(self):
        """Test is_valid_ip with invalid addresses."""
        assert is_valid_ip("256.256.256.256") is False
        assert is_valid_ip("not.an.ip") is False
        assert is_valid_ip("") is False


class TestSchemas:
    """Test JSON schema definitions."""
    
    def test_schemas_defined(self):
        """Test that schemas are defined."""
        assert 'generation_request' in SCHEMAS
        assert 'molecule_constraints' in SCHEMAS
        assert 'batch_request' in SCHEMAS
    
    def test_generation_request_schema_structure(self):
        """Test generation_request schema structure."""
        schema = SCHEMAS['generation_request']
        
        assert schema['type'] == 'object'
        assert 'prompt' in schema['properties']
        assert 'num_molecules' in schema['properties']
        assert 'safety_threshold' in schema['properties']
        assert 'prompt' in schema['required']
    
    def test_molecule_constraints_schema_structure(self):
        """Test molecule_constraints schema structure."""
        schema = SCHEMAS['molecule_constraints']
        
        assert schema['type'] == 'object'
        assert 'molecular_weight' in schema['properties']
        assert 'logP' in schema['properties']
        assert 'allergenic' in schema['properties']
    
    def test_batch_request_schema_structure(self):
        """Test batch_request schema structure."""
        schema = SCHEMAS['batch_request']
        
        assert schema['type'] == 'object'
        assert 'prompts' in schema['properties']
        assert 'num_molecules' in schema['properties']
        assert 'prompts' in schema['required']


class TestValidationIntegration:
    """Integration tests for validation system."""
    
    def test_full_generation_request_validation(self):
        """Test complete generation request validation flow."""
        data = {
            "prompt": "  Generate a fresh citrus scent  ",
            "num_molecules": 5,
            "safety_threshold": 0.1,
            "use_cache": True
        }
        
        # Should pass validation and sanitization
        result = validate_generation_request(data)
        
        assert isinstance(result, dict)
        assert result["prompt"].strip() == "Generate a fresh citrus scent"
        assert result["num_molecules"] == 5
        assert result["safety_threshold"] == 0.1
        assert result["use_cache"] is True
    
    def test_full_batch_request_validation(self):
        """Test complete batch request validation flow."""
        data = {
            "prompts": [
                "  floral scent  ",
                "citrus fragrance",
                "woody aroma"
            ],
            "num_molecules": 3,
            "priority": 1
        }
        
        # Should pass validation and sanitization
        result = validate_batch_request(data)
        
        assert isinstance(result, dict)
        assert len(result["prompts"]) == 3
        assert all(isinstance(p, str) for p in result["prompts"])
        assert result["num_molecules"] == 3
        assert result["priority"] == 1
    
    def test_security_validation_pipeline(self):
        """Test security validation pipeline."""
        dangerous_data = {
            "prompt": "<script>alert('xss')</script>Generate scent",
            "num_molecules": 5,
            "malicious_field": "'; DROP TABLE users; --"
        }
        
        # Should sanitize dangerous content
        with pytest.raises(ValidationError):
            # Dangerous script should be blocked
            validate_generation_request(dangerous_data)