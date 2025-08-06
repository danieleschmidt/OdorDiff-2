"""
Input validation and sanitization utilities.
"""

import re
from typing import Any, Dict, List, Optional, Union, Tuple
from rdkit import Chem
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class InputValidator:
    """Comprehensive input validation for OdorDiff-2."""
    
    @staticmethod
    def validate_prompt(prompt: str) -> str:
        """
        Validate and sanitize text prompts.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Sanitized prompt
            
        Raises:
            ValidationError: If prompt is invalid
        """
        if not isinstance(prompt, str):
            raise ValidationError("Prompt must be a string")
            
        # Remove leading/trailing whitespace
        prompt = prompt.strip()
        
        # Check length
        if len(prompt) == 0:
            raise ValidationError("Prompt cannot be empty")
        if len(prompt) > 1000:
            raise ValidationError("Prompt too long (max 1000 characters)")
            
        # Check for potentially harmful content
        dangerous_patterns = [
            r'<script[^>]*>',  # Script tags
            r'javascript:',    # JavaScript URLs
            r'vbscript:',      # VBScript URLs
            r'on\w+\s*=',      # Event handlers
            r'\\x[0-9a-fA-F]{2}',  # Hex escapes
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise ValidationError("Prompt contains potentially harmful content")
                
        # Basic profanity filter (simplified)
        profanity_words = ['toxin', 'poison', 'explosive', 'dangerous', 'harmful', 'illegal']
        prompt_lower = prompt.lower()
        for word in profanity_words:
            if word in prompt_lower:
                logger.warning(f"Prompt contains flagged word: {word}", prompt=prompt)
                
        return prompt
    
    @staticmethod
    def validate_smiles(smiles: str, strict: bool = True) -> str:
        """
        Validate SMILES string.
        
        Args:
            smiles: SMILES string to validate
            strict: If True, raise error for invalid SMILES
            
        Returns:
            Canonical SMILES
            
        Raises:
            ValidationError: If SMILES is invalid and strict=True
        """
        if not isinstance(smiles, str):
            raise ValidationError("SMILES must be a string")
            
        smiles = smiles.strip()
        
        if len(smiles) == 0:
            raise ValidationError("SMILES cannot be empty")
        if len(smiles) > 500:
            raise ValidationError("SMILES too long (max 500 characters)")
            
        # Check for dangerous characters
        allowed_chars = set('CNOSPFClBrI[]()=#+-.0123456789@/')
        if not set(smiles).issubset(allowed_chars):
            invalid_chars = set(smiles) - allowed_chars
            raise ValidationError(f"SMILES contains invalid characters: {invalid_chars}")
            
        # Validate with RDKit
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if strict:
                    raise ValidationError(f"Invalid SMILES: {smiles}")
                return smiles
                
            # Return canonical SMILES
            canonical = Chem.MolToSmiles(mol)
            return canonical
            
        except Exception as e:
            if strict:
                raise ValidationError(f"SMILES validation failed: {str(e)}")
            return smiles
    
    @staticmethod
    def validate_molecular_constraints(constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate molecular property constraints.
        
        Args:
            constraints: Dictionary of molecular constraints
            
        Returns:
            Validated constraints
            
        Raises:
            ValidationError: If constraints are invalid
        """
        if not isinstance(constraints, dict):
            raise ValidationError("Constraints must be a dictionary")
            
        valid_properties = {
            'molecular_weight': (50, 1000),
            'logP': (-5, 10),
            'tpsa': (0, 300),
            'hbd': (0, 20),
            'hba': (0, 20),
            'rotatable_bonds': (0, 30),
            'aromatic_rings': (0, 10),
            'vapor_pressure': (0.001, 100),
            'allergenic': (bool, bool),
            'biodegradable': (bool, bool)
        }
        
        validated = {}
        
        for prop, value in constraints.items():
            if prop not in valid_properties:
                logger.warning(f"Unknown constraint property: {prop}")
                continue
                
            prop_range = valid_properties[prop]
            
            # Handle boolean properties
            if prop_range == (bool, bool):
                if not isinstance(value, bool):
                    raise ValidationError(f"Property {prop} must be boolean")
                validated[prop] = value
                continue
                
            # Handle numeric ranges
            if isinstance(value, (tuple, list)) and len(value) == 2:
                min_val, max_val = value
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                    raise ValidationError(f"Range values for {prop} must be numeric")
                if min_val > max_val:
                    raise ValidationError(f"Invalid range for {prop}: min > max")
                if min_val < prop_range[0] or max_val > prop_range[1]:
                    raise ValidationError(f"Range for {prop} outside valid bounds {prop_range}")
                validated[prop] = (min_val, max_val)
                
            elif isinstance(value, (int, float)):
                if value < prop_range[0] or value > prop_range[1]:
                    raise ValidationError(f"Value for {prop} outside valid bounds {prop_range}")
                validated[prop] = value
                
            else:
                raise ValidationError(f"Invalid value type for {prop}: must be number or range")
                
        return validated
    
    @staticmethod
    def validate_generation_parameters(
        num_molecules: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
        max_length: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate molecule generation parameters.
        
        Returns:
            Validated parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate num_molecules
        if not isinstance(num_molecules, int) or num_molecules < 1:
            raise ValidationError("num_molecules must be positive integer")
        if num_molecules > 100:
            raise ValidationError("num_molecules too large (max 100)")
            
        # Validate temperature
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValidationError("temperature must be positive number")
        if temperature > 10:
            raise ValidationError("temperature too high (max 10)")
            
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1:
            raise ValidationError("top_k must be positive integer")
        if top_k > 1000:
            raise ValidationError("top_k too large (max 1000)")
            
        # Validate max_length
        if not isinstance(max_length, int) or max_length < 10:
            raise ValidationError("max_length must be at least 10")
        if max_length > 500:
            raise ValidationError("max_length too large (max 500)")
            
        return {
            'num_molecules': num_molecules,
            'temperature': temperature,
            'top_k': top_k,
            'max_length': max_length,
            **{k: v for k, v in kwargs.items() if k in ['safety_filter', 'synthesizability_min']}
        }
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for safe filesystem operations.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            raise ValidationError("Filename must be a string")
            
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\.\.+', '.', filename)  # Prevent directory traversal
        filename = filename.strip('. ')  # Remove leading/trailing dots and spaces
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
            
        # Ensure not empty
        if not filename:
            filename = "unnamed"
            
        return filename
    
    @staticmethod
    def validate_safety_threshold(threshold: float) -> float:
        """
        Validate safety threshold value.
        
        Args:
            threshold: Safety threshold (0-1)
            
        Returns:
            Validated threshold
            
        Raises:
            ValidationError: If threshold is invalid
        """
        if not isinstance(threshold, (int, float)):
            raise ValidationError("Safety threshold must be numeric")
        if not 0 <= threshold <= 1:
            raise ValidationError("Safety threshold must be between 0 and 1")
        return float(threshold)
    
    @staticmethod
    def validate_file_path(filepath: str, allowed_extensions: List[str] = None) -> str:
        """
        Validate file path for security.
        
        Args:
            filepath: File path to validate
            allowed_extensions: List of allowed file extensions
            
        Returns:
            Validated file path
            
        Raises:
            ValidationError: If path is invalid
        """
        if not isinstance(filepath, str):
            raise ValidationError("File path must be a string")
            
        # Check for directory traversal attempts
        if '..' in filepath or filepath.startswith('/'):
            raise ValidationError("Invalid file path: directory traversal detected")
            
        # Check extension if specified
        if allowed_extensions:
            extension = filepath.lower().split('.')[-1] if '.' in filepath else ''
            if extension not in [ext.lower() for ext in allowed_extensions]:
                raise ValidationError(f"File extension not allowed. Allowed: {allowed_extensions}")
                
        return filepath


def validate_input(validation_func):
    """Decorator to automatically validate function inputs."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Apply validation based on function name or parameters
                if 'prompt' in kwargs:
                    kwargs['prompt'] = InputValidator.validate_prompt(kwargs['prompt'])
                if 'smiles' in kwargs:
                    kwargs['smiles'] = InputValidator.validate_smiles(kwargs['smiles'])
                if 'constraints' in kwargs and kwargs['constraints']:
                    kwargs['constraints'] = InputValidator.validate_molecular_constraints(kwargs['constraints'])
                    
                return func(*args, **kwargs)
                
            except ValidationError as e:
                logger.error(f"Validation error in {func.__name__}: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                raise
                
        return wrapper
    return decorator