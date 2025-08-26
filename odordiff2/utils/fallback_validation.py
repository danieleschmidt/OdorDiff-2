"""
Comprehensive validation system for OdorDiff-2.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

class ComprehensiveValidator:
    """Comprehensive validation for all system inputs and outputs."""
    
    def __init__(self):
        self.smiles_pattern = re.compile(r'^[A-Za-z0-9@+\-\[\]\(\)=#:]+$')
        
    def validate_molecule_data(self, smiles: str, confidence: float = None) -> ValidationResult:
        """Validate molecule data comprehensively."""
        errors = []
        warnings = []
        
        # SMILES validation
        if not isinstance(smiles, str):
            errors.append(f"SMILES must be string, got {type(smiles)}")
        elif not smiles.strip():
            errors.append("SMILES cannot be empty")
        elif len(smiles) > 1000:
            errors.append("SMILES too long (>1000 characters)")
        elif not self.smiles_pattern.match(smiles):
            errors.append(f"Invalid SMILES characters: {smiles}")
        
        # Confidence validation
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                errors.append(f"Confidence must be numeric, got {type(confidence)}")
            elif not 0.0 <= confidence <= 1.0:
                errors.append(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            elif confidence < 0.5:
                warnings.append(f"Low confidence score: {confidence}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_generation_prompt(self, prompt: str) -> ValidationResult:
        """Validate text prompt for molecule generation."""
        errors = []
        warnings = []
        
        if not isinstance(prompt, str):
            errors.append(f"Prompt must be string, got {type(prompt)}")
        elif not prompt.strip():
            errors.append("Prompt cannot be empty")
        elif len(prompt) < 5:
            warnings.append("Very short prompt may not generate good results")
        elif len(prompt) > 500:
            warnings.append("Very long prompt may be truncated")
        
        # Check for potentially harmful content
        harmful_keywords = ['toxic', 'explosive', 'poison', 'dangerous', 'illegal']
        if any(keyword in prompt.lower() for keyword in harmful_keywords):
            errors.append("Prompt contains potentially harmful content")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_safety_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate safety filter parameters."""
        errors = []
        warnings = []
        
        required_params = ['toxicity_threshold']
        for param in required_params:
            if param not in params:
                errors.append(f"Missing required safety parameter: {param}")
        
        if 'toxicity_threshold' in params:
            threshold = params['toxicity_threshold']
            if not isinstance(threshold, (int, float)):
                errors.append(f"Toxicity threshold must be numeric, got {type(threshold)}")
            elif not 0.0 <= threshold <= 1.0:
                errors.append(f"Toxicity threshold must be between 0.0 and 1.0")
            elif threshold > 0.5:
                warnings.append("High toxicity threshold may allow unsafe molecules")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_synthesis_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate synthesis planning parameters."""
        errors = []
        warnings = []
        
        if 'max_steps' in params:
            max_steps = params['max_steps']
            if not isinstance(max_steps, int):
                errors.append(f"max_steps must be integer, got {type(max_steps)}")
            elif max_steps <= 0:
                errors.append("max_steps must be positive")
            elif max_steps > 10:
                warnings.append("High max_steps may result in complex synthesis")
        
        if 'starting_materials' in params:
            materials = params['starting_materials']
            if materials not in ['commercial', 'basic', 'any']:
                errors.append(f"Invalid starting_materials: {materials}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

# Global validator instance
validator = ComprehensiveValidator()

def validate_and_log(data: Any, validation_func: callable, operation_name: str) -> ValidationResult:
    """Validate data and log results."""
    result = validation_func(data)
    
    if result.has_errors:
        logger.error(f"{operation_name} validation failed: {result.errors}")
    
    if result.has_warnings:
        logger.warning(f"{operation_name} validation warnings: {result.warnings}")
    
    if result.is_valid:
        logger.info(f"{operation_name} validation passed")
    
    return result
