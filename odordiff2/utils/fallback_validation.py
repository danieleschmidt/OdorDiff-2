"""
Fallback validation system for environments without full dependencies.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    confidence: float = 1.0
    
    def __bool__(self) -> bool:
        return self.is_valid


class FallbackValidator:
    """Lightweight validation without external dependencies."""
    
    def __init__(self):
        # Common SMILES patterns
        self.valid_atoms = set([
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'F', 'Cl', 'Br', 'I',
            'B', 'Si', 'Se', 'As', 'Mg', 'Ca', 'Na', 'K', 'H'
        ])
        
        self.valid_characters = set('CNOSPFBrClIHBSiSeAsMgCaNaK[]()=+\\-#@/1234567890cnops')
        
        # Dangerous patterns to filter out
        self.dangerous_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'<iframe.*?>',
            r'<object.*?>',
            r'<embed.*?>',
        ]
    
    def validate_smiles(self, smiles: str) -> ValidationResult:
        """Validate SMILES string using basic rules."""
        if not smiles or not isinstance(smiles, str):
            return ValidationResult(False, ['Empty or invalid SMILES'], [])
        
        errors = []
        warnings = []
        
        # Basic character validation
        invalid_chars = set(smiles) - self.valid_characters
        if invalid_chars:
            errors.append(f"Invalid characters: {', '.join(invalid_chars)}")
        
        # Bracket matching
        if not self._check_brackets(smiles):
            errors.append("Unmatched brackets in SMILES")
        
        # Basic structural checks
        if smiles.count('=') > len(smiles) // 3:
            warnings.append("Unusually high number of double bonds")
        
        if smiles.count('#') > len(smiles) // 5:
            warnings.append("Unusually high number of triple bonds")
        
        # Ring closure validation
        if not self._check_ring_closures(smiles):
            errors.append("Invalid ring closure numbers")
        
        # Length validation
        if len(smiles) > 200:
            warnings.append("Very long SMILES string")
        elif len(smiles) < 2:
            errors.append("SMILES string too short")
        
        confidence = 0.8 if warnings else 0.95
        return ValidationResult(len(errors) == 0, errors, warnings, confidence)
    
    def _check_brackets(self, smiles: str) -> bool:
        """Check if brackets are properly matched."""
        stack = []
        bracket_pairs = {'(': ')', '[': ']'}
        
        for char in smiles:
            if char in bracket_pairs:
                stack.append(char)
            elif char in bracket_pairs.values():
                if not stack:
                    return False
                last = stack.pop()
                if bracket_pairs.get(last) != char:
                    return False
        
        return len(stack) == 0
    
    def _check_ring_closures(self, smiles: str) -> bool:
        """Check ring closure number validity."""
        import re
        
        # Find all ring closure numbers
        ring_numbers = re.findall(r'[0-9]', smiles)
        
        # Each ring number should appear exactly twice
        for num in set(ring_numbers):
            if ring_numbers.count(num) != 2:
                return False
        
        return True
    
    def validate_molecule_properties(self, properties: Dict[str, Any]) -> ValidationResult:
        """Validate molecular properties."""
        errors = []
        warnings = []
        
        # Molecular weight validation
        mw = properties.get('molecular_weight', 0)
        if mw and (mw < 10 or mw > 1000):
            if mw < 10:
                errors.append("Molecular weight too low")
            else:
                warnings.append("High molecular weight")
        
        # LogP validation
        logp = properties.get('logP', 0)
        if logp and (logp < -5 or logp > 10):
            warnings.append("LogP value outside typical range")
        
        # TPSA validation
        tpsa = properties.get('tpsa', 0)
        if tpsa and tpsa > 200:
            warnings.append("High topological polar surface area")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text for security."""
        if not isinstance(text, str):
            return str(text)
        
        # Remove dangerous patterns
        sanitized = text
        for pattern in self.dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
        
        return sanitized.strip()
    
    def validate_prompt(self, prompt: str) -> ValidationResult:
        """Validate user prompt for molecule generation."""
        if not prompt or not isinstance(prompt, str):
            return ValidationResult(False, ['Empty prompt'], [])
        
        errors = []
        warnings = []
        
        # Length validation
        if len(prompt) < 3:
            errors.append("Prompt too short")
        elif len(prompt) > 500:
            warnings.append("Very long prompt - may affect generation quality")
        
        # Content validation
        sanitized = self.sanitize_input(prompt)
        if len(sanitized) < len(prompt) * 0.8:
            warnings.append("Prompt contained potentially unsafe content")
        
        # Check for reasonable content
        if not any(c.isalpha() for c in sanitized):
            errors.append("Prompt must contain alphabetic characters")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


# Global validator instance
_validator = None

def get_validator() -> FallbackValidator:
    """Get global validator instance."""
    global _validator
    if _validator is None:
        _validator = FallbackValidator()
    return _validator


# Convenience functions
def validate_smiles(smiles: str) -> ValidationResult:
    """Validate SMILES string."""
    return get_validator().validate_smiles(smiles)


def validate_molecule_properties(properties: Dict[str, Any]) -> ValidationResult:
    """Validate molecular properties."""
    return get_validator().validate_molecule_properties(properties)


def sanitize_input(text: str) -> str:
    """Sanitize input text."""
    return get_validator().sanitize_input(text)


def validate_prompt(prompt: str) -> ValidationResult:
    """Validate generation prompt."""
    return get_validator().validate_prompt(prompt)