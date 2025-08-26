"""
Advanced error handling and recovery utilities for OdorDiff-2.
"""

import logging
import functools
import traceback
from typing import Any, Callable, Optional, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class OdorDiffException(Exception):
    """Base exception for OdorDiff system."""
    pass

class MoleculeGenerationError(OdorDiffException):
    """Error during molecule generation."""
    pass

class SafetyValidationError(OdorDiffException):
    """Error during safety validation."""
    pass

class SynthesisError(OdorDiffException):
    """Error during synthesis planning."""
    pass

def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Any, Optional[str]]:
    """Safely execute function with comprehensive error handling."""
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        error_msg = f"Error in {func.__name__}: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return False, None, error_msg

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final retry failed for {func.__name__}: {e}")
                        raise
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying...")
                        time.sleep(delay)
            
        return wrapper
    return decorator

@contextmanager
def error_context(operation_name: str):
    """Context manager for operation-specific error handling."""
    try:
        logger.info(f"Starting operation: {operation_name}")
        yield
        logger.info(f"Completed operation: {operation_name}")
    except Exception as e:
        logger.error(f"Operation '{operation_name}' failed: {e}")
        logger.debug(traceback.format_exc())
        raise

def validate_input(validation_func: Callable[[Any], bool], error_message: str):
    """Decorator for input validation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate all arguments
            for arg in args:
                if not validation_func(arg):
                    raise ValueError(f"{error_message}: {arg}")
            
            # Validate keyword arguments
            for key, value in kwargs.items():
                if not validation_func(value):
                    raise ValueError(f"{error_message} in {key}: {value}")
                    
            return func(*args, **kwargs)
        return wrapper
    return decorator

class RobustMoleculeValidator:
    """Enhanced molecule validation with detailed error reporting."""
    
    @staticmethod
    def validate_smiles(smiles: str) -> tuple[bool, Optional[str]]:
        """Validate SMILES string with detailed error reporting."""
        if not isinstance(smiles, str):
            return False, f"SMILES must be string, got {type(smiles)}"
        
        if not smiles.strip():
            return False, "SMILES cannot be empty"
        
        # Check for basic SMILES characters
        valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]=-+#@.:')
        invalid_chars = set(smiles) - valid_chars
        
        if invalid_chars:
            return False, f"Invalid characters in SMILES: {invalid_chars}"
        
        # Check for balanced parentheses
        paren_count = smiles.count('(') - smiles.count(')')
        bracket_count = smiles.count('[') - smiles.count(']')
        
        if paren_count != 0:
            return False, f"Unbalanced parentheses in SMILES: {paren_count}"
        
        if bracket_count != 0:
            return False, f"Unbalanced brackets in SMILES: {bracket_count}"
        
        return True, None
    
    @staticmethod
    def validate_generation_parameters(params: dict) -> tuple[bool, Optional[str]]:
        """Validate molecule generation parameters."""
        required_params = ['prompt', 'num_molecules']
        
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        if not isinstance(params['prompt'], str) or not params['prompt'].strip():
            return False, "Prompt must be non-empty string"
        
        if not isinstance(params['num_molecules'], int) or params['num_molecules'] <= 0:
            return False, "num_molecules must be positive integer"
        
        if params['num_molecules'] > 100:
            return False, "num_molecules cannot exceed 100 for safety"
        
        return True, None
