"""
Enhanced input validation and sanitization utilities with JSON schema support.
"""

import re
import json
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime
import ipaddress

try:
    import jsonschema
    from jsonschema import validate, ValidationError as JSONSchemaError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    JSONSchemaError = Exception

from rdkit import Chem
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


class SanitizationError(Exception):
    """Exception raised when sanitization fails."""
    pass


# JSON Schema definitions for common data structures
SCHEMAS = {
    'generation_request': {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1000,
                "pattern": "^[\\w\\s\\-.,!?;:()\\[\\]{}'\"/]+$"
            },
            "num_molecules": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100
            },
            "safety_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "synthesizability_min": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "use_cache": {
                "type": "boolean"
            }
        },
        "required": ["prompt"],
        "additionalProperties": False
    },
    
    'molecule_constraints': {
        "type": "object",
        "properties": {
            "molecular_weight": {
                "oneOf": [
                    {"type": "number", "minimum": 50, "maximum": 1000},
                    {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
                ]
            },
            "logP": {
                "oneOf": [
                    {"type": "number", "minimum": -5, "maximum": 10},
                    {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
                ]
            },
            "tpsa": {
                "oneOf": [
                    {"type": "number", "minimum": 0, "maximum": 300},
                    {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
                ]
            },
            "hbd": {
                "oneOf": [
                    {"type": "integer", "minimum": 0, "maximum": 20},
                    {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}
                ]
            },
            "hba": {
                "oneOf": [
                    {"type": "integer", "minimum": 0, "maximum": 20},
                    {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}
                ]
            },
            "allergenic": {"type": "boolean"},
            "biodegradable": {"type": "boolean"}
        },
        "additionalProperties": False
    },
    
    'batch_request': {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 1000},
                "minItems": 1,
                "maxItems": 50
            },
            "num_molecules": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20
            },
            "safety_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "priority": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10
            }
        },
        "required": ["prompts"],
        "additionalProperties": False
    }
}


class Sanitizer:
    """Advanced sanitization utilities."""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """
        Comprehensive text sanitization.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            raise SanitizationError("Input must be a string")
        
        # Strip whitespace
        text = text.strip()
        
        # Length check
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        # Remove null bytes and control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        if not allow_html:
            # HTML escape
            text = html.escape(text)
            
            # Remove potentially dangerous patterns
            dangerous_patterns = [
                r'javascript\s*:',
                r'vbscript\s*:',
                r'data\s*:',
                r'on\w+\s*=',
                r'<\s*script[^>]*>',
                r'<\s*iframe[^>]*>',
                r'<\s*object[^>]*>',
                r'<\s*embed[^>]*>',
                r'<\s*link[^>]*>',
                r'<\s*meta[^>]*>',
                r'\\x[0-9a-fA-F]{2}',
                r'\\u[0-9a-fA-F]{4}',
            ]
            
            for pattern in dangerous_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # URL decode to prevent encoding attacks
        try:
            decoded = urllib.parse.unquote(text)
            if decoded != text:
                logger.warning("URL-encoded content detected and decoded")
                text = decoded
        except:
            pass
        
        return text
    
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
            raise SanitizationError("Filename must be a string")
        
        # Remove dangerous characters and patterns
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        filename = re.sub(r'\.\.+', '.', filename)  # Prevent directory traversal
        filename = re.sub(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)', '_\\1\\2', filename, flags=re.IGNORECASE)
        
        # Strip problematic characters from start/end
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 200:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = 200 - len(ext) - 1 if ext else 200
            filename = name[:max_name_length] + ('.' + ext if ext else '')
        
        # Ensure not empty
        if not filename:
            filename = "unnamed"
        
        return filename
    
    @staticmethod
    def sanitize_json(data: Any, max_depth: int = 10, max_items: int = 1000) -> Any:
        """
        Recursively sanitize JSON data.
        
        Args:
            data: JSON data to sanitize
            max_depth: Maximum nesting depth
            max_items: Maximum number of items in collections
            
        Returns:
            Sanitized data
        """
        def _sanitize_recursive(obj, depth=0):
            if depth > max_depth:
                raise SanitizationError(f"Maximum nesting depth ({max_depth}) exceeded")
            
            if isinstance(obj, dict):
                if len(obj) > max_items:
                    raise SanitizationError(f"Dictionary too large (max {max_items} items)")
                return {
                    Sanitizer.sanitize_text(str(k), 100): _sanitize_recursive(v, depth + 1)
                    for k, v in obj.items()
                }
            elif isinstance(obj, (list, tuple)):
                if len(obj) > max_items:
                    raise SanitizationError(f"Array too large (max {max_items} items)")
                return [_sanitize_recursive(item, depth + 1) for item in obj]
            elif isinstance(obj, str):
                return Sanitizer.sanitize_text(obj)
            elif isinstance(obj, (int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        return _sanitize_recursive(data)
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """
        Validate IP address format.
        
        Args:
            ip: IP address string
            
        Returns:
            True if valid IP address
        """
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def sanitize_sql_input(text: str) -> str:
        """
        Basic SQL injection prevention (use parameterized queries instead).
        
        Args:
            text: Input text
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return str(text)
        
        # Remove common SQL injection patterns
        sql_patterns = [
            r"';.*--",
            r'";\s*--',
            r"';\s*#",
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+set',
            r'exec\s*\(',
            r'execute\s*\(',
            r'sp_\w+',
            r'xp_\w+',
        ]
        
        for pattern in sql_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text


class InputValidator:
    """Comprehensive input validation for OdorDiff-2."""
    
    @staticmethod
    def validate_with_schema(data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """
        Validate data against JSON schema.
        
        Args:
            data: Data to validate
            schema_name: Name of schema to use
            
        Returns:
            Validated data
            
        Raises:
            ValidationError: If validation fails
        """
        if not JSONSCHEMA_AVAILABLE:
            logger.warning("jsonschema not available, skipping schema validation")
            return data
        
        if schema_name not in SCHEMAS:
            raise ValidationError(f"Unknown schema: {schema_name}")
        
        try:
            validate(instance=data, schema=SCHEMAS[schema_name])
            return data
        except JSONSchemaError as e:
            raise ValidationError(f"Schema validation failed: {e.message}", field=e.path)
    
    @staticmethod
    def validate_prompt(prompt: str) -> str:
        """
        Validate and sanitize text prompts with enhanced security.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Sanitized prompt
            
        Raises:
            ValidationError: If prompt is invalid
        """
        if not isinstance(prompt, str):
            raise ValidationError("Prompt must be a string", field="prompt", value=type(prompt).__name__)
            
        # Sanitize first
        try:
            prompt = Sanitizer.sanitize_text(prompt, max_length=1000, allow_html=False)
        except SanitizationError as e:
            raise ValidationError(f"Prompt sanitization failed: {str(e)}", field="prompt")
        
        # Check length after sanitization
        if len(prompt) == 0:
            raise ValidationError("Prompt cannot be empty", field="prompt")
        if len(prompt) > 1000:
            raise ValidationError("Prompt too long (max 1000 characters)", field="prompt", value=len(prompt))
        
        # Enhanced content filtering
        dangerous_patterns = [
            (r'<script[^>]*>.*?</script>', 'script tags'),
            (r'javascript\s*:', 'javascript URLs'),
            (r'vbscript\s*:', 'vbscript URLs'),
            (r'data\s*:', 'data URLs'),
            (r'on\w+\s*=', 'event handlers'),
            (r'\\x[0-9a-fA-F]{2}', 'hex escapes'),
            (r'\\u[0-9a-fA-F]{4}', 'unicode escapes'),
            (r'eval\s*\(', 'eval function'),
            (r'document\s*\.', 'DOM access'),
            (r'window\s*\.', 'window object access'),
            (r'\bexec\b', 'exec statements'),
            (r'import\s+\w+', 'import statements'),
            (r'__\w+__', 'python magic methods'),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.warning(f"Blocked prompt with {description}", prompt_preview=prompt[:50])
                raise ValidationError(f"Prompt contains potentially harmful content: {description}", 
                                    field="prompt")
        
        # Content policy filter (enhanced)
        flagged_terms = {
            'toxin': 'toxic substances',
            'poison': 'poisonous substances',
            'explosive': 'explosive materials',
            'dangerous': 'dangerous substances',
            'harmful': 'harmful materials',
            'illegal': 'illegal substances',
            'drug': 'controlled substances',
            'narcotic': 'narcotic substances',
            'radioactive': 'radioactive materials',
            'carcinogen': 'carcinogenic substances',
            'pathogen': 'pathogenic materials',
        }
        
        prompt_lower = prompt.lower()
        flagged_words = []
        
        for term, description in flagged_terms.items():
            if term in prompt_lower:
                flagged_words.append(term)
                logger.warning(f"Prompt contains flagged term: {term} ({description})", 
                              prompt_preview=prompt[:50])
        
        # Allow prompts with flagged words but log them
        if flagged_words:
            logger.warning(f"Processed prompt with flagged terms: {flagged_words}", 
                          prompt_hash=hash(prompt))
        
        # Basic character set validation
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                           ' .,!?;:()[]{}\'"-_/\\@#$%^&*+=<>|~`\n\r\t')
        invalid_chars = set(prompt) - allowed_chars
        
        if invalid_chars:
            logger.warning(f"Prompt contains unusual characters: {invalid_chars}")
            # Remove invalid characters rather than rejecting
            prompt = ''.join(c for c in prompt if c in allowed_chars)
        
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


    @staticmethod
    def validate_request_data(data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """
        Comprehensive request data validation.
        
        Args:
            data: Request data
            schema_name: Schema to validate against
            
        Returns:
            Validated and sanitized data
        """
        # First sanitize the entire data structure
        try:
            sanitized_data = Sanitizer.sanitize_json(data)
        except SanitizationError as e:
            raise ValidationError(f"Data sanitization failed: {str(e)}")
        
        # Then validate against schema
        validated_data = InputValidator.validate_with_schema(sanitized_data, schema_name)
        
        # Additional custom validation based on schema type
        if schema_name == 'generation_request':
            if 'prompt' in validated_data:
                validated_data['prompt'] = InputValidator.validate_prompt(validated_data['prompt'])
        
        elif schema_name == 'batch_request':
            if 'prompts' in validated_data:
                validated_data['prompts'] = [
                    InputValidator.validate_prompt(p) for p in validated_data['prompts']
                ]
        
        return validated_data
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validate API key format and basic security.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid format
        """
        if not isinstance(api_key, str):
            return False
        
        # Basic format check - should be 32+ alphanumeric chars
        if not re.match(r'^[a-zA-Z0-9_-]{32,}$', api_key):
            return False
        
        # Check for suspicious patterns
        if api_key.lower() in ['test', 'demo', 'example', 'sample']:
            return False
        
        return True
    
    @staticmethod
    def validate_pagination_params(offset: int = 0, limit: int = 50, max_limit: int = 1000) -> Tuple[int, int]:
        """
        Validate pagination parameters.
        
        Args:
            offset: Starting offset
            limit: Number of items to return
            max_limit: Maximum allowed limit
            
        Returns:
            Validated (offset, limit) tuple
        """
        if not isinstance(offset, int) or offset < 0:
            offset = 0
        
        if not isinstance(limit, int) or limit < 1:
            limit = 50
        elif limit > max_limit:
            limit = max_limit
        
        return offset, limit


class ValidationMiddleware:
    """Middleware for automatic request validation."""
    
    def __init__(self, schema_mappings: Dict[str, str] = None):
        """
        Initialize validation middleware.
        
        Args:
            schema_mappings: Mapping of endpoint paths to schema names
        """
        self.schema_mappings = schema_mappings or {
            '/generate': 'generation_request',
            '/generate/batch': 'batch_request',
        }
    
    def validate_request(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate request based on path.
        
        Args:
            path: Request path
            data: Request data
            
        Returns:
            Validated data
        """
        schema_name = self.schema_mappings.get(path)
        
        if schema_name:
            return InputValidator.validate_request_data(data, schema_name)
        else:
            # Basic sanitization for unmapped endpoints
            return Sanitizer.sanitize_json(data)


def validate_input(schema_name: str = None, strict: bool = True):
    """
    Decorator to automatically validate function inputs.
    
    Args:
        schema_name: JSON schema to validate against
        strict: Whether to raise errors or just log warnings
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Schema validation if specified
                if schema_name and len(args) > 0 and isinstance(args[0], dict):
                    args = (InputValidator.validate_request_data(args[0], schema_name),) + args[1:]
                elif schema_name and 'data' in kwargs:
                    kwargs['data'] = InputValidator.validate_request_data(kwargs['data'], schema_name)
                
                # Field-specific validation
                if 'prompt' in kwargs:
                    kwargs['prompt'] = InputValidator.validate_prompt(kwargs['prompt'])
                if 'smiles' in kwargs:
                    kwargs['smiles'] = InputValidator.validate_smiles(kwargs['smiles'])
                if 'constraints' in kwargs and kwargs['constraints']:
                    kwargs['constraints'] = InputValidator.validate_molecular_constraints(kwargs['constraints'])
                    
                return func(*args, **kwargs)
                
            except ValidationError as e:
                if strict:
                    logger.error(f"Validation error in {func.__name__}: {str(e)}", 
                               field=e.field, value=e.value)
                    raise
                else:
                    logger.warning(f"Validation warning in {func.__name__}: {str(e)}")
                    return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                if strict:
                    raise
                return None
                
        return wrapper
    return decorator


# Convenience functions
def sanitize_user_input(data: Any) -> Any:
    """Sanitize user input data."""
    return Sanitizer.sanitize_json(data)


def validate_generation_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate generation request data."""
    return InputValidator.validate_request_data(data, 'generation_request')


def validate_batch_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate batch request data.""" 
    return InputValidator.validate_request_data(data, 'batch_request')


def is_safe_filename(filename: str) -> bool:
    """Check if filename is safe."""
    try:
        sanitized = Sanitizer.sanitize_filename(filename)
        return sanitized == filename
    except SanitizationError:
        return False


def is_valid_ip(ip: str) -> bool:
    """Check if IP address is valid."""
    return Sanitizer.validate_ip_address(ip)