"""
Environment-based configuration management system with validation.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.validation import ValidationError

logger = get_logger(__name__)


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigSource(Enum):
    """Configuration source types."""
    ENV_VAR = "env_var"
    FILE = "file"
    DEFAULT = "default"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    connection_pool_size: int = 10
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0
    retry_on_timeout: bool = True
    max_connections: int = 50
    
    def get_url(self) -> str:
        """Get database connection URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        else:
            return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "dev-secret-key-change-in-production"
    api_key_length: int = 32
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 300  # 5 minutes
    password_min_length: int = 8
    require_https: bool = False
    allowed_hosts: List[str] = None
    cors_origins: List[str] = None
    jwt_expiry: int = 86400  # 24 hours
    
    def __post_init__(self):
        if self.allowed_hosts is None:
            self.allowed_hosts = ["localhost", "127.0.0.1"]
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8000"]


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    default_rate: int = 100  # requests per window
    default_window: int = 60  # seconds
    burst_multiplier: float = 1.5
    ip_whitelist: List[str] = None
    ip_blacklist: List[str] = None
    api_key_rate: int = 1000
    global_rate: int = 10000
    
    def __post_init__(self):
        if self.ip_whitelist is None:
            self.ip_whitelist = ["127.0.0.1", "::1"]
        if self.ip_blacklist is None:
            self.ip_blacklist = []


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enabled: bool = True
    metrics_export_interval: int = 60
    health_check_interval: int = 30
    log_level: str = "INFO"
    structured_logging: bool = True
    correlation_ids: bool = True
    performance_tracking: bool = True
    error_reporting: bool = True
    prometheus_enabled: bool = False
    prometheus_port: int = 9090


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    default_ttl: int = 3600  # 1 hour
    max_size: int = 10000  # number of entries
    cleanup_interval: int = 300  # 5 minutes
    compression_enabled: bool = True
    serialization: str = "json"  # json, pickle, msgpack
    backup_enabled: bool = True
    backup_interval: int = 86400  # 24 hours


@dataclass
class ModelConfig:
    """Model configuration."""
    device: str = "cpu"
    max_workers: int = 4
    batch_size: int = 8
    max_sequence_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    safety_threshold: float = 0.1
    model_path: Optional[str] = None
    preload_cache: bool = True


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout: int = 60
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    enable_docs: bool = True
    enable_metrics: bool = True
    request_id_header: str = "X-Request-ID"
    api_prefix: str = "/api/v1"


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    testing: bool = False
    version: str = "1.0.0"
    
    # Sub-configurations
    database: DatabaseConfig = None
    security: SecurityConfig = None
    rate_limit: RateLimitConfig = None
    monitoring: MonitoringConfig = None
    cache: CacheConfig = None
    model: ModelConfig = None
    api: APIConfig = None
    
    def __post_init__(self):
        # Initialize sub-configurations if not provided
        if self.database is None:
            self.database = DatabaseConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.rate_limit is None:
            self.rate_limit = RateLimitConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.api is None:
            self.api = APIConfig()


class ConfigManager:
    """Configuration manager with environment support."""
    
    def __init__(self, config_dir: str = "config", env_prefix: str = "ODORDIFF_"):
        self.config_dir = Path(config_dir)
        self.env_prefix = env_prefix
        self.config: Optional[AppConfig] = None
        self._config_sources: Dict[str, ConfigSource] = {}
        
    def load_config(self, config_file: str = None, environment: str = None) -> AppConfig:
        """
        Load configuration from multiple sources.
        
        Args:
            config_file: Specific config file to load
            environment: Environment name (overrides auto-detection)
            
        Returns:
            Loaded configuration
        """
        # Determine environment
        env = self._detect_environment(environment)
        logger.info(f"Loading configuration for environment: {env.value}")
        
        # Load base configuration
        config_data = self._load_default_config()
        
        # Load environment-specific configuration
        if config_file:
            env_config = self._load_config_file(config_file)
            config_data.update(env_config)
        else:
            # Try to load environment-specific config file
            env_config_file = self.config_dir / f"{env.value}.yaml"
            if env_config_file.exists():
                env_config = self._load_config_file(str(env_config_file))
                config_data.update(env_config)
        
        # Override with environment variables
        env_config = self._load_env_vars()
        config_data.update(env_config)
        
        # Create configuration object
        self.config = self._create_config_object(config_data, env)
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Configuration loaded successfully")
        return self.config
    
    def _detect_environment(self, environment: str = None) -> Environment:
        """Detect current environment."""
        if environment:
            return Environment(environment.lower())
        
        # Check environment variables
        env_var = os.getenv(f"{self.env_prefix}ENV", os.getenv("ENVIRONMENT", "development"))
        
        try:
            return Environment(env_var.lower())
        except ValueError:
            logger.warning(f"Unknown environment '{env_var}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "environment": "development",
            "debug": True,
            "database": {
                "host": "localhost",
                "port": 6379
            },
            "security": {
                "secret_key": "dev-secret-key"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
    
    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            if config_path.is_absolute():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            else:
                # Try in config directory
                config_path = self.config_dir / config_file
                if not config_path.exists():
                    raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from file: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                elif config_path.suffix.lower() == '.toml' and TOML_AVAILABLE:
                    config_data = toml.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            self._record_config_sources(config_data, ConfigSource.FILE)
            return config_data
            
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            raise
    
    def _load_env_vars(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        env_vars = {}
        
        # Collect all environment variables with our prefix
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                env_key = key[len(self.env_prefix):].lower()
                env_vars[env_key] = value
        
        # Map environment variables to config structure
        mappings = {
            'debug': ('debug', bool),
            'secret_key': ('security.secret_key', str),
            'database_host': ('database.host', str),
            'database_port': ('database.port', int),
            'database_password': ('database.password', str),
            'api_host': ('api.host', str),
            'api_port': ('api.port', int),
            'model_device': ('model.device', str),
            'model_workers': ('model.max_workers', int),
            'log_level': ('monitoring.log_level', str),
            'cache_enabled': ('cache.enabled', bool),
            'rate_limit_enabled': ('rate_limit.enabled', bool),
        }
        
        for env_key, (config_path, value_type) in mappings.items():
            if env_key in env_vars:
                value = self._convert_env_value(env_vars[env_key], value_type)
                self._set_nested_config(config, config_path, value)
                self._config_sources[config_path] = ConfigSource.ENV_VAR
        
        if env_vars:
            logger.info(f"Loaded {len(env_vars)} environment variables")
        
        return config
    
    def _convert_env_value(self, value: str, value_type: Type) -> Any:
        """Convert environment variable string to typed value."""
        if value_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif value_type == int:
            return int(value)
        elif value_type == float:
            return float(value)
        elif value_type == list:
            return [item.strip() for item in value.split(',')]
        else:
            return value
    
    def _set_nested_config(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested configuration value."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _create_config_object(self, config_data: Dict[str, Any], env: Environment) -> AppConfig:
        """Create configuration object from data."""
        # Set environment
        config_data['environment'] = env
        
        # Recursively create configuration objects
        def create_dataclass_from_dict(dataclass_type: Type, data: Dict[str, Any]):
            field_types = {f.name: f.type for f in fields(dataclass_type)}
            kwargs = {}
            
            for field_name, field_type in field_types.items():
                if field_name in data:
                    value = data[field_name]
                    
                    # Handle nested dataclasses
                    if hasattr(field_type, '__dataclass_fields__'):
                        if isinstance(value, dict):
                            kwargs[field_name] = create_dataclass_from_dict(field_type, value)
                        else:
                            kwargs[field_name] = value
                    else:
                        kwargs[field_name] = value
            
            return dataclass_type(**kwargs)
        
        return create_dataclass_from_dict(AppConfig, config_data)
    
    def _record_config_sources(self, config_data: Dict[str, Any], source: ConfigSource, prefix: str = ""):
        """Record configuration sources for debugging."""
        for key, value in config_data.items():
            config_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._record_config_sources(value, source, config_path)
            else:
                self._config_sources[config_path] = source
    
    def _validate_config(self):
        """Validate loaded configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        # Validate required fields
        if self.config.environment == Environment.PRODUCTION:
            if self.config.security.secret_key == "dev-secret-key":
                raise ValidationError("Production secret key must be changed from default")
            
            if not self.config.security.require_https:
                logger.warning("HTTPS not required in production - security risk")
            
            if self.config.debug:
                logger.warning("Debug mode enabled in production - not recommended")
        
        # Validate database configuration
        if self.config.database.port < 1 or self.config.database.port > 65535:
            raise ValidationError(f"Invalid database port: {self.config.database.port}")
        
        # Validate API configuration
        if self.config.api.port < 1 or self.config.api.port > 65535:
            raise ValidationError(f"Invalid API port: {self.config.api.port}")
        
        # Validate rate limiting
        if self.config.rate_limit.default_rate <= 0:
            raise ValidationError("Rate limit must be positive")
        
        logger.info("Configuration validation passed")
    
    def get_config_sources(self) -> Dict[str, str]:
        """Get configuration sources for debugging."""
        return {path: source.value for path, source in self._config_sources.items()}
    
    def save_config(self, output_file: str, format: str = "yaml"):
        """Save current configuration to file."""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        output_path = Path(output_file)
        
        # Convert config to dict
        config_dict = self._config_to_dict(self.config)
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() in ['yml', 'yaml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                elif format.lower() == 'toml' and TOML_AVAILABLE:
                    toml.dump(config_dict, f)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def _config_to_dict(self, config_obj) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        if hasattr(config_obj, '__dataclass_fields__'):
            result = {}
            for field in fields(config_obj):
                value = getattr(config_obj, field.name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field.name] = self._config_to_dict(value)
                elif isinstance(value, Enum):
                    result[field.name] = value.value
                else:
                    result[field.name] = value
            return result
        else:
            return config_obj


# Global configuration manager
_config_manager: Optional[ConfigManager] = None
_current_config: Optional[AppConfig] = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config(config_file: str = None, environment: str = None) -> AppConfig:
    """Load configuration using global manager."""
    global _current_config
    manager = get_config_manager()
    _current_config = manager.load_config(config_file, environment)
    return _current_config

def get_config() -> Optional[AppConfig]:
    """Get current configuration."""
    return _current_config

def reload_config():
    """Reload configuration from sources."""
    global _current_config
    if _current_config:
        manager = get_config_manager()
        _current_config = manager.load_config()
    else:
        raise ValueError("No configuration loaded to reload")

# Convenience functions for common configurations
def is_production() -> bool:
    """Check if running in production."""
    config = get_config()
    return config and config.environment == Environment.PRODUCTION

def is_debug() -> bool:
    """Check if debug mode is enabled."""
    config = get_config()
    return config and config.debug

def get_database_url() -> str:
    """Get database connection URL."""
    config = get_config()
    if not config:
        raise ValueError("Configuration not loaded")
    return config.database.get_url()

def get_api_base_url() -> str:
    """Get API base URL."""
    config = get_config()
    if not config:
        raise ValueError("Configuration not loaded")
    
    protocol = "https" if config.security.require_https else "http"
    return f"{protocol}://{config.api.host}:{config.api.port}{config.api.api_prefix}"