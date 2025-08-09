"""
Unit tests for configuration management system.
"""

import pytest
import os
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from dataclasses import fields

from odordiff2.config.settings import (
    Environment, ConfigSource, DatabaseConfig, SecurityConfig, 
    RateLimitConfig, MonitoringConfig, CacheConfig, ModelConfig,
    APIConfig, AppConfig, ConfigManager, 
    get_config_manager, load_config, get_config, reload_config,
    is_production, is_debug, get_database_url, get_api_base_url
)
from odordiff2.utils.validation import ValidationError


class TestEnums:
    """Test enum classes."""
    
    def test_environment_enum(self):
        """Test Environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
    
    def test_config_source_enum(self):
        """Test ConfigSource enum values."""
        assert ConfigSource.ENV_VAR.value == "env_var"
        assert ConfigSource.FILE.value == "file"
        assert ConfigSource.DEFAULT.value == "default"


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.password is None
        assert config.db == 0
        assert config.connection_pool_size == 10
        assert config.connection_timeout == 5.0
        assert config.socket_timeout == 5.0
        assert config.retry_on_timeout is True
        assert config.max_connections == 50
    
    def test_get_url_without_password(self):
        """Test URL generation without password."""
        config = DatabaseConfig(host="redis.example.com", port=6380, db=1)
        url = config.get_url()
        
        assert url == "redis://redis.example.com:6380/1"
    
    def test_get_url_with_password(self):
        """Test URL generation with password."""
        config = DatabaseConfig(
            host="redis.example.com", 
            port=6380, 
            db=1, 
            password="secret123"
        )
        url = config.get_url()
        
        assert url == "redis://:secret123@redis.example.com:6380/1"
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DatabaseConfig(
            host="custom-host",
            port=1234,
            password="custom-password",
            db=5,
            connection_pool_size=20
        )
        
        assert config.host == "custom-host"
        assert config.port == 1234
        assert config.password == "custom-password"
        assert config.db == 5
        assert config.connection_pool_size == 20


class TestSecurityConfig:
    """Test SecurityConfig dataclass."""
    
    def test_default_values(self):
        """Test default security configuration."""
        config = SecurityConfig()
        
        assert config.secret_key == "dev-secret-key-change-in-production"
        assert config.api_key_length == 32
        assert config.session_timeout == 3600
        assert config.max_login_attempts == 5
        assert config.lockout_duration == 300
        assert config.password_min_length == 8
        assert config.require_https is False
        assert config.jwt_expiry == 86400
    
    def test_post_init_default_lists(self):
        """Test that post_init sets default lists."""
        config = SecurityConfig()
        
        assert config.allowed_hosts == ["localhost", "127.0.0.1"]
        assert config.cors_origins == ["http://localhost:3000", "http://localhost:8000"]
    
    def test_post_init_custom_lists(self):
        """Test that post_init preserves custom lists."""
        config = SecurityConfig(
            allowed_hosts=["custom.example.com"],
            cors_origins=["https://custom.example.com"]
        )
        
        assert config.allowed_hosts == ["custom.example.com"]
        assert config.cors_origins == ["https://custom.example.com"]


class TestRateLimitConfig:
    """Test RateLimitConfig dataclass."""
    
    def test_default_values(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        
        assert config.enabled is True
        assert config.default_rate == 100
        assert config.default_window == 60
        assert config.burst_multiplier == 1.5
        assert config.api_key_rate == 1000
        assert config.global_rate == 10000
    
    def test_post_init_default_lists(self):
        """Test that post_init sets default lists."""
        config = RateLimitConfig()
        
        assert config.ip_whitelist == ["127.0.0.1", "::1"]
        assert config.ip_blacklist == []


class TestMonitoringConfig:
    """Test MonitoringConfig dataclass."""
    
    def test_default_values(self):
        """Test default monitoring configuration."""
        config = MonitoringConfig()
        
        assert config.enabled is True
        assert config.metrics_export_interval == 60
        assert config.health_check_interval == 30
        assert config.log_level == "INFO"
        assert config.structured_logging is True
        assert config.correlation_ids is True
        assert config.performance_tracking is True
        assert config.error_reporting is True
        assert config.prometheus_enabled is False
        assert config.prometheus_port == 9090


class TestCacheConfig:
    """Test CacheConfig dataclass."""
    
    def test_default_values(self):
        """Test default cache configuration."""
        config = CacheConfig()
        
        assert config.enabled is True
        assert config.default_ttl == 3600
        assert config.max_size == 10000
        assert config.cleanup_interval == 300
        assert config.compression_enabled is True
        assert config.serialization == "json"
        assert config.backup_enabled is True
        assert config.backup_interval == 86400


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.device == "cpu"
        assert config.max_workers == 4
        assert config.batch_size == 8
        assert config.max_sequence_length == 512
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.safety_threshold == 0.1
        assert config.model_path is None
        assert config.preload_cache is True


class TestAPIConfig:
    """Test APIConfig dataclass."""
    
    def test_default_values(self):
        """Test default API configuration."""
        config = APIConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.timeout == 60
        assert config.max_request_size == 16 * 1024 * 1024
        assert config.enable_docs is True
        assert config.enable_metrics is True
        assert config.request_id_header == "X-Request-ID"
        assert config.api_prefix == "/api/v1"


class TestAppConfig:
    """Test AppConfig dataclass."""
    
    def test_default_values(self):
        """Test default app configuration."""
        config = AppConfig()
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.testing is False
        assert config.version == "1.0.0"
    
    def test_post_init_creates_sub_configs(self):
        """Test that post_init creates sub-configurations."""
        config = AppConfig()
        
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.rate_limit, RateLimitConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.api, APIConfig)
    
    def test_post_init_preserves_existing_configs(self):
        """Test that post_init preserves existing sub-configurations."""
        custom_db = DatabaseConfig(host="custom-host")
        config = AppConfig(database=custom_db)
        
        assert config.database is custom_db
        assert config.database.host == "custom-host"


class TestConfigManager:
    """Test ConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager instance."""
        return ConfigManager(config_dir=str(temp_config_dir))
    
    def test_initialization(self, temp_config_dir):
        """Test ConfigManager initialization."""
        manager = ConfigManager(config_dir=str(temp_config_dir), env_prefix="TEST_")
        
        assert manager.config_dir == temp_config_dir
        assert manager.env_prefix == "TEST_"
        assert manager.config is None
        assert manager._config_sources == {}
    
    def test_detect_environment_default(self, config_manager):
        """Test environment detection with defaults."""
        env = config_manager._detect_environment()
        assert env == Environment.DEVELOPMENT
    
    def test_detect_environment_explicit(self, config_manager):
        """Test environment detection with explicit value."""
        env = config_manager._detect_environment("production")
        assert env == Environment.PRODUCTION
    
    @patch.dict(os.environ, {'ODORDIFF_ENV': 'testing'})
    def test_detect_environment_from_env_var(self, config_manager):
        """Test environment detection from environment variable."""
        env = config_manager._detect_environment()
        assert env == Environment.TESTING
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'staging'})
    def test_detect_environment_from_generic_env_var(self, config_manager):
        """Test environment detection from generic ENVIRONMENT variable."""
        env = config_manager._detect_environment()
        assert env == Environment.STAGING
    
    def test_detect_environment_invalid_fallback(self, config_manager):
        """Test fallback for invalid environment."""
        with patch.dict(os.environ, {'ODORDIFF_ENV': 'invalid'}):
            env = config_manager._detect_environment()
            assert env == Environment.DEVELOPMENT
    
    def test_load_default_config(self, config_manager):
        """Test loading default configuration."""
        config_data = config_manager._load_default_config()
        
        assert config_data["environment"] == "development"
        assert config_data["debug"] is True
        assert "database" in config_data
        assert "security" in config_data
        assert "api" in config_data
    
    def test_load_config_file_yaml(self, config_manager, temp_config_dir):
        """Test loading YAML config file."""
        config_data = {
            "debug": True,
            "database": {"host": "test-host", "port": 6380},
            "api": {"port": 9000}
        }
        
        config_file = temp_config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded_config = config_manager._load_config_file(str(config_file))
        
        assert loaded_config["debug"] is True
        assert loaded_config["database"]["host"] == "test-host"
        assert loaded_config["api"]["port"] == 9000
    
    def test_load_config_file_json(self, config_manager, temp_config_dir):
        """Test loading JSON config file."""
        config_data = {
            "debug": False,
            "database": {"host": "json-host"},
        }
        
        config_file = temp_config_dir / "test.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loaded_config = config_manager._load_config_file(str(config_file))
        
        assert loaded_config["debug"] is False
        assert loaded_config["database"]["host"] == "json-host"
    
    def test_load_config_file_not_found(self, config_manager):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            config_manager._load_config_file("non_existent.yaml")
    
    def test_load_config_file_unsupported_format(self, config_manager, temp_config_dir):
        """Test loading unsupported config file format."""
        config_file = temp_config_dir / "test.txt"
        config_file.write_text("some text")
        
        with pytest.raises(ValueError, match="Unsupported config file format"):
            config_manager._load_config_file(str(config_file))
    
    @patch.dict(os.environ, {
        'ODORDIFF_DEBUG': 'true',
        'ODORDIFF_DATABASE_HOST': 'env-host',
        'ODORDIFF_DATABASE_PORT': '6380',
        'ODORDIFF_API_PORT': '9000',
        'ODORDIFF_LOG_LEVEL': 'DEBUG'
    })
    def test_load_env_vars(self, config_manager):
        """Test loading configuration from environment variables."""
        env_config = config_manager._load_env_vars()
        
        assert env_config["debug"] is True
        assert env_config["database"]["host"] == "env-host"
        assert env_config["database"]["port"] == 6380
        assert env_config["api"]["port"] == 9000
        assert env_config["monitoring"]["log_level"] == "DEBUG"
    
    def test_convert_env_value_bool(self, config_manager):
        """Test environment value conversion for booleans."""
        assert config_manager._convert_env_value("true", bool) is True
        assert config_manager._convert_env_value("1", bool) is True
        assert config_manager._convert_env_value("yes", bool) is True
        assert config_manager._convert_env_value("on", bool) is True
        assert config_manager._convert_env_value("false", bool) is False
        assert config_manager._convert_env_value("0", bool) is False
    
    def test_convert_env_value_int(self, config_manager):
        """Test environment value conversion for integers."""
        assert config_manager._convert_env_value("123", int) == 123
        assert config_manager._convert_env_value("-456", int) == -456
    
    def test_convert_env_value_float(self, config_manager):
        """Test environment value conversion for floats."""
        assert config_manager._convert_env_value("1.23", float) == 1.23
        assert config_manager._convert_env_value("-4.56", float) == -4.56
    
    def test_convert_env_value_list(self, config_manager):
        """Test environment value conversion for lists."""
        result = config_manager._convert_env_value("item1,item2,item3", list)
        assert result == ["item1", "item2", "item3"]
    
    def test_convert_env_value_string(self, config_manager):
        """Test environment value conversion for strings."""
        assert config_manager._convert_env_value("test string", str) == "test string"
    
    def test_set_nested_config(self, config_manager):
        """Test setting nested configuration values."""
        config = {}
        config_manager._set_nested_config(config, "database.host", "test-host")
        config_manager._set_nested_config(config, "api.security.enabled", True)
        
        assert config["database"]["host"] == "test-host"
        assert config["api"]["security"]["enabled"] is True
    
    def test_create_config_object(self, config_manager):
        """Test creating configuration object from data."""
        config_data = {
            "environment": "testing",
            "debug": True,
            "database": {
                "host": "test-host",
                "port": 6380
            },
            "api": {
                "port": 9000
            }
        }
        
        config = config_manager._create_config_object(config_data, Environment.TESTING)
        
        assert isinstance(config, AppConfig)
        assert config.environment == Environment.TESTING
        assert config.debug is True
        assert config.database.host == "test-host"
        assert config.database.port == 6380
        assert config.api.port == 9000
    
    def test_validate_config_development(self, config_manager):
        """Test configuration validation in development."""
        config_data = {
            "environment": Environment.DEVELOPMENT,
            "debug": True,
            "security": {"secret_key": "dev-secret-key"}
        }
        config = config_manager._create_config_object(config_data, Environment.DEVELOPMENT)
        config_manager.config = config
        
        # Should not raise exception in development
        config_manager._validate_config()
    
    def test_validate_config_production_valid(self, config_manager):
        """Test valid production configuration."""
        config_data = {
            "environment": Environment.PRODUCTION,
            "debug": False,
            "security": {
                "secret_key": "production-secret-key-123",
                "require_https": True
            },
            "database": {"port": 6379},
            "api": {"port": 8000},
            "rate_limit": {"default_rate": 100}
        }
        config = config_manager._create_config_object(config_data, Environment.PRODUCTION)
        config_manager.config = config
        
        # Should not raise exception
        config_manager._validate_config()
    
    def test_validate_config_production_invalid_secret(self, config_manager):
        """Test production configuration with default secret key."""
        config_data = {
            "environment": Environment.PRODUCTION,
            "security": {"secret_key": "dev-secret-key"}
        }
        config = config_manager._create_config_object(config_data, Environment.PRODUCTION)
        config_manager.config = config
        
        with pytest.raises(ValidationError, match="Production secret key must be changed"):
            config_manager._validate_config()
    
    def test_validate_config_invalid_database_port(self, config_manager):
        """Test validation with invalid database port."""
        config_data = {
            "database": {"port": 70000}  # Invalid port
        }
        config = config_manager._create_config_object(config_data, Environment.DEVELOPMENT)
        config_manager.config = config
        
        with pytest.raises(ValidationError, match="Invalid database port"):
            config_manager._validate_config()
    
    def test_validate_config_invalid_api_port(self, config_manager):
        """Test validation with invalid API port."""
        config_data = {
            "api": {"port": -1}  # Invalid port
        }
        config = config_manager._create_config_object(config_data, Environment.DEVELOPMENT)
        config_manager.config = config
        
        with pytest.raises(ValidationError, match="Invalid API port"):
            config_manager._validate_config()
    
    def test_validate_config_invalid_rate_limit(self, config_manager):
        """Test validation with invalid rate limit."""
        config_data = {
            "rate_limit": {"default_rate": 0}  # Invalid rate
        }
        config = config_manager._create_config_object(config_data, Environment.DEVELOPMENT)
        config_manager.config = config
        
        with pytest.raises(ValidationError, match="Rate limit must be positive"):
            config_manager._validate_config()
    
    def test_load_config_integration(self, config_manager, temp_config_dir):
        """Test complete configuration loading."""
        # Create config file
        config_data = {
            "debug": True,
            "database": {"host": "config-host"},
            "api": {"port": 9000}
        }
        
        config_file = temp_config_dir / "development.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load configuration
        with patch.dict(os.environ, {'ODORDIFF_API_PORT': '9001'}):
            config = config_manager.load_config()
        
        # Environment variable should override file
        assert config.api.port == 9001
        assert config.database.host == "config-host"
        assert config.debug is True
    
    def test_get_config_sources(self, config_manager):
        """Test getting configuration sources."""
        config_manager._config_sources = {
            "debug": ConfigSource.FILE,
            "database.host": ConfigSource.ENV_VAR,
            "api.port": ConfigSource.DEFAULT
        }
        
        sources = config_manager.get_config_sources()
        
        assert sources["debug"] == "file"
        assert sources["database.host"] == "env_var"
        assert sources["api.port"] == "default"
    
    def test_config_to_dict(self, config_manager):
        """Test converting config object to dictionary."""
        config = AppConfig()
        config_dict = config_manager._config_to_dict(config)
        
        assert isinstance(config_dict, dict)
        assert config_dict["environment"] == "development"
        assert isinstance(config_dict["database"], dict)
        assert config_dict["database"]["host"] == "localhost"
    
    def test_save_config_yaml(self, config_manager, temp_config_dir):
        """Test saving configuration to YAML file."""
        config = AppConfig()
        config_manager.config = config
        
        output_file = temp_config_dir / "output.yaml"
        config_manager.save_config(str(output_file), "yaml")
        
        assert output_file.exists()
        
        # Verify content
        with open(output_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["environment"] == "development"
        assert saved_data["database"]["host"] == "localhost"
    
    def test_save_config_json(self, config_manager, temp_config_dir):
        """Test saving configuration to JSON file."""
        config = AppConfig()
        config_manager.config = config
        
        output_file = temp_config_dir / "output.json"
        config_manager.save_config(str(output_file), "json")
        
        assert output_file.exists()
        
        # Verify content
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["environment"] == "development"
        assert saved_data["database"]["host"] == "localhost"
    
    def test_save_config_no_config_loaded(self, config_manager):
        """Test saving when no configuration is loaded."""
        with pytest.raises(ValueError, match="No configuration loaded"):
            config_manager.save_config("output.yaml")


class TestGlobalFunctions:
    """Test global configuration functions."""
    
    def setup_method(self):
        """Reset global state before each test."""
        from odordiff2.config import settings
        settings._config_manager = None
        settings._current_config = None
    
    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns singleton."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2
    
    @patch('odordiff2.config.settings.ConfigManager')
    def test_load_config_global(self, mock_manager_class):
        """Test global load_config function."""
        mock_manager = Mock()
        mock_config = AppConfig()
        mock_manager.load_config.return_value = mock_config
        mock_manager_class.return_value = mock_manager
        
        config = load_config("test.yaml", "production")
        
        assert config is mock_config
        assert get_config() is mock_config
        mock_manager.load_config.assert_called_once_with("test.yaml", "production")
    
    def test_get_config_none_loaded(self):
        """Test get_config when none loaded."""
        assert get_config() is None
    
    def test_reload_config_no_config(self):
        """Test reload_config when no config loaded."""
        with pytest.raises(ValueError, match="No configuration loaded to reload"):
            reload_config()
    
    def test_is_production_true(self):
        """Test is_production when in production."""
        from odordiff2.config import settings
        config = AppConfig(environment=Environment.PRODUCTION)
        settings._current_config = config
        
        assert is_production() is True
    
    def test_is_production_false(self):
        """Test is_production when not in production."""
        from odordiff2.config import settings
        config = AppConfig(environment=Environment.DEVELOPMENT)
        settings._current_config = config
        
        assert is_production() is False
    
    def test_is_production_no_config(self):
        """Test is_production when no config loaded."""
        assert is_production() is False
    
    def test_is_debug_true(self):
        """Test is_debug when debug enabled."""
        from odordiff2.config import settings
        config = AppConfig(debug=True)
        settings._current_config = config
        
        assert is_debug() is True
    
    def test_is_debug_false(self):
        """Test is_debug when debug disabled."""
        from odordiff2.config import settings
        config = AppConfig(debug=False)
        settings._current_config = config
        
        assert is_debug() is False
    
    def test_get_database_url(self):
        """Test get_database_url function."""
        from odordiff2.config import settings
        config = AppConfig()
        config.database = DatabaseConfig(host="test-host", port=6380)
        settings._current_config = config
        
        url = get_database_url()
        assert url == "redis://test-host:6380/0"
    
    def test_get_database_url_no_config(self):
        """Test get_database_url when no config loaded."""
        with pytest.raises(ValueError, match="Configuration not loaded"):
            get_database_url()
    
    def test_get_api_base_url_http(self):
        """Test get_api_base_url with HTTP."""
        from odordiff2.config import settings
        config = AppConfig()
        config.api = APIConfig(host="localhost", port=8000, api_prefix="/api/v1")
        config.security = SecurityConfig(require_https=False)
        settings._current_config = config
        
        url = get_api_base_url()
        assert url == "http://localhost:8000/api/v1"
    
    def test_get_api_base_url_https(self):
        """Test get_api_base_url with HTTPS."""
        from odordiff2.config import settings
        config = AppConfig()
        config.api = APIConfig(host="api.example.com", port=443, api_prefix="/api/v2")
        config.security = SecurityConfig(require_https=True)
        settings._current_config = config
        
        url = get_api_base_url()
        assert url == "https://api.example.com:443/api/v2"
    
    def test_get_api_base_url_no_config(self):
        """Test get_api_base_url when no config loaded."""
        with pytest.raises(ValueError, match="Configuration not loaded"):
            get_api_base_url()


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    @pytest.fixture
    def temp_config_setup(self):
        """Setup temporary configuration environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create development config
            dev_config = {
                "debug": True,
                "database": {"host": "dev-redis", "port": 6379},
                "api": {"port": 8000},
                "security": {"secret_key": "dev-secret-123"}
            }
            
            dev_file = config_dir / "development.yaml"
            with open(dev_file, 'w') as f:
                yaml.dump(dev_config, f)
            
            # Create production config
            prod_config = {
                "debug": False,
                "database": {"host": "prod-redis", "port": 6380},
                "api": {"port": 80},
                "security": {
                    "secret_key": "prod-secret-456",
                    "require_https": True
                }
            }
            
            prod_file = config_dir / "production.yaml"
            with open(prod_file, 'w') as f:
                yaml.dump(prod_config, f)
            
            yield config_dir
    
    def test_environment_specific_loading(self, temp_config_setup):
        """Test loading environment-specific configurations."""
        manager = ConfigManager(config_dir=str(temp_config_setup))
        
        # Load development config
        dev_config = manager.load_config(environment="development")
        assert dev_config.debug is True
        assert dev_config.database.host == "dev-redis"
        assert dev_config.api.port == 8000
        
        # Load production config
        prod_config = manager.load_config(environment="production")
        assert prod_config.debug is False
        assert prod_config.database.host == "prod-redis"
        assert prod_config.security.require_https is True
    
    def test_environment_variable_override(self, temp_config_setup):
        """Test environment variables overriding config files."""
        manager = ConfigManager(config_dir=str(temp_config_setup))
        
        with patch.dict(os.environ, {
            'ODORDIFF_DATABASE_HOST': 'env-override-host',
            'ODORDIFF_API_PORT': '9000',
            'ODORDIFF_DEBUG': 'true'
        }):
            config = manager.load_config(environment="production")
            
            # Environment variables should override file values
            assert config.database.host == "env-override-host"
            assert config.api.port == 9000
            assert config.debug is True  # overridden from prod config
    
    def test_configuration_validation_flow(self, temp_config_setup):
        """Test full configuration validation workflow."""
        manager = ConfigManager(config_dir=str(temp_config_setup))
        
        # Valid configuration should load successfully
        config = manager.load_config(environment="production")
        assert config.environment == Environment.PRODUCTION
        
        # Invalid configuration should raise error
        with patch.dict(os.environ, {'ODORDIFF_DATABASE_PORT': '70000'}):
            with pytest.raises(ValidationError):
                manager.load_config(environment="production")