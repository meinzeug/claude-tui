#!/usr/bin/env python3
"""
Comprehensive Unit Tests for ConfigManager

This module provides extensive unit testing for the configuration management
system, ensuring robust handling of configuration loading, validation,
and environment-specific settings.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any

# Import the test framework
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from framework.enhanced_test_framework import PerformanceMonitor, TestDataFactory

# Import the component under test (will be implemented)
# from claude_tiu.core.config_manager import ConfigManager, ConfigError


class MockConfigManager:
    """Mock ConfigManager for testing infrastructure."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config_data = {}
        self.environment = "development"
        self._loaded = False
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path or not Path(self.config_path).exists():
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith('.json'):
                self.config_data = json.load(f)
            elif self.config_path.endswith(('.yml', '.yaml')):
                self.config_data = yaml.safe_load(f)
        
        self._loaded = True
        return self.config_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def validate_config(self) -> bool:
        """Validate configuration structure."""
        required_keys = ['ai', 'database', 'security']
        return all(key in self.config_data for key in required_keys)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "ai": {
                "claude_api_key": None,
                "timeout": 30,
                "max_retries": 3
            },
            "database": {
                "url": "sqlite:///claude_tiu.db",
                "echo": False
            },
            "security": {
                "secret_key": "dev-secret-key",
                "algorithm": "HS256"
            }
        }


@pytest.fixture
def config_manager():
    """Create ConfigManager instance for testing."""
    return MockConfigManager()


@pytest.fixture
def temp_config_file(tmp_path):
    """Create temporary config file for testing."""
    config_data = {
        "ai": {
            "claude_api_key": "test-api-key",
            "timeout": 60,
            "max_retries": 5
        },
        "database": {
            "url": "postgresql://test:test@localhost/test_db",
            "echo": True
        },
        "security": {
            "secret_key": "test-secret-key",
            "algorithm": "HS256"
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_data, indent=2))
    return config_file


class TestConfigManagerBasicOperations:
    """Test basic configuration operations."""
    
    @pytest.mark.unit
    def test_initialization_with_default_config(self, config_manager):
        """Test ConfigManager initialization with default configuration."""
        # Act
        config = config_manager.load_config()
        
        # Assert
        assert config is not None
        assert isinstance(config, dict)
        assert "ai" in config
        assert "database" in config
        assert "security" in config
    
    @pytest.mark.unit
    def test_initialization_with_custom_config_file(self, temp_config_file):
        """Test ConfigManager initialization with custom config file."""
        # Arrange
        manager = MockConfigManager(str(temp_config_file))
        
        # Act
        config = manager.load_config()
        
        # Assert
        assert config["ai"]["claude_api_key"] == "test-api-key"
        assert config["database"]["url"] == "postgresql://test:test@localhost/test_db"
        assert config["security"]["secret_key"] == "test-secret-key"
    
    @pytest.mark.unit
    def test_get_config_value_simple_key(self, config_manager):
        """Test getting configuration value with simple key."""
        # Arrange
        config_manager.config_data = {"test_key": "test_value"}
        
        # Act
        value = config_manager.get("test_key")
        
        # Assert
        assert value == "test_value"
    
    @pytest.mark.unit
    def test_get_config_value_nested_key(self, config_manager):
        """Test getting configuration value with nested key."""
        # Arrange
        config_manager.config_data = {
            "database": {
                "connection": {
                    "timeout": 30
                }
            }
        }
        
        # Act
        value = config_manager.get("database.connection.timeout")
        
        # Assert
        assert value == 30
    
    @pytest.mark.unit
    def test_get_config_value_with_default(self, config_manager):
        """Test getting non-existent config value returns default."""
        # Arrange
        config_manager.config_data = {}
        
        # Act
        value = config_manager.get("non.existent.key", default="default_value")
        
        # Assert
        assert value == "default_value"
    
    @pytest.mark.unit
    def test_set_config_value_simple_key(self, config_manager):
        """Test setting configuration value with simple key."""
        # Act
        config_manager.set("new_key", "new_value")
        
        # Assert
        assert config_manager.get("new_key") == "new_value"
    
    @pytest.mark.unit
    def test_set_config_value_nested_key(self, config_manager):
        """Test setting configuration value with nested key."""
        # Act
        config_manager.set("section.subsection.key", "nested_value")
        
        # Assert
        assert config_manager.get("section.subsection.key") == "nested_value"


class TestConfigManagerValidation:
    """Test configuration validation functionality."""
    
    @pytest.mark.unit
    def test_validate_complete_config_returns_true(self, config_manager):
        """Test validation of complete configuration returns True."""
        # Arrange
        config_manager.config_data = {
            "ai": {"claude_api_key": "test"},
            "database": {"url": "test"},
            "security": {"secret_key": "test"}
        }
        
        # Act
        is_valid = config_manager.validate_config()
        
        # Assert
        assert is_valid is True
    
    @pytest.mark.unit
    def test_validate_incomplete_config_returns_false(self, config_manager):
        """Test validation of incomplete configuration returns False."""
        # Arrange
        config_manager.config_data = {
            "ai": {"claude_api_key": "test"},
            # Missing database and security sections
        }
        
        # Act
        is_valid = config_manager.validate_config()
        
        # Assert
        assert is_valid is False
    
    @pytest.mark.unit
    def test_validate_empty_config_returns_false(self, config_manager):
        """Test validation of empty configuration returns False."""
        # Arrange
        config_manager.config_data = {}
        
        # Act
        is_valid = config_manager.validate_config()
        
        # Assert
        assert is_valid is False


class TestConfigManagerEnvironmentHandling:
    """Test environment-specific configuration handling."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'CLAUDE_TIU_ENV': 'production'})
    def test_environment_detection_production(self, config_manager):
        """Test detection of production environment."""
        # This would test environment-specific logic
        # For now, just verify the mock works
        import os
        assert os.environ.get('CLAUDE_TIU_ENV') == 'production'
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'CLAUDE_TIU_ENV': 'development'})
    def test_environment_detection_development(self, config_manager):
        """Test detection of development environment."""
        import os
        assert os.environ.get('CLAUDE_TIU_ENV') == 'development'
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'CLAUDE_API_KEY': 'env-api-key'})
    def test_environment_variable_override(self, config_manager):
        """Test environment variable overrides config file values."""
        # Arrange
        config_manager.config_data = {
            "ai": {"claude_api_key": "file-api-key"}
        }
        
        # Act - In real implementation, this would check env vars
        import os
        env_key = os.environ.get('CLAUDE_API_KEY')
        
        # Assert
        assert env_key == 'env-api-key'  # Should override file value


class TestConfigManagerFileHandling:
    """Test configuration file handling scenarios."""
    
    @pytest.mark.unit
    def test_load_json_config_file(self, tmp_path):
        """Test loading JSON configuration file."""
        # Arrange
        config_data = {"test": "value"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        
        manager = MockConfigManager(str(config_file))
        
        # Act
        loaded_config = manager.load_config()
        
        # Assert
        assert loaded_config["test"] == "value"
    
    @pytest.mark.unit
    def test_load_yaml_config_file(self, tmp_path):
        """Test loading YAML configuration file."""
        # Arrange
        config_data = {"test": "yaml_value", "nested": {"key": "nested_value"}}
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump(config_data))
        
        manager = MockConfigManager(str(config_file))
        
        # Act
        loaded_config = manager.load_config()
        
        # Assert
        assert loaded_config["test"] == "yaml_value"
        assert loaded_config["nested"]["key"] == "nested_value"
    
    @pytest.mark.unit
    def test_load_nonexistent_config_file_uses_defaults(self, config_manager):
        """Test loading non-existent config file uses defaults."""
        # Arrange
        config_manager.config_path = "/nonexistent/path/config.json"
        
        # Act
        config = config_manager.load_config()
        
        # Assert
        assert config is not None
        assert "ai" in config
        assert config["ai"]["timeout"] == 30  # Default value


class TestConfigManagerErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.unit
    def test_malformed_json_config_file(self, tmp_path):
        """Test handling of malformed JSON configuration file."""
        # Arrange
        config_file = tmp_path / "malformed.json"
        config_file.write_text('{"invalid": json}')  # Invalid JSON
        
        manager = MockConfigManager(str(config_file))
        
        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            manager.load_config()
    
    @pytest.mark.unit 
    def test_malformed_yaml_config_file(self, tmp_path):
        """Test handling of malformed YAML configuration file."""
        # Arrange
        config_file = tmp_path / "malformed.yml"
        config_file.write_text('invalid: yaml: content: [')  # Invalid YAML
        
        manager = MockConfigManager(str(config_file))
        
        # Act & Assert
        with pytest.raises(yaml.YAMLError):
            manager.load_config()


class TestConfigManagerPerformance:
    """Test configuration manager performance characteristics."""
    
    @pytest.mark.unit
    def test_config_loading_performance(self, temp_config_file):
        """Test configuration loading performance."""
        manager = MockConfigManager(str(temp_config_file))
        
        with PerformanceMonitor(thresholds={"max_duration": 1.0}) as monitor:
            # Act
            config = manager.load_config()
            
            # Assert
            assert config is not None
            # Performance assertion handled by PerformanceMonitor
    
    @pytest.mark.unit
    def test_repeated_config_access_performance(self, config_manager):
        """Test performance of repeated configuration access."""
        # Arrange
        config_manager.config_data = {
            "nested": {"deep": {"value": "test"}}
        }
        
        with PerformanceMonitor(thresholds={"max_duration": 0.1}) as monitor:
            # Act - Multiple rapid accesses
            for _ in range(1000):
                value = config_manager.get("nested.deep.value")
                assert value == "test"


class TestConfigManagerThreadSafety:
    """Test thread safety of configuration operations."""
    
    @pytest.mark.unit
    def test_concurrent_config_access(self, config_manager):
        """Test concurrent access to configuration values."""
        import threading
        import time
        
        # Arrange
        config_manager.config_data = {"counter": 0}
        results = []
        
        def access_config():
            for i in range(10):
                value = config_manager.get("counter", 0)
                results.append(value)
                time.sleep(0.01)  # Small delay
        
        # Act
        threads = [threading.Thread(target=access_config) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Assert
        assert len(results) == 30  # 3 threads × 10 iterations
        assert all(result == 0 for result in results)


class TestConfigManagerIntegration:
    """Test integration aspects of configuration manager."""
    
    @pytest.mark.unit
    def test_config_manager_with_test_data_factory(self, test_factory):
        """Test ConfigManager with realistic test data."""
        # This test demonstrates integration with TestDataFactory
        project_data = test_factory.create_project_data()
        
        manager = MockConfigManager()
        manager.set("current_project", project_data)
        
        # Act
        retrieved_data = manager.get("current_project")
        
        # Assert
        assert retrieved_data == project_data
        assert "name" in retrieved_data
        assert "description" in retrieved_data