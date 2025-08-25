"""Enhanced comprehensive tests for ConfigManager."""

import pytest
import asyncio
import os
import json
import yaml
from unittest.mock import patch, mock_open, Mock, AsyncMock
from pathlib import Path
from cryptography.fernet import Fernet

from claude_tiu.core.config_manager import (
    ConfigManager, AppConfig, AIServiceConfig, 
    ProjectDefaults, UIPreferences, SecurityConfig
)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary configuration directory."""
    config_dir = tmp_path / "claude-tiu-test"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def config_manager(temp_config_dir):
    """Create ConfigManager instance with temporary directory."""
    return ConfigManager(config_dir=temp_config_dir)


@pytest.fixture
def sample_config():
    """Sample configuration data."""
    return {
        'version': '0.1.0',
        'config_version': '1.0',
        'ai_services': {
            'claude': {
                'service_name': 'claude',
                'endpoint_url': 'https://api.anthropic.com',
                'timeout': 300,
                'max_retries': 3
            }
        },
        'project_defaults': {
            'default_template': 'python',
            'auto_validation': True,
            'backup_enabled': True
        },
        'ui_preferences': {
            'theme': 'dark',
            'font_size': 14,
            'show_line_numbers': True
        },
        'security': {
            'sandbox_enabled': True,
            'max_file_size_mb': 100,
            'audit_logging': True
        }
    }


class TestConfigManagerInitialization:
    """Test ConfigManager initialization."""
    
    def test_init_default_config_dir(self):
        """Test initialization with default config directory."""
        manager = ConfigManager()
        
        # Should use platform-appropriate default directory
        expected_name = 'claude-tiu'
        assert manager.config_dir.name == expected_name
        assert manager.config_dir.exists()
    
    def test_init_custom_config_dir(self, temp_config_dir):
        """Test initialization with custom config directory."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        assert manager.config_dir == temp_config_dir
        assert manager.config_file == temp_config_dir / "config.yaml"
        assert manager.encrypted_config_file == temp_config_dir / "secrets.enc"
    
    def test_config_dir_creation(self, tmp_path):
        """Test that config directory is created if it doesn't exist."""
        config_dir = tmp_path / "nonexistent" / "config"
        manager = ConfigManager(config_dir=config_dir)
        
        assert config_dir.exists()
    
    @pytest.mark.asyncio
    async def test_initialize_new_config(self, config_manager):
        """Test initialization with no existing config."""
        await config_manager.initialize()
        
        assert config_manager.config is not None
        assert isinstance(config_manager.config, AppConfig)
        assert config_manager.config_file.exists()
    
    @pytest.mark.asyncio
    async def test_initialize_existing_config(self, config_manager, sample_config):
        """Test initialization with existing config file."""
        # Create existing config file
        with open(config_manager.config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        await config_manager.initialize()
        
        assert config_manager.config.version == '0.1.0'
        assert 'claude' in config_manager.config.ai_services
    
    @pytest.mark.asyncio
    async def test_initialize_encryption(self, config_manager):
        """Test encryption initialization."""
        await config_manager.initialize()
        
        key_file = config_manager.config_dir / '.encryption_key'
        assert key_file.exists()
        assert config_manager._encryption_key is not None


class TestConfigManagerConfigHandling:
    """Test configuration loading and saving."""
    
    @pytest.mark.asyncio
    async def test_load_valid_config(self, config_manager, sample_config):
        """Test loading valid configuration."""
        with open(config_manager.config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        await config_manager.load_config()
        
        assert config_manager.config.version == '0.1.0'
        assert config_manager.config.ui_preferences.theme == 'dark'
    
    @pytest.mark.asyncio
    async def test_load_invalid_config_fallback(self, config_manager):
        """Test fallback to default config when loading fails."""
        # Create invalid config file
        with open(config_manager.config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        await config_manager.load_config()
        
        # Should fallback to default config
        assert config_manager.config is not None
        assert isinstance(config_manager.config, AppConfig)
    
    @pytest.mark.asyncio
    async def test_save_config(self, config_manager):
        """Test configuration saving."""
        config_manager.config = AppConfig(version="test_version")
        
        await config_manager.save_config()
        
        assert config_manager.config_file.exists()
        
        # Verify saved content
        with open(config_manager.config_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['version'] == 'test_version'
    
    @pytest.mark.asyncio
    async def test_save_config_no_config_raises_error(self, config_manager):
        """Test that saving without config raises error."""
        config_manager.config = None
        
        with pytest.raises(RuntimeError, match="No configuration to save"):
            await config_manager.save_config()


class TestAIServiceConfiguration:
    """Test AI service configuration management."""
    
    @pytest.mark.asyncio
    async def test_set_ai_service_config(self, config_manager):
        """Test setting AI service configuration."""
        await config_manager.initialize()
        
        service_config = AIServiceConfig(
            service_name='test_service',
            endpoint_url='https://api.test.com',
            timeout=600,
            max_retries=5
        )
        
        await config_manager.set_ai_service_config('test_service', service_config)
        
        assert 'test_service' in config_manager.config.ai_services
        assert config_manager.config.ai_services['test_service'].timeout == 600
    
    @pytest.mark.asyncio
    async def test_get_ai_service_config(self, config_manager):
        """Test retrieving AI service configuration."""
        await config_manager.initialize()
        
        service_config = AIServiceConfig(
            service_name='test_service',
            endpoint_url='https://api.test.com'
        )
        config_manager.config.ai_services['test_service'] = service_config
        
        retrieved = await config_manager.get_ai_service_config('test_service')
        
        assert retrieved is not None
        assert retrieved.service_name == 'test_service'
        assert retrieved.endpoint_url == 'https://api.test.com'
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_ai_service_config(self, config_manager):
        """Test retrieving nonexistent AI service configuration."""
        await config_manager.initialize()
        
        retrieved = await config_manager.get_ai_service_config('nonexistent')
        
        assert retrieved is None


class TestEncryptionAndSecurity:
    """Test encryption and secure data handling."""
    
    @pytest.mark.asyncio
    async def test_store_api_key(self, config_manager):
        """Test storing API key securely."""
        await config_manager.initialize()
        
        api_key = "test_api_key_12345"
        service_name = "test_service"
        
        await config_manager.store_api_key(service_name, api_key)
        
        # Check that service config is created/updated
        assert service_name in config_manager.config.ai_services
        service_config = config_manager.config.ai_services[service_name]
        assert service_config.api_key_encrypted is not None
        
        # Check that encrypted data is stored
        assert f"api_key_{service_name}" in config_manager._encrypted_data
    
    @pytest.mark.asyncio
    async def test_retrieve_api_key(self, config_manager):
        """Test retrieving stored API key."""
        await config_manager.initialize()
        
        api_key = "test_api_key_12345"
        service_name = "test_service"
        
        await config_manager.store_api_key(service_name, api_key)
        retrieved_key = await config_manager.get_api_key(service_name)
        
        assert retrieved_key == api_key
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_api_key(self, config_manager):
        """Test retrieving nonexistent API key."""
        await config_manager.initialize()
        
        retrieved_key = await config_manager.get_api_key('nonexistent')
        
        assert retrieved_key is None
    
    @pytest.mark.asyncio
    async def test_encryption_key_persistence(self, config_manager):
        """Test that encryption key persists between sessions."""
        await config_manager.initialize()
        original_key = config_manager._encryption_key
        
        # Create new manager instance with same config dir
        new_manager = ConfigManager(config_dir=config_manager.config_dir)
        await new_manager.initialize()
        
        assert new_manager._encryption_key == original_key
    
    @pytest.mark.asyncio
    async def test_encryption_key_permissions(self, config_manager):
        """Test that encryption key file has correct permissions."""
        await config_manager.initialize()
        
        key_file = config_manager.config_dir / '.encryption_key'
        assert key_file.exists()
        
        # Check permissions (owner read/write only)
        import stat
        file_mode = key_file.stat().st_mode
        assert stat.filemode(file_mode) == '-rw-------'


class TestSettingsManagement:
    """Test configuration settings management."""
    
    @pytest.mark.asyncio
    async def test_update_setting_nested(self, config_manager):
        """Test updating nested configuration settings."""
        await config_manager.initialize()
        
        await config_manager.update_setting('ui_preferences.theme', 'light')
        
        assert config_manager.config.ui_preferences.theme == 'light'
    
    @pytest.mark.asyncio
    async def test_update_setting_top_level(self, config_manager):
        """Test updating top-level configuration settings."""
        await config_manager.initialize()
        
        await config_manager.update_setting('version', '2.0.0')
        
        assert config_manager.config.version == '2.0.0'
    
    @pytest.mark.asyncio
    async def test_get_setting_existing(self, config_manager):
        """Test retrieving existing configuration setting."""
        await config_manager.initialize()
        
        theme = await config_manager.get_setting('ui_preferences.theme')
        
        assert theme == 'dark'  # Default value
    
    @pytest.mark.asyncio
    async def test_get_setting_nonexistent_with_default(self, config_manager):
        """Test retrieving nonexistent setting with default value."""
        await config_manager.initialize()
        
        value = await config_manager.get_setting('nonexistent.setting', 'default_value')
        
        assert value == 'default_value'
    
    @pytest.mark.asyncio
    async def test_update_setting_creates_nested_structure(self, config_manager):
        """Test that updating nested settings creates structure."""
        await config_manager.initialize()
        
        await config_manager.update_setting('custom_settings.new_feature.enabled', True)
        
        assert config_manager.config.custom_settings['new_feature']['enabled'] is True


class TestConfigurationGetters:
    """Test configuration getter methods."""
    
    @pytest.mark.asyncio
    async def test_get_project_defaults(self, config_manager):
        """Test retrieving project defaults."""
        await config_manager.initialize()
        
        defaults = config_manager.get_project_defaults()
        
        assert isinstance(defaults, ProjectDefaults)
        assert defaults.default_template == 'basic'
        assert defaults.auto_validation is True
    
    @pytest.mark.asyncio
    async def test_get_ui_preferences(self, config_manager):
        """Test retrieving UI preferences."""
        await config_manager.initialize()
        
        prefs = config_manager.get_ui_preferences()
        
        assert isinstance(prefs, UIPreferences)
        assert prefs.theme == 'dark'
        assert prefs.font_size == 12
    
    @pytest.mark.asyncio
    async def test_get_security_config(self, config_manager):
        """Test retrieving security configuration."""
        await config_manager.initialize()
        
        security = config_manager.get_security_config()
        
        assert isinstance(security, SecurityConfig)
        assert security.sandbox_enabled is True
        assert security.audit_logging is True
    
    def test_get_config_dir(self, config_manager):
        """Test retrieving configuration directory."""
        config_dir = config_manager.get_config_dir()
        
        assert isinstance(config_dir, Path)
        assert config_dir.exists()


class TestConfigurationReset:
    """Test configuration reset functionality."""
    
    @pytest.mark.asyncio
    async def test_reset_to_defaults(self, config_manager):
        """Test resetting configuration to defaults."""
        await config_manager.initialize()
        
        # Modify configuration
        await config_manager.update_setting('ui_preferences.theme', 'custom')
        await config_manager.store_api_key('test', 'key')
        
        # Store original encrypted data
        original_encrypted = config_manager._encrypted_data.copy()
        
        # Reset to defaults
        await config_manager.reset_to_defaults()
        
        # Check that config is reset but encrypted data preserved
        assert config_manager.config.ui_preferences.theme == 'dark'  # Default
        assert config_manager._encrypted_data == original_encrypted


class TestCleanup:
    """Test configuration manager cleanup."""
    
    @pytest.mark.asyncio
    async def test_cleanup(self, config_manager):
        """Test configuration manager cleanup."""
        await config_manager.initialize()
        await config_manager.store_api_key('test', 'key')
        
        await config_manager.cleanup()
        
        # Check that sensitive data is cleared
        assert config_manager._encryption_key is None
        assert len(config_manager._encrypted_data) == 0


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_store_api_key_without_encryption(self, config_manager):
        """Test storing API key without initialized encryption."""
        # Don't initialize - no encryption key available
        with pytest.raises(RuntimeError, match="Encryption not initialized"):
            await config_manager.store_api_key('test', 'key')
    
    @pytest.mark.asyncio
    async def test_set_ai_service_config_without_initialization(self, config_manager):
        """Test setting service config without initialization."""
        service_config = AIServiceConfig(service_name='test')
        
        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            await config_manager.set_ai_service_config('test', service_config)
    
    @pytest.mark.asyncio
    async def test_update_setting_without_initialization(self, config_manager):
        """Test updating setting without initialization."""
        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            await config_manager.update_setting('test', 'value')
    
    @pytest.mark.asyncio
    async def test_corrupted_encrypted_file_handling(self, config_manager):
        """Test handling of corrupted encrypted file."""
        await config_manager.initialize()
        
        # Create corrupted encrypted file
        with open(config_manager.encrypted_config_file, 'w') as f:
            f.write("corrupted json content")
        
        # Should handle gracefully and start with empty encrypted data
        await config_manager._load_encrypted_data()
        
        assert config_manager._encrypted_data == {}


class TestPlatformSpecific:
    """Test platform-specific behavior."""
    
    @patch('os.name', 'nt')
    @patch.dict(os.environ, {'APPDATA': '/windows/appdata'})
    def test_default_config_dir_windows(self):
        """Test default config directory on Windows."""
        manager = ConfigManager()
        
        assert '/windows/appdata/claude-tiu' in str(manager.config_dir)
    
    @patch('os.name', 'posix')
    @patch.dict(os.environ, {'XDG_CONFIG_HOME': '/unix/config'})
    def test_default_config_dir_unix(self):
        """Test default config directory on Unix-like systems."""
        manager = ConfigManager()
        
        assert '/unix/config/claude-tiu' in str(manager.config_dir)
    
    @patch('os.name', 'posix')
    @patch.dict(os.environ, {}, clear=True)
    def test_default_config_dir_unix_fallback(self):
        """Test default config directory fallback on Unix systems."""
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path('/home/user')
            manager = ConfigManager()
            
            assert '/home/user/.config/claude-tiu' in str(manager.config_dir)


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_ai_service_config_validation(self):
        """Test AI service configuration validation."""
        # Valid config
        config = AIServiceConfig(
            service_name='claude',
            timeout=300,
            max_retries=3
        )
        assert config.service_name == 'claude'
        
        # Invalid timeout (too low)
        with pytest.raises(ValueError):
            AIServiceConfig(
                service_name='claude',
                timeout=5  # Below minimum of 10
            )
        
        # Invalid max_retries (too high)
        with pytest.raises(ValueError):
            AIServiceConfig(
                service_name='claude',
                max_retries=15  # Above maximum of 10
            )
    
    def test_project_defaults_validation(self):
        """Test project defaults validation."""
        # Valid config
        defaults = ProjectDefaults(
            code_quality_threshold=0.8,
            backup_interval_minutes=60
        )
        assert defaults.code_quality_threshold == 0.8
        
        # Invalid quality threshold
        with pytest.raises(ValueError):
            ProjectDefaults(code_quality_threshold=1.5)  # Above 1.0
        
        # Invalid backup interval
        with pytest.raises(ValueError):
            ProjectDefaults(backup_interval_minutes=2000)  # Above 1440
    
    def test_ui_preferences_validation(self):
        """Test UI preferences validation."""
        # Valid config
        prefs = UIPreferences(
            font_size=14,
            update_interval_seconds=30
        )
        assert prefs.font_size == 14
        
        # Invalid font size
        with pytest.raises(ValueError):
            UIPreferences(font_size=50)  # Above maximum of 24
        
        # Invalid log level
        with pytest.raises(ValueError):
            UIPreferences(log_level="INVALID")


class TestConcurrencyAndThreadSafety:
    """Test concurrent access and thread safety."""
    
    @pytest.mark.asyncio
    async def test_concurrent_config_updates(self, config_manager):
        """Test concurrent configuration updates."""
        await config_manager.initialize()
        
        # Define concurrent update tasks
        async def update_theme():
            await config_manager.update_setting('ui_preferences.theme', 'light')
        
        async def update_font_size():
            await config_manager.update_setting('ui_preferences.font_size', 16)
        
        async def store_api_key():
            await config_manager.store_api_key('concurrent_service', 'test_key')
        
        # Run tasks concurrently
        await asyncio.gather(
            update_theme(),
            update_font_size(), 
            store_api_key()
        )
        
        # Verify all updates were applied
        assert config_manager.config.ui_preferences.theme == 'light'
        assert config_manager.config.ui_preferences.font_size == 16
        assert await config_manager.get_api_key('concurrent_service') == 'test_key'


@pytest.mark.performance
class TestPerformance:
    """Test configuration manager performance."""
    
    @pytest.mark.asyncio
    async def test_large_config_performance(self, config_manager):
        """Test performance with large configuration."""
        await config_manager.initialize()
        
        # Create large configuration
        import time
        start_time = time.time()
        
        for i in range(100):
            service_config = AIServiceConfig(
                service_name=f'service_{i}',
                endpoint_url=f'https://api{i}.example.com'
            )
            await config_manager.set_ai_service_config(f'service_{i}', service_config)
        
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert end_time - start_time < 5.0  # 5 seconds
        
        # Verify all services were added
        assert len(config_manager.config.ai_services) == 100
    
    @pytest.mark.asyncio
    async def test_encryption_performance(self, config_manager):
        """Test encryption/decryption performance."""
        await config_manager.initialize()
        
        import time
        start_time = time.time()
        
        # Store and retrieve multiple API keys
        for i in range(50):
            await config_manager.store_api_key(f'service_{i}', f'key_{i}_very_long_api_key_value')
            retrieved = await config_manager.get_api_key(f'service_{i}')
            assert retrieved == f'key_{i}_very_long_api_key_value'
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10.0  # 10 seconds
