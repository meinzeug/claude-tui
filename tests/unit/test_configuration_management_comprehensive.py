"""
Comprehensive Configuration Management Tests

Tests for configuration loading, validation, merging, environment handling,
and security features in the Claude TIU application.

This module ensures robust configuration management across different
environments and deployment scenarios.
"""

import pytest
import json
import yaml
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any, List, Optional

# Import test fixtures
from tests.fixtures.comprehensive_test_fixtures import (
    TestDataGenerator,
    MockComponents,
    TestFileSystem
)


class TestConfigurationLoader:
    """Tests for configuration file loading and parsing."""
    
    @pytest.fixture
    def config_loader(self):
        """Mock configuration loader."""
        from claude_tiu.config.loader import ConfigLoader
        return ConfigLoader()
    
    @pytest.fixture
    def sample_configs(self):
        """Sample configuration data for testing."""
        return {
            'json_config': {
                'app': {'name': 'claude-tiu', 'version': '1.0.0'},
                'ai': {'provider': 'anthropic', 'model': 'claude-3'},
                'ui': {'theme': 'dark', 'refresh_rate': 60}
            },
            'yaml_config': {
                'database': {'host': 'localhost', 'port': 5432},
                'cache': {'redis_url': 'redis://localhost:6379'},
                'logging': {'level': 'INFO', 'format': 'json'}
            },
            'env_config': {
                'CLAUDE_API_KEY': 'test-key-12345',
                'DEBUG': 'true',
                'DATABASE_URL': 'postgresql://user:pass@localhost/db'
            }
        }
    
    @pytest.mark.asyncio
    async def test_json_config_loading(self, config_loader, sample_configs):
        """Test loading JSON configuration files."""
        json_data = json.dumps(sample_configs['json_config'])
        
        with patch('builtins.open', mock_open(read_data=json_data)):
            with patch('pathlib.Path.exists', return_value=True):
                config = await config_loader.load_json('config.json')
        
        assert config['app']['name'] == 'claude-tiu'
        assert config['ai']['provider'] == 'anthropic'
        assert config['ui']['theme'] == 'dark'
    
    @pytest.mark.asyncio
    async def test_yaml_config_loading(self, config_loader, sample_configs):
        """Test loading YAML configuration files."""
        yaml_data = yaml.dump(sample_configs['yaml_config'])
        
        with patch('builtins.open', mock_open(read_data=yaml_data)):
            with patch('pathlib.Path.exists', return_value=True):
                config = await config_loader.load_yaml('config.yml')
        
        assert config['database']['host'] == 'localhost'
        assert config['cache']['redis_url'] == 'redis://localhost:6379'
        assert config['logging']['level'] == 'INFO'
    
    @pytest.mark.asyncio
    async def test_environment_config_loading(self, config_loader, sample_configs):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, sample_configs['env_config']):
            config = await config_loader.load_environment()
        
        assert config['CLAUDE_API_KEY'] == 'test-key-12345'
        assert config['DEBUG'] == 'true'
        assert config['DATABASE_URL'] == 'postgresql://user:pass@localhost/db'
    
    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, config_loader):
        """Test handling of invalid JSON configuration."""
        invalid_json = '{"invalid": json, "missing": quote}'
        
        with patch('builtins.open', mock_open(read_data=invalid_json)):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(json.JSONDecodeError):
                    await config_loader.load_json('invalid.json')
    
    @pytest.mark.asyncio
    async def test_missing_file_handling(self, config_loader):
        """Test handling of missing configuration files."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                await config_loader.load_json('missing.json')
    
    @pytest.mark.asyncio
    async def test_config_file_permissions(self, config_loader):
        """Test handling of configuration files with restricted permissions."""
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(PermissionError):
                    await config_loader.load_json('restricted.json')


class TestConfigurationValidator:
    """Tests for configuration validation and schema checking."""
    
    @pytest.fixture
    def config_validator(self):
        """Mock configuration validator."""
        from claude_tiu.config.validator import ConfigValidator
        return ConfigValidator()
    
    @pytest.fixture
    def valid_config(self):
        """Valid configuration for testing."""
        return {
            'app': {
                'name': 'claude-tiu',
                'version': '1.0.0',
                'debug': False
            },
            'ai': {
                'provider': 'anthropic',
                'model': 'claude-3-sonnet',
                'api_key': 'sk-test-key',
                'max_tokens': 4096,
                'temperature': 0.7
            },
            'ui': {
                'theme': 'dark',
                'refresh_rate': 60,
                'auto_save': True
            },
            'performance': {
                'max_workers': 4,
                'timeout': 30,
                'cache_size': 1000
            }
        }
    
    @pytest.fixture
    def config_schema(self):
        """Configuration validation schema."""
        return {
            'type': 'object',
            'properties': {
                'app': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'version': {'type': 'string'},
                        'debug': {'type': 'boolean'}
                    },
                    'required': ['name', 'version']
                },
                'ai': {
                    'type': 'object',
                    'properties': {
                        'provider': {'type': 'string', 'enum': ['anthropic', 'openai']},
                        'model': {'type': 'string'},
                        'api_key': {'type': 'string'},
                        'max_tokens': {'type': 'integer', 'minimum': 1, 'maximum': 100000},
                        'temperature': {'type': 'number', 'minimum': 0, 'maximum': 2}
                    },
                    'required': ['provider', 'model', 'api_key']
                }
            },
            'required': ['app', 'ai']
        }
    
    @pytest.mark.asyncio
    async def test_valid_configuration_validation(self, config_validator, valid_config, config_schema):
        """Test validation of valid configuration."""
        result = await config_validator.validate(valid_config, config_schema)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.normalized_config == valid_config
    
    @pytest.mark.asyncio
    async def test_missing_required_field_validation(self, config_validator, valid_config, config_schema):
        """Test validation with missing required fields."""
        invalid_config = valid_config.copy()
        del invalid_config['ai']['api_key']
        
        result = await config_validator.validate(invalid_config, config_schema)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('api_key' in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_invalid_type_validation(self, config_validator, valid_config, config_schema):
        """Test validation with invalid data types."""
        invalid_config = valid_config.copy()
        invalid_config['ai']['max_tokens'] = 'invalid_number'
        
        result = await config_validator.validate(invalid_config, config_schema)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('max_tokens' in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_enum_validation(self, config_validator, valid_config, config_schema):
        """Test validation with invalid enum values."""
        invalid_config = valid_config.copy()
        invalid_config['ai']['provider'] = 'invalid_provider'
        
        result = await config_validator.validate(invalid_config, config_schema)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('provider' in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_range_validation(self, config_validator, valid_config, config_schema):
        """Test validation with out-of-range values."""
        invalid_config = valid_config.copy()
        invalid_config['ai']['temperature'] = 5.0  # Above maximum of 2.0
        
        result = await config_validator.validate(invalid_config, config_schema)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('temperature' in error for error in result.errors)


class TestConfigurationMerger:
    """Tests for configuration merging and precedence."""
    
    @pytest.fixture
    def config_merger(self):
        """Mock configuration merger."""
        from claude_tiu.config.merger import ConfigMerger
        return ConfigMerger()
    
    @pytest.fixture
    def base_config(self):
        """Base configuration."""
        return {
            'app': {'name': 'claude-tiu', 'version': '1.0.0', 'debug': False},
            'ai': {'provider': 'anthropic', 'model': 'claude-3', 'temperature': 0.7},
            'ui': {'theme': 'light', 'refresh_rate': 30}
        }
    
    @pytest.fixture
    def override_config(self):
        """Override configuration."""
        return {
            'app': {'debug': True},
            'ai': {'temperature': 0.9, 'max_tokens': 2000},
            'ui': {'theme': 'dark'}
        }
    
    @pytest.fixture
    def env_config(self):
        """Environment configuration."""
        return {
            'app': {'version': '2.0.0'},
            'ai': {'api_key': 'env-api-key'}
        }
    
    @pytest.mark.asyncio
    async def test_simple_config_merge(self, config_merger, base_config, override_config):
        """Test simple configuration merging."""
        merged = await config_merger.merge(base_config, override_config)
        
        # Check overrides
        assert merged['app']['debug'] is True
        assert merged['ai']['temperature'] == 0.9
        assert merged['ui']['theme'] == 'dark'
        
        # Check preserved values
        assert merged['app']['name'] == 'claude-tiu'
        assert merged['ai']['provider'] == 'anthropic'
        
        # Check new values
        assert merged['ai']['max_tokens'] == 2000
    
    @pytest.mark.asyncio
    async def test_multiple_config_merge(self, config_merger, base_config, override_config, env_config):
        """Test merging multiple configurations with precedence."""
        merged = await config_merger.merge_multiple([base_config, override_config, env_config])
        
        # Environment should have highest precedence
        assert merged['app']['version'] == '2.0.0'
        assert merged['ai']['api_key'] == 'env-api-key'
        
        # Override should override base
        assert merged['app']['debug'] is True
        assert merged['ui']['theme'] == 'dark'
        
        # Base values should be preserved
        assert merged['app']['name'] == 'claude-tiu'
    
    @pytest.mark.asyncio
    async def test_deep_merge_nested_objects(self, config_merger):
        """Test deep merging of nested configuration objects."""
        config1 = {
            'database': {
                'connection': {
                    'host': 'localhost',
                    'port': 5432,
                    'ssl': True
                },
                'pool': {'min_size': 5, 'max_size': 20}
            }
        }
        
        config2 = {
            'database': {
                'connection': {
                    'host': 'production.db',
                    'timeout': 30
                },
                'pool': {'max_size': 50}
            }
        }
        
        merged = await config_merger.merge(config1, config2)
        
        # Check deep merging
        assert merged['database']['connection']['host'] == 'production.db'
        assert merged['database']['connection']['port'] == 5432
        assert merged['database']['connection']['ssl'] is True
        assert merged['database']['connection']['timeout'] == 30
        assert merged['database']['pool']['min_size'] == 5
        assert merged['database']['pool']['max_size'] == 50
    
    @pytest.mark.asyncio
    async def test_list_merge_strategies(self, config_merger):
        """Test different list merging strategies."""
        config1 = {'plugins': ['plugin1', 'plugin2']}
        config2 = {'plugins': ['plugin3', 'plugin4']}
        
        # Test append strategy
        merged_append = await config_merger.merge(
            config1, config2, list_strategy='append'
        )
        assert merged_append['plugins'] == ['plugin1', 'plugin2', 'plugin3', 'plugin4']
        
        # Test replace strategy
        merged_replace = await config_merger.merge(
            config1, config2, list_strategy='replace'
        )
        assert merged_replace['plugins'] == ['plugin3', 'plugin4']


class TestEnvironmentConfigManager:
    """Tests for environment-specific configuration management."""
    
    @pytest.fixture
    def env_manager(self):
        """Mock environment configuration manager."""
        from claude_tiu.config.environment import EnvironmentManager
        return EnvironmentManager()
    
    @pytest.fixture
    def env_configs(self):
        """Environment-specific configurations."""
        return {
            'development': {
                'app': {'debug': True},
                'ai': {'temperature': 1.0},
                'database': {'host': 'localhost'}
            },
            'staging': {
                'app': {'debug': False},
                'ai': {'temperature': 0.7},
                'database': {'host': 'staging.db'}
            },
            'production': {
                'app': {'debug': False},
                'ai': {'temperature': 0.5},
                'database': {'host': 'production.db'}
            }
        }
    
    @pytest.mark.asyncio
    async def test_environment_detection(self, env_manager):
        """Test automatic environment detection."""
        # Test with environment variable
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            env = await env_manager.detect_environment()
            assert env == 'production'
        
        # Test with default
        with patch.dict(os.environ, {}, clear=True):
            env = await env_manager.detect_environment()
            assert env == 'development'
    
    @pytest.mark.asyncio
    async def test_environment_config_loading(self, env_manager, env_configs):
        """Test loading environment-specific configurations."""
        with patch.object(env_manager, '_load_env_configs', return_value=env_configs):
            # Test development environment
            dev_config = await env_manager.load_config('development')
            assert dev_config['app']['debug'] is True
            assert dev_config['database']['host'] == 'localhost'
            
            # Test production environment
            prod_config = await env_manager.load_config('production')
            assert prod_config['app']['debug'] is False
            assert prod_config['database']['host'] == 'production.db'
    
    @pytest.mark.asyncio
    async def test_environment_variable_override(self, env_manager, env_configs):
        """Test environment variable overrides."""
        env_vars = {
            'CLAUDE_DEBUG': 'true',
            'CLAUDE_DATABASE_HOST': 'override.db',
            'CLAUDE_AI_TEMPERATURE': '0.9'
        }
        
        with patch.dict(os.environ, env_vars):
            with patch.object(env_manager, '_load_env_configs', return_value=env_configs):
                config = await env_manager.load_config('production')
                overridden = await env_manager.apply_env_overrides(config)
        
        assert overridden['app']['debug'] is True  # Overridden by env var
        assert overridden['database']['host'] == 'override.db'
        assert overridden['ai']['temperature'] == 0.9
    
    @pytest.mark.asyncio
    async def test_invalid_environment_handling(self, env_manager, env_configs):
        """Test handling of invalid environment names."""
        with patch.object(env_manager, '_load_env_configs', return_value=env_configs):
            with pytest.raises(ValueError, match="Unknown environment"):
                await env_manager.load_config('invalid_env')


class TestConfigurationSecurity:
    """Tests for configuration security features."""
    
    @pytest.fixture
    def security_manager(self):
        """Mock configuration security manager."""
        from claude_tiu.config.security import SecurityManager
        return SecurityManager()
    
    @pytest.mark.asyncio
    async def test_sensitive_data_masking(self, security_manager):
        """Test masking of sensitive configuration data."""
        config = {
            'ai': {'api_key': 'sk-very-secret-key-12345'},
            'database': {'password': 'super-secret-password'},
            'app': {'name': 'claude-tiu'}
        }
        
        masked = await security_manager.mask_sensitive_data(config)
        
        assert masked['ai']['api_key'] == 'sk-***'
        assert masked['database']['password'] == '***'
        assert masked['app']['name'] == 'claude-tiu'  # Not sensitive
    
    @pytest.mark.asyncio
    async def test_configuration_encryption(self, security_manager):
        """Test configuration encryption and decryption."""
        config = {'sensitive_data': 'very-important-secret'}
        
        # Test encryption
        encrypted = await security_manager.encrypt_config(config, 'test-key')
        assert encrypted != config
        assert 'sensitive_data' not in str(encrypted)
        
        # Test decryption
        decrypted = await security_manager.decrypt_config(encrypted, 'test-key')
        assert decrypted == config
    
    @pytest.mark.asyncio
    async def test_configuration_signing(self, security_manager):
        """Test configuration signing and verification."""
        config = {'app': {'name': 'claude-tiu', 'version': '1.0.0'}}
        
        # Test signing
        signed_config = await security_manager.sign_config(config)
        assert 'signature' in signed_config
        assert signed_config['data'] == config
        
        # Test verification
        is_valid = await security_manager.verify_config(signed_config)
        assert is_valid is True
        
        # Test tampering detection
        signed_config['data']['app']['name'] = 'tampered'
        is_valid_tampered = await security_manager.verify_config(signed_config)
        assert is_valid_tampered is False
    
    @pytest.mark.asyncio
    async def test_secret_injection_prevention(self, security_manager):
        """Test prevention of secret injection in configurations."""
        malicious_config = {
            'command': '$(cat /etc/passwd)',
            'template': '${env:SECRET_KEY}',
            'eval': 'eval(malicious_code)',
            'safe_value': 'normal_value'
        }
        
        sanitized = await security_manager.sanitize_config(malicious_config)
        
        # Should sanitize dangerous patterns
        assert sanitized['command'] != '$(cat /etc/passwd)'
        assert sanitized['template'] != '${env:SECRET_KEY}'
        assert sanitized['eval'] != 'eval(malicious_code)'
        assert sanitized['safe_value'] == 'normal_value'


class TestConfigurationPerformance:
    """Tests for configuration loading and processing performance."""
    
    @pytest.fixture
    def perf_tester(self):
        """Performance testing utilities."""
        from claude_tiu.utils.performance import PerformanceTester
        return PerformanceTester()
    
    @pytest.mark.asyncio
    async def test_config_loading_performance(self, perf_tester):
        """Test configuration loading performance benchmarks."""
        # Generate large configuration for testing
        large_config = TestDataGenerator.generate_large_config(
            num_sections=100,
            section_size=50
        )
        
        # Mock file operations
        with patch('builtins.open', mock_open(read_data=json.dumps(large_config))):
            with patch('pathlib.Path.exists', return_value=True):
                
                # Test loading performance
                start_time = perf_tester.start_timer()
                
                from claude_tiu.config.loader import ConfigLoader
                loader = ConfigLoader()
                config = await loader.load_json('large_config.json')
                
                duration = perf_tester.end_timer(start_time)
        
        # Should load large config in reasonable time
        assert duration < 1.0  # Less than 1 second
        assert len(config) == 100
    
    @pytest.mark.asyncio
    async def test_config_validation_performance(self, perf_tester):
        """Test configuration validation performance."""
        # Generate complex schema and config
        config = TestDataGenerator.generate_complex_config()
        schema = TestDataGenerator.generate_config_schema()
        
        from claude_tiu.config.validator import ConfigValidator
        validator = ConfigValidator()
        
        start_time = perf_tester.start_timer()
        result = await validator.validate(config, schema)
        duration = perf_tester.end_timer(start_time)
        
        # Should validate complex config quickly
        assert duration < 0.5  # Less than 500ms
        assert result.is_valid is True
    
    @pytest.mark.asyncio
    async def test_config_merge_performance(self, perf_tester):
        """Test configuration merging performance with multiple configs."""
        # Generate multiple large configs
        configs = [
            TestDataGenerator.generate_large_config(50, 25)
            for _ in range(10)
        ]
        
        from claude_tiu.config.merger import ConfigMerger
        merger = ConfigMerger()
        
        start_time = perf_tester.start_timer()
        merged = await merger.merge_multiple(configs)
        duration = perf_tester.end_timer(start_time)
        
        # Should merge multiple configs efficiently
        assert duration < 2.0  # Less than 2 seconds
        assert isinstance(merged, dict)
        assert len(merged) > 0


class TestConfigurationManager:
    """Integration tests for the complete configuration management system."""
    
    @pytest.fixture
    def config_manager(self):
        """Mock configuration manager integrating all components."""
        from claude_tiu.config.manager import ConfigurationManager
        return ConfigurationManager()
    
    @pytest.mark.asyncio
    async def test_complete_config_lifecycle(self, config_manager):
        """Test complete configuration lifecycle from loading to validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create test configuration files
            base_config = {
                'app': {'name': 'claude-tiu', 'version': '1.0.0'},
                'ai': {'provider': 'anthropic', 'model': 'claude-3'}
            }
            
            env_config = {
                'app': {'debug': True},
                'ai': {'temperature': 0.9}
            }
            
            # Write config files
            (config_dir / 'config.json').write_text(json.dumps(base_config))
            (config_dir / 'development.json').write_text(json.dumps(env_config))
            
            # Test loading and merging
            with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
                config = await config_manager.load_configuration(config_dir)
        
        # Verify merged configuration
        assert config['app']['name'] == 'claude-tiu'
        assert config['app']['debug'] is True
        assert config['ai']['provider'] == 'anthropic'
        assert config['ai']['temperature'] == 0.9
    
    @pytest.mark.asyncio
    async def test_configuration_hot_reload(self, config_manager):
        """Test configuration hot reloading on file changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.json'
            
            # Initial configuration
            initial_config = {'app': {'debug': False}}
            config_file.write_text(json.dumps(initial_config))
            
            # Load initial configuration
            await config_manager.load_configuration(temp_dir)
            initial_value = config_manager.get('app.debug')
            
            # Modify configuration file
            updated_config = {'app': {'debug': True}}
            config_file.write_text(json.dumps(updated_config))
            
            # Trigger reload (simulate file watcher)
            await config_manager.reload_configuration()
            updated_value = config_manager.get('app.debug')
            
            # Verify hot reload worked
            assert initial_value is False
            assert updated_value is True
    
    @pytest.mark.asyncio
    async def test_configuration_caching(self, config_manager):
        """Test configuration caching and cache invalidation."""
        # Mock expensive configuration loading
        load_count = 0
        
        async def mock_load_config():
            nonlocal load_count
            load_count += 1
            return {'loaded_at': load_count}
        
        with patch.object(config_manager, '_load_raw_config', side_effect=mock_load_config):
            # First load should hit the source
            config1 = await config_manager.get_config()
            assert config1['loaded_at'] == 1
            assert load_count == 1
            
            # Second load should use cache
            config2 = await config_manager.get_config()
            assert config2['loaded_at'] == 1
            assert load_count == 1
            
            # Cache invalidation should force reload
            await config_manager.invalidate_cache()
            config3 = await config_manager.get_config()
            assert config3['loaded_at'] == 2
            assert load_count == 2


@pytest.mark.performance
class TestConfigurationPerformanceBenchmarks:
    """Performance benchmarks for configuration management."""
    
    @pytest.mark.asyncio
    async def test_startup_config_loading_benchmark(self):
        """Benchmark configuration loading during application startup."""
        from claude_tiu.config.manager import ConfigurationManager
        
        manager = ConfigurationManager()
        
        # Generate realistic startup configuration
        config_data = TestDataGenerator.generate_startup_config()
        
        with patch.object(manager, '_load_raw_config', return_value=config_data):
            # Measure startup config loading time
            import time
            start_time = time.perf_counter()
            
            await manager.initialize()
            config = await manager.get_config()
            
            end_time = time.perf_counter()
            duration = end_time - start_time
        
        # Should load startup config very quickly
        assert duration < 0.1  # Less than 100ms
        assert config is not None
        assert 'app' in config
    
    @pytest.mark.asyncio
    async def test_concurrent_config_access_benchmark(self):
        """Benchmark concurrent configuration access performance."""
        from claude_tiu.config.manager import ConfigurationManager
        import asyncio
        
        manager = ConfigurationManager()
        await manager.initialize()
        
        async def access_config():
            """Simulate concurrent config access."""
            for _ in range(100):
                value = manager.get('app.name')
                assert value is not None
        
        # Run multiple concurrent config access tasks
        tasks = [access_config() for _ in range(50)]
        
        start_time = time.perf_counter()
        await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        
        # Should handle concurrent access efficiently
        assert duration < 2.0  # Less than 2 seconds for 5000 accesses