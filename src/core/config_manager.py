"""
Configuration Management System for claude-tiu.

This module handles project configuration, settings persistence, and environment
management with validation and type safety.

Key Features:
- Hierarchical configuration (global, project, user)
- Type-safe configuration with Pydantic models
- Environment variable integration
- Configuration validation and defaults
- Hot-reloading configuration updates
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ValidationError

from .types import ConfigDict, PathStr, ProjectConfig


T = TypeVar('T', bound=BaseModel)


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


class ConfigManager:
    """
    Centralized configuration management system.
    
    Manages hierarchical configuration with support for:
    - Global system settings
    - Project-specific configurations
    - User preferences
    - Environment variable overrides
    """
    
    def __init__(
        self, 
        config_dir: Optional[PathStr] = None,
        auto_create_dirs: bool = True
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Root configuration directory
            auto_create_dirs: Automatically create config directories
        """
        self.config_dir = Path(config_dir) if config_dir else self._get_default_config_dir()
        self._config_cache: Dict[str, Any] = {}
        self._watchers: Dict[str, Any] = {}
        
        if auto_create_dirs:
            self._ensure_config_dirs()
    
    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory based on OS."""
        if os.name == 'nt':  # Windows
            base_dir = Path(os.getenv('APPDATA', '~/.config'))
        else:  # Unix-like
            base_dir = Path(os.getenv('XDG_CONFIG_HOME', '~/.config'))
        
        return (base_dir / 'claude-tiu').expanduser()
    
    def _ensure_config_dirs(self) -> None:
        """Ensure configuration directories exist."""
        dirs_to_create = [
            self.config_dir,
            self.config_dir / 'projects',
            self.config_dir / 'templates',
            self.config_dir / 'cache'
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_global_config(self) -> Dict[str, Any]:
        """
        Get global system configuration.
        
        Returns:
            Global configuration dictionary
        """
        config_file = self.config_dir / 'global.yaml'
        return self._load_config_file(config_file, self._get_global_defaults())
    
    def set_global_config(self, config: Dict[str, Any]) -> None:
        """
        Set global system configuration.
        
        Args:
            config: Configuration dictionary to save
        """
        config_file = self.config_dir / 'global.yaml'
        self._save_config_file(config_file, config)
        self._invalidate_cache('global')
    
    def get_project_config(self, project_id: str) -> ProjectConfig:
        """
        Get project-specific configuration.
        
        Args:
            project_id: Unique project identifier
            
        Returns:
            Project configuration model
            
        Raises:
            ConfigurationError: If project config is invalid
        """
        config_file = self.config_dir / 'projects' / f'{project_id}.yaml'
        
        if not config_file.exists():
            # Return default project config
            return ProjectConfig(name=f'Project {project_id}')
        
        try:
            config_data = self._load_config_file(config_file, {})
            return ProjectConfig(**config_data)
        except ValidationError as e:
            raise ConfigurationError(
                f"Invalid project configuration for {project_id}: {e}"
            ) from e
    
    def set_project_config(self, project_id: str, config: ProjectConfig) -> None:
        """
        Set project-specific configuration.
        
        Args:
            project_id: Unique project identifier
            config: Project configuration model
        """
        config_file = self.config_dir / 'projects' / f'{project_id}.yaml'
        config_data = config.dict(exclude_unset=True)
        self._save_config_file(config_file, config_data)
        self._invalidate_cache(f'project_{project_id}')
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """
        Get user preference settings.
        
        Returns:
            User preferences dictionary
        """
        config_file = self.config_dir / 'preferences.yaml'
        return self._load_config_file(config_file, self._get_user_defaults())
    
    def set_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Set user preference settings.
        
        Args:
            preferences: User preferences dictionary
        """
        config_file = self.config_dir / 'preferences.yaml'
        self._save_config_file(config_file, preferences)
        self._invalidate_cache('preferences')
    
    def get_template_config(self, template_name: str) -> Dict[str, Any]:
        """
        Get project template configuration.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template configuration dictionary
        """
        config_file = self.config_dir / 'templates' / f'{template_name}.yaml'
        
        if not config_file.exists():
            raise ConfigurationError(f"Template '{template_name}' not found")
        
        return self._load_config_file(config_file, {})
    
    def save_template_config(
        self, 
        template_name: str, 
        config: Dict[str, Any]
    ) -> None:
        """
        Save project template configuration.
        
        Args:
            template_name: Name of the template
            config: Template configuration dictionary
        """
        config_file = self.config_dir / 'templates' / f'{template_name}.yaml'
        self._save_config_file(config_file, config)
        self._invalidate_cache(f'template_{template_name}')
    
    def list_templates(self) -> List[str]:
        """
        List available project templates.
        
        Returns:
            List of template names
        """
        template_dir = self.config_dir / 'templates'
        if not template_dir.exists():
            return []
        
        return [
            f.stem for f in template_dir.glob('*.yaml')
            if f.is_file()
        ]
    
    def get_config_value(
        self,
        key_path: str,
        default: Any = None,
        config_type: str = 'global'
    ) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'ai.timeout')
            default: Default value if key not found
            config_type: Type of config ('global', 'preferences')
            
        Returns:
            Configuration value or default
            
        Example:
            >>> manager.get_config_value('ai.claude_code.timeout', 300)
        """
        if config_type == 'global':
            config = self.get_global_config()
        elif config_type == 'preferences':
            config = self.get_user_preferences()
        else:
            raise ValueError(f"Invalid config type: {config_type}")
        
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_config_value(
        self,
        key_path: str,
        value: Any,
        config_type: str = 'global'
    ) -> None:
        """
        Set nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
            config_type: Type of config ('global', 'preferences')
        """
        if config_type == 'global':
            config = self.get_global_config()
        elif config_type == 'preferences':
            config = self.get_user_preferences()
        else:
            raise ValueError(f"Invalid config type: {config_type}")
        
        keys = key_path.split('.')
        current = config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
        
        # Save updated config
        if config_type == 'global':
            self.set_global_config(config)
        elif config_type == 'preferences':
            self.set_user_preferences(config)
    
    def merge_configs(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to merge in
            
        Returns:
            Merged configuration dictionary
        """
        result = base_config.copy()
        
        for key, value in override_config.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(
        self,
        config: Dict[str, Any],
        model_class: Type[T]
    ) -> T:
        """
        Validate configuration against Pydantic model.
        
        Args:
            config: Configuration dictionary
            model_class: Pydantic model class for validation
            
        Returns:
            Validated model instance
            
        Raises:
            ConfigurationError: If validation fails
        """
        try:
            return model_class(**config)
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
    
    def resolve_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve environment variables in configuration values.
        
        Args:
            config: Configuration dictionary with potential env vars
            
        Returns:
            Configuration with resolved environment variables
        """
        def resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                # Handle ${VAR_NAME} or ${VAR_NAME:default} syntax
                import re
                pattern = r'\$\{([^}]+)\}'
                
                def replacer(match):
                    var_spec = match.group(1)
                    if ':' in var_spec:
                        var_name, default_val = var_spec.split(':', 1)
                    else:
                        var_name, default_val = var_spec, ''
                    
                    return os.getenv(var_name, default_val)
                
                return re.sub(pattern, replacer, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value
        
        return resolve_value(config)
    
    def backup_config(self, backup_name: Optional[str] = None) -> Path:
        """
        Create backup of all configuration files.
        
        Args:
            backup_name: Optional backup name, defaults to timestamp
            
        Returns:
            Path to backup directory
        """
        import shutil
        from datetime import datetime
        
        if backup_name is None:
            backup_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        backup_dir = self.config_dir / 'backups' / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all config files
        for item in self.config_dir.iterdir():
            if item.is_file() and item.suffix in ['.yaml', '.yml', '.json']:
                shutil.copy2(item, backup_dir / item.name)
            elif item.is_dir() and item.name in ['projects', 'templates']:
                shutil.copytree(item, backup_dir / item.name, dirs_exist_ok=True)
        
        return backup_dir
    
    def restore_config(self, backup_name: str) -> None:
        """
        Restore configuration from backup.
        
        Args:
            backup_name: Name of backup to restore from
            
        Raises:
            ConfigurationError: If backup doesn't exist
        """
        import shutil
        
        backup_dir = self.config_dir / 'backups' / backup_name
        if not backup_dir.exists():
            raise ConfigurationError(f"Backup '{backup_name}' not found")
        
        # Clear cache before restoring
        self._config_cache.clear()
        
        # Restore files
        for item in backup_dir.iterdir():
            target_path = self.config_dir / item.name
            if item.is_file():
                shutil.copy2(item, target_path)
            elif item.is_dir():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(item, target_path)
    
    def _load_config_file(
        self,
        file_path: Path,
        default_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load configuration from file with caching."""
        cache_key = str(file_path)
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        if not file_path.exists():
            config = default_config.copy()
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix.lower() == '.json':
                        config = json.load(f)
                    else:  # YAML
                        config = yaml.safe_load(f) or {}
                
                # Merge with defaults to ensure all required keys exist
                config = self.merge_configs(default_config, config)
                
                # Resolve environment variables
                config = self.resolve_env_vars(config)
                
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                raise ConfigurationError(
                    f"Failed to parse config file {file_path}: {e}"
                ) from e
        
        self._config_cache[cache_key] = config
        return config
    
    def _save_config_file(
        self,
        file_path: Path,
        config: Dict[str, Any]
    ) -> None:
        """Save configuration to file."""
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2, sort_keys=True)
                else:  # YAML
                    yaml.safe_dump(
                        config, f,
                        default_flow_style=False,
                        sort_keys=True,
                        indent=2
                    )
        except (IOError, OSError) as e:
            raise ConfigurationError(
                f"Failed to save config file {file_path}: {e}"
            ) from e
    
    def _invalidate_cache(self, cache_key: str) -> None:
        """Invalidate specific cache entry."""
        keys_to_remove = [
            key for key in self._config_cache.keys()
            if cache_key in key
        ]
        for key in keys_to_remove:
            del self._config_cache[key]
    
    def _get_global_defaults(self) -> Dict[str, Any]:
        """Get default global configuration."""
        return {
            'system': {
                'log_level': 'INFO',
                'max_workers': 5,
                'timeout_seconds': 300,
                'memory_limit_mb': 1024
            },
            'ai': {
                'claude_code': {
                    'enabled': True,
                    'timeout': 300,
                    'max_retries': 3,
                    'max_tokens': 4000
                },
                'claude_flow': {
                    'enabled': True,
                    'topology': 'mesh',
                    'max_agents': 8,
                    'coordination_enabled': True
                }
            },
            'validation': {
                'enabled': True,
                'anti_hallucination': True,
                'placeholder_tolerance': 5,
                'auto_fix_enabled': True,
                'validation_interval': 30
            },
            'security': {
                'sandbox_execution': True,
                'secret_scanning': True,
                'input_validation': True
            }
        }
    
    def _get_user_defaults(self) -> Dict[str, Any]:
        """Get default user preferences."""
        return {
            'ui': {
                'theme': 'dark',
                'show_progress_details': True,
                'auto_refresh_interval': 10,
                'notifications_enabled': True
            },
            'editor': {
                'tab_size': 4,
                'insert_final_newline': True,
                'trim_trailing_whitespace': True
            },
            'shortcuts': {
                'quit': 'ctrl+q',
                'refresh': 'f5',
                'help': 'f1'
            }
        }


# Convenience functions for common configuration tasks

def load_project_config(project_id: str, config_dir: Optional[PathStr] = None) -> ProjectConfig:
    """
    Convenience function to load project configuration.
    
    Args:
        project_id: Project identifier
        config_dir: Optional configuration directory
        
    Returns:
        Project configuration model
    """
    manager = ConfigManager(config_dir)
    return manager.get_project_config(project_id)


def save_project_config(
    project_id: str,
    config: ProjectConfig,
    config_dir: Optional[PathStr] = None
) -> None:
    """
    Convenience function to save project configuration.
    
    Args:
        project_id: Project identifier
        config: Project configuration
        config_dir: Optional configuration directory
    """
    manager = ConfigManager(config_dir)
    manager.set_project_config(project_id, config)