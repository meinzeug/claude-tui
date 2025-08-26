"""Enhanced Configuration Service Implementation."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml
from dataclasses import dataclass, asdict

from ..interfaces.service_interfaces import ConfigInterface

logger = logging.getLogger(__name__)


@dataclass
class UIConfig:
    """UI configuration settings."""
    theme: str = "dark"
    auto_refresh: bool = True
    show_line_numbers: bool = True
    font_size: int = 12
    font_family: str = "monospace"
    show_hidden_files: bool = False
    terminal_height: int = 10


@dataclass
class AIConfig:
    """AI service configuration."""
    provider: str = "anthropic"
    model: str = "claude-3-sonnet"
    max_tokens: int = 4096
    temperature: float = 0.7
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    timeout: int = 30


@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = False
    log_level: str = "INFO"
    auto_save: bool = True
    backup_count: int = 5
    recent_projects_count: int = 10
    check_updates: bool = True


@dataclass
class ProjectConfig:
    """Project-specific configuration."""
    default_language: str = "python"
    auto_detect_language: bool = True
    git_integration: bool = True
    lint_on_save: bool = True
    format_on_save: bool = False
    test_framework: Optional[str] = None


class ConfigService(ConfigInterface):
    """Enhanced configuration service with proper file handling and validation."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration service."""
        self.config_dir = config_dir or self._get_default_config_dir()
        self.config_file = self.config_dir / "config.yaml"
        self.config_backup = self.config_dir / "config.backup.yaml"
        
        # Configuration sections
        self.ui = UIConfig()
        self.ai = AIConfig()
        self.app = AppConfig()
        self.project = ProjectConfig()
        
        self._initialized = False
        self._config_data: Dict[str, Any] = {}
        
    async def initialize(self) -> None:
        """Initialize configuration system."""
        if self._initialized:
            return
            
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Load configuration
            await self.load_config()
            
            self._initialized = True
            logger.info(f"Configuration service initialized: {self.config_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize config service: {e}")
            # Use defaults if initialization fails
            self._initialized = True
    
    async def load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = config_path or self.config_file
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.suffix.lower() == '.json':
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f) or {}
                
                # Update configuration sections
                if 'ui' in data:
                    self._update_dataclass(self.ui, data['ui'])
                if 'ai' in data:
                    self._update_dataclass(self.ai, data['ai'])
                if 'app' in data:
                    self._update_dataclass(self.app, data['app'])
                if 'project' in data:
                    self._update_dataclass(self.project, data['project'])
                
                self._config_data = data
                logger.debug(f"Configuration loaded from: {config_path}")
                
                return data
            else:
                # Create default configuration
                logger.info(f"No config file found, creating default: {config_path}")
                await self.save_config({})
                return {}
                
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Try to load from backup
            if config_path == self.config_file and self.config_backup.exists():
                logger.info("Attempting to load from backup configuration")
                return await self.load_config(self.config_backup)
            return {}
    
    async def save_config(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        config_path = config_path or self.config_file
        
        try:
            # Prepare configuration data
            if config is None:
                config = {
                    'ui': asdict(self.ui),
                    'ai': asdict(self.ai),
                    'app': asdict(self.app),
                    'project': asdict(self.project)
                }
            
            # Create backup before saving
            if config_path == self.config_file and config_path.exists():
                try:
                    import shutil
                    shutil.copy2(config_path, self.config_backup)
                except Exception as e:
                    logger.warning(f"Failed to create config backup: {e}")
            
            # Save configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self._config_data = config
            logger.debug(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        try:
            keys = key.split('.')
            
            # Handle section-based access
            if keys[0] == 'ui':
                obj = self.ui
            elif keys[0] == 'ai':
                obj = self.ai
            elif keys[0] == 'app':
                obj = self.app
            elif keys[0] == 'project':
                obj = self.project
            else:
                # Fallback to raw config data
                obj = self._config_data
                keys = [keys[0]]  # Reset to first key only
            
            # Navigate through the keys
            value = obj
            for k in keys[1:] if len(keys) > 1 else keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to get config value for key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support."""
        try:
            keys = key.split('.')
            
            # Handle section-based access
            if keys[0] == 'ui' and len(keys) > 1:
                if hasattr(self.ui, keys[1]):
                    setattr(self.ui, keys[1], value)
                    return
            elif keys[0] == 'ai' and len(keys) > 1:
                if hasattr(self.ai, keys[1]):
                    setattr(self.ai, keys[1], value)
                    return
            elif keys[0] == 'app' and len(keys) > 1:
                if hasattr(self.app, keys[1]):
                    setattr(self.app, keys[1], value)
                    return
            elif keys[0] == 'project' and len(keys) > 1:
                if hasattr(self.project, keys[1]):
                    setattr(self.project, keys[1], value)
                    return
            
            # Fallback to raw config data
            config = self._config_data
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
            
            logger.debug(f"Configuration value set: {key} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to set config value for key '{key}': {e}")
    
    async def reload(self) -> None:
        """Reload configuration from file."""
        try:
            await self.load_config()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration section."""
        return self.ui
    
    def get_ai_config(self) -> AIConfig:
        """Get AI configuration section."""
        return self.ai
    
    def get_app_config(self) -> AppConfig:
        """Get app configuration section."""
        return self.app
    
    def get_project_config(self) -> ProjectConfig:
        """Get project configuration section."""
        return self.project
    
    def export_config(self, export_path: Path, include_sensitive: bool = False) -> None:
        """Export configuration to a file."""
        try:
            config = {
                'ui': asdict(self.ui),
                'ai': asdict(self.ai),
                'app': asdict(self.app),
                'project': asdict(self.project)
            }
            
            # Remove sensitive data if requested
            if not include_sensitive:
                if 'api_key' in config.get('ai', {}):
                    config['ai']['api_key'] = None
            
            with open(export_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration exported to: {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise
    
    def import_config(self, import_path: Path, merge: bool = True) -> None:
        """Import configuration from a file."""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                if import_path.suffix.lower() == '.json':
                    imported_config = json.load(f)
                else:
                    imported_config = yaml.safe_load(f)
            
            if merge:
                # Merge with existing configuration
                for section, values in imported_config.items():
                    if section in ['ui', 'ai', 'app', 'project']:
                        if hasattr(self, section):
                            section_obj = getattr(self, section)
                            self._update_dataclass(section_obj, values)
            else:
                # Replace existing configuration
                if 'ui' in imported_config:
                    self._update_dataclass(self.ui, imported_config['ui'])
                if 'ai' in imported_config:
                    self._update_dataclass(self.ai, imported_config['ai'])
                if 'app' in imported_config:
                    self._update_dataclass(self.app, imported_config['app'])
                if 'project' in imported_config:
                    self._update_dataclass(self.project, imported_config['project'])
            
            logger.info(f"Configuration imported from: {import_path}")
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            raise
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        try:
            self.ui = UIConfig()
            self.ai = AIConfig()
            self.app = AppConfig()
            self.project = ProjectConfig()
            self._config_data = {}
            logger.info("Configuration reset to defaults")
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return list of issues."""
        issues = []
        
        try:
            # Validate UI config
            if self.ui.font_size < 8 or self.ui.font_size > 72:
                issues.append("UI font_size should be between 8 and 72")
            
            if self.ui.terminal_height < 5 or self.ui.terminal_height > 50:
                issues.append("UI terminal_height should be between 5 and 50")
            
            # Validate AI config
            if self.ai.max_tokens < 1 or self.ai.max_tokens > 100000:
                issues.append("AI max_tokens should be between 1 and 100000")
            
            if self.ai.temperature < 0 or self.ai.temperature > 2:
                issues.append("AI temperature should be between 0 and 2")
            
            if self.ai.timeout < 1 or self.ai.timeout > 300:
                issues.append("AI timeout should be between 1 and 300 seconds")
            
            # Validate app config
            if self.app.backup_count < 0 or self.app.backup_count > 100:
                issues.append("App backup_count should be between 0 and 100")
            
            if self.app.recent_projects_count < 0 or self.app.recent_projects_count > 50:
                issues.append("App recent_projects_count should be between 0 and 50")
            
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        return issues
    
    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory."""
        if os.name == 'nt':  # Windows
            config_dir = Path(os.environ.get('APPDATA', Path.home())) / "ClaudeTUI"
        else:  # Unix-like
            config_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / "claude-tui"
        
        return config_dir
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """Update dataclass object with dictionary data."""
        try:
            for key, value in data.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
        except Exception as e:
            logger.error(f"Failed to update dataclass: {e}")
    
    async def auto_save(self) -> None:
        """Auto-save configuration if enabled."""
        if self.app.auto_save:
            try:
                await self.save_config()
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration service information."""
        return {
            'config_dir': str(self.config_dir),
            'config_file': str(self.config_file),
            'initialized': self._initialized,
            'backup_exists': self.config_backup.exists(),
            'sections': ['ui', 'ai', 'app', 'project']
        }