"""
Configuration Manager - Centralized configuration management system.

Handles all application configuration including:
- User preferences and settings
- AI service configurations
- Project templates and defaults
- Security and API key management
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import yaml
import json
from cryptography.fernet import Fernet

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AIServiceConfig(BaseModel):
    """Configuration for AI service integration."""
    service_name: str
    api_key_encrypted: Optional[str] = None
    endpoint_url: Optional[str] = None
    timeout: int = Field(default=300, ge=10, le=3600)
    max_retries: int = Field(default=3, ge=1, le=10)
    rate_limit: Optional[int] = Field(default=None, ge=1)
    model_preferences: Dict[str, str] = Field(default_factory=dict)


class ProjectDefaults(BaseModel):
    """Default settings for new projects."""
    default_template: str = "basic"
    auto_validation: bool = True
    auto_completion: bool = True
    code_quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    placeholder_detection_sensitivity: float = Field(default=0.95, ge=0.0, le=1.0)
    backup_enabled: bool = True
    backup_interval_minutes: int = Field(default=30, ge=5, le=1440)


class UIPreferences(BaseModel):
    """User interface preferences."""
    theme: str = "dark"
    font_size: int = Field(default=12, ge=8, le=24)
    show_line_numbers: bool = True
    auto_save: bool = True
    vim_mode: bool = False
    animations_enabled: bool = True
    terminal_size_min: tuple[int, int] = (80, 24)
    update_interval_seconds: int = Field(default=10, ge=1, le=60)
    show_progress_details: bool = True
    keyboard_shortcuts: Dict[str, str] = Field(default_factory=dict)
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class SecurityConfig(BaseModel):
    """Security and safety configuration."""
    sandbox_enabled: bool = True
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    allowed_file_extensions: List[str] = Field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json", ".yaml", ".yml", ".md", ".txt"
    ])
    blocked_commands: List[str] = Field(default_factory=lambda: [
        "rm -rf", "del", "format", "fdisk", "dd"
    ])
    api_key_rotation_days: int = Field(default=90, ge=30, le=365)
    audit_logging: bool = True


class AppConfig(BaseModel):
    """Main application configuration."""
    version: str = "0.1.0"
    config_version: str = "1.0"
    ai_services: Dict[str, AIServiceConfig] = Field(default_factory=dict)
    project_defaults: ProjectDefaults = Field(default_factory=ProjectDefaults)
    ui_preferences: UIPreferences = Field(default_factory=UIPreferences)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('ai_services')
    def validate_ai_services(cls, v):
        if not v:
            logger.warning("No AI services configured")
        return v


class ConfigManager:
    """
    Centralized configuration management system.
    
    The ConfigManager handles all aspects of application configuration including
    user preferences, AI service settings, security parameters, and project defaults.
    It provides secure storage for sensitive data like API keys and maintains
    configuration consistency across application restarts.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_dir: Custom configuration directory (uses default if None)
        """
        self.config_dir = config_dir or self._get_default_config_dir()
        self.config_file = self.config_dir / "config.yaml"
        self.encrypted_config_file = self.config_dir / "secrets.enc"
        
        # Runtime configuration
        self.config: Optional[AppConfig] = None
        self._encryption_key: Optional[bytes] = None
        self._encrypted_data: Dict[str, str] = {}
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ConfigManager initialized with config dir: {self.config_dir}")
    
    async def initialize(self) -> None:
        """
        Initialize configuration system.
        
        Loads existing configuration or creates defaults if none exists.
        Sets up encryption for sensitive data.
        """
        logger.info("Initializing configuration system")
        
        try:
            # Initialize encryption
            await self._initialize_encryption()
            
            # Load or create configuration
            if self.config_file.exists():
                await self.load_config()
            else:
                logger.info("No existing config found, creating default configuration")
                self.config = AppConfig()
                await self.save_config()
            
            # Load encrypted data
            await self._load_encrypted_data()
            
            logger.info("Configuration system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise
    
    async def load_config(self) -> None:
        """
        Load configuration from disk.
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            self.config = AppConfig(**config_data)
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Fall back to default config
            self.config = AppConfig()
            logger.info("Using default configuration due to load failure")
    
    async def save_config(self) -> None:
        """
        Save configuration to disk.
        """
        if not self.config:
            raise RuntimeError("No configuration to save")
        
        try:
            config_data = self.config.dict(exclude_unset=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=True)
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    async def get_ai_service_config(self, service_name: str) -> Optional[AIServiceConfig]:
        """
        Get configuration for a specific AI service.
        
        Args:
            service_name: Name of the AI service
            
        Returns:
            AIServiceConfig if found, None otherwise
        """
        if not self.config:
            return None
        
        return self.config.ai_services.get(service_name)
    
    async def set_ai_service_config(self, service_name: str, config: AIServiceConfig) -> None:
        """
        Set configuration for an AI service.
        
        Args:
            service_name: Name of the AI service
            config: Service configuration
        """
        if not self.config:
            raise RuntimeError("Configuration not initialized")
        
        self.config.ai_services[service_name] = config
        await self.save_config()
        logger.info(f"AI service '{service_name}' configuration updated")
    
    async def store_api_key(self, service_name: str, api_key: str) -> None:
        """
        Securely store an API key.
        
        Args:
            service_name: Name of the service
            api_key: API key to store
        """
        if not self._encryption_key:
            raise RuntimeError("Encryption not initialized")
        
        try:
            # Encrypt the API key
            fernet = Fernet(self._encryption_key)
            encrypted_key = fernet.encrypt(api_key.encode())
            
            # Store encrypted key
            self._encrypted_data[f"api_key_{service_name}"] = encrypted_key.decode()
            
            # Update service config with encrypted reference
            if not self.config:
                raise RuntimeError("Configuration not initialized")
            
            if service_name not in self.config.ai_services:
                self.config.ai_services[service_name] = AIServiceConfig(
                    service_name=service_name
                )
            
            self.config.ai_services[service_name].api_key_encrypted = f"api_key_{service_name}"
            
            # Save both configs
            await self.save_config()
            await self._save_encrypted_data()
            
            logger.info(f"API key for '{service_name}' stored securely")
            
        except Exception as e:
            logger.error(f"Failed to store API key for '{service_name}': {e}")
            raise
    
    async def get_api_key(self, service_name: str) -> Optional[str]:
        """
        Retrieve a stored API key.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Decrypted API key if found, None otherwise
        """
        if not self._encryption_key or not self.config:
            return None
        
        try:
            service_config = self.config.ai_services.get(service_name)
            if not service_config or not service_config.api_key_encrypted:
                return None
            
            encrypted_key = self._encrypted_data.get(service_config.api_key_encrypted)
            if not encrypted_key:
                return None
            
            # Decrypt the API key
            fernet = Fernet(self._encryption_key)
            decrypted_key = fernet.decrypt(encrypted_key.encode())
            
            return decrypted_key.decode()
            
        except Exception as e:
            logger.error(f"Failed to retrieve API key for '{service_name}': {e}")
            return None
    
    async def update_setting(self, setting_path: str, value: Any) -> None:
        """
        Update a specific configuration setting.
        
        Args:
            setting_path: Dot-separated path to the setting (e.g., 'ui_preferences.theme')
            value: New value for the setting
        """
        if not self.config:
            raise RuntimeError("Configuration not initialized")
        
        try:
            # Navigate to the setting using dot notation
            config_dict = self.config.dict()
            path_parts = setting_path.split('.')
            current = config_dict
            
            # Navigate to parent of target setting
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[path_parts[-1]] = value
            
            # Recreate config from modified dict
            self.config = AppConfig(**config_dict)
            
            await self.save_config()
            logger.info(f"Setting '{setting_path}' updated to: {value}")
            
        except Exception as e:
            logger.error(f"Failed to update setting '{setting_path}': {e}")
            raise
    
    async def get_setting(self, setting_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration setting.
        
        Args:
            setting_path: Dot-separated path to the setting
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        if not self.config:
            return default
        
        try:
            config_dict = self.config.dict()
            current = config_dict
            
            for part in setting_path.split('.'):
                if part not in current:
                    return default
                current = current[part]
            
            return current
            
        except Exception:
            return default
    
    def get_config_dir(self) -> Path:
        """Get the configuration directory path."""
        return self.config_dir
    
    def get_project_defaults(self) -> ProjectDefaults:
        """Get project default settings."""
        if not self.config:
            return ProjectDefaults()
        return self.config.project_defaults
    
    def get_ui_preferences(self) -> UIPreferences:
        """Get UI preferences."""
        if not self.config:
            return UIPreferences()
        return self.config.ui_preferences
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        if not self.config:
            return SecurityConfig()
        return self.config.security
    
    async def reset_to_defaults(self) -> None:
        """
        Reset configuration to default values.
        
        This will preserve encrypted data but reset all other settings.
        """
        logger.warning("Resetting configuration to defaults")
        
        # Backup current encrypted data
        encrypted_backup = self._encrypted_data.copy()
        
        # Create new default config
        self.config = AppConfig()
        
        # Restore encrypted data
        self._encrypted_data = encrypted_backup
        
        await self.save_config()
        await self._save_encrypted_data()
        
        logger.info("Configuration reset to defaults completed")
    
    async def cleanup(self) -> None:
        """
        Cleanup configuration manager.
        """
        logger.info("Cleaning up ConfigManager")
        
        if self.config:
            await self.save_config()
        
        # Clear sensitive data from memory
        self._encryption_key = None
        self._encrypted_data.clear()
        
        logger.info("ConfigManager cleanup completed")
    
    # Private helper methods
    
    def _get_default_config_dir(self) -> Path:
        """
        Get the default configuration directory based on platform.
        """
        if os.name == 'nt':  # Windows
            base_dir = Path(os.environ.get('APPDATA', Path.home()))
        else:  # Unix-like (Linux, macOS)
            base_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
        
        return base_dir / 'claude-tui'
    
    async def _initialize_encryption(self) -> None:
        """
        Initialize encryption for sensitive data.
        """
        key_file = self.config_dir / '.encryption_key'
        
        if key_file.exists():
            # Load existing key
            self._encryption_key = key_file.read_bytes()
        else:
            # Generate new key
            self._encryption_key = Fernet.generate_key()
            
            # Save key with restricted permissions
            key_file.write_bytes(self._encryption_key)
            key_file.chmod(0o600)  # Owner read/write only
        
        logger.info("Encryption initialized")
    
    async def _load_encrypted_data(self) -> None:
        """
        Load encrypted configuration data.
        """
        if not self.encrypted_config_file.exists():
            self._encrypted_data = {}
            return
        
        try:
            with open(self.encrypted_config_file, 'r', encoding='utf-8') as f:
                self._encrypted_data = json.load(f)
            
            logger.debug("Encrypted data loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load encrypted data: {e}")
            self._encrypted_data = {}
    
    async def _save_encrypted_data(self) -> None:
        """
        Save encrypted configuration data.
        """
        try:
            with open(self.encrypted_config_file, 'w', encoding='utf-8') as f:
                json.dump(self._encrypted_data, f, indent=2)
            
            # Set restrictive permissions
            self.encrypted_config_file.chmod(0o600)
            
            logger.debug("Encrypted data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save encrypted data: {e}")
            raise