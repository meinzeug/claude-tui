"""
Unified Configuration Management System
Centralized configuration management for Universal Development Environment Intelligence
"""

import asyncio
import json
import logging
import os
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
import weakref
import threading
from cryptography.fernet import Fernet
import base64
from collections import defaultdict
import jsonschema
from jsonschema import validate, ValidationError
import toml

from pydantic import BaseModel, Field, ConfigDict, validator


class ConfigFormat(str, Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


class ConfigScope(str, Enum):
    """Configuration scope levels"""
    SYSTEM = "system"
    USER = "user"
    PROJECT = "project"
    ENVIRONMENT = "environment"
    SESSION = "session"


class ConfigPriority(int, Enum):
    """Configuration priority levels (higher number = higher priority)"""
    DEFAULT = 0
    SYSTEM = 10
    USER = 20
    PROJECT = 30
    ENVIRONMENT = 40
    SESSION = 50
    RUNTIME = 100


@dataclass
class ConfigSource:
    """Configuration source specification"""
    name: str
    path: Optional[Path] = None
    url: Optional[str] = None
    format: ConfigFormat = ConfigFormat.JSON
    scope: ConfigScope = ConfigScope.USER
    priority: ConfigPriority = ConfigPriority.USER
    encrypted: bool = False
    watch_changes: bool = True
    readonly: bool = False
    schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.path:
            self.path = Path(self.path)


@dataclass
class ConfigValue:
    """Configuration value with metadata"""
    value: Any
    source: str
    priority: ConfigPriority
    timestamp: datetime = field(default_factory=datetime.now)
    encrypted: bool = False
    schema_validated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "encrypted": self.encrypted,
            "schema_validated": self.schema_validated,
            "metadata": self.metadata
        }


class ConfigParser(ABC):
    """Abstract base class for configuration parsers"""
    
    @abstractmethod
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        pass
        
    @abstractmethod
    def save(self, data: Dict[str, Any], file_path: Path):
        """Save configuration to file"""
        pass
        
    @abstractmethod
    def validate_format(self, file_path: Path) -> bool:
        """Validate file format"""
        pass


class JSONConfigParser(ConfigParser):
    """JSON configuration parser"""
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load JSON config {file_path}: {e}")
            return {}
            
    def save(self, data: Dict[str, Any], file_path: Path):
        """Save JSON configuration"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save JSON config {file_path}: {e}")
            raise
            
    def validate_format(self, file_path: Path) -> bool:
        """Validate JSON format"""
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            return True
        except:
            return False


class YAMLConfigParser(ConfigParser):
    """YAML configuration parser"""
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"Failed to load YAML config {file_path}: {e}")
            return {}
            
    def save(self, data: Dict[str, Any], file_path: Path):
        """Save YAML configuration"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, indent=2, default_flow_style=False)
        except Exception as e:
            logging.error(f"Failed to save YAML config {file_path}: {e}")
            raise
            
    def validate_format(self, file_path: Path) -> bool:
        """Validate YAML format"""
        try:
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
            return True
        except:
            return False


class TOMLConfigParser(ConfigParser):
    """TOML configuration parser"""
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load TOML configuration"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except Exception as e:
            logging.error(f"Failed to load TOML config {file_path}: {e}")
            return {}
            
    def save(self, data: Dict[str, Any], file_path: Path):
        """Save TOML configuration"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                toml.dump(data, f)
        except Exception as e:
            logging.error(f"Failed to save TOML config {file_path}: {e}")
            raise
            
    def validate_format(self, file_path: Path) -> bool:
        """Validate TOML format"""
        try:
            with open(file_path, 'r') as f:
                toml.load(f)
            return True
        except:
            return False


class ENVConfigParser(ConfigParser):
    """Environment variables configuration parser"""
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load environment variables from .env file"""
        config = {}
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            config[key] = self._parse_env_value(value)
            
            # Also load from actual environment
            for key, value in os.environ.items():
                if key.startswith('CLAUDE_TUI_'):
                    config_key = key[11:].lower()  # Remove CLAUDE_TUI_ prefix
                    config[config_key] = self._parse_env_value(value)
                    
        except Exception as e:
            logging.error(f"Failed to load ENV config {file_path}: {e}")
            
        return config
        
    def save(self, data: Dict[str, Any], file_path: Path):
        """Save environment variables to .env file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Generated on {datetime.now().isoformat()}\n")
                for key, value in data.items():
                    f.write(f"{key.upper()}={self._format_env_value(value)}\n")
        except Exception as e:
            logging.error(f"Failed to save ENV config {file_path}: {e}")
            raise
            
    def validate_format(self, file_path: Path) -> bool:
        """Validate environment file format"""
        try:
            self.load(file_path)
            return True
        except:
            return False
            
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        elif value.isdigit():
            return int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            return float(value)
        else:
            return value
            
    def _format_env_value(self, value: Any) -> str:
        """Format value for environment variable"""
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        else:
            return str(value)


class ConfigEncryption:
    """Configuration encryption/decryption utilities"""
    
    def __init__(self, key: Optional[bytes] = None):
        self.logger = logging.getLogger(__name__)
        
        if key:
            self._fernet = Fernet(key)
        else:
            # Generate or load key
            key_file = Path.home() / ".claude-tui" / "config.key"
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                key_file.parent.mkdir(parents=True, exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                key_file.chmod(0o600)  # Restrict permissions
                
            self._fernet = Fernet(key)
            
    def encrypt_value(self, value: Any) -> str:
        """Encrypt configuration value"""
        try:
            serialized = json.dumps(value).encode()
            encrypted = self._fernet.encrypt(serialized)
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"Failed to encrypt value: {e}")
            raise
            
    def decrypt_value(self, encrypted_value: str) -> Any:
        """Decrypt configuration value"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return json.loads(decrypted.decode())
        except Exception as e:
            self.logger.error(f"Failed to decrypt value: {e}")
            raise
            
    def encrypt_config(self, config: Dict[str, Any], 
                      sensitive_keys: Set[str]) -> Dict[str, Any]:
        """Encrypt sensitive keys in configuration"""
        encrypted_config = {}
        
        for key, value in config.items():
            if key in sensitive_keys:
                encrypted_config[key] = {
                    "_encrypted": True,
                    "_value": self.encrypt_value(value)
                }
            elif isinstance(value, dict):
                encrypted_config[key] = self.encrypt_config(value, sensitive_keys)
            else:
                encrypted_config[key] = value
                
        return encrypted_config
        
    def decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted configuration"""
        decrypted_config = {}
        
        for key, value in config.items():
            if isinstance(value, dict) and value.get("_encrypted"):
                decrypted_config[key] = self.decrypt_value(value["_value"])
            elif isinstance(value, dict):
                decrypted_config[key] = self.decrypt_config(value)
            else:
                decrypted_config[key] = value
                
        return decrypted_config


class ConfigValidator:
    """Configuration validation using JSON Schema"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._schemas: Dict[str, Dict[str, Any]] = {}
        
    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register configuration schema"""
        try:
            # Validate the schema itself
            jsonschema.Draft7Validator.check_schema(schema)
            self._schemas[name] = schema
        except Exception as e:
            self.logger.error(f"Invalid schema {name}: {e}")
            raise
            
    def validate_config(self, config: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """Validate configuration against schema"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if schema_name not in self._schemas:
            result["warnings"].append(f"Schema '{schema_name}' not found")
            return result
            
        schema = self._schemas[schema_name]
        
        try:
            validate(instance=config, schema=schema)
        except ValidationError as e:
            result["valid"] = False
            result["errors"].append(f"Validation error: {e.message}")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Schema validation failed: {str(e)}")
            
        return result
        
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get registered schema"""
        return self._schemas.get(name)
        
    def list_schemas(self) -> List[str]:
        """List registered schemas"""
        return list(self._schemas.keys())


class ConfigWatcher:
    """File system watcher for configuration changes"""
    
    def __init__(self, callback: Callable[[Path], None]):
        self.callback = callback
        self.logger = logging.getLogger(__name__)
        self._watched_files: Dict[Path, float] = {}  # path -> last_modified
        self._watch_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
    async def start_watching(self, watch_interval: float = 1.0):
        """Start watching for file changes"""
        if self._watch_task is not None:
            return
            
        self._stop_event.clear()
        self._watch_task = asyncio.create_task(
            self._watch_loop(watch_interval)
        )
        
    async def stop_watching(self):
        """Stop watching for file changes"""
        if self._watch_task:
            self._stop_event.set()
            await self._watch_task
            self._watch_task = None
            
    def add_file(self, file_path: Path):
        """Add file to watch list"""
        if file_path.exists():
            self._watched_files[file_path] = file_path.stat().st_mtime
            
    def remove_file(self, file_path: Path):
        """Remove file from watch list"""
        self._watched_files.pop(file_path, None)
        
    async def _watch_loop(self, interval: float):
        """Watch loop for file changes"""
        while not self._stop_event.is_set():
            try:
                for file_path, last_modified in list(self._watched_files.items()):
                    if not file_path.exists():
                        continue
                        
                    current_modified = file_path.stat().st_mtime
                    if current_modified > last_modified:
                        self._watched_files[file_path] = current_modified
                        self.logger.info(f"Config file changed: {file_path}")
                        
                        try:
                            self.callback(file_path)
                        except Exception as e:
                            self.logger.error(f"Error in config change callback: {e}")
                            
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in config watch loop: {e}")
                await asyncio.sleep(interval)


class UnifiedConfigManager:
    """
    Unified Configuration Management System
    Centralized configuration management for Universal Development Environment Intelligence
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration directory
        self.config_dir = config_dir or Path.home() / ".claude-tui" / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration sources
        self._sources: List[ConfigSource] = []
        self._config_data: Dict[str, ConfigValue] = {}
        
        # File parsers
        self._parsers: Dict[ConfigFormat, ConfigParser] = {
            ConfigFormat.JSON: JSONConfigParser(),
            ConfigFormat.YAML: YAMLConfigParser(),
            ConfigFormat.TOML: TOMLConfigParser(),
            ConfigFormat.ENV: ENVConfigParser()
        }
        
        # Core components
        self._encryption = ConfigEncryption()
        self._validator = ConfigValidator()
        self._watcher = ConfigWatcher(self._on_config_changed)
        
        # Event callbacks
        self._change_callbacks: List[Callable[[str, Any, Any], None]] = []
        
        # Configuration cache
        self._cache: Dict[str, Any] = {}
        self._cache_dirty = True
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Default configuration
        self._defaults: Dict[str, Any] = {}
        
        self._initialize_default_sources()
        self._register_default_schemas()
        
    def _initialize_default_sources(self):
        """Initialize default configuration sources"""
        # System configuration
        system_config = ConfigSource(
            name="system",
            path=Path("/etc/claude-tui/config.yaml"),
            format=ConfigFormat.YAML,
            scope=ConfigScope.SYSTEM,
            priority=ConfigPriority.SYSTEM,
            readonly=True
        )
        self._sources.append(system_config)
        
        # User configuration
        user_config = ConfigSource(
            name="user",
            path=self.config_dir / "config.yaml",
            format=ConfigFormat.YAML,
            scope=ConfigScope.USER,
            priority=ConfigPriority.USER
        )
        self._sources.append(user_config)
        
        # Environment variables
        env_config = ConfigSource(
            name="environment",
            path=self.config_dir / ".env",
            format=ConfigFormat.ENV,
            scope=ConfigScope.ENVIRONMENT,
            priority=ConfigPriority.ENVIRONMENT
        )
        self._sources.append(env_config)
        
    def _register_default_schemas(self):
        """Register default configuration schemas"""
        # Main application schema
        main_schema = {
            "type": "object",
            "properties": {
                "debug": {"type": "boolean", "default": False},
                "log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                    "default": "INFO"
                },
                "plugins": {
                    "type": "object",
                    "properties": {
                        "auto_update": {"type": "boolean", "default": False},
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"},
                                    "name": {"type": "string"},
                                    "auth_token": {"type": "string"}
                                },
                                "required": ["url", "name"]
                            }
                        }
                    }
                },
                "integrations": {
                    "type": "object",
                    "properties": {
                        "ide": {
                            "type": "object",
                            "properties": {
                                "auto_connect": {"type": "boolean", "default": True},
                                "sync_interval": {"type": "number", "default": 0.1}
                            }
                        },
                        "cicd": {
                            "type": "object",
                            "properties": {
                                "platforms": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "cloud": {
                            "type": "object",
                            "properties": {
                                "providers": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        self._validator.register_schema("main", main_schema)
        
    async def initialize(self) -> bool:
        """Initialize configuration manager"""
        try:
            self.logger.info("Initializing Unified Configuration Manager")
            
            # Load configurations from all sources
            await self._load_all_configurations()
            
            # Start file watching
            await self._watcher.start_watching()
            
            # Validate loaded configuration
            await self._validate_configurations()
            
            self.logger.info("Unified Configuration Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration manager: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown configuration manager"""
        self.logger.info("Shutting down Unified Configuration Manager")
        
        # Stop file watching
        await self._watcher.stop_watching()
        
        # Save any pending changes
        await self.save_configuration()
        
    def add_source(self, source: ConfigSource):
        """Add configuration source"""
        with self._lock:
            self._sources.append(source)
            self._cache_dirty = True
            
            # Add to file watcher if it's a file source
            if source.path and source.watch_changes:
                self._watcher.add_file(source.path)
                
    def remove_source(self, source_name: str):
        """Remove configuration source"""
        with self._lock:
            source = next((s for s in self._sources if s.name == source_name), None)
            if source:
                self._sources.remove(source)
                self._cache_dirty = True
                
                # Remove from file watcher
                if source.path:
                    self._watcher.remove_file(source.path)
                    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        with self._lock:
            # Update cache if dirty
            if self._cache_dirty:
                self._rebuild_cache()
                
            # Try cache first
            if key in self._cache:
                return self._cache[key]
                
            # Try nested key access (dot notation)
            if '.' in key:
                try:
                    value = self._cache
                    for part in key.split('.'):
                        value = value[part]
                    return value
                except (KeyError, TypeError):
                    pass
                    
            # Return default
            return default
            
    def set(self, key: str, value: Any, source: str = "runtime", 
            priority: ConfigPriority = ConfigPriority.RUNTIME,
            encrypted: bool = False) -> bool:
        """Set configuration value"""
        try:
            with self._lock:
                # Create config value
                config_value = ConfigValue(
                    value=value,
                    source=source,
                    priority=priority,
                    encrypted=encrypted
                )
                
                # Store in config data
                old_value = self._config_data.get(key)
                self._config_data[key] = config_value
                self._cache_dirty = True
                
                # Trigger change callbacks
                old_val = old_value.value if old_value else None
                self._trigger_change_callbacks(key, old_val, value)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to set config value {key}: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """Delete configuration value"""
        try:
            with self._lock:
                if key in self._config_data:
                    old_value = self._config_data[key].value
                    del self._config_data[key]
                    self._cache_dirty = True
                    
                    # Trigger change callbacks
                    self._trigger_change_callbacks(key, old_value, None)
                    
                    return True
                    
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete config value {key}: {e}")
            return False
            
    def has(self, key: str) -> bool:
        """Check if configuration key exists"""
        return self.get(key) is not None
        
    def keys(self, prefix: str = "") -> List[str]:
        """Get all configuration keys with optional prefix filter"""
        with self._lock:
            if self._cache_dirty:
                self._rebuild_cache()
                
            if prefix:
                return [k for k in self._cache.keys() if k.startswith(prefix)]
            else:
                return list(self._cache.keys())
                
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        with self._lock:
            if self._cache_dirty:
                self._rebuild_cache()
                
            section_data = {}
            section_prefix = f"{section}."
            
            for key, value in self._cache.items():
                if key.startswith(section_prefix):
                    section_key = key[len(section_prefix):]
                    section_data[section_key] = value
                elif key == section and isinstance(value, dict):
                    section_data.update(value)
                    
            return section_data
            
    def set_defaults(self, defaults: Dict[str, Any]):
        """Set default configuration values"""
        with self._lock:
            self._defaults.update(defaults)
            self._cache_dirty = True
            
    def register_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Register callback for configuration changes"""
        self._change_callbacks.append(callback)
        
    def unregister_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Unregister change callback"""
        try:
            self._change_callbacks.remove(callback)
        except ValueError:
            pass
            
    async def load_configuration(self, source_name: Optional[str] = None):
        """Load configuration from sources"""
        if source_name:
            # Load specific source
            source = next((s for s in self._sources if s.name == source_name), None)
            if source:
                await self._load_source_configuration(source)
        else:
            # Load all sources
            await self._load_all_configurations()
            
    async def save_configuration(self, source_name: Optional[str] = None):
        """Save configuration to sources"""
        try:
            sources_to_save = self._sources
            if source_name:
                sources_to_save = [s for s in self._sources if s.name == source_name]
                
            for source in sources_to_save:
                if source.readonly:
                    continue
                    
                await self._save_source_configuration(source)
                
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
            
    def validate_configuration(self, schema_name: str = "main") -> Dict[str, Any]:
        """Validate current configuration against schema"""
        with self._lock:
            if self._cache_dirty:
                self._rebuild_cache()
                
            return self._validator.validate_config(self._cache, schema_name)
            
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get configuration manager information"""
        with self._lock:
            source_info = []
            for source in self._sources:
                info = {
                    "name": source.name,
                    "format": source.format.value,
                    "scope": source.scope.value,
                    "priority": source.priority.value,
                    "readonly": source.readonly,
                    "encrypted": source.encrypted,
                    "watch_changes": source.watch_changes
                }
                
                if source.path:
                    info["path"] = str(source.path)
                    info["exists"] = source.path.exists()
                    
                if source.url:
                    info["url"] = source.url
                    
                source_info.append(info)
                
            return {
                "config_dir": str(self.config_dir),
                "sources": source_info,
                "total_keys": len(self._config_data),
                "schemas": self._validator.list_schemas(),
                "cache_dirty": self._cache_dirty
            }
            
    async def _load_all_configurations(self):
        """Load configurations from all sources"""
        # Sort sources by priority (lowest first so higher priority overrides)
        sorted_sources = sorted(self._sources, key=lambda s: s.priority.value)
        
        for source in sorted_sources:
            await self._load_source_configuration(source)
            
        self._cache_dirty = True
        
    async def _load_source_configuration(self, source: ConfigSource):
        """Load configuration from a single source"""
        try:
            config_data = {}
            
            if source.path and source.path.exists():
                # Load from file
                parser = self._parsers.get(source.format)
                if parser:
                    config_data = parser.load(source.path)
                    
            elif source.url:
                # Load from URL (would implement HTTP loading)
                pass
                
            # Decrypt if encrypted
            if source.encrypted and config_data:
                config_data = self._encryption.decrypt_config(config_data)
                
            # Store config values
            for key, value in config_data.items():
                config_value = ConfigValue(
                    value=value,
                    source=source.name,
                    priority=source.priority,
                    encrypted=source.encrypted
                )
                
                # Only override if higher priority
                existing = self._config_data.get(key)
                if not existing or config_value.priority.value >= existing.priority.value:
                    self._config_data[key] = config_value
                    
            self.logger.debug(f"Loaded {len(config_data)} config values from {source.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {source.name}: {e}")
            
    async def _save_source_configuration(self, source: ConfigSource):
        """Save configuration to a single source"""
        try:
            if not source.path:
                return
                
            # Collect values from this source
            source_data = {}
            for key, config_value in self._config_data.items():
                if config_value.source == source.name:
                    source_data[key] = config_value.value
                    
            if not source_data:
                return
                
            # Encrypt if needed
            if source.encrypted:
                sensitive_keys = set()  # Would be configured
                source_data = self._encryption.encrypt_config(source_data, sensitive_keys)
                
            # Save to file
            parser = self._parsers.get(source.format)
            if parser:
                parser.save(source_data, source.path)
                self.logger.debug(f"Saved {len(source_data)} config values to {source.name}")
                
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {source.name}: {e}")
            
    def _rebuild_cache(self):
        """Rebuild configuration cache"""
        self._cache = dict(self._defaults)
        
        # Sort by priority and merge
        sorted_values = sorted(
            self._config_data.items(),
            key=lambda x: x[1].priority.value
        )
        
        for key, config_value in sorted_values:
            self._cache[key] = config_value.value
            
        self._cache_dirty = False
        
    def _on_config_changed(self, file_path: Path):
        """Handle configuration file changes"""
        # Find source for changed file
        source = next((s for s in self._sources if s.path == file_path), None)
        if source:
            asyncio.create_task(self._reload_source(source))
            
    async def _reload_source(self, source: ConfigSource):
        """Reload configuration from changed source"""
        try:
            self.logger.info(f"Reloading configuration from {source.name}")
            
            # Remove old values from this source
            keys_to_remove = [
                key for key, config_value in self._config_data.items()
                if config_value.source == source.name
            ]
            
            for key in keys_to_remove:
                del self._config_data[key]
                
            # Reload source
            await self._load_source_configuration(source)
            self._cache_dirty = True
            
            # Trigger callbacks for changed values
            # This is simplified - a full implementation would track specific changes
            for callback in self._change_callbacks:
                try:
                    callback("*", None, None)  # Notify of bulk change
                except Exception as e:
                    self.logger.error(f"Error in change callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to reload configuration from {source.name}: {e}")
            
    async def _validate_configurations(self):
        """Validate all loaded configurations"""
        validation_result = self.validate_configuration()
        
        if not validation_result["valid"]:
            for error in validation_result["errors"]:
                self.logger.error(f"Configuration validation error: {error}")
        
        for warning in validation_result["warnings"]:
            self.logger.warning(f"Configuration validation warning: {warning}")
            
    def _trigger_change_callbacks(self, key: str, old_value: Any, new_value: Any):
        """Trigger configuration change callbacks"""
        for callback in self._change_callbacks:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                self.logger.error(f"Error in change callback for {key}: {e}")


# Utility functions and decorators
def config_property(key: str, default: Any = None, 
                   config_manager: Optional[UnifiedConfigManager] = None):
    """Decorator to create configuration property"""
    def decorator(cls):
        def getter(self):
            if config_manager:
                return config_manager.get(key, default)
            elif hasattr(self, '_config_manager'):
                return self._config_manager.get(key, default)
            else:
                return default
                
        def setter(self, value):
            if config_manager:
                config_manager.set(key, value)
            elif hasattr(self, '_config_manager'):
                self._config_manager.set(key, value)
                
        setattr(cls, key.replace('.', '_'), property(getter, setter))
        return cls
        
    return decorator


class ConfigurableComponent:
    """Base class for components that use configuration"""
    
    def __init__(self, config_manager: UnifiedConfigManager, config_prefix: str = ""):
        self._config_manager = config_manager
        self._config_prefix = config_prefix
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with prefix"""
        full_key = f"{self._config_prefix}.{key}" if self._config_prefix else key
        return self._config_manager.get(full_key, default)
        
    def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value with prefix"""
        full_key = f"{self._config_prefix}.{key}" if self._config_prefix else key
        return self._config_manager.set(full_key, value)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create configuration manager
        config_manager = UnifiedConfigManager()
        
        try:
            await config_manager.initialize()
            
            # Set some configuration values
            config_manager.set("app.debug", True)
            config_manager.set("app.log_level", "DEBUG")
            config_manager.set("plugins.auto_update", False)
            
            # Get configuration values
            debug = config_manager.get("app.debug")
            log_level = config_manager.get("app.log_level", "INFO")
            
            print(f"Debug: {debug}")
            print(f"Log Level: {log_level}")
            
            # Get configuration section
            app_config = config_manager.get_section("app")
            print(f"App config: {app_config}")
            
            # Validate configuration
            validation_result = config_manager.validate_configuration()
            print(f"Validation result: {validation_result}")
            
            # Get manager info
            info = config_manager.get_configuration_info()
            print(f"Config manager info: {info}")
            
            # Save configuration
            await config_manager.save_configuration()
            
        finally:
            await config_manager.shutdown()
            
    # Run example
    # asyncio.run(main())