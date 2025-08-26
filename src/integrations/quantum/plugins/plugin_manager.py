"""
Plugin Management and Registry System
Extensible plugin architecture for Universal Development Environment Intelligence
"""

import asyncio
import importlib
import importlib.util
import json
import logging
import os
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
import weakref
import threading
from urllib.parse import urlparse
import zipfile
import tempfile
import shutil
import pkg_resources
import subprocess

from pydantic import BaseModel, Field, ConfigDict, validator
import semver
import requests


class PluginStatus(str, Enum):
    """Plugin status states"""
    UNINSTALLED = "uninstalled"
    INSTALLED = "installed"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"
    UPDATING = "updating"


class PluginType(str, Enum):
    """Plugin types"""
    ADAPTER = "adapter"
    EXTENSION = "extension"
    THEME = "theme"
    LANGUAGE = "language"
    TOOL = "tool"
    INTEGRATION = "integration"
    MIDDLEWARE = "middleware"


class PluginPriority(str, Enum):
    """Plugin priority levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    NORMAL = "normal"
    LOW = "low"


@dataclass
class PluginDependency:
    """Plugin dependency specification"""
    name: str
    version: str
    optional: bool = False
    source: Optional[str] = None
    
    def is_satisfied(self, installed_version: str) -> bool:
        """Check if dependency version requirement is satisfied"""
        try:
            return semver.match(installed_version, self.version)
        except Exception:
            return False


@dataclass
class PluginMetadata:
    """Plugin metadata information"""
    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    repository: str = ""
    keywords: List[str] = field(default_factory=list)
    plugin_type: PluginType = PluginType.EXTENSION
    priority: PluginPriority = PluginPriority.NORMAL
    dependencies: List[PluginDependency] = field(default_factory=list)
    python_requires: str = ">=3.8"
    platforms: List[str] = field(default_factory=lambda: ["any"])
    entry_point: str = "main"
    config_schema: Optional[Dict[str, Any]] = None
    permissions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "repository": self.repository,
            "keywords": self.keywords,
            "plugin_type": self.plugin_type.value,
            "priority": self.priority.value,
            "dependencies": [
                {
                    "name": dep.name,
                    "version": dep.version,
                    "optional": dep.optional,
                    "source": dep.source
                }
                for dep in self.dependencies
            ],
            "python_requires": self.python_requires,
            "platforms": self.platforms,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
            "permissions": self.permissions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Create from dictionary"""
        dependencies = [
            PluginDependency(**dep) for dep in data.get("dependencies", [])
        ]
        
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", ""),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            keywords=data.get("keywords", []),
            plugin_type=PluginType(data.get("plugin_type", "extension")),
            priority=PluginPriority(data.get("priority", "normal")),
            dependencies=dependencies,
            python_requires=data.get("python_requires", ">=3.8"),
            platforms=data.get("platforms", ["any"]),
            entry_point=data.get("entry_point", "main"),
            config_schema=data.get("config_schema"),
            permissions=data.get("permissions", [])
        )


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, metadata: PluginMetadata, config: Dict[str, Any]):
        self.metadata = metadata
        self.config = config
        self.logger = logging.getLogger(f"plugin.{metadata.name}")
        self._status = PluginStatus.LOADED
        self._hooks: Dict[str, List[Callable]] = {}
        
    @property
    def status(self) -> PluginStatus:
        return self._status
        
    @property
    def name(self) -> str:
        return self.metadata.name
        
    @property
    def version(self) -> str:
        return self.metadata.version
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin"""
        pass
        
    @abstractmethod
    async def activate(self) -> bool:
        """Activate the plugin"""
        pass
        
    @abstractmethod
    async def deactivate(self) -> bool:
        """Deactivate the plugin"""
        pass
        
    @abstractmethod
    async def cleanup(self):
        """Cleanup plugin resources"""
        pass
        
    def register_hook(self, hook_name: str, callback: Callable):
        """Register hook callback"""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
        
    def unregister_hook(self, hook_name: str, callback: Callable):
        """Unregister hook callback"""
        if hook_name in self._hooks:
            try:
                self._hooks[hook_name].remove(callback)
            except ValueError:
                pass
                
    async def execute_hooks(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all callbacks for a hook"""
        results = []
        
        for callback in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Hook execution failed for {hook_name}: {e}")
                
        return results
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
        
    def set_config(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value


class PluginInstallationSource:
    """Plugin installation source management"""
    
    def __init__(self, source_url: str, auth_token: Optional[str] = None):
        self.source_url = source_url
        self.auth_token = auth_token
        self.logger = logging.getLogger(__name__)
        
    async def search_plugins(self, query: str, plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
        """Search for plugins"""
        try:
            # This would implement actual search against the plugin repository
            # For now, return mock results
            return [
                {
                    "name": f"plugin-{query}",
                    "version": "1.0.0",
                    "description": f"Plugin matching {query}",
                    "author": "Developer",
                    "plugin_type": plugin_type.value if plugin_type else "extension"
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Plugin search failed: {e}")
            return []
            
    async def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed plugin information"""
        try:
            # This would fetch plugin metadata from repository
            return {
                "name": plugin_name,
                "version": "1.0.0",
                "description": "Plugin description",
                "download_url": f"{self.source_url}/plugins/{plugin_name}/download",
                "metadata": {}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get plugin info for {plugin_name}: {e}")
            return None
            
    async def download_plugin(self, plugin_name: str, version: str, target_dir: Path) -> bool:
        """Download plugin to target directory"""
        try:
            plugin_info = await self.get_plugin_info(plugin_name)
            if not plugin_info:
                return False
                
            download_url = plugin_info.get("download_url")
            if not download_url:
                return False
                
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
                
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Download and extract plugin
            plugin_archive = target_dir / f"{plugin_name}-{version}.zip"
            with open(plugin_archive, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Extract archive
            plugin_dir = target_dir / plugin_name
            plugin_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(plugin_archive, 'r') as zip_ref:
                zip_ref.extractall(plugin_dir)
                
            # Clean up archive
            plugin_archive.unlink()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download plugin {plugin_name}: {e}")
            return False


class PluginValidator:
    """Plugin validation and security scanner"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def validate_plugin(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate plugin structure and security"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "security_score": 100,
            "metadata": None
        }
        
        try:
            # Check plugin structure
            await self._validate_structure(plugin_path, validation_result)
            
            # Validate metadata
            await self._validate_metadata(plugin_path, validation_result)
            
            # Security scan
            await self._security_scan(plugin_path, validation_result)
            
            # Code quality check
            await self._code_quality_check(plugin_path, validation_result)
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            
        return validation_result
        
    async def _validate_structure(self, plugin_path: Path, result: Dict[str, Any]):
        """Validate plugin directory structure"""
        required_files = ["plugin.json", "__init__.py"]
        
        for required_file in required_files:
            file_path = plugin_path / required_file
            if not file_path.exists():
                result["errors"].append(f"Missing required file: {required_file}")
                result["valid"] = False
                
    async def _validate_metadata(self, plugin_path: Path, result: Dict[str, Any]):
        """Validate plugin metadata"""
        try:
            metadata_file = plugin_path / "plugin.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_data = json.load(f)
                    
                # Validate required metadata fields
                required_fields = ["name", "version", "description"]
                for field in required_fields:
                    if field not in metadata_data:
                        result["errors"].append(f"Missing metadata field: {field}")
                        result["valid"] = False
                        
                # Validate version format
                version = metadata_data.get("version")
                if version:
                    try:
                        semver.VersionInfo.parse(version)
                    except ValueError:
                        result["errors"].append(f"Invalid version format: {version}")
                        result["valid"] = False
                        
                result["metadata"] = PluginMetadata.from_dict(metadata_data)
                
        except Exception as e:
            result["errors"].append(f"Metadata validation error: {str(e)}")
            result["valid"] = False
            
    async def _security_scan(self, plugin_path: Path, result: Dict[str, Any]):
        """Perform security scan of plugin code"""
        security_issues = []
        
        # Check for suspicious imports
        suspicious_imports = [
            "subprocess", "os.system", "eval", "exec", 
            "__import__", "compile", "open"
        ]
        
        for py_file in plugin_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for suspicious in suspicious_imports:
                    if suspicious in content:
                        security_issues.append(
                            f"Suspicious import '{suspicious}' in {py_file.name}"
                        )
                        result["security_score"] -= 10
                        
            except Exception as e:
                result["warnings"].append(f"Could not scan {py_file}: {e}")
                
        if security_issues:
            result["warnings"].extend(security_issues)
            
        # Additional security checks would go here
        
    async def _code_quality_check(self, plugin_path: Path, result: Dict[str, Any]):
        """Check code quality metrics"""
        try:
            # This would run code quality tools like flake8, pylint, etc.
            # For now, just check basic Python syntax
            
            for py_file in plugin_path.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        compile(f.read(), py_file, 'exec')
                except SyntaxError as e:
                    result["errors"].append(f"Syntax error in {py_file}: {e}")
                    result["valid"] = False
                    
        except Exception as e:
            result["warnings"].append(f"Code quality check error: {e}")


class PluginManager:
    """
    Plugin Management and Registry System
    Handles plugin installation, loading, activation, and lifecycle management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Plugin directories
        self.plugin_dir = Path(config.get("plugin_dir", "plugins"))
        self.plugin_dir.mkdir(exist_ok=True)
        
        self.system_plugin_dir = Path(config.get("system_plugin_dir", "system_plugins"))
        self.user_plugin_dir = Path(config.get("user_plugin_dir", "user_plugins"))
        
        # Plugin registry
        self._installed_plugins: Dict[str, PluginMetadata] = {}
        self._loaded_plugins: Dict[str, BasePlugin] = {}
        self._active_plugins: Dict[str, BasePlugin] = {}
        
        # Plugin sources
        self._sources: List[PluginInstallationSource] = []
        self._validator = PluginValidator()
        
        # Hook system
        self._global_hooks: Dict[str, List[Callable]] = {}
        
        # Configuration
        self.auto_update = config.get("auto_update", False)
        self.security_level = config.get("security_level", "medium")
        
        self._initialize_sources()
        
    def _initialize_sources(self):
        """Initialize plugin installation sources"""
        default_sources = self.config.get("sources", [
            {"url": "https://plugins.claude-tui.com", "name": "official"}
        ])
        
        for source_config in default_sources:
            source = PluginInstallationSource(
                source_config["url"],
                source_config.get("auth_token")
            )
            self._sources.append(source)
            
    async def initialize(self) -> bool:
        """Initialize plugin manager"""
        try:
            self.logger.info("Initializing Plugin Manager")
            
            # Load installed plugins metadata
            await self._discover_installed_plugins()
            
            # Auto-load system plugins
            await self._load_system_plugins()
            
            self.logger.info(f"Plugin Manager initialized with {len(self._installed_plugins)} plugins")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin manager: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown plugin manager"""
        self.logger.info("Shutting down Plugin Manager")
        
        # Deactivate all plugins
        for plugin in list(self._active_plugins.values()):
            await self.deactivate_plugin(plugin.name)
            
        # Cleanup all plugins
        for plugin in list(self._loaded_plugins.values()):
            await plugin.cleanup()
            
        self._loaded_plugins.clear()
        self._active_plugins.clear()
        
    async def search_plugins(self, query: str, plugin_type: Optional[PluginType] = None,
                           source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for plugins across all sources"""
        all_results = []
        
        sources_to_search = self._sources
        if source:
            sources_to_search = [s for s in self._sources if s.source_url == source]
            
        for plugin_source in sources_to_search:
            try:
                results = await plugin_source.search_plugins(query, plugin_type)
                for result in results:
                    result["source"] = plugin_source.source_url
                all_results.extend(results)
                
            except Exception as e:
                self.logger.error(f"Search failed for source {plugin_source.source_url}: {e}")
                
        # Remove duplicates and sort by relevance
        unique_results = {}
        for result in all_results:
            plugin_id = f"{result['name']}@{result.get('source', 'unknown')}"
            if plugin_id not in unique_results:
                unique_results[plugin_id] = result
                
        return list(unique_results.values())
        
    async def install_plugin(self, plugin_name: str, version: str = "latest",
                           source: Optional[str] = None) -> bool:
        """Install plugin from source"""
        try:
            self.logger.info(f"Installing plugin: {plugin_name}@{version}")
            
            # Check if already installed
            if plugin_name in self._installed_plugins:
                installed_version = self._installed_plugins[plugin_name].version
                if version == "latest" or semver.compare(version, installed_version) <= 0:
                    self.logger.info(f"Plugin {plugin_name} already installed with version {installed_version}")
                    return True
                    
            # Find source
            plugin_source = None
            if source:
                plugin_source = next((s for s in self._sources if s.source_url == source), None)
            else:
                # Try all sources
                for src in self._sources:
                    plugin_info = await src.get_plugin_info(plugin_name)
                    if plugin_info:
                        plugin_source = src
                        break
                        
            if not plugin_source:
                self.logger.error(f"Plugin {plugin_name} not found in any source")
                return False
                
            # Download plugin
            plugin_install_dir = self.user_plugin_dir / plugin_name
            plugin_install_dir.mkdir(parents=True, exist_ok=True)
            
            if not await plugin_source.download_plugin(plugin_name, version, plugin_install_dir.parent):
                self.logger.error(f"Failed to download plugin {plugin_name}")
                return False
                
            # Validate plugin
            validation_result = await self._validator.validate_plugin(plugin_install_dir)
            if not validation_result["valid"]:
                self.logger.error(f"Plugin validation failed: {validation_result['errors']}")
                shutil.rmtree(plugin_install_dir)
                return False
                
            if validation_result["security_score"] < 70:
                self.logger.warning(f"Plugin {plugin_name} has security concerns")
                if self.security_level == "high":
                    shutil.rmtree(plugin_install_dir)
                    return False
                    
            # Install dependencies
            metadata = validation_result["metadata"]
            if metadata and metadata.dependencies:
                for dependency in metadata.dependencies:
                    if not await self._install_dependency(dependency):
                        if not dependency.optional:
                            self.logger.error(f"Failed to install required dependency: {dependency.name}")
                            shutil.rmtree(plugin_install_dir)
                            return False
                            
            # Register plugin
            self._installed_plugins[plugin_name] = metadata
            
            # Save plugin registry
            await self._save_plugin_registry()
            
            self.logger.info(f"Successfully installed plugin: {plugin_name}@{version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to install plugin {plugin_name}: {e}")
            return False
            
    async def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall plugin"""
        try:
            self.logger.info(f"Uninstalling plugin: {plugin_name}")
            
            # Deactivate if active
            if plugin_name in self._active_plugins:
                await self.deactivate_plugin(plugin_name)
                
            # Unload if loaded
            if plugin_name in self._loaded_plugins:
                await self._loaded_plugins[plugin_name].cleanup()
                del self._loaded_plugins[plugin_name]
                
            # Remove from registry
            if plugin_name in self._installed_plugins:
                del self._installed_plugins[plugin_name]
                
            # Remove plugin files
            plugin_dir = self.user_plugin_dir / plugin_name
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
                
            # Save registry
            await self._save_plugin_registry()
            
            self.logger.info(f"Successfully uninstalled plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to uninstall plugin {plugin_name}: {e}")
            return False
            
    async def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load plugin into memory"""
        try:
            if plugin_name in self._loaded_plugins:
                self.logger.info(f"Plugin already loaded: {plugin_name}")
                return True
                
            if plugin_name not in self._installed_plugins:
                self.logger.error(f"Plugin not installed: {plugin_name}")
                return False
                
            metadata = self._installed_plugins[plugin_name]
            plugin_config = config or {}
            
            # Find plugin directory
            plugin_dir = None
            for search_dir in [self.user_plugin_dir, self.system_plugin_dir]:
                candidate = search_dir / plugin_name
                if candidate.exists():
                    plugin_dir = candidate
                    break
                    
            if not plugin_dir:
                self.logger.error(f"Plugin directory not found: {plugin_name}")
                return False
                
            # Load plugin module
            plugin_module = await self._load_plugin_module(plugin_dir, metadata)
            if not plugin_module:
                return False
                
            # Create plugin instance
            plugin_class = getattr(plugin_module, metadata.entry_point)
            plugin_instance = plugin_class(metadata, plugin_config)
            
            # Initialize plugin
            if not await plugin_instance.initialize():
                self.logger.error(f"Plugin initialization failed: {plugin_name}")
                return False
                
            self._loaded_plugins[plugin_name] = plugin_instance
            
            self.logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
            
    async def activate_plugin(self, plugin_name: str) -> bool:
        """Activate loaded plugin"""
        try:
            if plugin_name in self._active_plugins:
                self.logger.info(f"Plugin already active: {plugin_name}")
                return True
                
            # Load plugin if not loaded
            if plugin_name not in self._loaded_plugins:
                if not await self.load_plugin(plugin_name):
                    return False
                    
            plugin = self._loaded_plugins[plugin_name]
            
            # Check dependencies
            if not await self._check_plugin_dependencies(plugin.metadata):
                self.logger.error(f"Plugin dependencies not satisfied: {plugin_name}")
                return False
                
            # Activate plugin
            if not await plugin.activate():
                self.logger.error(f"Plugin activation failed: {plugin_name}")
                return False
                
            self._active_plugins[plugin_name] = plugin
            plugin._status = PluginStatus.ACTIVE
            
            # Execute activation hooks
            await self._execute_global_hooks("plugin_activated", plugin)
            
            self.logger.info(f"Successfully activated plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to activate plugin {plugin_name}: {e}")
            return False
            
    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate active plugin"""
        try:
            if plugin_name not in self._active_plugins:
                self.logger.info(f"Plugin not active: {plugin_name}")
                return True
                
            plugin = self._active_plugins[plugin_name]
            
            # Deactivate plugin
            if not await plugin.deactivate():
                self.logger.error(f"Plugin deactivation failed: {plugin_name}")
                return False
                
            del self._active_plugins[plugin_name]
            plugin._status = PluginStatus.LOADED
            
            # Execute deactivation hooks
            await self._execute_global_hooks("plugin_deactivated", plugin)
            
            self.logger.info(f"Successfully deactivated plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deactivate plugin {plugin_name}: {e}")
            return False
            
    async def update_plugin(self, plugin_name: str) -> bool:
        """Update plugin to latest version"""
        try:
            if plugin_name not in self._installed_plugins:
                self.logger.error(f"Plugin not installed: {plugin_name}")
                return False
                
            current_metadata = self._installed_plugins[plugin_name]
            current_version = current_metadata.version
            
            # Find latest version
            search_results = await self.search_plugins(plugin_name)
            if not search_results:
                self.logger.info(f"No updates available for plugin: {plugin_name}")
                return True
                
            latest_plugin = max(search_results, key=lambda x: semver.VersionInfo.parse(x["version"]))
            latest_version = latest_plugin["version"]
            
            if semver.compare(latest_version, current_version) <= 0:
                self.logger.info(f"Plugin {plugin_name} is up to date")
                return True
                
            # Backup current plugin
            backup_dir = self.plugin_dir / f"{plugin_name}.backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
                
            current_dir = self.user_plugin_dir / plugin_name
            if current_dir.exists():
                shutil.copytree(current_dir, backup_dir)
                
            # Deactivate plugin
            was_active = plugin_name in self._active_plugins
            if was_active:
                await self.deactivate_plugin(plugin_name)
                
            try:
                # Install new version
                if await self.install_plugin(plugin_name, latest_version, latest_plugin.get("source")):
                    # Reactivate if it was active
                    if was_active:
                        await self.activate_plugin(plugin_name)
                        
                    # Remove backup
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                        
                    self.logger.info(f"Successfully updated plugin {plugin_name} to {latest_version}")
                    return True
                else:
                    # Restore backup on failure
                    if backup_dir.exists():
                        if current_dir.exists():
                            shutil.rmtree(current_dir)
                        shutil.move(backup_dir, current_dir)
                        
                    self.logger.error(f"Failed to update plugin: {plugin_name}")
                    return False
                    
            except Exception as e:
                # Restore backup on error
                if backup_dir.exists():
                    if current_dir.exists():
                        shutil.rmtree(current_dir)
                    shutil.move(backup_dir, current_dir)
                raise e
                
        except Exception as e:
            self.logger.error(f"Failed to update plugin {plugin_name}: {e}")
            return False
            
    def get_installed_plugins(self) -> List[Dict[str, Any]]:
        """Get list of installed plugins"""
        return [
            {
                **metadata.to_dict(),
                "status": self._get_plugin_status(metadata.name),
                "loaded": metadata.name in self._loaded_plugins,
                "active": metadata.name in self._active_plugins
            }
            for metadata in self._installed_plugins.values()
        ]
        
    def get_active_plugins(self) -> List[Dict[str, Any]]:
        """Get list of active plugins"""
        return [
            {
                **plugin.metadata.to_dict(),
                "status": plugin.status.value
            }
            for plugin in self._active_plugins.values()
        ]
        
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed plugin information"""
        if plugin_name not in self._installed_plugins:
            return None
            
        metadata = self._installed_plugins[plugin_name]
        plugin = self._loaded_plugins.get(plugin_name)
        
        info = {
            **metadata.to_dict(),
            "status": self._get_plugin_status(plugin_name),
            "loaded": plugin is not None,
            "active": plugin_name in self._active_plugins
        }
        
        if plugin:
            info["config"] = plugin.config
            info["hooks"] = list(plugin._hooks.keys())
            
        return info
        
    def register_global_hook(self, hook_name: str, callback: Callable):
        """Register global hook callback"""
        if hook_name not in self._global_hooks:
            self._global_hooks[hook_name] = []
        self._global_hooks[hook_name].append(callback)
        
    def unregister_global_hook(self, hook_name: str, callback: Callable):
        """Unregister global hook callback"""
        if hook_name in self._global_hooks:
            try:
                self._global_hooks[hook_name].remove(callback)
            except ValueError:
                pass
                
    async def _discover_installed_plugins(self):
        """Discover installed plugins"""
        plugin_registry_file = self.plugin_dir / "registry.json"
        
        if plugin_registry_file.exists():
            try:
                with open(plugin_registry_file, 'r') as f:
                    registry_data = json.load(f)
                    
                for plugin_data in registry_data.get("plugins", []):
                    metadata = PluginMetadata.from_dict(plugin_data)
                    self._installed_plugins[metadata.name] = metadata
                    
            except Exception as e:
                self.logger.error(f"Failed to load plugin registry: {e}")
                
        # Also scan plugin directories for metadata
        for plugin_dir in [self.system_plugin_dir, self.user_plugin_dir]:
            if not plugin_dir.exists():
                continue
                
            for plugin_path in plugin_dir.iterdir():
                if not plugin_path.is_dir():
                    continue
                    
                metadata_file = plugin_path / "plugin.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_data = json.load(f)
                        metadata = PluginMetadata.from_dict(metadata_data)
                        self._installed_plugins[metadata.name] = metadata
                    except Exception as e:
                        self.logger.error(f"Failed to load plugin metadata from {plugin_path}: {e}")
                        
    async def _save_plugin_registry(self):
        """Save plugin registry to file"""
        try:
            registry_file = self.plugin_dir / "registry.json"
            registry_data = {
                "version": "1.0",
                "plugins": [
                    metadata.to_dict()
                    for metadata in self._installed_plugins.values()
                ]
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save plugin registry: {e}")
            
    async def _load_system_plugins(self):
        """Load system plugins automatically"""
        system_plugins = self.config.get("system_plugins", [])
        
        for plugin_name in system_plugins:
            try:
                await self.load_plugin(plugin_name)
                await self.activate_plugin(plugin_name)
            except Exception as e:
                self.logger.error(f"Failed to load system plugin {plugin_name}: {e}")
                
    async def _load_plugin_module(self, plugin_dir: Path, metadata: PluginMetadata):
        """Load plugin module from directory"""
        try:
            # Add plugin directory to Python path
            sys.path.insert(0, str(plugin_dir))
            
            try:
                # Load plugin module
                module_name = f"plugin_{metadata.name}"
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    plugin_dir / "__init__.py"
                )
                
                if not spec or not spec.loader:
                    self.logger.error(f"Could not create module spec for {metadata.name}")
                    return None
                    
                plugin_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plugin_module)
                
                return plugin_module
                
            finally:
                # Remove from path
                sys.path.remove(str(plugin_dir))
                
        except Exception as e:
            self.logger.error(f"Failed to load plugin module {metadata.name}: {e}")
            return None
            
    async def _install_dependency(self, dependency: PluginDependency) -> bool:
        """Install plugin dependency"""
        try:
            # Check if dependency is already satisfied
            if dependency.name in self._installed_plugins:
                installed_version = self._installed_plugins[dependency.name].version
                if dependency.is_satisfied(installed_version):
                    return True
                    
            # Try to install dependency
            return await self.install_plugin(
                dependency.name,
                dependency.version,
                dependency.source
            )
            
        except Exception as e:
            self.logger.error(f"Failed to install dependency {dependency.name}: {e}")
            return False
            
    async def _check_plugin_dependencies(self, metadata: PluginMetadata) -> bool:
        """Check if plugin dependencies are satisfied"""
        for dependency in metadata.dependencies:
            if dependency.name not in self._installed_plugins:
                if not dependency.optional:
                    return False
                continue
                
            installed_version = self._installed_plugins[dependency.name].version
            if not dependency.is_satisfied(installed_version):
                if not dependency.optional:
                    return False
                    
        return True
        
    def _get_plugin_status(self, plugin_name: str) -> str:
        """Get current status of plugin"""
        if plugin_name in self._active_plugins:
            return PluginStatus.ACTIVE.value
        elif plugin_name in self._loaded_plugins:
            return PluginStatus.LOADED.value
        elif plugin_name in self._installed_plugins:
            return PluginStatus.INSTALLED.value
        else:
            return PluginStatus.UNINSTALLED.value
            
    async def _execute_global_hooks(self, hook_name: str, *args, **kwargs):
        """Execute global hook callbacks"""
        for callback in self._global_hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Global hook execution failed for {hook_name}: {e}")


# Example plugin implementation
class ExamplePlugin(BasePlugin):
    """Example plugin implementation"""
    
    async def initialize(self) -> bool:
        """Initialize the plugin"""
        self.logger.info(f"Initializing {self.name}")
        return True
        
    async def activate(self) -> bool:
        """Activate the plugin"""
        self.logger.info(f"Activating {self.name}")
        self._status = PluginStatus.ACTIVE
        return True
        
    async def deactivate(self) -> bool:
        """Deactivate the plugin"""
        self.logger.info(f"Deactivating {self.name}")
        self._status = PluginStatus.LOADED
        return True
        
    async def cleanup(self):
        """Cleanup plugin resources"""
        self.logger.info(f"Cleaning up {self.name}")


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "plugin_dir": "plugins",
            "user_plugin_dir": "user_plugins",
            "system_plugin_dir": "system_plugins",
            "sources": [
                {"url": "https://plugins.example.com", "name": "official"}
            ],
            "system_plugins": ["core-utils", "syntax-highlighter"],
            "auto_update": True,
            "security_level": "medium"
        }
        
        manager = PluginManager(config)
        
        try:
            await manager.initialize()
            
            # Search for plugins
            search_results = await manager.search_plugins("code-formatter")
            print(f"Search results: {search_results}")
            
            # Install a plugin
            await manager.install_plugin("example-plugin", "1.0.0")
            
            # Load and activate plugin
            await manager.load_plugin("example-plugin")
            await manager.activate_plugin("example-plugin")
            
            # Get plugin info
            info = manager.get_plugin_info("example-plugin")
            print(f"Plugin info: {info}")
            
            # List active plugins
            active_plugins = manager.get_active_plugins()
            print(f"Active plugins: {active_plugins}")
            
        finally:
            await manager.shutdown()
            
    # Run example
    # asyncio.run(main())