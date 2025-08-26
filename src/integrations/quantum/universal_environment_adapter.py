"""
Universal Development Environment Intelligence Adapter
Core engine for seamless Claude-TUI integration across all development platforms
"""

import asyncio
import json
import logging
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
import weakref
import threading
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, ConfigDict


class EnvironmentType(str, Enum):
    """Supported development environment types"""
    IDE = "ide"
    CICD = "cicd" 
    CLOUD = "cloud"
    CONTAINER = "container"
    TERMINAL = "terminal"
    WEB = "web"
    MOBILE = "mobile"


class IntegrationStatus(str, Enum):
    """Integration status states"""
    INACTIVE = "inactive"
    CONNECTING = "connecting"
    ACTIVE = "active"
    ERROR = "error"
    DEGRADED = "degraded"


@dataclass
class EnvironmentCapability:
    """Defines what capabilities an environment supports"""
    name: str
    version: str
    features: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnvironmentContext(BaseModel):
    """Context information about current environment state"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    environment_id: str
    environment_type: EnvironmentType
    platform: str
    version: str
    capabilities: List[EnvironmentCapability] = Field(default_factory=list)
    current_project: Optional[str] = None
    active_files: List[str] = Field(default_factory=list)
    workspace_path: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    session_data: Dict[str, Any] = Field(default_factory=dict)


class AdapterPlugin(ABC):
    """Base class for environment-specific adapter plugins"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._status = IntegrationStatus.INACTIVE
        self._capabilities: Set[EnvironmentCapability] = set()
        
    @property
    def status(self) -> IntegrationStatus:
        return self._status
        
    @property 
    def capabilities(self) -> Set[EnvironmentCapability]:
        return self._capabilities
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the adapter plugin"""
        pass
        
    @abstractmethod
    async def connect(self, context: EnvironmentContext) -> bool:
        """Connect to the development environment"""
        pass
        
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the environment"""
        pass
        
    @abstractmethod
    async def send_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to the environment"""
        pass
        
    @abstractmethod
    async def receive_events(self) -> List[Dict[str, Any]]:
        """Receive events from the environment"""
        pass
        
    @abstractmethod
    async def sync_state(self, state: Dict[str, Any]) -> bool:
        """Synchronize state with environment"""
        pass
        
    async def health_check(self) -> bool:
        """Check if the connection is healthy"""
        try:
            response = await self.send_command("ping", {})
            return response.get("status") == "ok"
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False


class EventBus:
    """Event bus for cross-platform communication"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events of a specific type"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from events"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass
                    
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish event to all subscribers"""
        callbacks = []
        with self._lock:
            if event_type in self._subscribers:
                callbacks = self._subscribers[event_type].copy()
                
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logging.error(f"Error in event callback: {e}")


class PluginRegistry:
    """Registry for managing adapter plugins"""
    
    def __init__(self):
        self._plugins: Dict[str, Type[AdapterPlugin]] = {}
        self._instances: Dict[str, AdapterPlugin] = {}
        
    def register(self, name: str, plugin_class: Type[AdapterPlugin]):
        """Register a plugin class"""
        self._plugins[name] = plugin_class
        
    def unregister(self, name: str):
        """Unregister a plugin"""
        if name in self._plugins:
            del self._plugins[name]
        if name in self._instances:
            del self._instances[name]
            
    def create_instance(self, name: str, config: Dict[str, Any]) -> Optional[AdapterPlugin]:
        """Create plugin instance"""
        if name not in self._plugins:
            return None
            
        plugin_class = self._plugins[name]
        instance = plugin_class(config)
        self._instances[name] = instance
        return instance
        
    def get_instance(self, name: str) -> Optional[AdapterPlugin]:
        """Get existing plugin instance"""
        return self._instances.get(name)
        
    def list_plugins(self) -> List[str]:
        """List all registered plugins"""
        return list(self._plugins.keys())


class UniversalEnvironmentAdapter:
    """
    Universal Development Environment Intelligence Adapter
    Core engine for seamless integration across all development platforms
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/quantum_config.json"
        self.config = self._load_config()
        
        # Core components
        self.event_bus = EventBus()
        self.plugin_registry = PluginRegistry()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # State management
        self._environments: Dict[str, EnvironmentContext] = {}
        self._active_plugins: Dict[str, AdapterPlugin] = {}
        self._sync_tasks: Dict[str, asyncio.Task] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._metrics = {
            "connections": 0,
            "commands_sent": 0,
            "events_received": 0,
            "sync_operations": 0,
            "errors": 0
        }
        
        self._initialize_core_plugins()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
            
        return {
            "sync_interval": 5.0,
            "health_check_interval": 30.0,
            "max_retry_attempts": 3,
            "plugin_timeout": 10.0,
            "environments": {}
        }
        
    def _initialize_core_plugins(self):
        """Initialize core adapter plugins"""
        # This would be extended by specific plugin implementations
        self.logger.info("Core adapter plugins initialized")
        
    async def initialize(self) -> bool:
        """Initialize the universal adapter"""
        try:
            self.logger.info("Initializing Universal Environment Adapter")
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Initialize configured environments
            for env_id, env_config in self.config.get("environments", {}).items():
                await self._initialize_environment(env_id, env_config)
                
            self.logger.info("Universal Environment Adapter initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize adapter: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the adapter and cleanup resources"""
        self.logger.info("Shutting down Universal Environment Adapter")
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            
        # Cancel sync tasks
        for task in self._sync_tasks.values():
            task.cancel()
            
        # Disconnect all plugins
        for plugin in self._active_plugins.values():
            try:
                await plugin.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting plugin: {e}")
                
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Universal Environment Adapter shutdown complete")
        
    async def connect_environment(self, env_id: str, env_type: EnvironmentType, 
                                config: Dict[str, Any]) -> bool:
        """Connect to a development environment"""
        try:
            self.logger.info(f"Connecting to environment: {env_id} ({env_type})")
            
            # Get or create plugin
            plugin_name = config.get("plugin", env_type.value)
            plugin = self.plugin_registry.get_instance(plugin_name)
            
            if not plugin:
                plugin = self.plugin_registry.create_instance(plugin_name, config)
                if not plugin:
                    self.logger.error(f"No plugin available for: {plugin_name}")
                    return False
                    
            # Initialize plugin if needed
            if plugin.status == IntegrationStatus.INACTIVE:
                await plugin.initialize()
                
            # Create environment context
            context = EnvironmentContext(
                environment_id=env_id,
                environment_type=env_type,
                platform=config.get("platform", "unknown"),
                version=config.get("version", "1.0.0"),
                workspace_path=config.get("workspace_path"),
                user_preferences=config.get("preferences", {})
            )
            
            # Connect plugin
            if await plugin.connect(context):
                self._environments[env_id] = context
                self._active_plugins[env_id] = plugin
                
                # Start sync task
                self._sync_tasks[env_id] = asyncio.create_task(
                    self._sync_environment_loop(env_id)
                )
                
                self._metrics["connections"] += 1
                
                # Publish connection event
                await self.event_bus.publish("environment_connected", {
                    "environment_id": env_id,
                    "environment_type": env_type.value,
                    "context": context.model_dump()
                })
                
                self.logger.info(f"Successfully connected to environment: {env_id}")
                return True
            else:
                self.logger.error(f"Failed to connect plugin to environment: {env_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to environment {env_id}: {e}")
            self._metrics["errors"] += 1
            return False
            
    async def disconnect_environment(self, env_id: str) -> bool:
        """Disconnect from a development environment"""
        try:
            self.logger.info(f"Disconnecting from environment: {env_id}")
            
            # Cancel sync task
            if env_id in self._sync_tasks:
                self._sync_tasks[env_id].cancel()
                del self._sync_tasks[env_id]
                
            # Disconnect plugin
            if env_id in self._active_plugins:
                plugin = self._active_plugins[env_id]
                await plugin.disconnect()
                del self._active_plugins[env_id]
                
            # Remove environment
            if env_id in self._environments:
                env_type = self._environments[env_id].environment_type
                del self._environments[env_id]
                
                # Publish disconnection event
                await self.event_bus.publish("environment_disconnected", {
                    "environment_id": env_id,
                    "environment_type": env_type.value
                })
                
            self.logger.info(f"Successfully disconnected from environment: {env_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from environment {env_id}: {e}")
            self._metrics["errors"] += 1
            return False
            
    async def send_universal_command(self, env_id: str, command: str, 
                                   params: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to specific environment"""
        try:
            if env_id not in self._active_plugins:
                raise ValueError(f"Environment not connected: {env_id}")
                
            plugin = self._active_plugins[env_id]
            result = await plugin.send_command(command, params)
            
            self._metrics["commands_sent"] += 1
            
            # Publish command event
            await self.event_bus.publish("command_sent", {
                "environment_id": env_id,
                "command": command,
                "params": params,
                "result": result
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error sending command to {env_id}: {e}")
            self._metrics["errors"] += 1
            raise
            
    async def broadcast_command(self, command: str, params: Dict[str, Any], 
                              env_types: Optional[List[EnvironmentType]] = None) -> Dict[str, Dict[str, Any]]:
        """Broadcast command to multiple environments"""
        results = {}
        tasks = []
        
        for env_id, context in self._environments.items():
            if env_types is None or context.environment_type in env_types:
                task = self.send_universal_command(env_id, command, params)
                tasks.append((env_id, task))
                
        # Wait for all commands to complete
        for env_id, task in tasks:
            try:
                results[env_id] = await task
            except Exception as e:
                results[env_id] = {"error": str(e)}
                
        return results
        
    async def sync_all_environments(self, state: Dict[str, Any]) -> Dict[str, bool]:
        """Synchronize state across all connected environments"""
        results = {}
        
        for env_id, plugin in self._active_plugins.items():
            try:
                results[env_id] = await plugin.sync_state(state)
                self._metrics["sync_operations"] += 1
            except Exception as e:
                self.logger.error(f"Sync failed for {env_id}: {e}")
                results[env_id] = False
                self._metrics["errors"] += 1
                
        return results
        
    def get_environment_status(self, env_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific environment"""
        if env_id not in self._environments:
            return None
            
        context = self._environments[env_id]
        plugin = self._active_plugins.get(env_id)
        
        return {
            "environment_id": env_id,
            "environment_type": context.environment_type.value,
            "platform": context.platform,
            "status": plugin.status.value if plugin else "unknown",
            "capabilities": [cap.name for cap in plugin.capabilities] if plugin else [],
            "workspace_path": context.workspace_path,
            "active_files": context.active_files
        }
        
    def get_all_environments_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all environments"""
        return {
            env_id: self.get_environment_status(env_id)
            for env_id in self._environments.keys()
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics"""
        return {
            **self._metrics,
            "active_environments": len(self._environments),
            "active_plugins": len(self._active_plugins),
            "registered_plugins": len(self.plugin_registry.list_plugins())
        }
        
    async def _initialize_environment(self, env_id: str, config: Dict[str, Any]):
        """Initialize environment from configuration"""
        try:
            env_type = EnvironmentType(config.get("type", "terminal"))
            await self.connect_environment(env_id, env_type, config)
        except Exception as e:
            self.logger.error(f"Failed to initialize environment {env_id}: {e}")
            
    async def _sync_environment_loop(self, env_id: str):
        """Background sync loop for environment"""
        interval = self.config.get("sync_interval", 5.0)
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                if env_id not in self._active_plugins:
                    break
                    
                plugin = self._active_plugins[env_id]
                
                # Receive and process events
                events = await plugin.receive_events()
                for event in events:
                    await self.event_bus.publish(f"environment_event_{env_id}", event)
                    
                self._metrics["events_received"] += len(events)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in sync loop for {env_id}: {e}")
                self._metrics["errors"] += 1
                await asyncio.sleep(1)  # Brief pause before retry
                
    async def _health_check_loop(self):
        """Background health check loop"""
        interval = self.config.get("health_check_interval", 30.0)
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                unhealthy_envs = []
                for env_id, plugin in self._active_plugins.items():
                    try:
                        if not await plugin.health_check():
                            unhealthy_envs.append(env_id)
                    except Exception as e:
                        self.logger.warning(f"Health check failed for {env_id}: {e}")
                        unhealthy_envs.append(env_id)
                        
                # Attempt to reconnect unhealthy environments
                for env_id in unhealthy_envs:
                    self.logger.info(f"Attempting to reconnect unhealthy environment: {env_id}")
                    try:
                        await self.disconnect_environment(env_id)
                        # Re-initialize from config
                        env_config = self.config.get("environments", {}).get(env_id)
                        if env_config:
                            await self._initialize_environment(env_id, env_config)
                    except Exception as e:
                        self.logger.error(f"Failed to reconnect {env_id}: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")


# Utility functions for plugin development
def create_adapter_plugin(name: str, config: Dict[str, Any]) -> Callable:
    """Decorator for creating adapter plugins"""
    def decorator(cls):
        if not issubclass(cls, AdapterPlugin):
            raise TypeError("Plugin must inherit from AdapterPlugin")
        
        # Auto-register the plugin
        registry = PluginRegistry()
        registry.register(name, cls)
        
        return cls
    return decorator


class QuantumIntelligenceAPI:
    """High-level API for quantum intelligence features"""
    
    def __init__(self, adapter: UniversalEnvironmentAdapter):
        self.adapter = adapter
        self.logger = logging.getLogger(__name__)
        
    async def auto_complete_context(self, env_id: str, context: str) -> List[str]:
        """Get context-aware auto-completion suggestions"""
        try:
            result = await self.adapter.send_universal_command(
                env_id, "auto_complete", {"context": context}
            )
            return result.get("suggestions", [])
        except Exception as e:
            self.logger.error(f"Auto-completion failed: {e}")
            return []
            
    async def intelligent_refactoring(self, env_id: str, code: str, 
                                    refactor_type: str) -> Optional[str]:
        """Perform intelligent code refactoring"""
        try:
            result = await self.adapter.send_universal_command(
                env_id, "refactor", {
                    "code": code,
                    "type": refactor_type
                }
            )
            return result.get("refactored_code")
        except Exception as e:
            self.logger.error(f"Refactoring failed: {e}")
            return None
            
    async def cross_platform_sync(self, workspace_state: Dict[str, Any]) -> bool:
        """Synchronize workspace state across all platforms"""
        try:
            results = await self.adapter.sync_all_environments(workspace_state)
            return all(results.values())
        except Exception as e:
            self.logger.error(f"Cross-platform sync failed: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    async def main():
        adapter = UniversalEnvironmentAdapter()
        api = QuantumIntelligenceAPI(adapter)
        
        try:
            await adapter.initialize()
            
            # Connect to VS Code
            await adapter.connect_environment(
                "vscode-main", 
                EnvironmentType.IDE,
                {
                    "plugin": "vscode",
                    "platform": "vscode",
                    "workspace_path": "/home/user/project"
                }
            )
            
            # Example API usage
            suggestions = await api.auto_complete_context(
                "vscode-main", 
                "def process_"
            )
            print(f"Auto-completion suggestions: {suggestions}")
            
        finally:
            await adapter.shutdown()
            
    # Run example
    # asyncio.run(main())