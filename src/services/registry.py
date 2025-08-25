"""
Service Registry for Dependency Injection

Production-ready service registry with:
- Service lifecycle management
- Dependency injection
- Service health monitoring
- Configuration management
- Resource cleanup
- Service discovery
"""

import asyncio
import logging
from typing import Dict, Any, Type, Optional, Callable, AsyncContextManager
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

from ..core.logger import get_logger
from ..core.exceptions import ClaudeTIUException

logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ServiceRegistryError(ClaudeTIUException):
    """Service registry specific error."""
    
    def __init__(self, message: str, error_code: str = "SERVICE_REGISTRY_ERROR", details: Optional[Dict] = None):
        super().__init__(message, error_code, details)


class ServiceDescriptor:
    """Service descriptor for registry management."""
    
    def __init__(
        self,
        name: str,
        service_class: Type,
        factory: Optional[Callable] = None,
        singleton: bool = True,
        dependencies: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        health_check: Optional[Callable] = None
    ):
        """
        Initialize service descriptor.
        
        Args:
            name: Service name
            service_class: Service class type
            factory: Factory function for service creation
            singleton: Whether service should be singleton
            dependencies: List of service dependencies
            config: Service configuration
            health_check: Service health check function
        """
        self.name = name
        self.service_class = service_class
        self.factory = factory
        self.singleton = singleton
        self.dependencies = dependencies or []
        self.config = config or {}
        self.health_check = health_check
        
        self.instance = None
        self.status = ServiceStatus.UNREGISTERED
        self.created_at = None
        self.last_health_check = None
        self.health_status = None


class ServiceRegistry:
    """
    Production-ready service registry for dependency injection.
    
    Features:
    - Service lifecycle management
    - Dependency resolution
    - Health monitoring
    - Configuration management
    - Resource cleanup
    """
    
    def __init__(self):
        """Initialize service registry."""
        self._services: Dict[str, ServiceDescriptor] = {}
        self._initialization_lock = asyncio.Lock()
        self._shutdown_in_progress = False
        
        logger.info("Service registry initialized")
    
    def register_service(
        self,
        name: str,
        service_class: Type,
        factory: Optional[Callable] = None,
        singleton: bool = True,
        dependencies: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        health_check: Optional[Callable] = None
    ) -> None:
        """
        Register a service in the registry.
        
        Args:
            name: Unique service name
            service_class: Service class
            factory: Factory function (optional)
            singleton: Whether service is singleton
            dependencies: Service dependencies
            config: Service configuration
            health_check: Health check function
        """
        if name in self._services:
            raise ServiceRegistryError(
                f"Service '{name}' is already registered",
                "DUPLICATE_SERVICE",
                {"service_name": name}
            )
        
        descriptor = ServiceDescriptor(
            name=name,
            service_class=service_class,
            factory=factory,
            singleton=singleton,
            dependencies=dependencies,
            config=config,
            health_check=health_check
        )
        
        self._services[name] = descriptor
        descriptor.status = ServiceStatus.REGISTERED
        
        logger.info(f"Service '{name}' registered successfully")
    
    async def get_service(self, name: str, **kwargs) -> Any:
        """
        Get service instance by name.
        
        Args:
            name: Service name
            **kwargs: Additional arguments for service creation
            
        Returns:
            Service instance
            
        Raises:
            ServiceRegistryError: If service not found or initialization fails
        """
        if self._shutdown_in_progress:
            raise ServiceRegistryError(
                "Cannot get service during shutdown",
                "SHUTDOWN_IN_PROGRESS",
                {"service_name": name}
            )
        
        if name not in self._services:
            raise ServiceRegistryError(
                f"Service '{name}' not registered",
                "SERVICE_NOT_FOUND",
                {"service_name": name}
            )
        
        descriptor = self._services[name]
        
        # For singleton services, return existing instance if available
        if descriptor.singleton and descriptor.instance is not None:
            if descriptor.status == ServiceStatus.RUNNING:
                return descriptor.instance
            elif descriptor.status == ServiceStatus.ERROR:
                raise ServiceRegistryError(
                    f"Service '{name}' is in error state",
                    "SERVICE_ERROR",
                    {"service_name": name}
                )
        
        # Initialize service with dependencies
        async with self._initialization_lock:
            # Double-check pattern for thread safety
            if descriptor.singleton and descriptor.instance is not None:
                return descriptor.instance
            
            return await self._initialize_service(descriptor, **kwargs)
    
    async def _initialize_service(self, descriptor: ServiceDescriptor, **kwargs) -> Any:
        """
        Initialize service with dependency resolution.
        
        Args:
            descriptor: Service descriptor
            **kwargs: Additional arguments
            
        Returns:
            Service instance
        """
        try:
            descriptor.status = ServiceStatus.INITIALIZING
            
            logger.info(f"Initializing service '{descriptor.name}'")
            
            # Resolve dependencies
            dependency_instances = {}
            for dep_name in descriptor.dependencies:
                if dep_name not in self._services:
                    raise ServiceRegistryError(
                        f"Dependency '{dep_name}' not found for service '{descriptor.name}'",
                        "DEPENDENCY_NOT_FOUND",
                        {"service_name": descriptor.name, "dependency": dep_name}
                    )
                
                dependency_instances[dep_name] = await self.get_service(dep_name)
            
            # Create service instance
            if descriptor.factory:
                # Use factory function
                if asyncio.iscoroutinefunction(descriptor.factory):
                    instance = await descriptor.factory(
                        config=descriptor.config,
                        dependencies=dependency_instances,
                        **kwargs
                    )
                else:
                    instance = descriptor.factory(
                        config=descriptor.config,
                        dependencies=dependency_instances,
                        **kwargs
                    )
            else:
                # Use class constructor
                instance = descriptor.service_class(
                    config=descriptor.config,
                    **dependency_instances,
                    **kwargs
                )
            
            # Initialize service if it has an initialize method
            if hasattr(instance, 'initialize') and callable(getattr(instance, 'initialize')):
                if asyncio.iscoroutinefunction(instance.initialize):
                    await instance.initialize()
                else:
                    instance.initialize()
            
            # Update descriptor
            descriptor.instance = instance
            descriptor.status = ServiceStatus.RUNNING
            descriptor.created_at = datetime.utcnow()
            
            logger.info(f"Service '{descriptor.name}' initialized successfully")
            
            return instance
            
        except Exception as e:
            descriptor.status = ServiceStatus.ERROR
            
            logger.error(f"Failed to initialize service '{descriptor.name}': {e}")
            
            raise ServiceRegistryError(
                f"Service initialization failed: {str(e)}",
                "SERVICE_INIT_ERROR",
                {"service_name": descriptor.name, "error": str(e)}
            )
    
    @asynccontextmanager
    async def get_service_context(self, name: str, **kwargs) -> AsyncContextManager:
        """
        Get service as async context manager.
        
        Args:
            name: Service name
            **kwargs: Additional arguments
            
        Yields:
            Service instance
        """
        service = await self.get_service(name, **kwargs)
        try:
            yield service
        finally:
            # For non-singleton services, cleanup might be needed
            descriptor = self._services.get(name)
            if descriptor and not descriptor.singleton:
                await self._cleanup_service_instance(service)
    
    async def _cleanup_service_instance(self, service: Any) -> None:
        """
        Cleanup service instance.
        
        Args:
            service: Service instance to cleanup
        """
        try:
            if hasattr(service, 'close') and callable(getattr(service, 'close')):
                if asyncio.iscoroutinefunction(service.close):
                    await service.close()
                else:
                    service.close()
        except Exception as e:
            logger.error(f"Error cleaning up service instance: {e}")
    
    async def health_check(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform health check on services.
        
        Args:
            service_name: Specific service to check (all if None)
            
        Returns:
            dict: Health check results
        """
        results = {}
        
        services_to_check = (
            [service_name] if service_name else list(self._services.keys())
        )
        
        for name in services_to_check:
            if name not in self._services:
                results[name] = {
                    'status': 'not_found',
                    'timestamp': datetime.utcnow().isoformat()
                }
                continue
            
            descriptor = self._services[name]
            
            try:
                # Basic status check
                if descriptor.status != ServiceStatus.RUNNING:
                    results[name] = {
                        'status': 'not_running',
                        'service_status': descriptor.status.value,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    continue
                
                # Custom health check
                if descriptor.health_check and descriptor.instance:
                    if asyncio.iscoroutinefunction(descriptor.health_check):
                        health_result = await descriptor.health_check(descriptor.instance)
                    else:
                        health_result = descriptor.health_check(descriptor.instance)
                    
                    results[name] = {
                        'status': 'healthy' if health_result else 'unhealthy',
                        'service_status': descriptor.status.value,
                        'custom_health_check': health_result,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                
                # Service-specific health check
                elif hasattr(descriptor.instance, 'health_check'):
                    health_method = getattr(descriptor.instance, 'health_check')
                    if asyncio.iscoroutinefunction(health_method):
                        health_result = await health_method()
                    else:
                        health_result = health_method()
                    
                    results[name] = {
                        'status': 'healthy',
                        'service_status': descriptor.status.value,
                        'health_result': health_result,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                
                else:
                    # Basic health check - service is running
                    results[name] = {
                        'status': 'healthy',
                        'service_status': descriptor.status.value,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                
                descriptor.last_health_check = datetime.utcnow()
                
            except Exception as e:
                results[name] = {
                    'status': 'unhealthy',
                    'service_status': descriptor.status.value,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        return results
    
    async def get_service_info(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get service information.
        
        Args:
            service_name: Specific service (all if None)
            
        Returns:
            dict: Service information
        """
        services_info = {}
        
        services_to_check = (
            [service_name] if service_name else list(self._services.keys())
        )
        
        for name in services_to_check:
            if name not in self._services:
                continue
            
            descriptor = self._services[name]
            
            services_info[name] = {
                'name': descriptor.name,
                'service_class': descriptor.service_class.__name__,
                'status': descriptor.status.value,
                'singleton': descriptor.singleton,
                'dependencies': descriptor.dependencies,
                'has_instance': descriptor.instance is not None,
                'created_at': descriptor.created_at.isoformat() if descriptor.created_at else None,
                'last_health_check': descriptor.last_health_check.isoformat() if descriptor.last_health_check else None,
                'config_keys': list(descriptor.config.keys()) if descriptor.config else []
            }
        
        return services_info
    
    async def shutdown(self) -> None:
        """
        Shutdown all services gracefully.
        """
        self._shutdown_in_progress = True
        
        logger.info("Starting service registry shutdown")
        
        # Get services in reverse dependency order
        shutdown_order = self._get_shutdown_order()
        
        for service_name in shutdown_order:
            descriptor = self._services[service_name]
            
            if descriptor.instance and descriptor.status == ServiceStatus.RUNNING:
                try:
                    descriptor.status = ServiceStatus.STOPPING
                    
                    logger.info(f"Shutting down service '{service_name}'")
                    
                    await self._cleanup_service_instance(descriptor.instance)
                    
                    descriptor.status = ServiceStatus.STOPPED
                    descriptor.instance = None
                    
                    logger.info(f"Service '{service_name}' shut down successfully")
                    
                except Exception as e:
                    descriptor.status = ServiceStatus.ERROR
                    logger.error(f"Error shutting down service '{service_name}': {e}")
        
        logger.info("Service registry shutdown completed")
    
    def _get_shutdown_order(self) -> list:
        """
        Get services in reverse dependency order for shutdown.
        
        Returns:
            list: Service names in shutdown order
        """
        # Simple approach: reverse registration order
        # In a more complex system, you'd implement topological sort
        return list(reversed(list(self._services.keys())))
    
    def unregister_service(self, name: str) -> None:
        """
        Unregister service from registry.
        
        Args:
            name: Service name
        """
        if name not in self._services:
            return
        
        descriptor = self._services[name]
        
        # Cleanup instance if exists
        if descriptor.instance:
            asyncio.create_task(self._cleanup_service_instance(descriptor.instance))
        
        del self._services[name]
        
        logger.info(f"Service '{name}' unregistered")
    
    def __len__(self) -> int:
        """Get number of registered services."""
        return len(self._services)
    
    def __contains__(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self._services
    
    def __iter__(self):
        """Iterate over service names."""
        return iter(self._services.keys())


# Global service registry instance
_service_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    """
    Get global service registry instance.
    
    Returns:
        ServiceRegistry: Global registry instance
    """
    return _service_registry


def register_service(
    name: str,
    service_class: Type,
    factory: Optional[Callable] = None,
    singleton: bool = True,
    dependencies: Optional[list] = None,
    config: Optional[Dict[str, Any]] = None,
    health_check: Optional[Callable] = None
) -> None:
    """
    Register service in global registry.
    
    Args:
        name: Service name
        service_class: Service class
        factory: Factory function
        singleton: Singleton flag
        dependencies: Dependencies list
        config: Service configuration
        health_check: Health check function
    """
    _service_registry.register_service(
        name=name,
        service_class=service_class,
        factory=factory,
        singleton=singleton,
        dependencies=dependencies,
        config=config,
        health_check=health_check
    )


async def get_service(name: str, **kwargs) -> Any:
    """
    Get service from global registry.
    
    Args:
        name: Service name
        **kwargs: Additional arguments
        
    Returns:
        Service instance
    """
    return await _service_registry.get_service(name, **kwargs)


@asynccontextmanager
async def get_service_context(name: str, **kwargs) -> AsyncContextManager:
    """
    Get service as context manager from global registry.
    
    Args:
        name: Service name
        **kwargs: Additional arguments
        
    Yields:
        Service instance
    """
    async with _service_registry.get_service_context(name, **kwargs) as service:
        yield service


async def shutdown_services() -> None:
    """Shutdown all services in global registry."""
    await _service_registry.shutdown()