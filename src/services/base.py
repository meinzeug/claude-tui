"""
Base Service Class with Dependency Injection for claude-tiu.

This module provides the foundation service class with:
- Dependency injection container
- Async/await pattern throughout
- Error handling and logging
- Context management
- Performance monitoring
"""

import asyncio
import inspect
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Dict, Optional, Type, TypeVar, Union
from uuid import uuid4

from ..core.exceptions import ClaudeTIUException, ValidationError
from ..core.logger import get_logger, get_performance_logger, get_security_logger

T = TypeVar('T')

# Service context for request tracking
service_context: ContextVar[Dict[str, Any]] = ContextVar('service_context', default={})


class DependencyContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
    
    def register(self, service_class: Type[T], instance: Optional[T] = None, singleton: bool = True) -> None:
        """Register a service in the container."""
        service_name = service_class.__name__
        
        if instance is not None:
            if singleton:
                self._singletons[service_name] = instance
            else:
                self._services[service_name] = instance
        else:
            self._factories[service_name] = service_class
    
    def get(self, service_class: Type[T]) -> T:
        """Get service instance from container."""
        service_name = service_class.__name__
        
        # Check singletons first
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check registered instances
        if service_name in self._services:
            return self._services[service_name]
        
        # Create from factory
        if service_name in self._factories:
            factory = self._factories[service_name]
            instance = factory()
            # Auto-inject dependencies if service is BaseService subclass
            if isinstance(instance, BaseService):
                instance._inject_dependencies(self)
            return instance
        
        # Auto-register if it's a service class
        if hasattr(service_class, '__bases__') and BaseService in service_class.__mro__:
            instance = service_class()
            instance._inject_dependencies(self)
            self._singletons[service_name] = instance
            return instance
        
        raise ValidationError(f"Service {service_name} not found in container")
    
    def has(self, service_class: Type) -> bool:
        """Check if service is registered."""
        service_name = service_class.__name__
        return (service_name in self._singletons or 
                service_name in self._services or 
                service_name in self._factories)


# Global dependency container
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Get global dependency injection container."""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


class BaseService(ABC):
    """
    Base service class providing common functionality.
    
    Features:
    - Dependency injection
    - Logging and performance monitoring
    - Error handling
    - Context management
    - Async/await support
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.performance_logger = get_performance_logger()
        self.security_logger = get_security_logger()
        self._dependencies: Dict[str, Any] = {}
        self._initialized = False
        
    def _inject_dependencies(self, container: DependencyContainer) -> None:
        """Inject dependencies from container."""
        self._container = container
        
        # Analyze constructor for dependencies
        init_signature = inspect.signature(self.__class__.__init__)
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
            
            # Look for type annotations that are service classes
            if param.annotation and hasattr(param.annotation, '__bases__'):
                if BaseService in param.annotation.__mro__:
                    if container.has(param.annotation):
                        self._dependencies[param_name] = container.get(param.annotation)
    
    async def initialize(self) -> None:
        """Initialize service (override in subclasses)."""
        if self._initialized:
            return
        
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
        try:
            await self._initialize_impl()
            self._initialized = True
            self.logger.info(f"Successfully initialized {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
            raise
    
    async def _initialize_impl(self) -> None:
        """Override in subclasses for custom initialization."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the service."""
        return {
            'service': self.__class__.__name__,
            'status': 'healthy' if self._initialized else 'not_initialized',
            'dependencies': len(self._dependencies),
            'initialized': self._initialized
        }
    
    def set_context(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        operation: Optional[str] = None
    ) -> None:
        """Set service context for current operation."""
        context = {
            'request_id': request_id or str(uuid4()),
            'user_id': user_id or 'anonymous',
            'project_id': project_id or 'none',
            'operation': operation or 'unknown',
            'service': self.__class__.__name__
        }
        service_context.set(context)
    
    def get_context(self) -> Dict[str, Any]:
        """Get current service context."""
        return service_context.get({})
    
    async def execute_with_monitoring(
        self,
        operation_name: str,
        operation_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with performance monitoring and error handling."""
        context = self.get_context()
        timer_id = self.performance_logger.start_timer(f"{self.__class__.__name__}.{operation_name}")
        
        try:
            self.logger.debug(f"Starting operation: {operation_name}")
            
            # Execute operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            self.performance_logger.end_timer(timer_id, {
                'operation': operation_name,
                'service': self.__class__.__name__,
                'success': True,
                'context': context
            })
            
            self.logger.debug(f"Completed operation: {operation_name}")
            return result
            
        except ClaudeTIUException as e:
            self.performance_logger.end_timer(timer_id, {
                'operation': operation_name,
                'service': self.__class__.__name__,
                'success': False,
                'error': str(e),
                'context': context
            })
            
            self.logger.error(
                f"Service operation failed: {operation_name}",
                extra={
                    'error_id': e.error_id,
                    'error_code': e.error_code,
                    'operation': operation_name
                }
            )
            raise
            
        except Exception as e:
            self.performance_logger.end_timer(timer_id, {
                'operation': operation_name,
                'service': self.__class__.__name__,
                'success': False,
                'error': str(e),
                'context': context
            })
            
            self.logger.error(
                f"Unexpected error in operation: {operation_name}",
                exc_info=True,
                extra={'operation': operation_name}
            )
            
            # Convert to structured exception
            from ..core.exceptions import handle_exception
            structured_exception = handle_exception(e, self.logger, context)
            raise structured_exception
    
    def get_dependency(self, service_class: Type[T]) -> T:
        """Get injected dependency."""
        service_name = service_class.__name__
        if service_name in self._dependencies:
            return self._dependencies[service_name]
        
        # Try to get from container
        if hasattr(self, '_container'):
            return self._container.get(service_class)
        
        raise ValidationError(f"Dependency {service_name} not found")
    
    def has_dependency(self, service_class: Type) -> bool:
        """Check if dependency is available."""
        service_name = service_class.__name__
        return (service_name in self._dependencies or 
                (hasattr(self, '_container') and self._container.has(service_class)))


class ServiceRegistry:
    """Registry for managing service lifecycle."""
    
    def __init__(self):
        self.container = get_container()
        self._services: Dict[str, BaseService] = {}
    
    async def register_service(
        self, 
        service_class: Type[BaseService], 
        instance: Optional[BaseService] = None,
        auto_initialize: bool = True
    ) -> BaseService:
        """Register and optionally initialize a service."""
        if instance is None:
            instance = service_class()
        
        # Inject dependencies
        instance._inject_dependencies(self.container)
        
        # Initialize if requested
        if auto_initialize:
            await instance.initialize()
        
        # Register in container
        self.container.register(service_class, instance, singleton=True)
        self._services[service_class.__name__] = instance
        
        return instance
    
    async def get_service(self, service_class: Type[T]) -> T:
        """Get initialized service instance."""
        service_name = service_class.__name__
        
        if service_name in self._services:
            return self._services[service_name]
        
        # Auto-register if not found
        if hasattr(service_class, '__bases__') and BaseService in service_class.__mro__:
            return await self.register_service(service_class)
        
        return self.container.get(service_class)
    
    async def initialize_all(self) -> None:
        """Initialize all registered services."""
        for service in self._services.values():
            if not service._initialized:
                await service.initialize()
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Run health checks on all services."""
        results = {}
        for name, service in self._services.items():
            try:
                results[name] = await service.health_check()
            except Exception as e:
                results[name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        return results


# Global service registry
_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get global service registry."""
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry


async def get_service(service_class: Type[T]) -> T:
    """Convenience function to get service instance."""
    registry = get_service_registry()
    return await registry.get_service(service_class)


def inject_dependency(service_class: Type):
    """Decorator for dependency injection."""
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if hasattr(self, '_container'):
                dependency = self._container.get(service_class)
                setattr(self, service_class.__name__.lower(), dependency)
        
        cls.__init__ = new_init
        return cls
    return decorator