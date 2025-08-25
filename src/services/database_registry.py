"""
Database Service Registration

Integration of database service with service registry for:
- Dependency injection
- Service lifecycle management
- Configuration management
- Health monitoring
- Resource cleanup
"""

import asyncio
import os
from typing import Dict, Any, Optional

from .registry import register_service, ServiceRegistryError
from ..database.service import DatabaseService
from ..database.session import DatabaseConfig
from ..core.logger import get_logger
from config.database import get_database_config, DatabaseEnvironment

logger = get_logger(__name__)


async def database_service_factory(
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
    **kwargs
) -> DatabaseService:
    """
    Factory function for creating database service instances.
    
    Args:
        config: Service configuration
        dependencies: Service dependencies
        **kwargs: Additional arguments
        
    Returns:
        DatabaseService: Configured database service instance
        
    Raises:
        ServiceRegistryError: If service creation fails
    """
    try:
        # Get environment-specific database configuration
        environment = None
        if config and 'environment' in config:
            env_name = config['environment']
            try:
                environment = DatabaseEnvironment(env_name.lower())
            except ValueError:
                logger.warning(f"Invalid environment '{env_name}', using auto-detection")
        
        # Create database configuration
        db_config = get_database_config(environment)
        
        # Override with any provided configuration
        if config:
            config_overrides = config.get('database_overrides', {})
            for key, value in config_overrides.items():
                if hasattr(db_config, key):
                    setattr(db_config, key, value)
                    logger.info(f"Database config override: {key} = {value}")
        
        # Create database service
        service = DatabaseService(config=db_config)
        
        # Initialize the service
        await service.initialize()
        
        logger.info("Database service created and initialized successfully")
        
        return service
        
    except Exception as e:
        logger.error(f"Failed to create database service: {e}")
        raise ServiceRegistryError(
            f"Database service factory failed: {str(e)}",
            "DATABASE_SERVICE_FACTORY_ERROR",
            {"error": str(e)}
        )


async def database_service_health_check(service: DatabaseService) -> Dict[str, Any]:
    """
    Health check function for database service.
    
    Args:
        service: Database service instance
        
    Returns:
        dict: Health check results
    """
    try:
        health_result = await service.health_check()
        
        # Extract key health indicators
        overall_status = health_result.get('status', 'unknown')
        
        # Detailed health information
        detailed_result = {
            'overall_status': overall_status,
            'service_initialized': health_result.get('initialized', False),
            'connection_pool_healthy': health_result.get('connection_pool_test', {}).get('test_passed', False),
            'repository_health': health_result.get('repository_health', {}),
            'database_info': health_result.get('database_info', {}),
            'timestamp': health_result.get('timestamp')
        }
        
        # Calculate health score
        health_indicators = [
            detailed_result['service_initialized'],
            detailed_result['connection_pool_healthy'],
            overall_status == 'healthy'
        ]
        
        repo_health = detailed_result['repository_health']
        if repo_health:
            healthy_repos = sum(1 for status in repo_health.values() if status == 'healthy')
            total_repos = len(repo_health)
            repo_health_score = healthy_repos / total_repos if total_repos > 0 else 0
            health_indicators.append(repo_health_score > 0.8)  # 80% of repos must be healthy
        
        health_score = sum(health_indicators) / len(health_indicators)
        detailed_result['health_score'] = health_score
        detailed_result['healthy'] = health_score >= 0.75  # 75% threshold
        
        return detailed_result
        
    except Exception as e:
        logger.error(f"Database service health check failed: {e}")
        return {
            'overall_status': 'unhealthy',
            'healthy': False,
            'health_score': 0.0,
            'error': str(e),
            'timestamp': None
        }


def register_database_service(
    service_name: str = 'database',
    environment: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register database service with service registry.
    
    Args:
        service_name: Name to register service under
        environment: Target environment
        config_overrides: Configuration overrides
    """
    # Prepare service configuration
    service_config = {
        'environment': environment or os.getenv('ENVIRONMENT', 'development'),
        'database_overrides': config_overrides or {}
    }
    
    # Register the database service
    register_service(
        name=service_name,
        service_class=DatabaseService,
        factory=database_service_factory,
        singleton=True,  # Database service should be singleton
        dependencies=[],  # Database service has no dependencies
        config=service_config,
        health_check=database_service_health_check
    )
    
    logger.info(f"Database service registered as '{service_name}'")


def register_database_services() -> None:
    """
    Register all database-related services.
    
    This function registers:
    - Main database service
    - Environment-specific database services
    - Database utility services
    """
    try:
        # Register main database service
        register_database_service('database')
        
        # Register environment-specific services
        environments = ['development', 'staging', 'production', 'testing']
        
        for env in environments:
            service_name = f'database_{env}'
            
            try:
                register_database_service(
                    service_name=service_name,
                    environment=env
                )
                logger.info(f"Registered database service for {env} environment")
                
            except Exception as e:
                logger.warning(f"Failed to register {env} database service: {e}")
        
        logger.info("Database services registration completed")
        
    except Exception as e:
        logger.error(f"Failed to register database services: {e}")
        raise ServiceRegistryError(
            "Database services registration failed",
            "DATABASE_SERVICES_REGISTRATION_ERROR",
            {"error": str(e)}
        )


class DatabaseServiceManager:
    """
    Database service manager for advanced service operations.
    
    Provides additional management capabilities beyond basic registry:
    - Service monitoring
    - Configuration management
    - Performance tracking
    - Maintenance operations
    """
    
    def __init__(self, service_registry=None):
        """
        Initialize database service manager.
        
        Args:
            service_registry: Service registry instance (uses global if None)
        """
        from .registry import get_service_registry
        
        self.registry = service_registry or get_service_registry()
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    async def get_database_service(self, environment: Optional[str] = None) -> DatabaseService:
        """
        Get database service for environment.
        
        Args:
            environment: Target environment (main service if None)
            
        Returns:
            DatabaseService: Database service instance
        """
        if environment:
            service_name = f'database_{environment}'
        else:
            service_name = 'database'
        
        return await self.registry.get_service(service_name)
    
    async def monitor_database_services(self) -> Dict[str, Any]:
        """
        Monitor all database services.
        
        Returns:
            dict: Monitoring results for all database services
        """
        monitoring_results = {}
        
        # Get all database services
        service_info = await self.registry.get_service_info()
        database_services = {
            name: info for name, info in service_info.items()
            if name.startswith('database')
        }
        
        for service_name in database_services:
            try:
                # Get service health
                health_results = await self.registry.health_check(service_name)
                service_health = health_results.get(service_name, {})
                
                # Get service performance metrics if available
                try:
                    service = await self.registry.get_service(service_name)
                    if hasattr(service, 'get_database_statistics'):
                        stats = await service.get_database_statistics()
                    else:
                        stats = {}
                except Exception:
                    stats = {}
                
                monitoring_results[service_name] = {
                    'health': service_health,
                    'statistics': stats,
                    'service_info': database_services[service_name]
                }
                
            except Exception as e:
                monitoring_results[service_name] = {
                    'error': str(e),
                    'service_info': database_services.get(service_name, {})
                }
        
        return monitoring_results
    
    async def run_maintenance_on_all_services(self) -> Dict[str, Any]:
        """
        Run maintenance tasks on all database services.
        
        Returns:
            dict: Maintenance results for all services
        """
        maintenance_results = {}
        
        # Get all database services
        service_info = await self.registry.get_service_info()
        database_services = [
            name for name in service_info.keys()
            if name.startswith('database')
        ]
        
        for service_name in database_services:
            try:
                service = await self.registry.get_service(service_name)
                
                if hasattr(service, 'execute_maintenance_tasks'):
                    results = await service.execute_maintenance_tasks()
                    maintenance_results[service_name] = {
                        'status': 'completed',
                        'results': results
                    }
                else:
                    maintenance_results[service_name] = {
                        'status': 'skipped',
                        'reason': 'No maintenance method available'
                    }
                    
            except Exception as e:
                maintenance_results[service_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return maintenance_results
    
    async def validate_all_configurations(self) -> Dict[str, Any]:
        """
        Validate configurations for all database services.
        
        Returns:
            dict: Validation results for all services
        """
        validation_results = {}
        
        environments = ['development', 'staging', 'production', 'testing']
        
        for env in environments:
            try:
                from config.database import get_database_config, validate_database_environment, DatabaseEnvironment
                
                # Validate environment configuration
                env_enum = DatabaseEnvironment(env)
                config = get_database_config(env_enum)
                
                # Basic configuration validation
                validation_result = {
                    'environment': env,
                    'valid': True,
                    'database_url_set': bool(config.database_url),
                    'pool_size': config.pool_size,
                    'warnings': []
                }
                
                # Environment-specific validations
                if env == 'production':
                    if config.pool_size < 10:
                        validation_result['warnings'].append('Pool size may be too small for production')
                    
                    if 'sqlite' in config.database_url:
                        validation_result['warnings'].append('SQLite not recommended for production')
                
                validation_results[env] = validation_result
                
            except Exception as e:
                validation_results[env] = {
                    'environment': env,
                    'valid': False,
                    'error': str(e)
                }
        
        return validation_results


# Global database service manager
_database_service_manager = DatabaseServiceManager()


def get_database_service_manager() -> DatabaseServiceManager:
    """
    Get global database service manager.
    
    Returns:
        DatabaseServiceManager: Global manager instance
    """
    return _database_service_manager