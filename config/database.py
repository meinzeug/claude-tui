"""
Database Configuration

Production-ready database configuration with:
- Environment-specific settings
- Connection pooling optimization
- Security configurations
- Performance tuning
- Monitoring integration
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.database.session import DatabaseConfig


class DatabaseEnvironment(Enum):
    """Database environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseSettings:
    """Database configuration settings."""
    
    # Connection settings
    database_url: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    pool_pre_ping: bool
    
    # Performance settings
    echo: bool
    echo_pool: bool
    query_cache_size: int
    
    # Security settings
    connect_args: Dict[str, Any]
    
    # Monitoring settings
    enable_metrics: bool
    slow_query_threshold: float
    
    # Backup settings
    backup_retention_days: int
    auto_backup_enabled: bool


class DatabaseConfigurationManager:
    """Database configuration management."""
    
    def __init__(self, environment: Optional[DatabaseEnvironment] = None):
        """
        Initialize configuration manager.
        
        Args:
            environment: Database environment (auto-detected if None)
        """
        self.environment = environment or self._detect_environment()
    
    def _detect_environment(self) -> DatabaseEnvironment:
        """
        Detect current environment from environment variables.
        
        Returns:
            DatabaseEnvironment: Detected environment
        """
        env_name = os.getenv('ENVIRONMENT', 'development').lower()
        
        try:
            return DatabaseEnvironment(env_name)
        except ValueError:
            # Default to development for unknown environments
            return DatabaseEnvironment.DEVELOPMENT
    
    def get_database_url(self) -> str:
        """
        Get database URL for current environment.
        
        Returns:
            str: Database connection URL
        """
        # Check for explicit DATABASE_URL first
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return database_url
        
        # Environment-specific database URLs
        if self.environment == DatabaseEnvironment.PRODUCTION:
            return self._build_production_url()
        elif self.environment == DatabaseEnvironment.STAGING:
            return self._build_staging_url()
        elif self.environment == DatabaseEnvironment.TESTING:
            return self._build_testing_url()
        else:  # DEVELOPMENT
            return self._build_development_url()
    
    def _build_production_url(self) -> str:
        """Build production database URL."""
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'claude_tiu_prod')
        username = os.getenv('DB_USER', 'claude_tiu')
        password = os.getenv('DB_PASSWORD')
        
        if not password:
            raise ValueError("DB_PASSWORD must be set for production environment")
        
        return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    
    def _build_staging_url(self) -> str:
        """Build staging database URL."""
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'claude_tiu_staging')
        username = os.getenv('DB_USER', 'claude_tiu')
        password = os.getenv('DB_PASSWORD')
        
        if not password:
            raise ValueError("DB_PASSWORD must be set for staging environment")
        
        return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    
    def _build_testing_url(self) -> str:
        """Build testing database URL."""
        # Use in-memory SQLite for tests
        return "sqlite+aiosqlite:///:memory:"
    
    def _build_development_url(self) -> str:
        """Build development database URL."""
        # Use local PostgreSQL or SQLite
        if os.getenv('USE_SQLITE', 'false').lower() == 'true':
            return "sqlite+aiosqlite:///./claude_tiu_dev.db"
        else:
            host = os.getenv('DB_HOST', 'localhost')
            port = os.getenv('DB_PORT', '5432')
            database = os.getenv('DB_NAME', 'claude_tiu_dev')
            username = os.getenv('DB_USER', 'claude_tiu')
            password = os.getenv('DB_PASSWORD', 'development')
            
            return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    
    def get_database_settings(self) -> DatabaseSettings:
        """
        Get database settings for current environment.
        
        Returns:
            DatabaseSettings: Database configuration settings
        """
        return {
            DatabaseEnvironment.PRODUCTION: self._get_production_settings,
            DatabaseEnvironment.STAGING: self._get_staging_settings,
            DatabaseEnvironment.TESTING: self._get_testing_settings,
            DatabaseEnvironment.DEVELOPMENT: self._get_development_settings,
        }[self.environment]()
    
    def _get_production_settings(self) -> DatabaseSettings:
        """Get production database settings."""
        return DatabaseSettings(
            database_url=self.get_database_url(),
            
            # Optimized connection pool for production
            pool_size=int(os.getenv('DB_POOL_SIZE', '20')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '10')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '3600')),  # 1 hour
            pool_pre_ping=True,
            
            # Performance settings
            echo=False,  # No SQL logging in production
            echo_pool=False,
            query_cache_size=int(os.getenv('DB_QUERY_CACHE_SIZE', '500')),
            
            # Security settings
            connect_args={
                'server_settings': {
                    'application_name': 'claude-tiu-prod',
                    'jit': 'off',  # Disable JIT for predictable performance
                },
                'command_timeout': 60,
                'ssl': 'require' if os.getenv('DB_SSL_REQUIRE', 'true').lower() == 'true' else 'prefer',
            },
            
            # Monitoring settings
            enable_metrics=True,
            slow_query_threshold=1.0,  # 1 second
            
            # Backup settings
            backup_retention_days=30,
            auto_backup_enabled=True,
        )
    
    def _get_staging_settings(self) -> DatabaseSettings:
        """Get staging database settings."""
        return DatabaseSettings(
            database_url=self.get_database_url(),
            
            # Moderate connection pool for staging
            pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '5')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '60')),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '1800')),  # 30 minutes
            pool_pre_ping=True,
            
            # Performance settings
            echo=os.getenv('DB_ECHO', 'false').lower() == 'true',
            echo_pool=False,
            query_cache_size=int(os.getenv('DB_QUERY_CACHE_SIZE', '200')),
            
            # Security settings
            connect_args={
                'server_settings': {
                    'application_name': 'claude-tiu-staging',
                },
                'command_timeout': 120,
                'ssl': 'prefer',
            },
            
            # Monitoring settings
            enable_metrics=True,
            slow_query_threshold=2.0,  # 2 seconds
            
            # Backup settings
            backup_retention_days=7,
            auto_backup_enabled=True,
        )
    
    def _get_development_settings(self) -> DatabaseSettings:
        """Get development database settings."""
        return DatabaseSettings(
            database_url=self.get_database_url(),
            
            # Smaller connection pool for development
            pool_size=int(os.getenv('DB_POOL_SIZE', '5')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '2')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '300')),  # 5 minutes
            pool_pre_ping=True,
            
            # Performance settings (with debugging)
            echo=os.getenv('DB_ECHO', 'false').lower() == 'true',
            echo_pool=os.getenv('DB_ECHO_POOL', 'false').lower() == 'true',
            query_cache_size=int(os.getenv('DB_QUERY_CACHE_SIZE', '50')),
            
            # Security settings (relaxed for development)
            connect_args={
                'server_settings': {
                    'application_name': 'claude-tiu-dev',
                },
                'command_timeout': 300,  # 5 minutes for debugging
            } if 'postgresql' in self.get_database_url() else {},
            
            # Monitoring settings
            enable_metrics=True,
            slow_query_threshold=5.0,  # 5 seconds (relaxed for development)
            
            # Backup settings
            backup_retention_days=3,
            auto_backup_enabled=False,
        )
    
    def _get_testing_settings(self) -> DatabaseSettings:
        """Get testing database settings."""
        return DatabaseSettings(
            database_url=self.get_database_url(),
            
            # Minimal connection pool for testing
            pool_size=1,
            max_overflow=0,
            pool_timeout=10,
            pool_recycle=-1,  # Disable recycling
            pool_pre_ping=False,
            
            # Performance settings (minimal for speed)
            echo=False,
            echo_pool=False,
            query_cache_size=10,
            
            # Security settings (none for in-memory testing)
            connect_args={},
            
            # Monitoring settings (disabled for testing)
            enable_metrics=False,
            slow_query_threshold=10.0,
            
            # Backup settings (disabled for testing)
            backup_retention_days=0,
            auto_backup_enabled=False,
        )
    
    def create_database_config(self) -> DatabaseConfig:
        """
        Create DatabaseConfig instance from settings.
        
        Returns:
            DatabaseConfig: Configured database config
        """
        settings = self.get_database_settings()
        
        return DatabaseConfig(
            database_url=settings.database_url,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            pool_pre_ping=settings.pool_pre_ping,
            echo=settings.echo,
            connect_args=settings.connect_args,
        )


# Global configuration instances
_config_manager = DatabaseConfigurationManager()


def get_database_config(environment: Optional[DatabaseEnvironment] = None) -> DatabaseConfig:
    """
    Get database configuration for environment.
    
    Args:
        environment: Target environment (auto-detected if None)
        
    Returns:
        DatabaseConfig: Configured database config
    """
    if environment:
        manager = DatabaseConfigurationManager(environment)
    else:
        manager = _config_manager
    
    return manager.create_database_config()


def get_database_settings(environment: Optional[DatabaseEnvironment] = None) -> DatabaseSettings:
    """
    Get database settings for environment.
    
    Args:
        environment: Target environment (auto-detected if None)
        
    Returns:
        DatabaseSettings: Database settings
    """
    if environment:
        manager = DatabaseConfigurationManager(environment)
    else:
        manager = _config_manager
    
    return manager.get_database_settings()


def validate_database_environment() -> Dict[str, Any]:
    """
    Validate database environment configuration.
    
    Returns:
        dict: Validation results
    """
    try:
        config = get_database_config()
        settings = get_database_settings()
        
        validation_results = {
            'valid': True,
            'environment': _config_manager.environment.value,
            'database_url_set': bool(config.database_url),
            'using_postgresql': 'postgresql' in config.database_url,
            'using_sqlite': 'sqlite' in config.database_url,
            'pool_size': config.pool_size,
            'ssl_required': settings.connect_args.get('ssl') == 'require',
            'metrics_enabled': settings.enable_metrics,
            'warnings': []
        }
        
        # Environment-specific validations
        if _config_manager.environment == DatabaseEnvironment.PRODUCTION:
            if not settings.connect_args.get('ssl'):
                validation_results['warnings'].append('SSL not configured for production')
            
            if config.pool_size < 10:
                validation_results['warnings'].append('Pool size may be too small for production')
            
            if 'sqlite' in config.database_url:
                validation_results['warnings'].append('SQLite not recommended for production')
        
        elif _config_manager.environment == DatabaseEnvironment.STAGING:
            if config.pool_size < 5:
                validation_results['warnings'].append('Pool size may be too small for staging')
        
        return validation_results
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'environment': _config_manager.environment.value,
        }


# Environment-specific config getters
def get_production_config() -> DatabaseConfig:
    """Get production database configuration."""
    return get_database_config(DatabaseEnvironment.PRODUCTION)


def get_staging_config() -> DatabaseConfig:
    """Get staging database configuration."""
    return get_database_config(DatabaseEnvironment.STAGING)


def get_development_config() -> DatabaseConfig:
    """Get development database configuration."""
    return get_database_config(DatabaseEnvironment.DEVELOPMENT)


def get_testing_config() -> DatabaseConfig:
    """Get testing database configuration."""
    return get_database_config(DatabaseEnvironment.TESTING)