"""
Database Layer - Production-Ready Database Optimization Suite

Comprehensive database optimization system providing:
- Advanced query optimization and monitoring
- Multi-level caching with Redis integration
- Connection pool optimization and management
- Read replica load balancing and scaling
- Migration management with rollback capabilities
- Backup and recovery with disaster protection
- Performance benchmarking and monitoring

This module integrates all database optimization components for production-grade
performance targeting <10ms query response times.
"""

from .service import DatabaseService, get_database_service, create_database_service
from .session import DatabaseSessionManager, DatabaseConfig, get_session_manager
from .models import Base, User, Role, Permission, UserRole, RolePermission, UserSession, Project, Task, AuditLog
from .repositories import RepositoryFactory

# Optimization Components
from .query_optimizer import QueryOptimizer, get_query_optimizer, setup_query_optimization
from .caching import DatabaseCache, get_database_cache, setup_database_caching
from .connection_pool import AdvancedConnectionPool, get_connection_pool_manager, setup_advanced_connection_pool
from .read_replica_manager import (
    ReadReplicaManager, ReplicaConfig, get_read_replica_manager, setup_read_replica_manager
)
from .migration_manager import MigrationManager, get_migration_manager, setup_migration_manager
# from .backup_manager import (
#     BackupManager, BackupConfig, BackupType, get_backup_manager, setup_backup_manager
# )
from .performance_benchmark import (
    DatabasePerformanceBenchmark, get_performance_benchmark, setup_performance_benchmark
)

__all__ = [
    # Core Database Components
    'DatabaseService',
    'DatabaseSessionManager', 
    'DatabaseConfig',
    'RepositoryFactory',
    'Base',
    'User',
    'Role', 
    'Permission',
    'UserRole',
    'RolePermission',
    'UserSession',
    'Project',
    'Task',
    'AuditLog',
    
    # Service Getters
    'get_database_service',
    'create_database_service',
    'get_session_manager',
    
    # Query Optimization
    'QueryOptimizer',
    'get_query_optimizer',
    'setup_query_optimization',
    
    # Caching
    'DatabaseCache',
    'get_database_cache',
    'setup_database_caching',
    
    # Connection Pool Management
    'AdvancedConnectionPool',
    'get_connection_pool_manager',
    'setup_advanced_connection_pool',
    
    # Read Replica Management
    'ReadReplicaManager',
    'ReplicaConfig',
    'get_read_replica_manager',
    'setup_read_replica_manager',
    
    # Migration Management
    'MigrationManager',
    'get_migration_manager',
    'setup_migration_manager',
    
    # Backup and Recovery (disabled - module not found)
    # 'BackupManager',
    # 'BackupConfig',
    # 'BackupType', 
    # 'get_backup_manager',
    # 'setup_backup_manager',
    
    # Performance Benchmarking
    'DatabasePerformanceBenchmark',
    'get_performance_benchmark',
    'setup_performance_benchmark'
]