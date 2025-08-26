#!/usr/bin/env python3
"""
Production Database Setup and Initialization Script
Complete setup for production database with all components
"""

import os
import sys
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from database.session import DatabaseManager, DatabaseConfig
from database.connection_pool import setup_advanced_connection_pool
from database.redis_cluster import setup_redis_cluster, RedisNodeConfig
from database.backup_recovery import setup_backup_manager, BackupConfig
from database.health_monitor import setup_health_monitor
from database.read_replica_manager import setup_read_replica_manager, LoadBalancingStrategy
from database.production_validator import ProductionDatabaseValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionDatabaseSetup:
    """Production database setup orchestrator."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.config = self._load_environment_config()
        self.results = {
            'setup_id': f"db_setup_{int(__import__('time').time())}",
            'status': 'initializing',
            'components': {},
            'errors': [],
            'warnings': []
        }
        
        logger.info("Production database setup initialized")
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            # Database configuration
            'database_url': os.getenv('DATABASE_URL', ''),
            'db_host': os.getenv('DB_HOST', 'localhost'),
            'db_port': int(os.getenv('DB_PORT', '5432')),
            'db_name': os.getenv('DB_NAME', 'claude_tui_prod'),
            'db_user': os.getenv('DB_USER', 'claude_tui_prod'),
            'db_password': os.getenv('DB_PASSWORD', ''),
            'db_pool_size': int(os.getenv('DB_POOL_SIZE', '25')),
            'db_max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '15')),
            'db_ssl_require': os.getenv('DB_SSL_REQUIRE', 'true').lower() == 'true',
            
            # Read replicas
            'read_replica_urls': [
                url.strip() for url in os.getenv('DB_READ_REPLICA_URLS', '').split(',') 
                if url.strip()
            ],
            
            # Redis configuration
            'redis_cluster_nodes': [
                node.strip() for node in os.getenv('REDIS_CLUSTER_NODES', '').split(',')
                if node.strip()
            ],
            'redis_password': os.getenv('REDIS_PASSWORD', ''),
            'redis_url': os.getenv('REDIS_URL', ''),
            
            # Backup configuration
            'backup_enabled': os.getenv('DB_BACKUP_ENABLED', 'true').lower() == 'true',
            'backup_s3_bucket': os.getenv('DB_BACKUP_S3_BUCKET', ''),
            'backup_retention_days': int(os.getenv('DB_BACKUP_RETENTION_DAYS', '30')),
            
            # Monitoring
            'monitoring_enabled': os.getenv('DB_ENABLE_METRICS', 'true').lower() == 'true',
            'health_check_interval': int(os.getenv('HEALTH_CHECK_INTERVAL', '30')),
            
            # Environment
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
    
    async def run_complete_setup(self) -> Dict[str, Any]:
        """Run complete production database setup."""
        logger.info("Starting complete production database setup")
        
        try:
            self.results['status'] = 'running'
            
            # Step 1: Validate environment and prerequisites
            await self._validate_prerequisites()
            
            # Step 2: Initialize database schema
            await self._initialize_database_schema()
            
            # Step 3: Set up primary database connection
            await self._setup_primary_database()
            
            # Step 4: Set up connection pooling
            await self._setup_connection_pooling()
            
            # Step 5: Set up read replicas (if configured)
            if self.config['read_replica_urls']:
                await self._setup_read_replicas()
            
            # Step 6: Set up Redis cluster (if configured)
            if self.config['redis_cluster_nodes'] or self.config['redis_url']:
                await self._setup_redis_cluster()
            
            # Step 7: Set up backup system
            if self.config['backup_enabled']:
                await self._setup_backup_system()
            
            # Step 8: Set up health monitoring
            if self.config['monitoring_enabled']:
                await self._setup_health_monitoring()
            
            # Step 9: Run validation suite
            await self._run_validation_suite()
            
            self.results['status'] = 'completed'
            logger.info("Production database setup completed successfully")
            
        except Exception as e:
            self.results['status'] = 'failed'
            self.results['fatal_error'] = str(e)
            logger.error(f"Production database setup failed: {e}")
            raise
        
        return self.results
    
    async def _validate_prerequisites(self):
        """Validate prerequisites and environment."""
        component = 'prerequisites'
        logger.info(f"Validating {component}")
        
        try:
            checks = {
                'database_url_set': bool(self.config['database_url']),
                'database_password_set': bool(self.config['db_password']),
                'environment_valid': self.config['environment'] in ['development', 'staging', 'production'],
                'postgresql_available': await self._check_postgresql_availability(),
                'required_packages': self._check_required_packages()
            }
            
            # Check if PostgreSQL is accessible
            if not checks['postgresql_available']:
                raise Exception("PostgreSQL is not accessible")
            
            # Check for required Python packages
            if not checks['required_packages']:
                raise Exception("Required packages are missing")
            
            self.results['components'][component] = {
                'status': 'completed',
                'checks': checks
            }
            
            logger.info(f"{component} validation passed")
            
        except Exception as e:
            self.results['components'][component] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['errors'].append(f"Prerequisites validation failed: {e}")
            raise
    
    async def _check_postgresql_availability(self) -> bool:
        """Check if PostgreSQL is available."""
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host=self.config['db_host'],
                port=self.config['db_port'],
                user=self.config['db_user'],
                password=self.config['db_password'],
                database='postgres'  # Connect to default database
            )
            await conn.close()
            return True
        except Exception:
            return False
    
    def _check_required_packages(self) -> bool:
        """Check if required Python packages are available."""
        required_packages = [
            'sqlalchemy',
            'asyncpg',
            'redis',
            'alembic',
            'boto3'  # For S3 backups
        ]
        
        try:
            for package in required_packages:
                __import__(package)
            return True
        except ImportError:
            return False
    
    async def _initialize_database_schema(self):
        """Initialize database schema using Alembic."""
        component = 'database_schema'
        logger.info(f"Initializing {component}")
        
        try:
            # Run Alembic migrations
            alembic_config_path = Path(__file__).parent.parent / 'alembic.ini'
            
            # Check if alembic.ini exists
            if not alembic_config_path.exists():
                logger.warning(f"Alembic config not found at {alembic_config_path}")
                self.results['warnings'].append("Alembic configuration not found - skipping migrations")
                self.results['components'][component] = {
                    'status': 'skipped',
                    'reason': 'Alembic configuration not found'
                }
                return
            
            # Set database URL for Alembic
            env = os.environ.copy()
            env['DATABASE_URL'] = self.config['database_url']
            
            # Run migrations
            result = subprocess.run([
                'alembic', 
                '-c', str(alembic_config_path),
                'upgrade', 'head'
            ], capture_output=True, text=True, env=env, cwd=str(alembic_config_path.parent))
            
            if result.returncode == 0:
                self.results['components'][component] = {
                    'status': 'completed',
                    'migration_output': result.stdout
                }
                logger.info("Database schema initialized successfully")
            else:
                raise Exception(f"Alembic migration failed: {result.stderr}")
                
        except Exception as e:
            self.results['components'][component] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['errors'].append(f"Database schema initialization failed: {e}")
            # Don't raise - continue with setup
            logger.warning(f"Database schema setup failed: {e}")
    
    async def _setup_primary_database(self):
        """Set up primary database connection."""
        component = 'primary_database'
        logger.info(f"Setting up {component}")
        
        try:
            # Create database config
            db_config = DatabaseConfig(
                database_url=self.config['database_url'],
                pool_size=self.config['db_pool_size'],
                max_overflow=self.config['db_max_overflow'],
                pool_timeout=30,
                echo=False
            )
            
            # Initialize database manager
            db_manager = DatabaseManager(db_config)
            await db_manager.initialize()
            
            # Test connection
            async with db_manager.get_session() as session:
                from sqlalchemy import text
                result = await session.execute(text("SELECT version()"))
                version = result.scalar()
            
            await db_manager.close()
            
            self.results['components'][component] = {
                'status': 'completed',
                'database_version': version,
                'pool_size': self.config['db_pool_size'],
                'max_overflow': self.config['db_max_overflow']
            }
            
            logger.info(f"Primary database setup completed")
            
        except Exception as e:
            self.results['components'][component] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['errors'].append(f"Primary database setup failed: {e}")
            raise
    
    async def _setup_connection_pooling(self):
        """Set up advanced connection pooling."""
        component = 'connection_pooling'
        logger.info(f"Setting up {component}")
        
        try:
            # Create database config
            db_config = DatabaseConfig(database_url=self.config['database_url'])
            db_manager = DatabaseManager(db_config)
            await db_manager.initialize()
            
            # Set up advanced connection pool
            pool_manager = await setup_advanced_connection_pool(
                db_manager.engine,
                min_pool_size=self.config['db_pool_size'],
                max_pool_size=self.config['db_pool_size'] + self.config['db_max_overflow'],
                auto_optimize=True
            )
            
            # Warm up connections
            await pool_manager.warm_connections(self.config['db_pool_size'])
            
            # Get initial statistics
            pool_stats = await pool_manager.get_pool_statistics()
            
            # Clean up for setup
            await pool_manager.close()
            await db_manager.close()
            
            self.results['components'][component] = {
                'status': 'completed',
                'initial_pool_stats': pool_stats
            }
            
            logger.info("Connection pooling setup completed")
            
        except Exception as e:
            self.results['components'][component] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['warnings'].append(f"Advanced connection pooling setup failed: {e}")
            # Don't raise - basic pooling will still work
    
    async def _setup_read_replicas(self):
        """Set up read replicas."""
        component = 'read_replicas'
        logger.info(f"Setting up {component}")
        
        try:
            # Create master config
            master_config = DatabaseConfig(database_url=self.config['database_url'])
            
            # Create replica configs
            replica_configs = [
                DatabaseConfig(database_url=replica_url)
                for replica_url in self.config['read_replica_urls']
            ]
            
            # Set up replica manager
            replica_manager = await setup_read_replica_manager(
                master_config=master_config,
                replica_configs=replica_configs,
                strategy=LoadBalancingStrategy.WEIGHTED_RANDOM,
                health_check_interval=self.config['health_check_interval']
            )
            
            # Force initial health check
            await replica_manager.force_health_check()
            
            # Get statistics
            replica_stats = await replica_manager.get_replica_statistics()
            
            # Clean up for setup
            await replica_manager.close()
            
            self.results['components'][component] = {
                'status': 'completed',
                'total_replicas': len(replica_configs),
                'replica_stats': replica_stats
            }
            
            logger.info(f"Read replicas setup completed ({len(replica_configs)} replicas)")
            
        except Exception as e:
            self.results['components'][component] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['warnings'].append(f"Read replicas setup failed: {e}")
            # Don't raise - system can work without replicas
    
    async def _setup_redis_cluster(self):
        """Set up Redis cluster."""
        component = 'redis_cluster'
        logger.info(f"Setting up {component}")
        
        try:
            if self.config['redis_cluster_nodes']:
                # Set up Redis cluster
                cluster_manager = await setup_redis_cluster(
                    cluster_nodes=self.config['redis_cluster_nodes'],
                    password=self.config['redis_password'],
                    max_connections_per_node=50,
                    retry_on_timeout=True
                )
                
                # Test basic operations
                test_key = "setup_test"
                await cluster_manager.set(test_key, "test_value", expire=60)
                test_value = await cluster_manager.get(test_key)
                await cluster_manager.delete(test_key)
                
                # Get cluster info
                cluster_info = await cluster_manager.get_cluster_info()
                
                # Clean up for setup
                await cluster_manager.close()
                
                self.results['components'][component] = {
                    'status': 'completed',
                    'cluster_type': 'cluster',
                    'nodes_count': len(self.config['redis_cluster_nodes']),
                    'test_successful': test_value == "test_value",
                    'cluster_info': cluster_info
                }
                
            else:
                # Single Redis instance fallback
                import redis.asyncio as aioredis
                
                redis_client = aioredis.from_url(self.config['redis_url'])
                
                # Test connection
                await redis_client.ping()
                await redis_client.set("setup_test", "test_value", ex=60)
                test_value = await redis_client.get("setup_test")
                await redis_client.delete("setup_test")
                await redis_client.close()
                
                self.results['components'][component] = {
                    'status': 'completed',
                    'cluster_type': 'single',
                    'test_successful': test_value == b"test_value"
                }
            
            logger.info("Redis setup completed")
            
        except Exception as e:
            self.results['components'][component] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['warnings'].append(f"Redis setup failed: {e}")
            # Don't raise - system can work without Redis
    
    async def _setup_backup_system(self):
        """Set up backup system."""
        component = 'backup_system'
        logger.info(f"Setting up {component}")
        
        try:
            backup_config = BackupConfig(
                enabled=True,
                retention_days=self.config['backup_retention_days'],
                local_path="/var/backups/claude-tui",
                s3_bucket=self.config['backup_s3_bucket'],
                compression=True,
                encryption=True
            )
            
            backup_manager = await setup_backup_manager(
                database_url=self.config['database_url'],
                config=backup_config
            )
            
            # Get backup status
            backup_status = await backup_manager.get_backup_status()
            
            self.results['components'][component] = {
                'status': 'completed',
                'backup_config': {
                    'retention_days': backup_config.retention_days,
                    'compression': backup_config.compression,
                    'encryption': backup_config.encryption,
                    's3_enabled': bool(backup_config.s3_bucket)
                },
                'backup_status': backup_status
            }
            
            logger.info("Backup system setup completed")
            
        except Exception as e:
            self.results['components'][component] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['warnings'].append(f"Backup system setup failed: {e}")
            # Don't raise - system can work without backups (but warn user)
    
    async def _setup_health_monitoring(self):
        """Set up health monitoring."""
        component = 'health_monitoring'
        logger.info(f"Setting up {component}")
        
        try:
            # Create database config
            db_config = DatabaseConfig(database_url=self.config['database_url'])
            db_manager = DatabaseManager(db_config)
            await db_manager.initialize()
            
            # Set up health monitor
            health_monitor = await setup_health_monitor(
                engine=db_manager.engine,
                alert_webhook=None,  # Configure externally
                recovery_actions_enabled=self.config['environment'] == 'production'
            )
            
            # Wait for initial health checks
            await asyncio.sleep(5)
            
            # Get health status
            health_status = await health_monitor.get_health_status()
            
            # Clean up for setup
            await health_monitor.stop_monitoring()
            await db_manager.close()
            
            self.results['components'][component] = {
                'status': 'completed',
                'health_checks_count': len(health_status['health_checks']),
                'initial_status': health_status['overall_status'],
                'recovery_actions_enabled': self.config['environment'] == 'production'
            }
            
            logger.info("Health monitoring setup completed")
            
        except Exception as e:
            self.results['components'][component] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['warnings'].append(f"Health monitoring setup failed: {e}")
            # Don't raise - system can work without monitoring
    
    async def _run_validation_suite(self):
        """Run comprehensive validation suite."""
        component = 'validation_suite'
        logger.info(f"Running {component}")
        
        try:
            validator = ProductionDatabaseValidator()
            validation_results = await validator.run_full_validation()
            
            self.results['components'][component] = {
                'status': 'completed',
                'validation_status': validation_results['status'],
                'tests_passed': sum(1 for test in validation_results['tests'].values() 
                                 if test.get('status') == 'passed'),
                'tests_failed': sum(1 for test in validation_results['tests'].values() 
                                 if test.get('status') == 'failed'),
                'validation_errors': validation_results.get('errors', []),
                'validation_warnings': validation_results.get('warnings', []),
                'performance_metrics': validation_results.get('performance_metrics', {}),
                'recommendations': validation_results.get('recommendations', [])
            }
            
            # Add validation warnings to overall warnings
            self.results['warnings'].extend(validation_results.get('warnings', []))
            
            if validation_results['status'] == 'failed':
                self.results['warnings'].append("Database validation suite detected issues")
            
            logger.info("Validation suite completed")
            
        except Exception as e:
            self.results['components'][component] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['warnings'].append(f"Validation suite failed: {e}")
            # Don't raise - setup can still be considered successful
    
    def save_results(self, output_file: str = None) -> Path:
        """Save setup results to file."""
        if not output_file:
            output_file = f"database_setup_report_{int(__import__('time').time())}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Setup results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print setup summary."""
        print("\n" + "="*80)
        print("PRODUCTION DATABASE SETUP SUMMARY")
        print("="*80)
        
        print(f"Setup ID: {self.results['setup_id']}")
        print(f"Status: {self.results['status'].upper()}")
        
        print(f"\nComponents:")
        for component, details in self.results['components'].items():
            status_icon = "✓" if details['status'] == 'completed' else "✗" if details['status'] == 'failed' else "-"
            print(f"  {status_icon} {component}: {details['status']}")
        
        if self.results['errors']:
            print(f"\nErrors ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"  ✗ {error}")
        
        if self.results['warnings']:
            print(f"\nWarnings ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"  ⚠ {warning}")
        
        print("\n" + "="*80)


async def main():
    """Main setup function."""
    print("Starting Production Database Setup")
    print("=" * 50)
    
    setup_manager = ProductionDatabaseSetup()
    
    try:
        results = await setup_manager.run_complete_setup()
        
        # Print summary
        setup_manager.print_summary()
        
        # Save results
        output_file = setup_manager.save_results()
        print(f"\nDetailed results saved to: {output_file}")
        
        # Exit code
        if results['status'] == 'failed':
            sys.exit(1)
        elif len(results['warnings']) > 0:
            print("\nSetup completed with warnings - review configuration")
            sys.exit(0)
        else:
            print("\nSetup completed successfully")
            sys.exit(0)
            
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())