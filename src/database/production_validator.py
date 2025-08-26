#!/usr/bin/env python3
"""
Production Database Validation and Testing Suite
Comprehensive validation of database configuration and performance
"""

import asyncio
import sys
import os
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import subprocess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection_pool import setup_advanced_connection_pool
from database.redis_cluster import setup_redis_cluster, RedisNodeConfig
from database.backup_recovery import setup_backup_manager, BackupConfig
from database.health_monitor import setup_health_monitor
from database.session import DatabaseManager, DatabaseConfig
from core.logger import get_logger

logger = get_logger(__name__)


class DatabaseValidationError(Exception):
    """Database validation specific error."""
    pass


class ProductionDatabaseValidator:
    """
    Production database validation and testing suite.
    
    Validates:
    - Database connectivity and authentication
    - Connection pool performance
    - Read replica functionality  
    - Redis cluster connectivity
    - Backup and recovery processes
    - Performance benchmarks
    - Security configurations
    """
    
    def __init__(self):
        """Initialize validator."""
        self.results: Dict[str, Any] = {
            'validation_id': f"db_validation_{int(time.time())}",
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'running',
            'tests': {},
            'errors': [],
            'warnings': [],
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Load configuration from environment
        self.config = self._load_configuration()
        
        logger.info("Production database validator initialized")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load database configuration from environment."""
        return {
            'database_url': os.getenv('DATABASE_URL', ''),
            'db_host': os.getenv('DB_HOST', 'localhost'),
            'db_port': int(os.getenv('DB_PORT', '5432')),
            'db_name': os.getenv('DB_NAME', 'claude_tui_prod'),
            'db_user': os.getenv('DB_USER', 'claude_tui_prod'),
            'db_password': os.getenv('DB_PASSWORD', ''),
            'db_pool_size': int(os.getenv('DB_POOL_SIZE', '25')),
            'db_max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '15')),
            'redis_cluster_nodes': os.getenv('REDIS_CLUSTER_NODES', '').split(',') if os.getenv('REDIS_CLUSTER_NODES') else [],
            'redis_password': os.getenv('REDIS_PASSWORD', ''),
            'backup_enabled': os.getenv('DB_BACKUP_ENABLED', 'true').lower() == 'true',
            'backup_s3_bucket': os.getenv('DB_BACKUP_S3_BUCKET', ''),
            'ssl_require': os.getenv('DB_SSL_REQUIRE', 'true').lower() == 'true'
        }
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("Starting full database validation suite")
        
        validation_start = time.time()
        
        try:
            # Core connectivity tests
            await self._test_database_connectivity()
            await self._test_database_authentication()
            await self._test_database_permissions()
            
            # Performance tests
            await self._test_connection_pool_performance()
            await self._test_query_performance()
            
            # Redis tests
            if self.config['redis_cluster_nodes']:
                await self._test_redis_connectivity()
                await self._test_redis_performance()
            
            # Backup tests
            if self.config['backup_enabled']:
                await self._test_backup_functionality()
            
            # Security tests
            await self._test_security_configuration()
            
            # Read replica tests (if configured)
            await self._test_read_replicas()
            
            # Health monitoring tests
            await self._test_health_monitoring()
            
            self.results['status'] = 'completed'
            self.results['total_duration_seconds'] = time.time() - validation_start
            
            # Generate recommendations
            self._generate_recommendations()
            
            logger.info(f"Database validation completed in {self.results['total_duration_seconds']:.2f}s")
            
        except Exception as e:
            self.results['status'] = 'failed'
            self.results['fatal_error'] = str(e)
            logger.error(f"Database validation failed: {e}")
        
        return self.results
    
    async def _test_database_connectivity(self):
        """Test basic database connectivity."""
        test_name = "database_connectivity"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            # Create database config
            db_config = DatabaseConfig(
                database_url=self.config['database_url'],
                pool_size=2,  # Small pool for testing
                max_overflow=1,
                pool_timeout=30,
                echo=False
            )
            
            # Initialize database manager
            db_manager = DatabaseManager(db_config)
            await db_manager.initialize()
            
            # Test basic query
            async with db_manager.get_session() as session:
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1 as test_value, version() as db_version"))
                row = result.fetchone()
                
                if row and row.test_value == 1:
                    self.results['tests'][test_name] = {
                        'status': 'passed',
                        'duration_ms': (time.time() - start_time) * 1000,
                        'db_version': row.db_version,
                        'connection_successful': True
                    }
                else:
                    raise DatabaseValidationError("Unexpected query result")
            
            await db_manager.close()
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
            self.results['errors'].append(f"Database connectivity failed: {e}")
            raise DatabaseValidationError(f"Database connectivity test failed: {e}")
    
    async def _test_database_authentication(self):
        """Test database authentication and user permissions."""
        test_name = "database_authentication"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            db_config = DatabaseConfig(database_url=self.config['database_url'])
            db_manager = DatabaseManager(db_config)
            await db_manager.initialize()
            
            async with db_manager.get_session() as session:
                from sqlalchemy import text
                
                # Test user permissions
                result = await session.execute(text("""
                    SELECT 
                        current_user as username,
                        session_user as session_user,
                        current_database() as database_name,
                        inet_server_addr() as server_ip
                """))
                row = result.fetchone()
                
                # Check if user can create/drop tables (basic DDL permissions)
                try:
                    await session.execute(text("CREATE TABLE test_permissions_check (id INTEGER)"))
                    await session.execute(text("DROP TABLE test_permissions_check"))
                    await session.commit()
                    ddl_permissions = True
                except Exception:
                    ddl_permissions = False
                    await session.rollback()
                
                self.results['tests'][test_name] = {
                    'status': 'passed',
                    'duration_ms': (time.time() - start_time) * 1000,
                    'username': row.username,
                    'database_name': row.database_name,
                    'server_ip': row.server_ip,
                    'ddl_permissions': ddl_permissions
                }
                
                if not ddl_permissions:
                    self.results['warnings'].append("User lacks DDL permissions - migrations may fail")
            
            await db_manager.close()
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
            self.results['errors'].append(f"Database authentication failed: {e}")
    
    async def _test_database_permissions(self):
        """Test required database permissions."""
        test_name = "database_permissions"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            db_config = DatabaseConfig(database_url=self.config['database_url'])
            db_manager = DatabaseManager(db_config)
            await db_manager.initialize()
            
            permissions = {
                'create_table': False,
                'insert_data': False,
                'update_data': False,
                'delete_data': False,
                'create_index': False,
                'create_sequence': False
            }
            
            async with db_manager.get_session() as session:
                from sqlalchemy import text
                
                # Test table creation
                try:
                    await session.execute(text("""
                        CREATE TABLE test_permissions (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    permissions['create_table'] = True
                    
                    # Test insert
                    await session.execute(text(
                        "INSERT INTO test_permissions (name) VALUES ('test')"
                    ))
                    permissions['insert_data'] = True
                    
                    # Test update
                    await session.execute(text(
                        "UPDATE test_permissions SET name = 'updated' WHERE name = 'test'"
                    ))
                    permissions['update_data'] = True
                    
                    # Test delete
                    await session.execute(text(
                        "DELETE FROM test_permissions WHERE name = 'updated'"
                    ))
                    permissions['delete_data'] = True
                    
                    # Test index creation
                    await session.execute(text(
                        "CREATE INDEX idx_test_permissions_name ON test_permissions (name)"
                    ))
                    permissions['create_index'] = True
                    
                    # Test sequence creation
                    await session.execute(text(
                        "CREATE SEQUENCE test_seq START 1"
                    ))
                    permissions['create_sequence'] = True
                    
                    # Clean up
                    await session.execute(text("DROP SEQUENCE test_seq"))
                    await session.execute(text("DROP TABLE test_permissions"))
                    await session.commit()
                    
                except Exception as e:
                    await session.rollback()
                    logger.warning(f"Permission test failed: {e}")
            
            await db_manager.close()
            
            self.results['tests'][test_name] = {
                'status': 'passed',
                'duration_ms': (time.time() - start_time) * 1000,
                'permissions': permissions
            }
            
            # Check if all required permissions are available
            required_permissions = ['create_table', 'insert_data', 'update_data', 'delete_data']
            missing_permissions = [p for p in required_permissions if not permissions[p]]
            
            if missing_permissions:
                self.results['errors'].append(f"Missing required permissions: {missing_permissions}")
                
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    async def _test_connection_pool_performance(self):
        """Test connection pool performance and optimization."""
        test_name = "connection_pool_performance"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            db_config = DatabaseConfig(
                database_url=self.config['database_url'],
                pool_size=self.config['db_pool_size'],
                max_overflow=self.config['db_max_overflow'],
                pool_timeout=30,
                echo=False
            )
            
            db_manager = DatabaseManager(db_config)
            await db_manager.initialize()
            
            # Set up advanced connection pool
            pool_manager = await setup_advanced_connection_pool(
                db_manager.engine,
                min_pool_size=self.config['db_pool_size'],
                max_pool_size=self.config['db_pool_size'] + self.config['db_max_overflow']
            )
            
            # Warm up connections
            await pool_manager.warm_connections(self.config['db_pool_size'])
            
            # Test concurrent connections
            concurrent_tasks = []
            for i in range(20):  # More than pool size to test overflow
                concurrent_tasks.append(self._test_single_connection(db_manager))
            
            # Execute concurrent tasks
            start_concurrent = time.time()
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            concurrent_duration = time.time() - start_concurrent
            
            # Analyze results
            successful_connections = sum(1 for r in results if not isinstance(r, Exception))
            failed_connections = len(results) - successful_connections
            
            # Get pool statistics
            pool_stats = await pool_manager.get_pool_statistics()
            
            await pool_manager.close()
            await db_manager.close()
            
            self.results['tests'][test_name] = {
                'status': 'passed',
                'duration_ms': (time.time() - start_time) * 1000,
                'concurrent_test_duration_ms': concurrent_duration * 1000,
                'successful_connections': successful_connections,
                'failed_connections': failed_connections,
                'pool_statistics': pool_stats,
                'connection_success_rate': successful_connections / len(results) * 100
            }
            
            # Performance warnings
            if concurrent_duration > 5:  # 5 seconds
                self.results['warnings'].append("Connection pool performance may be suboptimal")
            
            if failed_connections > 0:
                self.results['warnings'].append(f"{failed_connections} connections failed during load test")
                
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    async def _test_single_connection(self, db_manager):
        """Test a single database connection."""
        async with db_manager.get_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT pg_sleep(0.1), 1 as test"))
            return result.scalar()
    
    async def _test_query_performance(self):
        """Test database query performance."""
        test_name = "query_performance"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            db_config = DatabaseConfig(database_url=self.config['database_url'])
            db_manager = DatabaseManager(db_config)
            await db_manager.initialize()
            
            query_times = []
            
            async with db_manager.get_session() as session:
                from sqlalchemy import text
                
                # Test simple queries
                for i in range(10):
                    query_start = time.time()
                    await session.execute(text("SELECT 1"))
                    query_times.append((time.time() - query_start) * 1000)
                
                # Test complex query
                complex_query_start = time.time()
                await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        most_common_vals
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    LIMIT 10
                """))
                complex_query_time = (time.time() - complex_query_start) * 1000
            
            await db_manager.close()
            
            avg_query_time = sum(query_times) / len(query_times)
            max_query_time = max(query_times)
            min_query_time = min(query_times)
            
            self.results['tests'][test_name] = {
                'status': 'passed',
                'duration_ms': (time.time() - start_time) * 1000,
                'avg_simple_query_ms': avg_query_time,
                'max_simple_query_ms': max_query_time,
                'min_simple_query_ms': min_query_time,
                'complex_query_ms': complex_query_time
            }
            
            self.results['performance_metrics']['avg_query_time_ms'] = avg_query_time
            
            # Performance warnings
            if avg_query_time > 100:  # 100ms
                self.results['warnings'].append("Average query time is high (>100ms)")
            
            if complex_query_time > 1000:  # 1 second
                self.results['warnings'].append("Complex query performance may need optimization")
                
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    async def _test_redis_connectivity(self):
        """Test Redis cluster connectivity."""
        test_name = "redis_connectivity"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            # Set up Redis cluster
            cluster_manager = await setup_redis_cluster(
                cluster_nodes=self.config['redis_cluster_nodes'],
                password=self.config['redis_password']
            )
            
            # Test basic operations
            test_key = f"validation_test_{int(time.time())}"
            test_value = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
            
            # Test SET
            set_success = await cluster_manager.set(test_key, test_value, expire=300)
            
            # Test GET
            retrieved_value = await cluster_manager.get(test_key)
            
            # Test EXISTS
            exists = await cluster_manager.exists(test_key)
            
            # Test DELETE
            delete_success = await cluster_manager.delete(test_key)
            
            # Get cluster info
            cluster_info = await cluster_manager.get_cluster_info()
            
            await cluster_manager.close()
            
            self.results['tests'][test_name] = {
                'status': 'passed',
                'duration_ms': (time.time() - start_time) * 1000,
                'set_success': set_success,
                'get_success': retrieved_value == test_value,
                'exists_success': exists,
                'delete_success': delete_success,
                'cluster_info': cluster_info
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
            self.results['warnings'].append("Redis connectivity failed - caching disabled")
    
    async def _test_redis_performance(self):
        """Test Redis performance."""
        test_name = "redis_performance"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            cluster_manager = await setup_redis_cluster(
                cluster_nodes=self.config['redis_cluster_nodes'],
                password=self.config['redis_password']
            )
            
            # Performance test - multiple operations
            operation_times = []
            
            for i in range(100):
                op_start = time.time()
                await cluster_manager.set(f"perf_test_{i}", {"data": i}, expire=60)
                operation_times.append((time.time() - op_start) * 1000)
            
            # Batch get test
            batch_start = time.time()
            keys = [f"perf_test_{i}" for i in range(100)]
            batch_results = await cluster_manager.mget(keys)
            batch_time = (time.time() - batch_start) * 1000
            
            # Clean up
            for i in range(100):
                await cluster_manager.delete(f"perf_test_{i}")
            
            await cluster_manager.close()
            
            avg_op_time = sum(operation_times) / len(operation_times)
            
            self.results['tests'][test_name] = {
                'status': 'passed',
                'duration_ms': (time.time() - start_time) * 1000,
                'avg_operation_time_ms': avg_op_time,
                'max_operation_time_ms': max(operation_times),
                'batch_get_time_ms': batch_time,
                'batch_success_count': len(batch_results)
            }
            
            self.results['performance_metrics']['redis_avg_operation_ms'] = avg_op_time
            
            if avg_op_time > 10:  # 10ms
                self.results['warnings'].append("Redis operation time is high (>10ms)")
                
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    async def _test_backup_functionality(self):
        """Test backup functionality."""
        test_name = "backup_functionality"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            backup_config = BackupConfig(
                enabled=True,
                retention_days=7,
                local_path="/tmp/claude_tui_backup_test",
                s3_bucket=self.config['backup_s3_bucket'],
                compression=True,
                encryption=False  # Disabled for testing
            )
            
            backup_manager = await setup_backup_manager(
                database_url=self.config['database_url'],
                config=backup_config,
                encryption_key=None
            )
            
            # Create test backup (without upload to avoid S3 costs)
            test_backup = await backup_manager.create_full_backup(
                backup_name="validation_test",
                upload_to_s3=False
            )
            
            # Verify backup file exists
            backup_path = Path(test_backup.location)
            backup_exists = backup_path.exists()
            
            # Get backup status
            backup_status = await backup_manager.get_backup_status()
            
            # Clean up test backup
            if backup_path.exists():
                backup_path.unlink()
            
            self.results['tests'][test_name] = {
                'status': 'passed',
                'duration_ms': (time.time() - start_time) * 1000,
                'backup_created': test_backup.status == 'completed',
                'backup_file_exists': backup_exists,
                'backup_size_mb': round(test_backup.size_bytes / 1024 / 1024, 2),
                'backup_status': backup_status
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
            self.results['warnings'].append("Backup functionality not working - data loss risk")
    
    async def _test_security_configuration(self):
        """Test security configuration."""
        test_name = "security_configuration"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            security_checks = {
                'ssl_required': self.config['ssl_require'],
                'strong_password': len(self.config['db_password']) >= 12 if self.config['db_password'] else False,
                'non_default_user': self.config['db_user'] not in ['postgres', 'root', 'admin'],
                'non_default_database': self.config['db_name'] not in ['postgres', 'template1', 'template0'],
                'connection_encryption': 'sslmode=require' in self.config['database_url']
            }
            
            # Test SSL connection
            try:
                import ssl
                import asyncpg
                
                # Try to connect with SSL required
                conn = await asyncpg.connect(
                    host=self.config['db_host'],
                    port=self.config['db_port'],
                    user=self.config['db_user'],
                    password=self.config['db_password'],
                    database=self.config['db_name'],
                    ssl='require'
                )
                await conn.close()
                security_checks['ssl_connection_works'] = True
                
            except Exception:
                security_checks['ssl_connection_works'] = False
            
            passed_checks = sum(1 for check in security_checks.values() if check)
            total_checks = len(security_checks)
            
            self.results['tests'][test_name] = {
                'status': 'passed',
                'duration_ms': (time.time() - start_time) * 1000,
                'security_checks': security_checks,
                'security_score': f"{passed_checks}/{total_checks}",
                'security_percentage': (passed_checks / total_checks) * 100
            }
            
            # Security warnings
            if not security_checks['ssl_required']:
                self.results['warnings'].append("SSL not required - connection may not be encrypted")
            
            if not security_checks['strong_password']:
                self.results['warnings'].append("Database password may be weak")
            
            if security_checks['security_percentage'] < 80:
                self.results['warnings'].append("Security configuration needs improvement")
                
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    async def _test_read_replicas(self):
        """Test read replica configuration."""
        test_name = "read_replicas"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            # Check if read replica URLs are configured
            read_replica_urls = os.getenv('DB_READ_REPLICA_URLS', '').split(',')
            read_replica_urls = [url.strip() for url in read_replica_urls if url.strip()]
            
            if not read_replica_urls:
                self.results['tests'][test_name] = {
                    'status': 'skipped',
                    'duration_ms': (time.time() - start_time) * 1000,
                    'message': 'No read replicas configured'
                }
                return
            
            replica_results = []
            
            for i, replica_url in enumerate(read_replica_urls):
                try:
                    replica_config = DatabaseConfig(database_url=replica_url)
                    replica_manager = DatabaseManager(replica_config)
                    await replica_manager.initialize()
                    
                    async with replica_manager.get_session() as session:
                        from sqlalchemy import text
                        result = await session.execute(text(
                            "SELECT pg_is_in_recovery(), pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()"
                        ))
                        row = result.fetchone()
                        
                        is_replica = row[0] if row else False
                        
                        replica_results.append({
                            'replica_index': i,
                            'connection_successful': True,
                            'is_replica': is_replica,
                            'receive_lsn': str(row[1]) if row and row[1] else None,
                            'replay_lsn': str(row[2]) if row and row[2] else None
                        })
                    
                    await replica_manager.close()
                    
                except Exception as e:
                    replica_results.append({
                        'replica_index': i,
                        'connection_successful': False,
                        'error': str(e)
                    })
            
            successful_replicas = sum(1 for r in replica_results if r['connection_successful'])
            
            self.results['tests'][test_name] = {
                'status': 'passed',
                'duration_ms': (time.time() - start_time) * 1000,
                'total_replicas': len(read_replica_urls),
                'successful_replicas': successful_replicas,
                'replica_details': replica_results
            }
            
            if successful_replicas == 0:
                self.results['errors'].append("No read replicas are accessible")
            elif successful_replicas < len(read_replica_urls):
                self.results['warnings'].append(f"Only {successful_replicas}/{len(read_replica_urls)} read replicas are accessible")
                
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    async def _test_health_monitoring(self):
        """Test health monitoring functionality."""
        test_name = "health_monitoring"
        logger.info(f"Testing {test_name}")
        
        start_time = time.time()
        
        try:
            db_config = DatabaseConfig(database_url=self.config['database_url'])
            db_manager = DatabaseManager(db_config)
            await db_manager.initialize()
            
            # Set up health monitor
            health_monitor = await setup_health_monitor(
                engine=db_manager.engine,
                alert_webhook=None,  # Disable for testing
                recovery_actions_enabled=False  # Disable for testing
            )
            
            # Wait for initial health checks
            await asyncio.sleep(5)
            
            # Get health status
            health_status = await health_monitor.get_health_status()
            
            await health_monitor.stop_monitoring()
            await db_manager.close()
            
            self.results['tests'][test_name] = {
                'status': 'passed',
                'duration_ms': (time.time() - start_time) * 1000,
                'monitoring_active': health_status['monitoring_active'],
                'overall_status': health_status['overall_status'],
                'health_checks_count': len(health_status['health_checks']),
                'performance_summary': health_status['performance_summary']
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'failed',
                'duration_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
            self.results['warnings'].append("Health monitoring setup failed")
    
    def _generate_recommendations(self):
        """Generate optimization recommendations based on test results."""
        recommendations = []
        
        # Connection pool recommendations
        if 'connection_pool_performance' in self.results['tests']:
            pool_test = self.results['tests']['connection_pool_performance']
            if pool_test['status'] == 'passed':
                success_rate = pool_test.get('connection_success_rate', 0)
                if success_rate < 95:
                    recommendations.append("Consider increasing connection pool size or timeout values")
        
        # Query performance recommendations
        if 'avg_query_time_ms' in self.results['performance_metrics']:
            avg_time = self.results['performance_metrics']['avg_query_time_ms']
            if avg_time > 50:
                recommendations.append("Consider query optimization or adding database indexes")
            if avg_time > 100:
                recommendations.append("Database performance is suboptimal - urgent optimization needed")
        
        # Redis recommendations
        if 'redis_avg_operation_ms' in self.results['performance_metrics']:
            redis_time = self.results['performance_metrics']['redis_avg_operation_ms']
            if redis_time > 5:
                recommendations.append("Redis performance may benefit from optimization or closer proximity")
        
        # Security recommendations
        if len(self.results['warnings']) > 5:
            recommendations.append("Multiple security and configuration issues detected - review security settings")
        
        # Backup recommendations
        backup_test = self.results['tests'].get('backup_functionality', {})
        if backup_test.get('status') != 'passed':
            recommendations.append("Set up automated backups to prevent data loss")
        
        # High availability recommendations
        replica_test = self.results['tests'].get('read_replicas', {})
        if replica_test.get('status') == 'skipped':
            recommendations.append("Consider setting up read replicas for high availability and load distribution")
        
        self.results['recommendations'] = recommendations
    
    def save_results(self, output_file: Optional[str] = None):
        """Save validation results to file."""
        if not output_file:
            output_file = f"database_validation_report_{int(time.time())}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Validation results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print validation summary to console."""
        print("\n" + "="*80)
        print("DATABASE VALIDATION SUMMARY")
        print("="*80)
        
        print(f"Validation ID: {self.results['validation_id']}")
        print(f"Status: {self.results['status'].upper()}")
        print(f"Duration: {self.results.get('total_duration_seconds', 0):.2f}s")
        
        print(f"\nTests: {len(self.results['tests'])}")
        passed_tests = sum(1 for test in self.results['tests'].values() if test.get('status') == 'passed')
        failed_tests = sum(1 for test in self.results['tests'].values() if test.get('status') == 'failed')
        skipped_tests = sum(1 for test in self.results['tests'].values() if test.get('status') == 'skipped')
        
        print(f"  ✓ Passed: {passed_tests}")
        print(f"  ✗ Failed: {failed_tests}")
        print(f"  - Skipped: {skipped_tests}")
        
        if self.results['errors']:
            print(f"\nErrors ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"  ✗ {error}")
        
        if self.results['warnings']:
            print(f"\nWarnings ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"  ⚠ {warning}")
        
        if self.results['recommendations']:
            print(f"\nRecommendations ({len(self.results['recommendations'])}):")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


async def main():
    """Main validation function."""
    print("Starting Production Database Validation")
    print("=" * 50)
    
    # Initialize validator
    validator = ProductionDatabaseValidator()
    
    try:
        # Run full validation suite
        results = await validator.run_full_validation()
        
        # Print summary
        validator.print_summary()
        
        # Save detailed results
        output_file = validator.save_results()
        print(f"\nDetailed results saved to: {output_file}")
        
        # Exit with appropriate code
        if results['status'] == 'failed' or len(results['errors']) > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())