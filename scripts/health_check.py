#!/usr/bin/env python3
"""
Database Health Check Script

Comprehensive database health monitoring with:
- Connection pool status
- Repository health validation
- Performance metrics
- Migration status
- Security audits
- Resource utilization
- Alert generation
- Monitoring integration
"""

import asyncio
import argparse
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.service import DatabaseService, create_database_service
from src.core.logger import get_logger

logger = get_logger(__name__)


class DatabaseHealthChecker:
    """Comprehensive database health monitoring."""
    
    def __init__(self, database_url: str = None, environment: str = 'production'):
        """
        Initialize health checker.
        
        Args:
            database_url: Database connection URL
            environment: Environment name
        """
        self.database_url = database_url
        self.environment = environment
        self.service = None
        self.health_results = {}
        
    async def initialize(self) -> bool:
        """
        Initialize database service.
        
        Returns:
            bool: True if successful
        """
        try:
            self.service = await create_database_service(
                database_url=self.database_url,
                pool_size=10,  # Smaller pool for health checks
                max_overflow=2,
                pool_timeout=30
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            return False
    
    async def check_basic_connectivity(self) -> Dict[str, Any]:
        """
        Check basic database connectivity.
        
        Returns:
            dict: Connectivity check results
        """
        logger.info("Checking basic database connectivity...")
        
        start_time = time.time()
        
        try:
            async with self.service.get_session() as session:
                result = await session.execute("SELECT 1 as health_check")
                response_value = result.scalar()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy' if response_value == 1 else 'unhealthy',
                'response_time_ms': round(response_time, 2),
                'error': None,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'unhealthy',
                'response_time_ms': round(response_time, 2),
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def check_connection_pool(self) -> Dict[str, Any]:
        """
        Check connection pool health and performance.
        
        Returns:
            dict: Connection pool status
        """
        logger.info("Checking connection pool health...")
        
        try:
            # Get pool status
            pool_status = await self.service.session_manager.get_pool_status()
            
            # Run concurrent connection test
            pool_test = await self.service._test_connection_pool()
            
            # Calculate pool efficiency
            pool_size = pool_status.get('pool_size', 0)
            checked_out = pool_status.get('checked_out', 0)
            
            efficiency = (checked_out / pool_size * 100) if pool_size > 0 else 0
            
            # Determine health status
            status = 'healthy'
            warnings = []
            
            if pool_test.get('failed_connections', 0) > 0:
                status = 'degraded'
                warnings.append('Some connections failed')
            
            if efficiency > 80:
                status = 'degraded'
                warnings.append('High pool utilization')
            
            if pool_test.get('average_connection_time', 0) > 1000:  # > 1 second
                status = 'degraded'
                warnings.append('Slow connection times')
            
            return {
                'status': status,
                'pool_status': pool_status,
                'pool_test': pool_test,
                'efficiency_percent': round(efficiency, 2),
                'warnings': warnings,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def check_repositories(self) -> Dict[str, Any]:
        """
        Check repository health and basic operations.
        
        Returns:
            dict: Repository health status
        """
        logger.info("Checking repository health...")
        
        repository_status = {}
        overall_status = 'healthy'
        
        try:
            async with self.service.get_repositories() as repos:
                repo_types = ['user', 'project', 'task', 'audit', 'session']
                
                for repo_type in repo_types:
                    repo_name = f'{repo_type}_repository'
                    start_time = time.time()
                    
                    try:
                        repo = getattr(repos, f'get_{repo_type}_repository')()\n                        
                        # Test basic count operation
                        count = await repo.count()
                        
                        # Test health check method if available
                        if hasattr(repo, 'health_check'):
                            await repo.health_check()
                        
                        response_time = (time.time() - start_time) * 1000
                        
                        repository_status[repo_type] = {
                            'status': 'healthy',
                            'record_count': count,
                            'response_time_ms': round(response_time, 2),
                            'error': None
                        }
                        
                    except Exception as e:
                        response_time = (time.time() - start_time) * 1000
                        overall_status = 'unhealthy'
                        
                        repository_status[repo_type] = {
                            'status': 'unhealthy',
                            'response_time_ms': round(response_time, 2),
                            'error': str(e)
                        }
            
            return {
                'status': overall_status,
                'repositories': repository_status,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def check_migration_status(self) -> Dict[str, Any]:
        """
        Check database migration status.
        
        Returns:
            dict: Migration status information
        """
        logger.info("Checking migration status...")
        
        try:
            migration_status = await self.service.get_migration_status()
            
            # Determine health based on migration status
            status = 'healthy'
            warnings = []
            
            if migration_status.get('status') == 'error':
                status = 'unhealthy'
            elif migration_status.get('status') == 'configuration_missing':
                status = 'degraded'
                warnings.append('Alembic configuration missing')
            elif not migration_status.get('is_up_to_date', True):
                status = 'degraded'
                warnings.append('Database not up to date')
            
            return {
                'status': status,
                'migration_info': migration_status,
                'warnings': warnings,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def check_performance_metrics(self) -> Dict[str, Any]:
        """
        Check database performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        logger.info("Checking performance metrics...")
        
        try:
            # Get database statistics
            stats = await self.service.get_database_statistics()
            
            # Measure query performance
            start_time = time.time()
            async with self.service.get_session() as session:
                await session.execute("SELECT COUNT(*) FROM sqlite_master")  # SQLite compatible
            query_time = (time.time() - start_time) * 1000
            
            # Measure transaction performance
            start_time = time.time()
            async def dummy_transaction(repos):
                return True
            await self.service.execute_in_transaction(dummy_transaction)
            transaction_time = (time.time() - start_time) * 1000
            
            # Evaluate performance
            status = 'healthy'
            warnings = []
            
            if query_time > 500:  # > 500ms
                status = 'degraded'
                warnings.append('Slow query performance')
            
            if transaction_time > 1000:  # > 1 second
                status = 'degraded'
                warnings.append('Slow transaction performance')
            
            return {
                'status': status,
                'query_performance_ms': round(query_time, 2),
                'transaction_performance_ms': round(transaction_time, 2),
                'database_statistics': stats,
                'warnings': warnings,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def check_security_health(self) -> Dict[str, Any]:
        """
        Check database security health.
        
        Returns:
            dict: Security health status
        """
        logger.info("Checking security health...")
        
        try:
            warnings = []
            status = 'healthy'
            
            async with self.service.get_repositories() as repos:
                audit_repo = repos.get_audit_repository()
                session_repo = repos.get_session_repository()
                
                # Check for recent failed login attempts
                failed_logins = await audit_repo.get_failed_actions(
                    days=1,
                    action_filter='login_failed'
                )
                
                if len(failed_logins) > 50:  # More than 50 failed logins in 24h
                    status = 'degraded'
                    warnings.append(f'High number of failed login attempts: {len(failed_logins)}')
                
                # Check for suspicious session activity
                suspicious_sessions = await session_repo.get_suspicious_sessions(
                    days=1,
                    max_sessions_per_user=10,
                    max_sessions_per_ip=20
                )
                
                total_suspicious = sum(len(sessions) for sessions in suspicious_sessions.values())
                if total_suspicious > 10:
                    status = 'degraded'
                    warnings.append(f'Suspicious session activity detected: {total_suspicious} sessions')
                
                # Check for old active sessions
                old_threshold = datetime.utcnow() - timedelta(days=30)
                old_sessions = await session_repo.get_all(
                    filters={
                        'is_active': True,
                        'created_at__lt': old_threshold
                    },
                    limit=100
                )
                
                if len(old_sessions) > 5:
                    status = 'degraded'
                    warnings.append(f'Old active sessions found: {len(old_sessions)}')
            
            return {
                'status': status,
                'failed_logins_24h': len(failed_logins),
                'suspicious_sessions_24h': total_suspicious,
                'old_active_sessions': len(old_sessions),
                'warnings': warnings,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def check_resource_utilization(self) -> Dict[str, Any]:
        """
        Check database resource utilization.
        
        Returns:
            dict: Resource utilization status
        """
        logger.info("Checking resource utilization...")
        
        try:
            # Get database info
            db_info = await self.service.get_database_info()
            
            warnings = []
            status = 'healthy'
            
            # Check connection pool utilization
            pool_status = db_info.get('connection_pool', {})
            pool_size = pool_status.get('pool_size', 0)
            checked_out = pool_status.get('checked_out', 0)
            
            if pool_size > 0:
                utilization = (checked_out / pool_size) * 100
                if utilization > 80:
                    status = 'degraded'
                    warnings.append(f'High connection pool utilization: {utilization:.1f}%')
            
            # Check query response time
            response_time = db_info.get('query_response_time_ms', 0)
            if response_time > 100:  # > 100ms
                status = 'degraded'
                warnings.append(f'Slow query response time: {response_time}ms')
            
            return {
                'status': status,
                'connection_pool_utilization_percent': round((checked_out / pool_size) * 100, 2) if pool_size > 0 else 0,
                'query_response_time_ms': response_time,
                'database_info': db_info,
                'warnings': warnings,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check.
        
        Returns:
            dict: Complete health check results
        """
        logger.info("Running comprehensive database health check...")
        
        # Run all health checks concurrently
        checks = [
            ('connectivity', self.check_basic_connectivity()),
            ('connection_pool', self.check_connection_pool()),
            ('repositories', self.check_repositories()),
            ('migrations', self.check_migration_status()),
            ('performance', self.check_performance_metrics()),
            ('security', self.check_security_health()),
            ('resources', self.check_resource_utilization())
        ]
        
        results = {}
        for check_name, check_coro in checks:
            try:
                results[check_name] = await check_coro
            except Exception as e:
                results[check_name] = {
                    'status': 'unhealthy',
                    'error': f'Health check failed: {str(e)}',
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        # Calculate overall status
        statuses = [result.get('status', 'unhealthy') for result in results.values()]
        
        if all(status == 'healthy' for status in statuses):
            overall_status = 'healthy'
        elif any(status == 'unhealthy' for status in statuses):
            overall_status = 'unhealthy'
        else:
            overall_status = 'degraded'
        
        # Collect all warnings
        all_warnings = []
        for result in results.values():
            warnings = result.get('warnings', [])
            if isinstance(warnings, list):
                all_warnings.extend(warnings)
        
        self.health_results = {
            'overall_status': overall_status,
            'environment': self.environment,
            'checks': results,
            'summary': {
                'total_checks': len(results),
                'healthy_checks': sum(1 for s in statuses if s == 'healthy'),
                'degraded_checks': sum(1 for s in statuses if s == 'degraded'),
                'unhealthy_checks': sum(1 for s in statuses if s == 'unhealthy'),
                'total_warnings': len(all_warnings),
                'warnings': all_warnings
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return self.health_results
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.service:
            await self.service.close()


def format_health_results(results: Dict[str, Any], output_format: str = 'human') -> str:
    """
    Format health check results for output.
    
    Args:
        results: Health check results
        output_format: Output format ('human', 'json', 'prometheus')
        
    Returns:
        str: Formatted output
    """
    if output_format == 'json':
        return json.dumps(results, indent=2)
    
    elif output_format == 'prometheus':
        # Prometheus metrics format
        metrics = []
        
        # Overall status (1=healthy, 0.5=degraded, 0=unhealthy)
        status_value = {
            'healthy': 1,
            'degraded': 0.5,
            'unhealthy': 0
        }.get(results.get('overall_status'), 0)
        
        metrics.append(f'claude_tiu_database_health{{environment="{results.get("environment", "unknown")}"}} {status_value}')
        
        # Individual check statuses
        for check_name, check_result in results.get('checks', {}).items():
            check_status_value = {
                'healthy': 1,
                'degraded': 0.5,
                'unhealthy': 0
            }.get(check_result.get('status'), 0)
            
            metrics.append(f'claude_tiu_database_check_health{{check="{check_name}",environment="{results.get("environment", "unknown")}"}} {check_status_value}')
        
        return '\n'.join(metrics)
    
    else:  # human-readable format
        output = []
        output.append("=" * 60)
        output.append("CLAUDE-TIU DATABASE HEALTH CHECK REPORT")
        output.append("=" * 60)
        output.append(f"Environment: {results.get('environment', 'unknown')}")
        output.append(f"Timestamp: {results.get('timestamp', 'unknown')}")
        output.append(f"Overall Status: {results.get('overall_status', 'unknown').upper()}")
        output.append("")
        
        # Summary
        summary = results.get('summary', {})
        output.append("SUMMARY:")
        output.append(f"  Total Checks: {summary.get('total_checks', 0)}")
        output.append(f"  Healthy: {summary.get('healthy_checks', 0)}")
        output.append(f"  Degraded: {summary.get('degraded_checks', 0)}")
        output.append(f"  Unhealthy: {summary.get('unhealthy_checks', 0)}")
        output.append(f"  Warnings: {summary.get('total_warnings', 0)}")
        output.append("")
        
        # Individual checks
        output.append("DETAILED RESULTS:")
        for check_name, check_result in results.get('checks', {}).items():
            status = check_result.get('status', 'unknown').upper()
            output.append(f"  {check_name.upper()}: {status}")
            
            if check_result.get('error'):
                output.append(f"    Error: {check_result['error']}")
            
            warnings = check_result.get('warnings', [])
            if warnings:
                for warning in warnings:
                    output.append(f"    Warning: {warning}")
            
            # Add specific metrics for some checks
            if check_name == 'connectivity':
                response_time = check_result.get('response_time_ms')
                if response_time is not None:
                    output.append(f"    Response Time: {response_time}ms")
            
            elif check_name == 'performance':
                query_time = check_result.get('query_performance_ms')
                if query_time is not None:
                    output.append(f"    Query Performance: {query_time}ms")
        
        output.append("")
        output.append("=" * 60)
        
        return '\n'.join(output)


async def main():
    """Main health check execution."""
    parser = argparse.ArgumentParser(description="Claude-TIU Database Health Check")
    
    parser.add_argument(
        '--database-url', '-d',
        help='Database connection URL'
    )
    
    parser.add_argument(
        '--environment', '-e',
        default='production',
        help='Environment name'
    )
    
    parser.add_argument(
        '--output-format', '-f',
        choices=['human', 'json', 'prometheus'],
        default='human',
        help='Output format'
    )
    
    parser.add_argument(
        '--output-file', '-o',
        help='Output file path (default: stdout)'
    )
    
    parser.add_argument(
        '--exit-code-on-degraded',
        action='store_true',
        help='Exit with code 1 on degraded status (default: only unhealthy)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Health check timeout in seconds'
    )
    
    args = parser.parse_args()
    
    checker = DatabaseHealthChecker(
        database_url=args.database_url,
        environment=args.environment
    )
    
    try:
        # Initialize with timeout
        init_task = asyncio.create_task(checker.initialize())
        
        if not await asyncio.wait_for(init_task, timeout=30):
            logger.error("Failed to initialize database service")
            return 1
        
        # Run health checks with timeout
        health_task = asyncio.create_task(checker.run_comprehensive_health_check())
        results = await asyncio.wait_for(health_task, timeout=args.timeout)
        
        # Format output
        formatted_output = format_health_results(results, args.output_format)
        
        # Write output
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(formatted_output)
            logger.info(f"Health check results written to {args.output_file}")
        else:
            print(formatted_output)
        
        # Determine exit code
        overall_status = results.get('overall_status', 'unhealthy')
        
        if overall_status == 'healthy':
            return 0
        elif overall_status == 'degraded' and args.exit_code_on_degraded:
            return 1
        elif overall_status == 'unhealthy':
            return 1
        else:
            return 0
        
    except asyncio.TimeoutError:
        logger.error(f"Health check timed out after {args.timeout} seconds")
        return 1
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return 1
        
    finally:
        await checker.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))