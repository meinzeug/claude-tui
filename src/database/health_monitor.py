"""
Database Health Monitoring and Validation System
Real-time database health checks with automated recovery
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..core.logger import get_logger
from ..core.exceptions import ClaudeTUIException

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    description: str
    check_function: Callable
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True
    interval_seconds: int = 60
    timeout_seconds: int = 30
    
    # Runtime data
    last_run: Optional[datetime] = None
    last_result: Optional[Dict[str, Any]] = None
    status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0


@dataclass 
class HealthMetrics:
    """Database health metrics."""
    timestamp: datetime
    overall_status: HealthStatus
    connection_count: int
    active_queries: int
    idle_queries: int
    long_running_queries: int
    database_size_mb: float
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    cache_hit_ratio: float
    query_performance_ms: float
    replication_lag_ms: float
    error_rate_percent: float
    uptime_hours: float
    
    # Performance metrics
    queries_per_second: float = 0.0
    avg_response_time_ms: float = 0.0
    slowest_query_ms: float = 0.0
    deadlocks_per_hour: float = 0.0


class DatabaseHealthMonitor:
    """
    Comprehensive database health monitoring system.
    
    Features:
    - Real-time health checks and metrics collection
    - Automated alerting and recovery actions
    - Performance trend analysis
    - Connection pool monitoring
    - Query performance analysis
    - Replication health monitoring
    """
    
    def __init__(
        self,
        engine: AsyncEngine,
        read_replica_engines: Optional[List[AsyncEngine]] = None,
        alert_webhook: Optional[str] = None,
        recovery_actions_enabled: bool = True
    ):
        """
        Initialize health monitor.
        
        Args:
            engine: Primary database engine
            read_replica_engines: Read replica engines for monitoring
            alert_webhook: Webhook URL for alerts
            recovery_actions_enabled: Enable automated recovery actions
        """
        self.engine = engine
        self.read_replica_engines = read_replica_engines or []
        self.alert_webhook = alert_webhook
        self.recovery_actions_enabled = recovery_actions_enabled
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Metrics storage
        self.current_metrics: Optional[HealthMetrics] = None
        self.metrics_history: List[HealthMetrics] = []
        self.max_history = 1000  # Keep last 1000 metrics
        
        # Monitoring control
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._check_tasks: Dict[str, asyncio.Task] = {}
        
        # Alert management
        self._alert_cooldown: Dict[str, datetime] = {}
        self._alert_cooldown_minutes = 15
        
        # Performance tracking
        self._query_times: List[float] = []
        self._error_count = 0
        self._total_queries = 0
        self._start_time = datetime.utcnow()
        
        # Register default health checks
        self._register_default_health_checks()
        
        logger.info("Database health monitor initialized")
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        
        # Connection health check
        self.register_health_check(HealthCheck(
            name="connection",
            description="Database connection availability",
            check_function=self._check_connection_health,
            warning_threshold=1.0,
            critical_threshold=5.0,
            interval_seconds=30
        ))
        
        # Query performance check
        self.register_health_check(HealthCheck(
            name="query_performance", 
            description="Average query response time",
            check_function=self._check_query_performance,
            warning_threshold=100.0,  # 100ms
            critical_threshold=1000.0,  # 1s
            interval_seconds=60
        ))
        
        # Connection count check
        self.register_health_check(HealthCheck(
            name="connection_count",
            description="Active database connections",
            check_function=self._check_connection_count,
            warning_threshold=80.0,  # 80% of max connections
            critical_threshold=95.0,  # 95% of max connections
            interval_seconds=60
        ))
        
        # Database size check
        self.register_health_check(HealthCheck(
            name="database_size",
            description="Database storage usage",
            check_function=self._check_database_size,
            warning_threshold=80.0,  # 80% full
            critical_threshold=95.0,  # 95% full
            interval_seconds=300  # Check every 5 minutes
        ))
        
        # Long running queries check
        self.register_health_check(HealthCheck(
            name="long_queries",
            description="Long running queries",
            check_function=self._check_long_running_queries,
            warning_threshold=5.0,  # 5 long queries
            critical_threshold=10.0,  # 10 long queries
            interval_seconds=120
        ))
        
        # Replication lag check
        if self.read_replica_engines:
            self.register_health_check(HealthCheck(
                name="replication_lag",
                description="Replication lag in milliseconds", 
                check_function=self._check_replication_lag,
                warning_threshold=1000.0,  # 1 second
                critical_threshold=5000.0,  # 5 seconds
                interval_seconds=60
            ))
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check."""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self._monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        # Start individual health check tasks
        for name, check in self.health_checks.items():
            if check.enabled:
                self._check_tasks[name] = asyncio.create_task(
                    self._health_check_loop(check)
                )
        
        logger.info("Database health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
        
        for task in self._check_tasks.values():
            task.cancel()
        
        self._check_tasks.clear()
        
        logger.info("Database health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                await self._collect_metrics()
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self, health_check: HealthCheck):
        """Individual health check loop."""
        while self._monitoring_active:
            try:
                await self._run_health_check(health_check)
                await asyncio.sleep(health_check.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check {health_check.name} error: {e}")
                health_check.consecutive_failures += 1
                await asyncio.sleep(health_check.interval_seconds)
    
    async def _run_health_check(self, health_check: HealthCheck):
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            # Run the check function with timeout
            result = await asyncio.wait_for(
                health_check.check_function(),
                timeout=health_check.timeout_seconds
            )
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Update check status
            health_check.last_run = datetime.utcnow()
            health_check.last_result = result
            health_check.consecutive_failures = 0
            
            # Determine status based on thresholds
            value = result.get('value', 0)
            if value >= health_check.critical_threshold:
                health_check.status = HealthStatus.CRITICAL
            elif value >= health_check.warning_threshold:
                health_check.status = HealthStatus.WARNING  
            else:
                health_check.status = HealthStatus.HEALTHY
            
            # Send alerts if needed
            if health_check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                await self._send_alert(health_check, result)
            
            logger.debug(f"Health check {health_check.name}: {health_check.status.value} ({execution_time:.1f}ms)")
            
        except asyncio.TimeoutError:
            health_check.status = HealthStatus.CRITICAL
            health_check.consecutive_failures += 1
            health_check.last_result = {'error': 'Timeout'}
            logger.error(f"Health check {health_check.name} timed out")
            
        except Exception as e:
            health_check.status = HealthStatus.CRITICAL
            health_check.consecutive_failures += 1
            health_check.last_result = {'error': str(e)}
            logger.error(f"Health check {health_check.name} failed: {e}")
    
    async def _check_connection_health(self) -> Dict[str, Any]:
        """Check database connection health."""
        start_time = time.time()
        
        try:
            async with AsyncSession(self.engine) as session:
                result = await session.execute(text("SELECT 1"))
                row = result.fetchone()
                
                if row and row[0] == 1:
                    response_time = (time.time() - start_time) * 1000
                    return {
                        'status': 'healthy',
                        'response_time_ms': response_time,
                        'value': response_time
                    }
                else:
                    return {
                        'status': 'failed',
                        'error': 'Unexpected result',
                        'value': 999999
                    }
                    
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'value': 999999
            }
    
    async def _check_query_performance(self) -> Dict[str, Any]:
        """Check average query performance."""
        if not self._query_times:
            return {'status': 'no_data', 'avg_time_ms': 0, 'value': 0}
        
        # Use recent query times (last 100 queries)
        recent_times = self._query_times[-100:]
        avg_time = statistics.mean(recent_times)
        
        return {
            'status': 'measured',
            'avg_time_ms': avg_time,
            'sample_size': len(recent_times),
            'value': avg_time
        }
    
    async def _check_connection_count(self) -> Dict[str, Any]:
        """Check active connection count."""
        try:
            async with AsyncSession(self.engine) as session:
                # PostgreSQL specific query
                result = await session.execute(text("""
                    SELECT 
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections,
                        current_setting('max_connections')::int as max_connections
                    FROM pg_stat_activity 
                    WHERE pid <> pg_backend_pid()
                """))
                
                row = result.fetchone()
                if row:
                    total = row.total_connections
                    active = row.active_connections  
                    idle = row.idle_connections
                    max_conn = row.max_connections
                    
                    usage_percent = (total / max_conn) * 100
                    
                    return {
                        'status': 'measured',
                        'total_connections': total,
                        'active_connections': active,
                        'idle_connections': idle,
                        'max_connections': max_conn,
                        'usage_percent': usage_percent,
                        'value': usage_percent
                    }
                    
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'value': 0
            }
        
        return {'status': 'no_data', 'value': 0}
    
    async def _check_database_size(self) -> Dict[str, Any]:
        """Check database size and storage usage."""
        try:
            async with AsyncSession(self.engine) as session:
                # Get database size
                result = await session.execute(text("""
                    SELECT pg_database_size(current_database()) / 1024 / 1024 as size_mb
                """))
                row = result.fetchone()
                
                if row:
                    size_mb = row.size_mb
                    
                    # For simplicity, assume warning at 1GB, critical at 5GB
                    # In production, this should be configurable based on available storage
                    usage_percent = min((size_mb / 5120) * 100, 100)  # 5GB = 5120MB
                    
                    return {
                        'status': 'measured', 
                        'size_mb': size_mb,
                        'usage_percent': usage_percent,
                        'value': usage_percent
                    }
                    
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'value': 0
            }
        
        return {'status': 'no_data', 'value': 0}
    
    async def _check_long_running_queries(self) -> Dict[str, Any]:
        """Check for long running queries."""
        try:
            async with AsyncSession(self.engine) as session:
                # Find queries running longer than 5 minutes
                result = await session.execute(text("""
                    SELECT 
                        count(*) as long_query_count,
                        max(EXTRACT(EPOCH FROM (now() - query_start))) as longest_query_seconds
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                        AND query_start < now() - interval '5 minutes'
                        AND pid <> pg_backend_pid()
                """))
                
                row = result.fetchone()
                if row:
                    long_count = row.long_query_count or 0
                    longest_seconds = row.longest_query_seconds or 0
                    
                    return {
                        'status': 'measured',
                        'long_query_count': long_count,
                        'longest_query_seconds': longest_seconds,
                        'value': long_count
                    }
                    
        except Exception as e:
            return {
                'status': 'failed', 
                'error': str(e),
                'value': 0
            }
        
        return {'status': 'no_data', 'value': 0}
    
    async def _check_replication_lag(self) -> Dict[str, Any]:
        """Check replication lag for read replicas."""
        if not self.read_replica_engines:
            return {'status': 'no_replicas', 'value': 0}
        
        try:
            # Get master LSN
            async with AsyncSession(self.engine) as session:
                result = await session.execute(text("SELECT pg_current_wal_lsn()"))
                master_lsn = result.scalar()
            
            max_lag_ms = 0
            replica_lags = []
            
            # Check each replica
            for replica_engine in self.read_replica_engines:
                try:
                    async with AsyncSession(replica_engine) as session:
                        result = await session.execute(text("""
                            SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) * 1000
                        """))
                        lag_ms = result.scalar() or 0
                        replica_lags.append(lag_ms)
                        max_lag_ms = max(max_lag_ms, lag_ms)
                        
                except Exception as e:
                    logger.error(f"Failed to check replica lag: {e}")
                    replica_lags.append(999999)  # High lag for failed replica
                    max_lag_ms = 999999
            
            return {
                'status': 'measured',
                'max_lag_ms': max_lag_ms,
                'avg_lag_ms': statistics.mean(replica_lags) if replica_lags else 0,
                'replica_count': len(self.read_replica_engines),
                'value': max_lag_ms
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e), 
                'value': 999999
            }
    
    async def _collect_metrics(self):
        """Collect comprehensive database metrics."""
        try:
            # Calculate overall status
            overall_status = HealthStatus.HEALTHY
            critical_checks = sum(1 for check in self.health_checks.values() 
                                if check.status == HealthStatus.CRITICAL)
            warning_checks = sum(1 for check in self.health_checks.values()
                               if check.status == HealthStatus.WARNING)
            
            if critical_checks > 0:
                overall_status = HealthStatus.CRITICAL
            elif warning_checks > 0:
                overall_status = HealthStatus.WARNING
            
            # Get connection metrics
            connection_result = self.health_checks.get('connection_count', {}).last_result or {}
            
            # Get query performance metrics  
            query_result = self.health_checks.get('query_performance', {}).last_result or {}
            
            # Calculate uptime
            uptime_hours = (datetime.utcnow() - self._start_time).total_seconds() / 3600
            
            # Create metrics object
            metrics = HealthMetrics(
                timestamp=datetime.utcnow(),
                overall_status=overall_status,
                connection_count=connection_result.get('total_connections', 0),
                active_queries=connection_result.get('active_connections', 0), 
                idle_queries=connection_result.get('idle_connections', 0),
                long_running_queries=self.health_checks.get('long_queries', {}).last_result.get('long_query_count', 0) if self.health_checks.get('long_queries', {}).last_result else 0,
                database_size_mb=self.health_checks.get('database_size', {}).last_result.get('size_mb', 0) if self.health_checks.get('database_size', {}).last_result else 0,
                cpu_usage_percent=0,  # Would need system monitoring
                memory_usage_percent=0,  # Would need system monitoring
                disk_usage_percent=self.health_checks.get('database_size', {}).last_result.get('usage_percent', 0) if self.health_checks.get('database_size', {}).last_result else 0,
                cache_hit_ratio=0,  # Would need cache stats
                query_performance_ms=query_result.get('avg_time_ms', 0),
                replication_lag_ms=self.health_checks.get('replication_lag', {}).last_result.get('max_lag_ms', 0) if self.health_checks.get('replication_lag', {}).last_result else 0,
                error_rate_percent=(self._error_count / max(self._total_queries, 1)) * 100,
                uptime_hours=uptime_hours,
                queries_per_second=self._total_queries / max(uptime_hours * 3600, 1),
                avg_response_time_ms=statistics.mean(self._query_times) if self._query_times else 0,
                slowest_query_ms=max(self._query_times) if self._query_times else 0
            )
            
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Keep history size manageable
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
            
            logger.debug(f"Metrics collected: {overall_status.value}")
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    async def _send_alert(self, health_check: HealthCheck, result: Dict[str, Any]):
        """Send alert for health check failure."""
        alert_key = f"{health_check.name}_{health_check.status.value}"
        
        # Check cooldown
        if alert_key in self._alert_cooldown:
            time_since_last = datetime.utcnow() - self._alert_cooldown[alert_key]
            if time_since_last < timedelta(minutes=self._alert_cooldown_minutes):
                return  # Still in cooldown
        
        self._alert_cooldown[alert_key] = datetime.utcnow()
        
        if self.alert_webhook:
            try:
                import aiohttp
                
                alert_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'severity': health_check.status.value,
                    'check_name': health_check.name,
                    'description': health_check.description,
                    'value': result.get('value', 0),
                    'warning_threshold': health_check.warning_threshold,
                    'critical_threshold': health_check.critical_threshold,
                    'consecutive_failures': health_check.consecutive_failures,
                    'result': result
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.alert_webhook,
                        json=alert_data,
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Alert sent for {health_check.name}: {health_check.status.value}")
                        else:
                            logger.warning(f"Alert webhook failed: {response.status}")
                            
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
        
        # Attempt automated recovery if enabled
        if self.recovery_actions_enabled:
            await self._attempt_recovery(health_check, result)
    
    async def _attempt_recovery(self, health_check: HealthCheck, result: Dict[str, Any]):
        """Attempt automated recovery actions."""
        logger.info(f"Attempting recovery for {health_check.name}")
        
        try:
            if health_check.name == "connection" and health_check.status == HealthStatus.CRITICAL:
                # Try to refresh connection pool
                await self._refresh_connection_pool()
                
            elif health_check.name == "long_queries" and result.get('long_query_count', 0) > 10:
                # Could terminate long running queries (be very careful!)
                logger.warning("Many long running queries detected - manual intervention needed")
                
            # Add more recovery actions as needed
            
        except Exception as e:
            logger.error(f"Recovery attempt failed for {health_check.name}: {e}")
    
    async def _refresh_connection_pool(self):
        """Refresh database connection pool."""
        try:
            # Dispose of current pool and create new one
            await self.engine.dispose()
            logger.info("Connection pool refreshed")
            
        except Exception as e:
            logger.error(f"Failed to refresh connection pool: {e}")
    
    def record_query_time(self, execution_time_ms: float):
        """Record query execution time for performance tracking."""
        self._query_times.append(execution_time_ms)
        self._total_queries += 1
        
        # Keep only recent query times
        if len(self._query_times) > 1000:
            self._query_times = self._query_times[-1000:]
    
    def record_query_error(self):
        """Record query error for error rate tracking."""
        self._error_count += 1
        self._total_queries += 1
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_checks_status = {}
        
        for name, check in self.health_checks.items():
            health_checks_status[name] = {
                'status': check.status.value,
                'description': check.description,
                'last_run': check.last_run.isoformat() if check.last_run else None,
                'last_result': check.last_result,
                'consecutive_failures': check.consecutive_failures,
                'enabled': check.enabled
            }
        
        return {
            'overall_status': self.current_metrics.overall_status.value if self.current_metrics else 'unknown',
            'monitoring_active': self._monitoring_active,
            'uptime_hours': (datetime.utcnow() - self._start_time).total_seconds() / 3600,
            'health_checks': health_checks_status,
            'current_metrics': {
                'timestamp': self.current_metrics.timestamp.isoformat(),
                'connection_count': self.current_metrics.connection_count,
                'active_queries': self.current_metrics.active_queries,
                'database_size_mb': self.current_metrics.database_size_mb,
                'query_performance_ms': self.current_metrics.query_performance_ms,
                'error_rate_percent': self.current_metrics.error_rate_percent,
                'replication_lag_ms': self.current_metrics.replication_lag_ms
            } if self.current_metrics else None,
            'performance_summary': {
                'total_queries': self._total_queries,
                'total_errors': self._error_count,
                'avg_query_time_ms': statistics.mean(self._query_times) if self._query_times else 0,
                'queries_per_second': self._total_queries / max((datetime.utcnow() - self._start_time).total_seconds(), 1)
            }
        }


# Global health monitor
_health_monitor: Optional[DatabaseHealthMonitor] = None


def get_health_monitor() -> Optional[DatabaseHealthMonitor]:
    """Get global health monitor."""
    return _health_monitor


async def setup_health_monitor(
    engine: AsyncEngine,
    read_replica_engines: Optional[List[AsyncEngine]] = None,
    alert_webhook: Optional[str] = None,
    recovery_actions_enabled: bool = True
) -> DatabaseHealthMonitor:
    """Set up database health monitoring."""
    global _health_monitor
    
    _health_monitor = DatabaseHealthMonitor(
        engine=engine,
        read_replica_engines=read_replica_engines,
        alert_webhook=alert_webhook,
        recovery_actions_enabled=recovery_actions_enabled
    )
    
    await _health_monitor.start_monitoring()
    
    logger.info("Database health monitoring enabled")
    return _health_monitor