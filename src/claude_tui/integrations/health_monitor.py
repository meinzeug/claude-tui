"""Integration Health Monitor - Real-time monitoring and alerting system.

Provides comprehensive health monitoring for all integration services:
- Real-time health checks
- Performance monitoring
- SLA compliance tracking
- Alert generation
- Automatic recovery attempts
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold: float
    status: HealthStatus
    last_updated: datetime = field(default_factory=datetime.now)
    history: List[float] = field(default_factory=list)


@dataclass
class ServiceHealth:
    """Health status for a service."""
    service_name: str
    status: HealthStatus
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)
    uptime_percentage: float = 100.0
    error_count: int = 0
    recovery_attempts: int = 0
    is_recovering: bool = False


@dataclass
class Alert:
    """System alert."""
    id: str
    service_name: str
    severity: AlertSeverity
    message: str
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    is_resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SLATracker:
    """SLA compliance tracking."""
    
    def __init__(self, target_uptime: float = 99.9, target_response_time: float = 2.0):
        self.target_uptime = target_uptime
        self.target_response_time = target_response_time
        self.uptime_history: List[Tuple[datetime, bool]] = []
        self.response_time_history: List[Tuple[datetime, float]] = []
        
    def record_uptime(self, is_up: bool) -> None:
        """Record uptime status."""
        self.uptime_history.append((datetime.now(), is_up))
        
        # Keep only last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        self.uptime_history = [
            (ts, status) for ts, status in self.uptime_history
            if ts > cutoff
        ]
    
    def record_response_time(self, response_time: float) -> None:
        """Record response time."""
        self.response_time_history.append((datetime.now(), response_time))
        
        # Keep only last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        self.response_time_history = [
            (ts, rt) for ts, rt in self.response_time_history
            if ts > cutoff
        ]
    
    def get_uptime_percentage(self) -> float:
        """Calculate current uptime percentage."""
        if not self.uptime_history:
            return 100.0
        
        total_up = sum(1 for _, is_up in self.uptime_history if is_up)
        return (total_up / len(self.uptime_history)) * 100
    
    def get_avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_time_history:
            return 0.0
        
        return sum(rt for _, rt in self.response_time_history) / len(self.response_time_history)
    
    def is_sla_compliant(self) -> bool:
        """Check if SLA is being met."""
        uptime_ok = self.get_uptime_percentage() >= self.target_uptime
        response_time_ok = self.get_avg_response_time() <= self.target_response_time
        return uptime_ok and response_time_ok


class IntegrationHealthMonitor:
    """Comprehensive health monitoring system for all integrations."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.services: Dict[str, ServiceHealth] = {}
        self.alerts: List[Alert] = []
        self.sla_trackers: Dict[str, SLATracker] = {}
        
        # Health check functions
        self.health_checkers: Dict[str, Callable] = {}
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Monitoring state
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.max_alerts = 1000
        self.alert_retention_hours = 24
        self.recovery_attempt_limit = 3
        
        # Thresholds
        self.thresholds = {
            'response_time': 5.0,  # seconds
            'error_rate': 0.05,    # 5%
            'uptime': 95.0,        # 95%
            'consecutive_failures': 3
        }
        
        logger.info("Integration Health Monitor initialized")
    
    async def start_monitoring(self) -> None:
        """Start the health monitoring system."""
        if self.is_running:
            return
        
        logger.info("Starting integration health monitoring")
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop the health monitoring system."""
        if not self.is_running:
            return
        
        logger.info("Stopping integration health monitoring")
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    def register_service(self, service_name: str, health_checker: Callable) -> None:
        """Register a service for health monitoring."""
        self.services[service_name] = ServiceHealth(
            service_name=service_name,
            status=HealthStatus.HEALTHY
        )
        self.health_checkers[service_name] = health_checker
        self.sla_trackers[service_name] = SLATracker()
        
        logger.info(f"Registered service for monitoring: {service_name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler."""
        self.alert_handlers.append(handler)
    
    async def check_service_health(self, service_name: str) -> ServiceHealth:
        """Check health of a specific service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not registered")
        
        service_health = self.services[service_name]
        health_checker = self.health_checkers[service_name]
        
        try:
            start_time = time.time()
            health_result = await health_checker()
            response_time = time.time() - start_time
            
            # Update metrics
            await self._update_service_metrics(
                service_name, health_result, response_time
            )
            
            # Record SLA data
            sla_tracker = self.sla_trackers[service_name]
            sla_tracker.record_uptime(health_result.get('is_healthy', False))
            sla_tracker.record_response_time(response_time)
            
            # Update service health
            service_health.last_check = datetime.now()
            service_health.uptime_percentage = sla_tracker.get_uptime_percentage()
            
            # Determine overall status
            new_status = self._calculate_service_status(service_name, health_result)
            
            if new_status != service_health.status:
                await self._handle_status_change(service_name, service_health.status, new_status)
                service_health.status = new_status
            
            # Reset recovery flag if healthy
            if new_status == HealthStatus.HEALTHY:
                service_health.is_recovering = False
                service_health.recovery_attempts = 0
            
            return service_health
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            
            # Update error count and status
            service_health.error_count += 1
            service_health.status = HealthStatus.DOWN
            service_health.last_check = datetime.now()
            
            # Record SLA downtime
            self.sla_trackers[service_name].record_uptime(False)
            
            # Generate alert
            await self._create_alert(
                service_name,
                AlertSeverity.CRITICAL,
                f"Health check failed: {e}",
                {'error': str(e), 'error_count': service_health.error_count}
            )
            
            return service_health
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.services:
            return {
                'status': HealthStatus.HEALTHY.value,
                'services': {},
                'alerts': 0,
                'sla_compliance': True
            }
        
        service_statuses = [service.status for service in self.services.values()]
        
        # Determine overall status
        if any(status == HealthStatus.DOWN for status in service_statuses):
            overall_status = HealthStatus.DOWN
        elif any(status == HealthStatus.CRITICAL for status in service_statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.WARNING for status in service_statuses):
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Check SLA compliance
        sla_compliant = all(
            tracker.is_sla_compliant()
            for tracker in self.sla_trackers.values()
        )
        
        active_alerts = sum(1 for alert in self.alerts if not alert.is_resolved)
        
        return {
            'status': overall_status.value,
            'services': {
                name: {
                    'status': service.status.value,
                    'uptime': service.uptime_percentage,
                    'error_count': service.error_count,
                    'last_check': service.last_check.isoformat(),
                    'is_recovering': service.is_recovering
                }
                for name, service in self.services.items()
            },
            'alerts': active_alerts,
            'sla_compliance': sla_compliant,
            'sla_details': {
                name: {
                    'uptime_percentage': tracker.get_uptime_percentage(),
                    'avg_response_time': tracker.get_avg_response_time(),
                    'is_compliant': tracker.is_sla_compliant()
                }
                for name, tracker in self.sla_trackers.items()
            }
        }
    
    async def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get detailed metrics for a service."""
        if service_name not in self.services:
            return {}
        
        service = self.services[service_name]
        sla_tracker = self.sla_trackers[service_name]
        
        return {
            'service_name': service_name,
            'status': service.status.value,
            'uptime_percentage': service.uptime_percentage,
            'error_count': service.error_count,
            'recovery_attempts': service.recovery_attempts,
            'is_recovering': service.is_recovering,
            'last_check': service.last_check.isoformat(),
            'metrics': {
                name: {
                    'value': metric.value,
                    'threshold': metric.threshold,
                    'status': metric.status.value,
                    'last_updated': metric.last_updated.isoformat()
                }
                for name, metric in service.metrics.items()
            },
            'sla': {
                'uptime_percentage': sla_tracker.get_uptime_percentage(),
                'avg_response_time': sla_tracker.get_avg_response_time(),
                'is_compliant': sla_tracker.is_sla_compliant()
            }
        }
    
    async def get_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get system alerts."""
        alerts = self.alerts if include_resolved else [
            alert for alert in self.alerts if not alert.is_resolved
        ]
        
        return [
            {
                'id': alert.id,
                'service_name': alert.service_name,
                'severity': alert.severity.value,
                'message': alert.message,
                'created_at': alert.created_at.isoformat(),
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                'is_resolved': alert.is_resolved,
                'metadata': alert.metadata
            }
            for alert in alerts
        ]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.is_resolved:
                alert.is_resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert {alert_id} resolved")
                return True
        
        return False
    
    async def attempt_recovery(self, service_name: str) -> bool:
        """Attempt to recover a service."""
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        
        if service.recovery_attempts >= self.recovery_attempt_limit:
            logger.warning(f"Max recovery attempts reached for {service_name}")
            return False
        
        service.is_recovering = True
        service.recovery_attempts += 1
        
        logger.info(f"Attempting recovery for {service_name} (attempt {service.recovery_attempts})")
        
        try:
            # Attempt to check health again
            health_result = await self.check_service_health(service_name)
            
            if health_result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]:
                await self._create_alert(
                    service_name,
                    AlertSeverity.INFO,
                    f"Service recovery successful after {service.recovery_attempts} attempts"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Recovery attempt failed for {service_name}: {e}")
            return False
    
    async def export_health_report(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Export comprehensive health report."""
        overall_health = await self.get_overall_health()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'overall_health': overall_health,
            'service_details': {},
            'alerts': await self.get_alerts(include_resolved=True),
            'configuration': {
                'check_interval': self.check_interval,
                'thresholds': self.thresholds,
                'recovery_attempt_limit': self.recovery_attempt_limit
            }
        }
        
        # Add detailed service metrics
        for service_name in self.services:
            report['service_details'][service_name] = await self.get_service_metrics(service_name)
        
        # Save to file if path provided
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Health report exported to {file_path}")
        
        return report
    
    # Private implementation methods
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Health monitoring loop started")
        
        while self.is_running:
            try:
                # Check all registered services
                check_tasks = [
                    self.check_service_health(service_name)
                    for service_name in self.services
                ]
                
                if check_tasks:
                    await asyncio.gather(*check_tasks, return_exceptions=True)
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                # Sleep until next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.check_interval)
        
        logger.info("Health monitoring loop stopped")
    
    async def _update_service_metrics(
        self,
        service_name: str,
        health_result: Dict[str, Any],
        response_time: float
    ) -> None:
        """Update service metrics."""
        service = self.services[service_name]
        
        # Update response time metric
        if 'response_time' not in service.metrics:
            service.metrics['response_time'] = HealthMetric(
                name='response_time',
                value=response_time,
                threshold=self.thresholds['response_time']
            )
        
        metric = service.metrics['response_time']
        metric.value = response_time
        metric.last_updated = datetime.now()
        metric.history.append(response_time)
        
        # Keep only last 100 values
        if len(metric.history) > 100:
            metric.history = metric.history[-100:]
        
        # Update status
        if response_time > metric.threshold:
            metric.status = HealthStatus.WARNING
        else:
            metric.status = HealthStatus.HEALTHY
        
        # Add other metrics from health result
        for key, value in health_result.items():
            if isinstance(value, (int, float)) and key != 'is_healthy':
                if key not in service.metrics:
                    service.metrics[key] = HealthMetric(
                        name=key,
                        value=value,
                        threshold=self.thresholds.get(key, 0)
                    )
                else:
                    service.metrics[key].value = value
                    service.metrics[key].last_updated = datetime.now()
    
    def _calculate_service_status(
        self,
        service_name: str,
        health_result: Dict[str, Any]
    ) -> HealthStatus:
        """Calculate overall service status."""
        service = self.services[service_name]
        
        # Check if service reports itself as healthy
        if not health_result.get('is_healthy', False):
            return HealthStatus.DOWN
        
        # Check individual metrics
        warning_count = 0
        critical_count = 0
        
        for metric in service.metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                critical_count += 1
            elif metric.status == HealthStatus.WARNING:
                warning_count += 1
        
        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif warning_count > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def _handle_status_change(
        self,
        service_name: str,
        old_status: HealthStatus,
        new_status: HealthStatus
    ) -> None:
        """Handle service status changes."""
        logger.info(f"Service {service_name} status changed: {old_status.value} -> {new_status.value}")
        
        # Generate appropriate alert
        if new_status == HealthStatus.DOWN:
            severity = AlertSeverity.CRITICAL
            message = f"Service {service_name} is DOWN"
        elif new_status == HealthStatus.CRITICAL:
            severity = AlertSeverity.ERROR
            message = f"Service {service_name} is in CRITICAL state"
        elif new_status == HealthStatus.WARNING:
            severity = AlertSeverity.WARNING
            message = f"Service {service_name} performance degraded"
        elif old_status != HealthStatus.HEALTHY and new_status == HealthStatus.HEALTHY:
            severity = AlertSeverity.INFO
            message = f"Service {service_name} recovered and is HEALTHY"
        else:
            return  # No alert needed
        
        await self._create_alert(
            service_name,
            severity,
            message,
            {'old_status': old_status.value, 'new_status': new_status.value}
        )
        
        # Attempt recovery for critical services
        if new_status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
            asyncio.create_task(self.attempt_recovery(service_name))
    
    async def _create_alert(
        self,
        service_name: str,
        severity: AlertSeverity,
        message: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Create and handle an alert."""
        import uuid
        
        alert = Alert(
            id=str(uuid.uuid4())[:8],
            service_name=service_name,
            severity=severity,
            message=message,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Ensure we don't exceed max alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        logger.warning(f"Alert created: {alert.severity.value} - {alert.message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        cutoff = datetime.now() - timedelta(hours=self.alert_retention_hours)
        
        old_alert_count = len(self.alerts)
        self.alerts = [
            alert for alert in self.alerts
            if not alert.is_resolved or alert.resolved_at > cutoff
        ]
        
        removed_count = old_alert_count - len(self.alerts)
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} old alerts")