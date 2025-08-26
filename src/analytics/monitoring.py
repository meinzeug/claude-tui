"""
Real-time Monitoring and Alerting System.

Advanced monitoring system that provides real-time performance tracking,
intelligent alerting, and automated response capabilities.
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from ..core.types import Severity, Priority
from .models import (
    PerformanceMetrics, PerformanceAlert, BottleneckAnalysis,
    OptimizationRecommendation, AnalyticsConfiguration,
    AlertType
)


class MonitoringMode(str, Enum):
    """Monitoring operation modes."""
    PASSIVE = "passive"  # Monitor only, no automated actions
    ACTIVE = "active"    # Monitor with automated responses
    LEARNING = "learning"  # Monitor and learn patterns


class AlertChannel(str, Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"
    FILE = "file"


class EscalationLevel(str, Enum):
    """Alert escalation levels."""
    LEVEL_1 = "level_1"  # Team notification
    LEVEL_2 = "level_2"  # Manager notification  
    LEVEL_3 = "level_3"  # Executive notification
    LEVEL_4 = "level_4"  # External escalation


@dataclass
class AlertRule:
    """Configuration for alert rules."""
    rule_id: str
    name: str
    description: str
    
    # Conditions
    metric_name: str
    operator: str  # >, <, ==, !=, >=, <=
    threshold: float
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    
    # Alert properties
    severity: Severity = Severity.MEDIUM
    priority: Priority = Priority.MEDIUM
    alert_type: AlertType = AlertType.THRESHOLD_EXCEEDED
    
    # Notification settings
    channels: List[AlertChannel] = field(default_factory=list)
    escalation_rules: Dict[EscalationLevel, Dict[str, Any]] = field(default_factory=dict)
    
    # Suppression settings
    suppress_duration: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    max_alerts_per_hour: int = 10
    
    # Auto-resolution
    auto_resolve: bool = True
    resolve_threshold: Optional[float] = None
    
    # Conditions for rule activation
    enabled: bool = True
    environments: Set[str] = field(default_factory=set)
    time_ranges: List[Dict[str, Any]] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None


@dataclass
class MonitoringTarget:
    """Target system or component to monitor."""
    target_id: str
    name: str
    target_type: str  # system, application, service, etc.
    
    # Monitoring configuration
    enabled: bool = True
    monitoring_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    
    # Target-specific settings
    endpoints: List[str] = field(default_factory=list)
    credentials: Dict[str, str] = field(default_factory=dict)
    custom_metrics: List[str] = field(default_factory=list)
    
    # Health check configuration
    health_check_enabled: bool = True
    health_check_url: Optional[str] = None
    health_check_timeout: int = 30
    
    # Alert rules specific to this target
    alert_rules: List[str] = field(default_factory=list)  # Rule IDs
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlertNotification:
    """Alert notification record."""
    notification_id: str
    alert_id: str
    channel: AlertChannel
    recipient: str
    
    # Notification content
    subject: str
    message: str
    attachment_data: Optional[Dict[str, Any]] = None
    
    # Status tracking
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    status: str = "pending"  # pending, sent, delivered, failed
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    created_at: datetime = field(default_factory=datetime.utcnow)


class RealTimeMonitor:
    """
    Real-time performance monitoring system.
    
    Features:
    - Configurable alert rules and thresholds
    - Multiple notification channels
    - Alert escalation and suppression
    - Automated response capabilities
    - Health check monitoring
    - Performance baseline tracking
    - Anomaly detection integration
    """
    
    def __init__(
        self,
        config: Optional[AnalyticsConfiguration] = None,
        mode: MonitoringMode = MonitoringMode.ACTIVE
    ):
        """Initialize the real-time monitor."""
        self.config = config or AnalyticsConfiguration()
        self.mode = mode
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Alert rules and targets
        self.alert_rules: Dict[str, AlertRule] = {}
        self.monitoring_targets: Dict[str, MonitoringTarget] = {}
        
        # Active monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: Set[asyncio.Task] = set()
        self.alert_processor_task: Optional[asyncio.Task] = None
        
        # Alert management
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_suppressions: Dict[str, datetime] = {}
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.pending_notifications: Dict[str, AlertNotification] = {}
        
        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.baseline_learning_period = timedelta(days=7)
        
        # Escalation tracking
        self.escalation_timers: Dict[str, asyncio.Task] = {}
        
        # Notification channels configuration
        self.notification_handlers: Dict[AlertChannel, Callable] = {}
        self._initialize_notification_handlers()
        
        # Statistics
        self.monitoring_stats = {
            'alerts_generated': 0,
            'alerts_resolved': 0,
            'notifications_sent': 0,
            'false_positives': 0,
            'uptime': 0,
            'last_health_check': None
        }
    
    async def initialize(self) -> None:
        """Initialize the monitoring system."""
        try:
            self.logger.info("Initializing Real-time Monitor...")
            
            # Load default alert rules
            await self._load_default_alert_rules()
            
            # Load monitoring targets
            await self._load_monitoring_targets()
            
            # Initialize notification system
            await self._initialize_notification_system()
            
            # Load baselines
            await self._load_performance_baselines()
            
            self.logger.info(f"Real-time Monitor initialized with {len(self.alert_rules)} rules")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitor: {e}")
            raise
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already active")
            return
        
        self.logger.info("Starting real-time monitoring...")
        self.is_monitoring = True
        
        # Start alert processor
        self.alert_processor_task = asyncio.create_task(self._alert_processing_loop())
        
        # Start notification processor
        notification_task = asyncio.create_task(self._notification_processing_loop())
        self.monitoring_tasks.add(notification_task)
        
        # Start escalation processor
        escalation_task = asyncio.create_task(self._escalation_processing_loop())
        self.monitoring_tasks.add(escalation_task)
        
        # Start health check monitoring
        health_check_task = asyncio.create_task(self._health_check_loop())
        self.monitoring_tasks.add(health_check_task)
        
        # Clean up completed tasks
        for task in self.monitoring_tasks.copy():
            task.add_done_callback(self.monitoring_tasks.discard)
    
    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping real-time monitoring...")
        self.is_monitoring = False
        
        # Cancel alert processor
        if self.alert_processor_task:
            self.alert_processor_task.cancel()
            try:
                await self.alert_processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.copy():
            task.cancel()
        
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        # Cancel escalation timers
        for timer in self.escalation_timers.values():
            timer.cancel()
        
        self.escalation_timers.clear()
    
    async def process_metrics(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """Process new performance metrics and check for alerts."""
        try:
            alerts_generated = []
            
            # Update baselines if in learning mode
            if self.mode == MonitoringMode.LEARNING:
                await self._update_baselines(metrics)
            
            # Evaluate each alert rule
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check environment constraints
                if rule.environments and metrics.environment not in rule.environments:
                    continue
                
                # Check time range constraints
                if not self._is_in_time_range(rule):
                    continue
                
                # Evaluate rule condition
                if await self._evaluate_alert_rule(rule, metrics):
                    # Check if alert should be suppressed
                    if self._is_alert_suppressed(rule_id):
                        continue
                    
                    # Generate alert
                    alert = await self._generate_alert(rule, metrics)
                    alerts_generated.append(alert)
                    
                    # Store active alert
                    self.active_alerts[str(alert.id)] = alert
                    
                    # Update rule trigger time
                    rule.last_triggered = datetime.utcnow()
                    
                    # Schedule notifications
                    await self._schedule_alert_notifications(alert, rule)
                    
                    # Schedule escalation if configured
                    await self._schedule_escalation(alert, rule)
                    
                    # Update statistics
                    self.monitoring_stats['alerts_generated'] += 1
            
            # Check for auto-resolution of existing alerts
            await self._check_alert_auto_resolution(metrics)
            
            return alerts_generated
            
        except Exception as e:
            self.logger.error(f"Error processing metrics for monitoring: {e}")
            return []
    
    async def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    async def update_alert_rule(self, rule: AlertRule) -> bool:
        """Update an existing alert rule."""
        if rule.rule_id in self.alert_rules:
            self.alert_rules[rule.rule_id] = rule
            self.logger.info(f"Updated alert rule: {rule.name}")
            return True
        return False
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            
            # Cancel escalation timer if exists
            if alert_id in self.escalation_timers:
                self.escalation_timers[alert_id].cancel()
                del self.escalation_timers[alert_id]
            
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, notes: str = "") -> bool:
        """Manually resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            alert.resolution_notes = notes
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            # Cancel escalation timer
            if alert_id in self.escalation_timers:
                self.escalation_timers[alert_id].cancel()
                del self.escalation_timers[alert_id]
            
            self.monitoring_stats['alerts_resolved'] += 1
            self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
        return False
    
    async def suppress_alert(self, rule_id: str, duration: timedelta) -> None:
        """Suppress alerts for a specific rule."""
        suppress_until = datetime.utcnow() + duration
        self.alert_suppressions[rule_id] = suppress_until
        self.logger.info(f"Alert rule {rule_id} suppressed until {suppress_until}")
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            'is_monitoring': self.is_monitoring,
            'mode': self.mode.value,
            'active_alerts': len(self.active_alerts),
            'alert_rules': len(self.alert_rules),
            'monitoring_targets': len(self.monitoring_targets),
            'suppressed_rules': len(self.alert_suppressions),
            'pending_notifications': len(self.pending_notifications),
            'statistics': self.monitoring_stats.copy(),
            'uptime': (datetime.utcnow() - datetime.utcnow()).total_seconds() if self.is_monitoring else 0
        }
    
    # Private methods
    
    async def _alert_processing_loop(self) -> None:
        """Main alert processing loop."""
        while self.is_monitoring:
            try:
                # Process any pending alert logic
                await self._cleanup_expired_suppressions()
                await self._update_monitoring_statistics()
                
                # Wait before next iteration
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _notification_processing_loop(self) -> None:
        """Process notification queue."""
        while self.is_monitoring:
            try:
                # Get notification from queue (with timeout)
                try:
                    notification = await asyncio.wait_for(
                        self.notification_queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process notification
                await self._process_notification(notification)
                
                # Mark task as done
                self.notification_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in notification processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _escalation_processing_loop(self) -> None:
        """Process alert escalations."""
        while self.is_monitoring:
            try:
                # Check for escalations that need to be triggered
                current_time = datetime.utcnow()
                
                for alert_id, alert in self.active_alerts.items():
                    if (alert.acknowledged_at is None and 
                        alert_id not in self.escalation_timers):
                        
                        # Check if alert is old enough for escalation
                        alert_age = current_time - alert.created_at
                        if alert_age > timedelta(minutes=15):  # Escalate after 15 minutes
                            await self._trigger_escalation(alert)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in escalation processing loop: {e}")
                await asyncio.sleep(30)
    
    async def _health_check_loop(self) -> None:
        """Perform health checks on monitoring targets."""
        while self.is_monitoring:
            try:
                for target_id, target in self.monitoring_targets.items():
                    if target.enabled and target.health_check_enabled:
                        await self._perform_health_check(target)
                
                self.monitoring_stats['last_health_check'] = datetime.utcnow()
                
                # Wait for next health check cycle
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_alert_rule(self, rule: AlertRule, metrics: PerformanceMetrics) -> bool:
        """Evaluate if an alert rule condition is met."""
        try:
            # Get metric value
            if not hasattr(metrics, rule.metric_name):
                return False
            
            current_value = getattr(metrics, rule.metric_name)
            if current_value is None:
                return False
            
            # Evaluate condition
            threshold = rule.threshold
            
            if rule.operator == '>':
                return current_value > threshold
            elif rule.operator == '<':
                return current_value < threshold
            elif rule.operator == '>=':
                return current_value >= threshold
            elif rule.operator == '<=':
                return current_value <= threshold
            elif rule.operator == '==':
                return abs(current_value - threshold) < 0.001
            elif rule.operator == '!=':
                return abs(current_value - threshold) >= 0.001
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating alert rule {rule.rule_id}: {e}")
            return False
    
    async def _generate_alert(self, rule: AlertRule, metrics: PerformanceMetrics) -> PerformanceAlert:
        """Generate a performance alert from a rule."""
        current_value = getattr(metrics, rule.metric_name)
        
        alert = PerformanceAlert(
            alert_type=rule.alert_type,
            severity=rule.severity,
            priority=rule.priority,
            title=f"{rule.name} Alert",
            description=f"{rule.description}. Current value: {current_value}, Threshold: {rule.threshold}",
            affected_component=rule.metric_name,
            metric_name=rule.metric_name,
            threshold_value=rule.threshold,
            current_value=current_value,
            performance_metrics=metrics,
            environment=metrics.environment
        )
        
        return alert
    
    async def _schedule_alert_notifications(self, alert: PerformanceAlert, rule: AlertRule) -> None:
        """Schedule notifications for an alert."""
        for channel in rule.channels:
            notification = AlertNotification(
                notification_id=f"{alert.id}_{channel.value}_{datetime.utcnow().timestamp()}",
                alert_id=str(alert.id),
                channel=channel,
                recipient=self._get_channel_recipient(channel),
                subject=f"Performance Alert: {alert.title}",
                message=self._format_alert_message(alert)
            )
            
            # Add to queue
            await self.notification_queue.put(notification)
            self.pending_notifications[notification.notification_id] = notification
    
    async def _process_notification(self, notification: AlertNotification) -> None:
        """Process a single notification."""
        try:
            handler = self.notification_handlers.get(notification.channel)
            if not handler:
                notification.status = "failed"
                notification.error_message = f"No handler for channel {notification.channel}"
                return
            
            # Send notification
            success = await handler(notification)
            
            if success:
                notification.status = "sent"
                notification.sent_at = datetime.utcnow()
                self.monitoring_stats['notifications_sent'] += 1
            else:
                notification.status = "failed"
                notification.retry_count += 1
                
                # Retry if under limit
                if notification.retry_count < notification.max_retries:
                    # Re-queue with delay
                    await asyncio.sleep(30)  # Wait 30 seconds before retry
                    await self.notification_queue.put(notification)
                    return
            
            # Remove from pending
            if notification.notification_id in self.pending_notifications:
                del self.pending_notifications[notification.notification_id]
                
        except Exception as e:
            notification.status = "failed"
            notification.error_message = str(e)
            self.logger.error(f"Error processing notification {notification.notification_id}: {e}")
    
    def _is_alert_suppressed(self, rule_id: str) -> bool:
        """Check if alerts for a rule are currently suppressed."""
        if rule_id in self.alert_suppressions:
            suppress_until = self.alert_suppressions[rule_id]
            if datetime.utcnow() < suppress_until:
                return True
            else:
                del self.alert_suppressions[rule_id]
        return False
    
    def _is_in_time_range(self, rule: AlertRule) -> bool:
        """Check if current time is within rule's active time ranges."""
        if not rule.time_ranges:
            return True
        
        current_time = datetime.utcnow().time()
        current_day = datetime.utcnow().weekday()
        
        for time_range in rule.time_ranges:
            start_time = time_range.get('start_time')
            end_time = time_range.get('end_time')
            days = time_range.get('days', [])
            
            if days and current_day not in days:
                continue
            
            if start_time and end_time:
                if start_time <= current_time <= end_time:
                    return True
        
        return len(rule.time_ranges) == 0
    
    def _initialize_notification_handlers(self) -> None:
        """Initialize notification channel handlers."""
        self.notification_handlers = {
            AlertChannel.CONSOLE: self._send_console_notification,
            AlertChannel.EMAIL: self._send_email_notification,
            AlertChannel.FILE: self._send_file_notification,
            AlertChannel.WEBHOOK: self._send_webhook_notification
        }
    
    async def _send_console_notification(self, notification: AlertNotification) -> bool:
        """Send notification to console/log."""
        try:
            self.logger.warning(f"ALERT: {notification.subject} - {notification.message}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending console notification: {e}")
            return False
    
    async def _send_email_notification(self, notification: AlertNotification) -> bool:
        """Send email notification (placeholder)."""
        try:
            # This would integrate with actual email service
            self.logger.info(f"EMAIL ALERT to {notification.recipient}: {notification.subject}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")
            return False
    
    async def _send_file_notification(self, notification: AlertNotification) -> bool:
        """Write notification to file."""
        try:
            alert_file = Path("alerts.log")
            with open(alert_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.utcnow().isoformat()} - {notification.subject}: {notification.message}\n")
            return True
        except Exception as e:
            self.logger.error(f"Error writing alert to file: {e}")
            return False
    
    async def _send_webhook_notification(self, notification: AlertNotification) -> bool:
        """Send webhook notification (placeholder)."""
        try:
            # This would make HTTP POST to webhook URL
            self.logger.info(f"WEBHOOK ALERT: {notification.subject}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending webhook notification: {e}")
            return False
    
    def _get_channel_recipient(self, channel: AlertChannel) -> str:
        """Get default recipient for a channel."""
        recipients = {
            AlertChannel.EMAIL: "admin@example.com",
            AlertChannel.SLACK: "#alerts",
            AlertChannel.WEBHOOK: "http://webhook.example.com",
            AlertChannel.SMS: "+1234567890",
            AlertChannel.CONSOLE: "console",
            AlertChannel.FILE: "alerts.log"
        }
        return recipients.get(channel, "unknown")
    
    def _format_alert_message(self, alert: PerformanceAlert) -> str:
        """Format alert message for notifications."""
        return f"""
Performance Alert Details:
- Alert: {alert.title}
- Description: {alert.description}
- Severity: {alert.severity.value}
- Component: {alert.affected_component}
- Metric: {alert.metric_name}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold_value}
- Time: {alert.created_at.isoformat()}
- Environment: {alert.environment}
        """.strip()
    
    async def _load_default_alert_rules(self) -> None:
        """Load default alert rules."""
        # CPU usage alert
        cpu_rule = AlertRule(
            rule_id="cpu_high_usage",
            name="High CPU Usage",
            description="CPU usage exceeds 80%",
            metric_name="cpu_percent",
            operator=">",
            threshold=80.0,
            duration=timedelta(minutes=5),
            severity=Severity.HIGH,
            channels=[AlertChannel.CONSOLE, AlertChannel.FILE]
        )
        await self.add_alert_rule(cpu_rule)
        
        # Memory usage alert
        memory_rule = AlertRule(
            rule_id="memory_high_usage",
            name="High Memory Usage",
            description="Memory usage exceeds 85%",
            metric_name="memory_percent",
            operator=">",
            threshold=85.0,
            duration=timedelta(minutes=3),
            severity=Severity.HIGH,
            channels=[AlertChannel.CONSOLE, AlertChannel.FILE]
        )
        await self.add_alert_rule(memory_rule)
        
        # Error rate alert
        error_rule = AlertRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            description="Error rate exceeds 5%",
            metric_name="error_rate",
            operator=">",
            threshold=0.05,
            duration=timedelta(minutes=2),
            severity=Severity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.FILE]
        )
        await self.add_alert_rule(error_rule)
    
    async def _load_monitoring_targets(self) -> None:
        """Load monitoring targets."""
        # Default system target
        system_target = MonitoringTarget(
            target_id="system",
            name="System Performance",
            target_type="system",
            health_check_enabled=True
        )
        self.monitoring_targets[system_target.target_id] = system_target
    
    async def _initialize_notification_system(self) -> None:
        """Initialize notification system."""
        # Placeholder for notification system initialization
        pass
    
    async def _load_performance_baselines(self) -> None:
        """Load performance baselines."""
        # Placeholder for baseline loading
        pass
    
    async def _update_baselines(self, metrics: PerformanceMetrics) -> None:
        """Update performance baselines in learning mode."""
        # Placeholder for baseline updating
        pass
    
    async def _check_alert_auto_resolution(self, metrics: PerformanceMetrics) -> None:
        """Check for automatic alert resolution."""
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            rule = self.alert_rules.get(f"{alert.metric_name}_rule")
            if rule and rule.auto_resolve and rule.resolve_threshold:
                current_value = getattr(metrics, alert.metric_name, None)
                
                if current_value is not None:
                    # Check if value is back within acceptable range
                    if ((rule.operator in ['>', '>='] and current_value <= rule.resolve_threshold) or
                        (rule.operator in ['<', '<='] and current_value >= rule.resolve_threshold)):
                        alerts_to_resolve.append(alert_id)
        
        # Auto-resolve alerts
        for alert_id in alerts_to_resolve:
            await self.resolve_alert(alert_id, "auto-resolution", "Metric returned to normal range")
    
    async def _cleanup_expired_suppressions(self) -> None:
        """Clean up expired alert suppressions."""
        current_time = datetime.utcnow()
        expired_suppressions = [
            rule_id for rule_id, suppress_until in self.alert_suppressions.items()
            if current_time >= suppress_until
        ]
        
        for rule_id in expired_suppressions:
            del self.alert_suppressions[rule_id]
    
    async def _update_monitoring_statistics(self) -> None:
        """Update monitoring statistics."""
        if self.is_monitoring:
            self.monitoring_stats['uptime'] += 10  # Add 10 seconds
    
    async def _perform_health_check(self, target: MonitoringTarget) -> None:
        """Perform health check on a monitoring target."""
        try:
            # Placeholder for actual health check implementation
            self.logger.debug(f"Health check for target: {target.name}")
        except Exception as e:
            self.logger.error(f"Health check failed for {target.name}: {e}")
    
    async def _schedule_escalation(self, alert: PerformanceAlert, rule: AlertRule) -> None:
        """Schedule alert escalation."""
        if rule.escalation_rules:
            escalation_task = asyncio.create_task(
                self._escalation_timer(str(alert.id), rule.escalation_rules)
            )
            self.escalation_timers[str(alert.id)] = escalation_task
    
    async def _escalation_timer(self, alert_id: str, escalation_rules: Dict[EscalationLevel, Dict[str, Any]]) -> None:
        """Timer for alert escalation."""
        try:
            # Wait for escalation delay
            await asyncio.sleep(900)  # 15 minutes default
            
            # Check if alert is still active and not acknowledged
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                if alert.acknowledged_at is None:
                    await self._trigger_escalation(alert)
        except asyncio.CancelledError:
            pass
    
    async def _trigger_escalation(self, alert: PerformanceAlert) -> None:
        """Trigger alert escalation."""
        alert.escalation_level += 1
        alert.escalated_at = datetime.utcnow()
        
        self.logger.warning(f"Alert {alert.id} escalated to level {alert.escalation_level}")
        
        # Send escalation notifications (placeholder)
        # This would send notifications to higher-level recipients