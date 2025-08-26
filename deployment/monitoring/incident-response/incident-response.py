"""
Automated Incident Response System for Hive Mind Collective
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import asyncpg
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"

class ActionType(str, Enum):
    NOTIFICATION = "notification"
    SCALING = "scaling"
    RESTART = "restart"
    ROLLBACK = "rollback"
    CIRCUIT_BREAKER = "circuit_breaker"
    CUSTOM = "custom"

@dataclass
class Alert:
    alert_name: str
    severity: Severity
    message: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    starts_at: datetime
    ends_at: Optional[datetime] = None
    fingerprint: str = ""

@dataclass
class Incident:
    id: str
    title: str
    description: str
    severity: Severity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    alerts: List[Alert] = None
    actions_taken: List[str] = None
    affected_services: List[str] = None

class NotificationChannel:
    """Base class for notification channels"""
    
    async def send(self, incident: Incident, message: str) -> bool:
        raise NotImplementedError

class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str, channel: str = "#incidents"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send(self, incident: Incident, message: str) -> bool:
        try:
            payload = {
                "channel": self.channel,
                "username": "Incident Bot",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": self._get_color(incident.severity),
                        "title": f"Incident: {incident.title}",
                        "text": message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": incident.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": incident.status.upper(),
                                "short": True
                            },
                            {
                                "title": "Created",
                                "value": incident.created_at.isoformat(),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _get_color(self, severity: Severity) -> str:
        colors = {
            Severity.LOW: "good",
            Severity.MEDIUM: "warning",
            Severity.HIGH: "danger",
            Severity.CRITICAL: "#ff0000"
        }
        return colors.get(severity, "warning")

class PagerDutyNotificationChannel(NotificationChannel):
    """PagerDuty notification channel"""
    
    def __init__(self, routing_key: str):
        self.routing_key = routing_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
    
    async def send(self, incident: Incident, message: str) -> bool:
        try:
            payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger",
                "dedup_key": incident.id,
                "payload": {
                    "summary": f"{incident.title} - {incident.severity.upper()}",
                    "source": "hive-mind-monitoring",
                    "severity": self._map_severity(incident.severity),
                    "custom_details": {
                        "description": incident.description,
                        "affected_services": incident.affected_services or [],
                        "created_at": incident.created_at.isoformat()
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as response:
                    return response.status == 202
                    
        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")
            return False
    
    def _map_severity(self, severity: Severity) -> str:
        mapping = {
            Severity.LOW: "info",
            Severity.MEDIUM: "warning",
            Severity.HIGH: "error",
            Severity.CRITICAL: "critical"
        }
        return mapping.get(severity, "warning")

class AutoScaler:
    """Automatic scaling actions"""
    
    def __init__(self, kubernetes_config: Dict[str, Any]):
        self.k8s_config = kubernetes_config
    
    async def scale_deployment(self, deployment: str, namespace: str, replicas: int) -> bool:
        """Scale a Kubernetes deployment"""
        try:
            # In real implementation, use kubernetes client
            logger.info(f"Scaling {deployment} in {namespace} to {replicas} replicas")
            
            # Simulate scaling action
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False
    
    async def scale_based_on_metrics(self, service: str, cpu_threshold: float = 80.0) -> bool:
        """Auto-scale based on CPU metrics"""
        try:
            # Query Prometheus for current CPU usage
            current_cpu = await self._get_cpu_usage(service)
            
            if current_cpu > cpu_threshold:
                # Scale up
                current_replicas = await self._get_current_replicas(service)
                new_replicas = min(current_replicas * 2, 10)  # Max 10 replicas
                
                return await self.scale_deployment(service, "default", new_replicas)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to auto-scale {service}: {e}")
            return False
    
    async def _get_cpu_usage(self, service: str) -> float:
        """Get current CPU usage from Prometheus"""
        # Simulate Prometheus query
        return 85.0
    
    async def _get_current_replicas(self, service: str) -> int:
        """Get current replica count"""
        # Simulate Kubernetes API call
        return 2

class CircuitBreaker:
    """Circuit breaker for failing services"""
    
    def __init__(self):
        self.breakers: Dict[str, Dict[str, Any]] = {}
    
    async def trip_breaker(self, service: str, duration_minutes: int = 5) -> bool:
        """Trip circuit breaker for a service"""
        try:
            self.breakers[service] = {
                "status": "open",
                "tripped_at": datetime.utcnow(),
                "duration": duration_minutes
            }
            
            logger.info(f"Circuit breaker tripped for {service} for {duration_minutes} minutes")
            
            # In real implementation, update service mesh or load balancer
            await self._update_routing_rules(service, enabled=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to trip circuit breaker for {service}: {e}")
            return False
    
    async def reset_breaker(self, service: str) -> bool:
        """Reset circuit breaker for a service"""
        try:
            if service in self.breakers:
                del self.breakers[service]
            
            logger.info(f"Circuit breaker reset for {service}")
            
            # Re-enable service
            await self._update_routing_rules(service, enabled=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset circuit breaker for {service}: {e}")
            return False
    
    async def _update_routing_rules(self, service: str, enabled: bool):
        """Update routing rules in service mesh"""
        # Simulate updating Istio/Envoy routing rules
        logger.info(f"Updated routing rules for {service}: enabled={enabled}")

class IncidentResponseSystem:
    """Main incident response orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.notification_channels: List[NotificationChannel] = []
        self.auto_scaler = AutoScaler(config.get("kubernetes", {}))
        self.circuit_breaker = CircuitBreaker()
        self.active_incidents: Dict[str, Incident] = {}
        
        # Initialize notification channels
        self._setup_notification_channels()
        
        # Response playbooks
        self.playbooks = self._load_playbooks()
    
    def _setup_notification_channels(self):
        """Setup notification channels based on config"""
        if slack_config := self.config.get("slack"):
            self.notification_channels.append(
                SlackNotificationChannel(
                    webhook_url=slack_config["webhook_url"],
                    channel=slack_config.get("channel", "#incidents")
                )
            )
        
        if pagerduty_config := self.config.get("pagerduty"):
            self.notification_channels.append(
                PagerDutyNotificationChannel(
                    routing_key=pagerduty_config["routing_key"]
                )
            )
    
    def _load_playbooks(self) -> Dict[str, Dict[str, Any]]:
        """Load incident response playbooks"""
        return {
            "high_error_rate": {
                "name": "High Error Rate Response",
                "triggers": ["ClaudeTUIHighErrorRate"],
                "actions": [
                    {"type": ActionType.NOTIFICATION, "channels": ["slack", "pagerduty"]},
                    {"type": ActionType.SCALING, "service": "claude-tui", "factor": 2},
                    {"type": ActionType.CIRCUIT_BREAKER, "service": "external-api", "duration": 5}
                ]
            },
            "service_down": {
                "name": "Service Down Response",
                "triggers": ["ServiceDown", "PostgreSQLDown", "HiveMindCoordinatorDown"],
                "actions": [
                    {"type": ActionType.NOTIFICATION, "channels": ["slack", "pagerduty"], "severity": Severity.CRITICAL},
                    {"type": ActionType.RESTART, "service": "auto-detect"},
                    {"type": ActionType.ROLLBACK, "condition": "if_recent_deployment"}
                ]
            },
            "high_latency": {
                "name": "High Latency Response",
                "triggers": ["ClaudeTUIHighLatency"],
                "actions": [
                    {"type": ActionType.NOTIFICATION, "channels": ["slack"]},
                    {"type": ActionType.SCALING, "service": "claude-tui", "factor": 1.5},
                    {"type": ActionType.CUSTOM, "function": "optimize_database_queries"}
                ]
            },
            "resource_exhaustion": {
                "name": "Resource Exhaustion Response",
                "triggers": ["HighMemoryUsage", "HighCPUUsage", "HighDiskUsage"],
                "actions": [
                    {"type": ActionType.NOTIFICATION, "channels": ["slack"]},
                    {"type": ActionType.SCALING, "service": "auto-detect", "factor": 2},
                    {"type": ActionType.CUSTOM, "function": "cleanup_resources"}
                ]
            },
            "security_incident": {
                "name": "Security Incident Response",
                "triggers": ["SecurityAuditFailures", "UnauthorizedAccessAttempts"],
                "actions": [
                    {"type": ActionType.NOTIFICATION, "channels": ["slack", "pagerduty"], "severity": Severity.CRITICAL},
                    {"type": ActionType.CIRCUIT_BREAKER, "service": "api-gateway", "duration": 10},
                    {"type": ActionType.CUSTOM, "function": "enable_rate_limiting"}
                ]
            }
        }
    
    async def handle_webhook(self, alert_data: Dict[str, Any]):
        """Handle incoming webhook from AlertManager"""
        try:
            alerts = []
            for alert_info in alert_data.get("alerts", []):
                alert = Alert(
                    alert_name=alert_info["labels"]["alertname"],
                    severity=self._map_alert_severity(alert_info["labels"].get("severity", "medium")),
                    message=alert_info["annotations"].get("summary", ""),
                    labels=alert_info["labels"],
                    annotations=alert_info["annotations"],
                    starts_at=datetime.fromisoformat(alert_info["startsAt"].replace("Z", "+00:00")),
                    ends_at=datetime.fromisoformat(alert_info["endsAt"].replace("Z", "+00:00")) if alert_info.get("endsAt") else None,
                    fingerprint=alert_info.get("fingerprint", "")
                )
                alerts.append(alert)
            
            # Process each alert
            for alert in alerts:
                await self._process_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to handle webhook: {e}")
    
    def _map_alert_severity(self, severity_str: str) -> Severity:
        """Map alert severity string to enum"""
        mapping = {
            "low": Severity.LOW,
            "warning": Severity.MEDIUM,
            "high": Severity.HIGH,
            "critical": Severity.CRITICAL
        }
        return mapping.get(severity_str.lower(), Severity.MEDIUM)
    
    async def _process_alert(self, alert: Alert):
        """Process individual alert and trigger appropriate responses"""
        try:
            # Check if this alert matches any playbooks
            triggered_playbooks = []
            for playbook_name, playbook in self.playbooks.items():
                if alert.alert_name in playbook["triggers"]:
                    triggered_playbooks.append((playbook_name, playbook))
            
            if not triggered_playbooks:
                logger.info(f"No playbooks found for alert: {alert.alert_name}")
                return
            
            # Create or update incident
            incident_id = f"incident-{alert.fingerprint}-{int(alert.starts_at.timestamp())}"
            
            if incident_id not in self.active_incidents:
                incident = Incident(
                    id=incident_id,
                    title=f"Alert: {alert.alert_name}",
                    description=alert.message,
                    severity=alert.severity,
                    status=IncidentStatus.OPEN,
                    created_at=alert.starts_at,
                    updated_at=datetime.utcnow(),
                    alerts=[alert],
                    actions_taken=[],
                    affected_services=self._extract_affected_services(alert)
                )
                self.active_incidents[incident_id] = incident
            else:
                incident = self.active_incidents[incident_id]
                incident.alerts.append(alert)
                incident.updated_at = datetime.utcnow()
            
            # Execute playbook actions
            for playbook_name, playbook in triggered_playbooks:
                logger.info(f"Executing playbook: {playbook_name}")
                await self._execute_playbook(incident, playbook)
            
        except Exception as e:
            logger.error(f"Failed to process alert: {e}")
    
    def _extract_affected_services(self, alert: Alert) -> List[str]:
        """Extract affected services from alert labels"""
        services = []
        
        # Look for service indicators in labels
        if "service" in alert.labels:
            services.append(alert.labels["service"])
        if "job" in alert.labels:
            services.append(alert.labels["job"])
        if "instance" in alert.labels:
            services.append(alert.labels["instance"])
        
        return list(set(services))  # Remove duplicates
    
    async def _execute_playbook(self, incident: Incident, playbook: Dict[str, Any]):
        """Execute a response playbook"""
        try:
            for action in playbook["actions"]:
                action_type = ActionType(action["type"])
                
                if action_type == ActionType.NOTIFICATION:
                    await self._execute_notification_action(incident, action)
                elif action_type == ActionType.SCALING:
                    await self._execute_scaling_action(incident, action)
                elif action_type == ActionType.RESTART:
                    await self._execute_restart_action(incident, action)
                elif action_type == ActionType.CIRCUIT_BREAKER:
                    await self._execute_circuit_breaker_action(incident, action)
                elif action_type == ActionType.ROLLBACK:
                    await self._execute_rollback_action(incident, action)
                elif action_type == ActionType.CUSTOM:
                    await self._execute_custom_action(incident, action)
                
                # Record action taken
                action_desc = f"Executed {action_type.value}: {action}"
                incident.actions_taken.append(action_desc)
                logger.info(f"Action taken for incident {incident.id}: {action_desc}")
                
        except Exception as e:
            logger.error(f"Failed to execute playbook: {e}")
    
    async def _execute_notification_action(self, incident: Incident, action: Dict[str, Any]):
        """Execute notification action"""
        try:
            channels = action.get("channels", ["slack"])
            message = f"Incident detected: {incident.description}"
            
            for channel_name in channels:
                for channel in self.notification_channels:
                    if (channel_name == "slack" and isinstance(channel, SlackNotificationChannel)) or \
                       (channel_name == "pagerduty" and isinstance(channel, PagerDutyNotificationChannel)):
                        await channel.send(incident, message)
                        
        except Exception as e:
            logger.error(f"Failed to execute notification action: {e}")
    
    async def _execute_scaling_action(self, incident: Incident, action: Dict[str, Any]):
        """Execute scaling action"""
        try:
            service = action.get("service", "auto-detect")
            factor = action.get("factor", 2)
            
            if service == "auto-detect":
                # Auto-detect service from incident
                if incident.affected_services:
                    service = incident.affected_services[0]
                else:
                    logger.warning("Cannot auto-detect service for scaling")
                    return
            
            await self.auto_scaler.scale_based_on_metrics(service)
            
        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
    
    async def _execute_restart_action(self, incident: Incident, action: Dict[str, Any]):
        """Execute service restart action"""
        try:
            service = action.get("service", "auto-detect")
            
            if service == "auto-detect":
                if incident.affected_services:
                    service = incident.affected_services[0]
                else:
                    logger.warning("Cannot auto-detect service for restart")
                    return
            
            # In real implementation, restart the service
            logger.info(f"Restarting service: {service}")
            
        except Exception as e:
            logger.error(f"Failed to execute restart action: {e}")
    
    async def _execute_circuit_breaker_action(self, incident: Incident, action: Dict[str, Any]):
        """Execute circuit breaker action"""
        try:
            service = action.get("service")
            duration = action.get("duration", 5)
            
            if service:
                await self.circuit_breaker.trip_breaker(service, duration)
                
        except Exception as e:
            logger.error(f"Failed to execute circuit breaker action: {e}")
    
    async def _execute_rollback_action(self, incident: Incident, action: Dict[str, Any]):
        """Execute rollback action"""
        try:
            condition = action.get("condition")
            
            if condition == "if_recent_deployment":
                # Check if there was a recent deployment
                recent_deployment = await self._check_recent_deployment()
                
                if recent_deployment:
                    logger.info("Rolling back recent deployment")
                    # Execute rollback
                else:
                    logger.info("No recent deployment found, skipping rollback")
            else:
                logger.info("Executing unconditional rollback")
                # Execute rollback
                
        except Exception as e:
            logger.error(f"Failed to execute rollback action: {e}")
    
    async def _execute_custom_action(self, incident: Incident, action: Dict[str, Any]):
        """Execute custom action"""
        try:
            function_name = action.get("function")
            
            if function_name == "optimize_database_queries":
                await self._optimize_database_queries()
            elif function_name == "cleanup_resources":
                await self._cleanup_resources()
            elif function_name == "enable_rate_limiting":
                await self._enable_rate_limiting()
            else:
                logger.warning(f"Unknown custom function: {function_name}")
                
        except Exception as e:
            logger.error(f"Failed to execute custom action: {e}")
    
    async def _check_recent_deployment(self) -> bool:
        """Check if there was a recent deployment"""
        # In real implementation, check deployment history
        return False
    
    async def _optimize_database_queries(self):
        """Custom function to optimize database queries"""
        logger.info("Optimizing database queries")
        # Implement database optimization logic
    
    async def _cleanup_resources(self):
        """Custom function to cleanup resources"""
        logger.info("Cleaning up resources")
        # Implement resource cleanup logic
    
    async def _enable_rate_limiting(self):
        """Custom function to enable rate limiting"""
        logger.info("Enabling rate limiting")
        # Implement rate limiting logic

# Configuration
config = {
    "slack": {
        "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
        "channel": "#incidents"
    },
    "pagerduty": {
        "routing_key": "YOUR_PAGERDUTY_ROUTING_KEY"
    },
    "kubernetes": {
        "config_file": "/path/to/kubeconfig"
    }
}

# Initialize incident response system
incident_response = IncidentResponseSystem(config)

# Example usage
async def main():
    """Example of handling an alert"""
    
    # Simulate AlertManager webhook
    sample_alert = {
        "alerts": [
            {
                "labels": {
                    "alertname": "ClaudeTUIHighErrorRate",
                    "severity": "critical",
                    "service": "claude-tui"
                },
                "annotations": {
                    "summary": "High error rate detected in Claude-TUI",
                    "description": "Error rate is above threshold"
                },
                "startsAt": "2024-01-01T12:00:00Z",
                "endsAt": "",
                "fingerprint": "abc123"
            }
        ]
    }
    
    await incident_response.handle_webhook(sample_alert)

if __name__ == "__main__":
    asyncio.run(main())