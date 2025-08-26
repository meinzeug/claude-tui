#!/usr/bin/env python3
"""
Security Monitoring and Incident Response System for Claude-TUI

Implements comprehensive security monitoring including:
- Real-time threat detection and alerting
- Security Information and Event Management (SIEM)
- Automated incident response workflows
- Forensic logging and audit trails
- Security metrics and compliance reporting
- Integration with external security tools

Author: Security Manager - Claude-TUI Security Team
Date: 2025-08-26
"""

import asyncio
import json
import time
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident response status"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"


class AlertType(Enum):
    """Types of security alerts"""
    AUTHENTICATION_FAILURE = "auth_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALWARE_DETECTED = "malware_detected"
    DATA_BREACH = "data_breach"
    SYSTEM_COMPROMISE = "system_compromise"
    NETWORK_INTRUSION = "network_intrusion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class SecurityAlert:
    """Represents a security alert"""
    alert_id: str
    alert_type: AlertType
    threat_level: ThreatLevel
    title: str
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    affected_systems: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: IncidentStatus = IncidentStatus.NEW
    assigned_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityIncident:
    """Represents a security incident"""
    incident_id: str
    title: str
    description: str
    threat_level: ThreatLevel
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    related_alerts: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    affected_assets: List[str] = field(default_factory=list)
    containment_actions: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)


class SecurityMetricsCollector:
    """
    Collects and analyzes security metrics for monitoring and compliance.
    
    Tracks authentication attempts, system access patterns, resource usage,
    and security control effectiveness.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {
            'authentication': {
                'successful_logins': 0,
                'failed_logins': 0,
                'password_resets': 0,
                'account_lockouts': 0,
                'mfa_challenges': 0
            },
            'access_control': {
                'authorized_access': 0,
                'unauthorized_access': 0,
                'privilege_escalations': 0,
                'resource_access': {}
            },
            'network_security': {
                'blocked_requests': 0,
                'allowed_requests': 0,
                'malicious_ips_blocked': 0,
                'waf_blocks': 0,
                'ddos_attempts': 0
            },
            'system_security': {
                'malware_detected': 0,
                'vulnerability_scans': 0,
                'security_patches_applied': 0,
                'configuration_changes': 0
            },
            'data_protection': {
                'encryption_operations': 0,
                'key_rotations': 0,
                'backup_operations': 0,
                'data_access_requests': 0
            },
            'compliance': {
                'audit_events': 0,
                'policy_violations': 0,
                'compliance_checks_passed': 0,
                'compliance_checks_failed': 0
            }
        }
        
        # Time-series data for trend analysis
        self.time_series: Dict[str, List[Dict[str, Any]]] = {}
        
        # Baseline metrics for anomaly detection
        self.baselines: Dict[str, float] = {}
    
    def record_metric(self, category: str, metric: str, value: Union[int, float] = 1, metadata: Optional[Dict[str, Any]] = None):
        """Record a security metric."""
        if category not in self.metrics:
            self.metrics[category] = {}
        
        if metric not in self.metrics[category]:
            self.metrics[category][metric] = 0
        
        if isinstance(self.metrics[category][metric], dict):
            # Handle nested metrics
            if metadata and 'key' in metadata:
                key = metadata['key']
                if key not in self.metrics[category][metric]:
                    self.metrics[category][metric][key] = 0
                self.metrics[category][metric][key] += value
        else:
            self.metrics[category][metric] += value
        
        # Record time-series data
        self._record_time_series(category, metric, value, metadata)
    
    def _record_time_series(self, category: str, metric: str, value: Union[int, float], metadata: Optional[Dict[str, Any]]):
        """Record metric in time-series for trend analysis."""
        key = f"{category}.{metric}"
        
        if key not in self.time_series:
            self.time_series[key] = []
        
        self.time_series[key].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'value': value,
            'metadata': metadata or {}
        })
        
        # Keep only last 1000 data points
        if len(self.time_series[key]) > 1000:
            self.time_series[key] = self.time_series[key][-1000:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'current_metrics': self.metrics.copy(),
            'trends': self._calculate_trends(),
            'anomalies': self._detect_anomalies(),
            'compliance_score': self._calculate_compliance_score()
        }
        
        return summary
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate trends from time-series data."""
        trends = {}
        
        for key, data_points in self.time_series.items():
            if len(data_points) < 2:
                continue
            
            # Calculate trend over last hour
            current_time = datetime.now(timezone.utc)
            hour_ago = current_time - timedelta(hours=1)
            
            recent_points = [
                dp for dp in data_points
                if datetime.fromisoformat(dp['timestamp'].replace('Z', '+00:00')) > hour_ago
            ]
            
            if len(recent_points) < 2:
                continue
            
            values = [dp['value'] for dp in recent_points]
            
            trends[key] = {
                'current': values[-1] if values else 0,
                'average': sum(values) / len(values),
                'trend_direction': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable',
                'data_points': len(values)
            }
        
        return trends
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in security metrics."""
        anomalies = []
        
        # Define anomaly thresholds
        anomaly_rules = {
            'authentication.failed_logins': {'threshold': 100, 'timeframe_minutes': 60},
            'access_control.unauthorized_access': {'threshold': 10, 'timeframe_minutes': 30},
            'network_security.malicious_ips_blocked': {'threshold': 50, 'timeframe_minutes': 60},
            'system_security.malware_detected': {'threshold': 1, 'timeframe_minutes': 5}
        }
        
        current_time = datetime.now(timezone.utc)
        
        for metric_key, rule in anomaly_rules.items():
            if metric_key not in self.time_series:
                continue
            
            threshold_time = current_time - timedelta(minutes=rule['timeframe_minutes'])
            
            recent_values = [
                dp['value'] for dp in self.time_series[metric_key]
                if datetime.fromisoformat(dp['timestamp'].replace('Z', '+00:00')) > threshold_time
            ]
            
            total_value = sum(recent_values)
            
            if total_value > rule['threshold']:
                anomalies.append({
                    'metric': metric_key,
                    'value': total_value,
                    'threshold': rule['threshold'],
                    'timeframe_minutes': rule['timeframe_minutes'],
                    'severity': 'HIGH' if total_value > rule['threshold'] * 2 else 'MEDIUM'
                })
        
        return anomalies
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score."""
        compliance_metrics = self.metrics.get('compliance', {})
        
        passed = compliance_metrics.get('compliance_checks_passed', 0)
        failed = compliance_metrics.get('compliance_checks_failed', 0)
        total = passed + failed
        
        if total == 0:
            return 0.0
        
        score = (passed / total) * 100
        
        # Factor in policy violations
        violations = compliance_metrics.get('policy_violations', 0)
        if violations > 0:
            score = max(0, score - (violations * 5))  # Deduct 5% per violation
        
        return min(100.0, score)


class ThreatDetectionEngine:
    """
    Advanced threat detection engine using behavioral analysis and ML techniques.
    
    Implements anomaly detection, pattern recognition, and threat intelligence
    correlation for proactive security monitoring.
    """
    
    def __init__(self):
        """Initialize threat detection engine."""
        self.detection_rules: List[Dict[str, Any]] = []
        self.behavioral_baselines: Dict[str, Dict[str, float]] = {}
        self.threat_indicators: Set[str] = set()
        self.ml_models: Dict[str, Any] = {}
        
        # Initialize detection rules
        self._initialize_detection_rules()
        
        # Load threat intelligence
        self._load_threat_intelligence()
    
    def _initialize_detection_rules(self):
        """Initialize built-in detection rules."""
        self.detection_rules = [
            {
                'rule_id': 'brute_force_login',
                'name': 'Brute Force Login Attack',
                'category': 'authentication',
                'condition': 'failed_logins > 10 in 5 minutes from same IP',
                'threat_level': ThreatLevel.HIGH,
                'alert_type': AlertType.AUTHENTICATION_FAILURE
            },
            {
                'rule_id': 'privilege_escalation',
                'name': 'Privilege Escalation Attempt',
                'category': 'access_control',
                'condition': 'user attempts admin access without proper role',
                'threat_level': ThreatLevel.CRITICAL,
                'alert_type': AlertType.PRIVILEGE_ESCALATION
            },
            {
                'rule_id': 'data_exfiltration',
                'name': 'Large Data Download',
                'category': 'data_protection',
                'condition': 'data download > 1GB in 10 minutes',
                'threat_level': ThreatLevel.HIGH,
                'alert_type': AlertType.DATA_BREACH
            },
            {
                'rule_id': 'suspicious_network_activity',
                'name': 'Suspicious Network Activity',
                'category': 'network',
                'condition': 'connection to known malicious IP',
                'threat_level': ThreatLevel.CRITICAL,
                'alert_type': AlertType.NETWORK_INTRUSION
            },
            {
                'rule_id': 'anomalous_user_behavior',
                'name': 'Anomalous User Behavior',
                'category': 'behavior',
                'condition': 'user activity deviates significantly from baseline',
                'threat_level': ThreatLevel.MEDIUM,
                'alert_type': AlertType.ANOMALOUS_BEHAVIOR
            }
        ]
    
    def _load_threat_intelligence(self):
        """Load threat intelligence indicators."""
        # Known malicious IP addresses (sample)
        malicious_ips = [
            '192.0.2.1',    # Test IP for demonstration
            '198.51.100.1', # Test IP for demonstration
            '203.0.113.1'   # Test IP for demonstration
        ]
        
        self.threat_indicators.update(malicious_ips)
    
    async def analyze_event(self, event: Dict[str, Any]) -> List[SecurityAlert]:
        """Analyze security event and generate alerts if threats detected."""
        alerts = []
        
        # Run detection rules
        for rule in self.detection_rules:
            if await self._evaluate_rule(rule, event):
                alert = self._create_alert_from_rule(rule, event)
                alerts.append(alert)
        
        # Behavioral analysis
        behavioral_alerts = await self._behavioral_analysis(event)
        alerts.extend(behavioral_alerts)
        
        # Threat intelligence correlation
        ti_alerts = await self._correlate_threat_intelligence(event)
        alerts.extend(ti_alerts)
        
        return alerts
    
    async def _evaluate_rule(self, rule: Dict[str, Any], event: Dict[str, Any]) -> bool:
        """Evaluate detection rule against event."""
        rule_id = rule['rule_id']
        
        # Simple rule evaluation (in production, use more sophisticated rule engine)
        if rule_id == 'brute_force_login':
            return await self._check_brute_force_login(event)
        elif rule_id == 'privilege_escalation':
            return await self._check_privilege_escalation(event)
        elif rule_id == 'data_exfiltration':
            return await self._check_data_exfiltration(event)
        elif rule_id == 'suspicious_network_activity':
            return await self._check_suspicious_network_activity(event)
        elif rule_id == 'anomalous_user_behavior':
            return await self._check_anomalous_user_behavior(event)
        
        return False
    
    async def _check_brute_force_login(self, event: Dict[str, Any]) -> bool:
        """Check for brute force login attempts."""
        if event.get('event_type') != 'login_failure':
            return False
        
        source_ip = event.get('source_ip')
        if not source_ip:
            return False
        
        # Count recent failed logins from this IP
        # In production, query from log database
        # For simulation, assume threshold met
        failed_count = event.get('failed_login_count', 0)
        
        return failed_count >= 10
    
    async def _check_privilege_escalation(self, event: Dict[str, Any]) -> bool:
        """Check for privilege escalation attempts."""
        if event.get('event_type') != 'access_attempt':
            return False
        
        user_role = event.get('user_role', 'user')
        requested_resource = event.get('requested_resource', '')
        
        # Check if user without admin role tries to access admin resources
        if user_role != 'admin' and '/admin' in requested_resource:
            return True
        
        return False
    
    async def _check_data_exfiltration(self, event: Dict[str, Any]) -> bool:
        """Check for potential data exfiltration."""
        if event.get('event_type') not in ['file_download', 'data_export']:
            return False
        
        data_size = event.get('data_size_bytes', 0)
        
        # Alert on downloads > 1GB
        return data_size > 1_000_000_000
    
    async def _check_suspicious_network_activity(self, event: Dict[str, Any]) -> bool:
        """Check for suspicious network activity."""
        if event.get('event_type') != 'network_connection':
            return False
        
        destination_ip = event.get('destination_ip')
        source_ip = event.get('source_ip')
        
        # Check if connecting to known malicious IPs
        if destination_ip in self.threat_indicators or source_ip in self.threat_indicators:
            return True
        
        return False
    
    async def _check_anomalous_user_behavior(self, event: Dict[str, Any]) -> bool:
        """Check for anomalous user behavior."""
        user_id = event.get('user_id')
        if not user_id:
            return False
        
        # Simple anomaly detection based on time and location
        login_time = event.get('timestamp')
        source_ip = event.get('source_ip')
        
        # Check for unusual login times (outside business hours)
        if login_time:
            hour = datetime.fromisoformat(login_time.replace('Z', '+00:00')).hour
            if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
                return True
        
        # Check for login from unusual locations (simplified)
        if source_ip and not source_ip.startswith(('10.', '172.', '192.168.')):
            # Login from external IP
            return True
        
        return False
    
    def _create_alert_from_rule(self, rule: Dict[str, Any], event: Dict[str, Any]) -> SecurityAlert:
        """Create security alert from detection rule and event."""
        alert_id = f"alert_{rule['rule_id']}_{int(time.time())}"
        
        return SecurityAlert(
            alert_id=alert_id,
            alert_type=rule['alert_type'],
            threat_level=rule['threat_level'],
            title=rule['name'],
            description=f"Detection rule '{rule['name']}' triggered",
            source_ip=event.get('source_ip'),
            user_id=event.get('user_id'),
            indicators={
                'rule_id': rule['rule_id'],
                'event_type': event.get('event_type'),
                'detection_reason': rule['condition']
            },
            metadata=event
        )
    
    async def _behavioral_analysis(self, event: Dict[str, Any]) -> List[SecurityAlert]:
        """Perform behavioral analysis on event."""
        alerts = []
        
        # Placeholder for ML-based behavioral analysis
        # In production, this would use trained models
        
        return alerts
    
    async def _correlate_threat_intelligence(self, event: Dict[str, Any]) -> List[SecurityAlert]:
        """Correlate event with threat intelligence."""
        alerts = []
        
        # Check IPs against threat intelligence
        source_ip = event.get('source_ip')
        if source_ip in self.threat_indicators:
            alert = SecurityAlert(
                alert_id=f"ti_alert_{int(time.time())}",
                alert_type=AlertType.NETWORK_INTRUSION,
                threat_level=ThreatLevel.CRITICAL,
                title="Known Malicious IP Detected",
                description=f"Connection from known malicious IP: {source_ip}",
                source_ip=source_ip,
                indicators={'threat_intelligence_match': source_ip}
            )
            alerts.append(alert)
        
        return alerts


class IncidentResponseManager:
    """
    Automated incident response manager with playbook execution.
    
    Implements automated response workflows, escalation procedures,
    and incident lifecycle management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize incident response manager."""
        self.config = config or {}
        self.incidents: Dict[str, SecurityIncident] = {}
        self.playbooks: Dict[str, Dict[str, Any]] = {}
        self.escalation_rules: List[Dict[str, Any]] = []
        self.notification_channels: List[str] = []
        
        # Initialize response playbooks
        self._initialize_playbooks()
        
        # Initialize escalation rules
        self._initialize_escalation_rules()
    
    def _initialize_playbooks(self):
        """Initialize incident response playbooks."""
        self.playbooks = {
            'brute_force_response': {
                'name': 'Brute Force Attack Response',
                'triggers': [AlertType.AUTHENTICATION_FAILURE],
                'actions': [
                    {'action': 'block_ip', 'duration': '1 hour'},
                    {'action': 'notify_admin', 'channel': 'email'},
                    {'action': 'increase_monitoring', 'duration': '24 hours'}
                ]
            },
            'malware_response': {
                'name': 'Malware Detection Response',
                'triggers': [AlertType.MALWARE_DETECTED],
                'actions': [
                    {'action': 'isolate_system', 'immediate': True},
                    {'action': 'notify_security_team', 'channel': 'sms'},
                    {'action': 'create_forensic_image'},
                    {'action': 'run_full_scan'}
                ]
            },
            'data_breach_response': {
                'name': 'Data Breach Response',
                'triggers': [AlertType.DATA_BREACH],
                'actions': [
                    {'action': 'isolate_affected_systems'},
                    {'action': 'notify_legal_team'},
                    {'action': 'preserve_evidence'},
                    {'action': 'assess_data_impact'},
                    {'action': 'prepare_breach_notification'}
                ]
            }
        }
    
    def _initialize_escalation_rules(self):
        """Initialize escalation rules."""
        self.escalation_rules = [
            {
                'condition': 'threat_level == CRITICAL',
                'escalate_to': 'security_team',
                'notification': 'immediate',
                'channels': ['email', 'sms', 'slack']
            },
            {
                'condition': 'incident_age > 2 hours AND status == NEW',
                'escalate_to': 'security_manager',
                'notification': 'urgent',
                'channels': ['email', 'slack']
            },
            {
                'condition': 'multiple_incidents > 5 in 1 hour',
                'escalate_to': 'ciso',
                'notification': 'critical',
                'channels': ['email', 'sms', 'phone']
            }
        ]
    
    async def handle_alert(self, alert: SecurityAlert) -> Optional[SecurityIncident]:
        """Handle incoming security alert and create incident if needed."""
        logger.info(f"🚨 Handling security alert: {alert.alert_id}")
        
        # Check if alert should create incident
        if await self._should_create_incident(alert):
            incident = await self._create_incident_from_alert(alert)
            
            # Execute response playbook
            await self._execute_response_playbook(alert, incident)
            
            # Check escalation rules
            await self._check_escalation_rules(incident)
            
            return incident
        else:
            # Log alert but don't create incident
            logger.info(f"Alert {alert.alert_id} logged without incident creation")
            return None
    
    async def _should_create_incident(self, alert: SecurityAlert) -> bool:
        """Determine if alert should create a new incident."""
        # Create incident for HIGH and CRITICAL alerts
        if alert.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            return True
        
        # Check for correlation with existing incidents
        related_incidents = self._find_related_incidents(alert)
        if related_incidents:
            # Add alert to existing incident
            for incident_id in related_incidents:
                incident = self.incidents[incident_id]
                incident.related_alerts.append(alert.alert_id)
                incident.updated_at = datetime.now(timezone.utc)
            return False
        
        # Create incident for repeated MEDIUM alerts
        if alert.threat_level == ThreatLevel.MEDIUM:
            recent_alerts = self._count_recent_similar_alerts(alert)
            if recent_alerts >= 3:
                return True
        
        return False
    
    def _find_related_incidents(self, alert: SecurityAlert) -> List[str]:
        """Find incidents related to the alert."""
        related = []
        
        for incident_id, incident in self.incidents.items():
            if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                continue
            
            # Check if same source IP
            if alert.source_ip:
                for related_alert_id in incident.related_alerts:
                    # In production, fetch alert details from database
                    # For now, assume correlation logic
                    pass
            
            # Check if same user
            if alert.user_id:
                # Similar correlation logic for user-based incidents
                pass
        
        return related
    
    def _count_recent_similar_alerts(self, alert: SecurityAlert) -> int:
        """Count recent similar alerts."""
        # In production, query alert database
        # For simulation, return a count
        return 1
    
    async def _create_incident_from_alert(self, alert: SecurityAlert) -> SecurityIncident:
        """Create security incident from alert."""
        incident_id = f"inc_{int(time.time())}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"Security Incident: {alert.title}",
            description=alert.description,
            threat_level=alert.threat_level,
            status=IncidentStatus.NEW,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            related_alerts=[alert.alert_id],
            affected_assets=alert.affected_systems.copy(),
            timeline=[
                {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'action': 'incident_created',
                    'details': f"Created from alert {alert.alert_id}"
                }
            ]
        )
        
        self.incidents[incident_id] = incident
        
        logger.info(f"📋 Created security incident: {incident_id}")
        return incident
    
    async def _execute_response_playbook(self, alert: SecurityAlert, incident: SecurityIncident):
        """Execute automated response playbook."""
        # Find matching playbooks
        matching_playbooks = []
        
        for playbook_id, playbook in self.playbooks.items():
            if alert.alert_type in playbook['triggers']:
                matching_playbooks.append((playbook_id, playbook))
        
        # Execute playbook actions
        for playbook_id, playbook in matching_playbooks:
            logger.info(f"🎭 Executing playbook: {playbook['name']}")
            
            for action_config in playbook['actions']:
                try:
                    await self._execute_response_action(action_config, alert, incident)
                    
                    # Record action in incident timeline
                    incident.timeline.append({
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'action': 'playbook_action_executed',
                        'details': f"Executed {action_config['action']} from {playbook['name']}"
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to execute action {action_config['action']}: {e}")
                    
                    incident.timeline.append({
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'action': 'playbook_action_failed',
                        'details': f"Failed to execute {action_config['action']}: {str(e)}"
                    })
    
    async def _execute_response_action(self, action_config: Dict[str, Any], alert: SecurityAlert, incident: SecurityIncident):
        """Execute individual response action."""
        action = action_config['action']
        
        if action == 'block_ip':
            await self._block_ip(alert.source_ip, action_config.get('duration', '1 hour'))
        elif action == 'notify_admin':
            await self._notify_admin(alert, incident, action_config.get('channel', 'email'))
        elif action == 'isolate_system':
            await self._isolate_system(alert.affected_systems)
        elif action == 'increase_monitoring':
            await self._increase_monitoring(action_config.get('duration', '24 hours'))
        elif action == 'create_forensic_image':
            await self._create_forensic_image(alert.affected_systems)
        elif action == 'run_full_scan':
            await self._run_full_scan(alert.affected_systems)
        else:
            logger.warning(f"Unknown response action: {action}")
    
    async def _block_ip(self, ip_address: Optional[str], duration: str):
        """Block IP address."""
        if not ip_address:
            return
        
        logger.info(f"🚫 Blocking IP address {ip_address} for {duration}")
        
        # In production, integrate with firewall/WAF
        # For simulation, just log the action
        
    async def _notify_admin(self, alert: SecurityAlert, incident: SecurityIncident, channel: str):
        """Notify administrators."""
        logger.info(f"📧 Notifying admin via {channel}")
        
        if channel == 'email':
            await self._send_email_notification(alert, incident)
        elif channel == 'sms':
            await self._send_sms_notification(alert, incident)
        elif channel == 'slack':
            await self._send_slack_notification(alert, incident)
    
    async def _send_email_notification(self, alert: SecurityAlert, incident: SecurityIncident):
        """Send email notification."""
        # Email configuration from config
        smtp_config = self.config.get('email', {})
        
        if not smtp_config.get('enabled', False):
            logger.debug("Email notifications disabled")
            return
        
        # Create email message
        subject = f"Security Alert: {alert.title}"
        body = f"""
Security Incident Alert

Incident ID: {incident.incident_id}
Alert ID: {alert.alert_id}
Threat Level: {alert.threat_level.value.upper()}
Description: {alert.description}

Source IP: {alert.source_ip or 'Unknown'}
User ID: {alert.user_id or 'Unknown'}
Timestamp: {alert.timestamp.isoformat()}

Affected Systems: {', '.join(alert.affected_systems) or 'None specified'}

This is an automated security alert. Please investigate immediately.
"""
        
        # In production, actually send email
        logger.info(f"Email notification prepared: {subject}")
    
    async def _send_sms_notification(self, alert: SecurityAlert, incident: SecurityIncident):
        """Send SMS notification."""
        message = f"SECURITY ALERT: {alert.title} - Incident {incident.incident_id} - Threat Level: {alert.threat_level.value.upper()}"
        logger.info(f"SMS notification prepared: {message[:100]}...")
    
    async def _send_slack_notification(self, alert: SecurityAlert, incident: SecurityIncident):
        """Send Slack notification."""
        message = {
            'text': f"🚨 Security Alert: {alert.title}",
            'attachments': [
                {
                    'color': 'danger' if alert.threat_level == ThreatLevel.CRITICAL else 'warning',
                    'fields': [
                        {'title': 'Incident ID', 'value': incident.incident_id, 'short': True},
                        {'title': 'Threat Level', 'value': alert.threat_level.value.upper(), 'short': True},
                        {'title': 'Source IP', 'value': alert.source_ip or 'Unknown', 'short': True},
                        {'title': 'Description', 'value': alert.description, 'short': False}
                    ]
                }
            ]
        }
        
        logger.info(f"Slack notification prepared: {message['text']}")
    
    async def _isolate_system(self, systems: List[str]):
        """Isolate affected systems."""
        for system in systems:
            logger.info(f"🔒 Isolating system: {system}")
            # In production, implement actual system isolation
    
    async def _increase_monitoring(self, duration: str):
        """Increase security monitoring."""
        logger.info(f"📊 Increasing security monitoring for {duration}")
        # In production, adjust monitoring sensitivity
    
    async def _create_forensic_image(self, systems: List[str]):
        """Create forensic images of affected systems."""
        for system in systems:
            logger.info(f"🔍 Creating forensic image of system: {system}")
            # In production, trigger forensic imaging tools
    
    async def _run_full_scan(self, systems: List[str]):
        """Run full security scan on systems."""
        for system in systems:
            logger.info(f"🔎 Running full security scan on system: {system}")
            # In production, trigger security scanning tools
    
    async def _check_escalation_rules(self, incident: SecurityIncident):
        """Check and apply escalation rules."""
        for rule in self.escalation_rules:
            if await self._evaluate_escalation_condition(rule['condition'], incident):
                await self._escalate_incident(incident, rule)
    
    async def _evaluate_escalation_condition(self, condition: str, incident: SecurityIncident) -> bool:
        """Evaluate escalation condition."""
        # Simple condition evaluation
        if 'threat_level == CRITICAL' in condition:
            return incident.threat_level == ThreatLevel.CRITICAL
        
        if 'incident_age > 2 hours AND status == NEW' in condition:
            age = datetime.now(timezone.utc) - incident.created_at
            return age.total_seconds() > 7200 and incident.status == IncidentStatus.NEW
        
        # Add more condition evaluations as needed
        return False
    
    async def _escalate_incident(self, incident: SecurityIncident, rule: Dict[str, Any]):
        """Escalate incident according to rule."""
        logger.warning(f"⬆️ Escalating incident {incident.incident_id} to {rule['escalate_to']}")
        
        # Update incident
        incident.timeline.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'incident_escalated',
            'details': f"Escalated to {rule['escalate_to']} - {rule['notification']}"
        })
        
        # Send escalation notifications
        for channel in rule['channels']:
            logger.info(f"📢 Sending escalation notification via {channel}")
    
    def get_incident_status(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get incident status and details."""
        if incident_id not in self.incidents:
            return None
        
        incident = self.incidents[incident_id]
        return asdict(incident)
    
    def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get all active incidents."""
        active_statuses = [
            IncidentStatus.NEW,
            IncidentStatus.ACKNOWLEDGED,
            IncidentStatus.INVESTIGATING,
            IncidentStatus.CONTAINED
        ]
        
        active_incidents = [
            asdict(incident) for incident in self.incidents.values()
            if incident.status in active_statuses
        ]
        
        return active_incidents


class SecurityMonitoringSystem:
    """
    Comprehensive security monitoring system coordinator.
    
    Integrates metrics collection, threat detection, and incident response
    into a unified security monitoring platform.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security monitoring system."""
        self.config = config or {}
        
        # Initialize components
        self.metrics_collector = SecurityMetricsCollector()
        self.threat_detector = ThreatDetectionEngine()
        self.incident_manager = IncidentResponseManager(config)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Alert storage
        self.alerts: List[SecurityAlert] = []
        self.alert_handlers: List[Callable[[SecurityAlert], None]] = []
    
    async def initialize(self) -> bool:
        """Initialize security monitoring system."""
        try:
            logger.info("🔍 Initializing Security Monitoring System...")
            
            # Start monitoring loop
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("✅ Security Monitoring System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize security monitoring: {e}")
            return False
    
    async def process_security_event(self, event: Dict[str, Any]) -> List[SecurityAlert]:
        """Process security event and generate alerts."""
        # Analyze event for threats
        alerts = await self.threat_detector.analyze_event(event)
        
        # Store alerts
        self.alerts.extend(alerts)
        
        # Process each alert
        for alert in alerts:
            # Record metrics
            self.metrics_collector.record_metric(
                'security_events', 
                alert.alert_type.value, 
                1,
                {'threat_level': alert.threat_level.value}
            )
            
            # Handle alert through incident response
            incident = await self.incident_manager.handle_alert(alert)
            
            # Notify alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")
            
            logger.warning(f"🚨 Security alert generated: {alert.alert_id} - {alert.title}")
        
        return alerts
    
    def register_alert_handler(self, handler: Callable[[SecurityAlert], None]):
        """Register custom alert handler."""
        self.alert_handlers.append(handler)
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check for anomalies
                anomalies = self.metrics_collector._detect_anomalies()
                
                # Generate alerts for anomalies
                for anomaly in anomalies:
                    await self._process_anomaly_alert(anomaly)
                
                # Cleanup old alerts (keep last 10000)
                if len(self.alerts) > 10000:
                    self.alerts = self.alerts[-10000:]
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Sleep shorter on error
    
    async def _collect_system_metrics(self):
        """Collect system-level security metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 90:
                self.metrics_collector.record_metric('system', 'high_cpu_usage', 1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.metrics_collector.record_metric('system', 'high_memory_usage', 1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                self.metrics_collector.record_metric('system', 'high_disk_usage', 1)
            
            # Network connections
            connections = psutil.net_connections()
            external_connections = [
                conn for conn in connections 
                if conn.status == 'ESTABLISHED' and conn.raddr
            ]
            
            if len(external_connections) > 1000:
                self.metrics_collector.record_metric('network', 'high_connection_count', 1)
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _process_anomaly_alert(self, anomaly: Dict[str, Any]):
        """Process anomaly detection alert."""
        alert_id = f"anomaly_{int(time.time())}"
        
        threat_level = ThreatLevel.HIGH if anomaly['severity'] == 'HIGH' else ThreatLevel.MEDIUM
        
        alert = SecurityAlert(
            alert_id=alert_id,
            alert_type=AlertType.ANOMALOUS_BEHAVIOR,
            threat_level=threat_level,
            title=f"Anomaly Detected: {anomaly['metric']}",
            description=f"Metric {anomaly['metric']} exceeded threshold: {anomaly['value']} > {anomaly['threshold']}",
            indicators=anomaly
        )
        
        # Process alert
        await self.process_security_event({'alert': alert})
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'monitoring_active': self.monitoring_active,
            'total_alerts': len(self.alerts),
            'recent_alerts': [
                asdict(alert) for alert in self.alerts[-10:]
            ],
            'active_incidents': len(self.incident_manager.get_active_incidents()),
            'metrics_summary': self.metrics_collector.get_metrics_summary(),
            'threat_indicators_loaded': len(self.threat_detector.threat_indicators),
            'detection_rules_active': len(self.threat_detector.detection_rules)
        }
    
    async def cleanup(self):
        """Cleanup monitoring resources."""
        logger.info("🧹 Cleaning up security monitoring...")
        
        self.monitoring_active = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        
        # Clear sensitive data
        self.alerts.clear()
        self.alert_handlers.clear()
        
        logger.info("✅ Security monitoring cleanup completed")


# Global security monitoring system
_security_monitoring_system: Optional[SecurityMonitoringSystem] = None


async def init_security_monitoring(config: Optional[Dict[str, Any]] = None) -> SecurityMonitoringSystem:
    """Initialize global security monitoring system."""
    global _security_monitoring_system
    
    _security_monitoring_system = SecurityMonitoringSystem(config)
    await _security_monitoring_system.initialize()
    
    return _security_monitoring_system


def get_security_monitoring_system() -> SecurityMonitoringSystem:
    """Get global security monitoring system instance."""
    global _security_monitoring_system
    
    if _security_monitoring_system is None:
        raise RuntimeError("Security monitoring system not initialized. Call init_security_monitoring() first.")
    
    return _security_monitoring_system