"""
Security Audit Logger.

Provides comprehensive security event logging for authentication
and authorization activities with structured logging format.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from pathlib import Path

from ..core.config import get_settings


class SecurityEventType(Enum):
    """Security event types for classification."""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGIN_BLOCKED = "login_blocked"
    LOGOUT = "logout"
    LOGOUT_ALL = "logout_all"
    
    # Registration events
    REGISTRATION_SUCCESS = "registration_success"
    REGISTRATION_FAILED = "registration_failed"
    REGISTRATION_DUPLICATE = "registration_duplicate"
    
    # Token events
    TOKEN_ISSUED = "token_issued"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_REVOKED = "token_revoked"
    TOKEN_EXPIRED = "token_expired"
    
    # Session events
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SESSION_TERMINATED = "session_terminated"
    SESSION_IP_CHANGE = "session_ip_change"
    
    # Password events
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUEST = "password_reset_request"
    PASSWORD_RESET_SUCCESS = "password_reset_success"
    PASSWORD_RESET_FAILED = "password_reset_failed"
    
    # OAuth events
    OAUTH_LOGIN_SUCCESS = "oauth_login_success"
    OAUTH_LOGIN_FAILED = "oauth_login_failed"
    OAUTH_REGISTRATION = "oauth_registration"
    
    # Authorization events
    ACCESS_DENIED = "access_denied"
    PERMISSION_ESCALATION_ATTEMPT = "permission_escalation_attempt"
    ROLE_CHANGED = "role_changed"
    
    # Security events
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"
    BRUTE_FORCE_DETECTED = "brute_force_detected"


class SecurityLevel(Enum):
    """Security event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: datetime
    event_type: SecurityEventType
    level: SecurityLevel
    user_id: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    success: bool = True
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    risk_score: int = 0
    correlation_id: Optional[str] = None


class SecurityAuditLogger:
    """
    Security audit logger with structured logging and alerting.
    
    Features:
    - Structured security event logging
    - Risk scoring
    - Event correlation
    - Real-time alerting for critical events
    - Log rotation and archival
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_syslog: bool = False,
        alert_webhook: Optional[str] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10
    ):
        self.settings = get_settings()
        self.log_file = log_file or self.settings.SECURITY_LOG_FILE
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_syslog = enable_syslog
        self.alert_webhook = alert_webhook or self.settings.SECURITY_ALERT_WEBHOOK
        
        # Setup logger
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Setup file handler with rotation
        if enable_file and self.log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(self._get_json_formatter())
            self.logger.addHandler(file_handler)
        
        # Setup console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._get_console_formatter())
            self.logger.addHandler(console_handler)
        
        # Setup syslog handler
        if enable_syslog:
            try:
                from logging.handlers import SysLogHandler
                syslog_handler = SysLogHandler(address='/dev/log')
                syslog_handler.setFormatter(self._get_syslog_formatter())
                self.logger.addHandler(syslog_handler)
            except Exception:
                pass  # Syslog not available
    
    def _get_json_formatter(self) -> logging.Formatter:
        """Get JSON formatter for structured logging."""
        return logging.Formatter(
            '%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_console_formatter(self) -> logging.Formatter:
        """Get console formatter for human-readable output."""
        return logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_syslog_formatter(self) -> logging.Formatter:
        """Get syslog formatter."""
        return logging.Formatter(
            'claude-tiu-security: %(message)s'
        )
    
    async def log_event(self, event: SecurityEvent) -> None:\n        \"\"\"Log a security event.\"\"\"\n        try:\n            # Convert event to dict\n            event_dict = asdict(event)\n            event_dict['timestamp'] = event.timestamp.isoformat()\n            event_dict['event_type'] = event.event_type.value\n            event_dict['level'] = event.level.value\n            \n            # Log as JSON for structured logging\n            self.logger.info(json.dumps(event_dict, default=str))\n            \n            # Check for critical events that need immediate alerting\n            if event.level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:\n                await self._send_alert(event)\n            \n        except Exception as e:\n            # Never let audit logging break the application\n            self.logger.error(f\"Failed to log security event: {str(e)}\")\n    \n    async def log_authentication(self,\n        event_type: SecurityEventType,\n        user_id: Optional[str] = None,\n        username: Optional[str] = None,\n        ip_address: Optional[str] = None,\n        user_agent: Optional[str] = None,\n        session_id: Optional[str] = None,\n        success: bool = True,\n        message: Optional[str] = None,\n        details: Optional[Dict[str, Any]] = None\n    ) -> None:\n        \"\"\"Log authentication event.\"\"\"\n        level = self._determine_auth_level(event_type, success)\n        risk_score = self._calculate_auth_risk_score(event_type, success, details)\n        \n        event = SecurityEvent(\n            timestamp=datetime.now(timezone.utc),\n            event_type=event_type,\n            level=level,\n            user_id=user_id,\n            username=username,\n            ip_address=ip_address,\n            user_agent=user_agent,\n            session_id=session_id,\n            success=success,\n            message=message,\n            details=details,\n            risk_score=risk_score\n        )\n        \n        await self.log_event(event)\n    \n    async def log_authorization(\n        self,\n        event_type: SecurityEventType,\n        user_id: str,\n        username: str,\n        resource: str,\n        action: str,\n        success: bool,\n        ip_address: Optional[str] = None,\n        message: Optional[str] = None,\n        details: Optional[Dict[str, Any]] = None\n    ) -> None:\n        \"\"\"Log authorization event.\"\"\"\n        level = SecurityLevel.MEDIUM if success else SecurityLevel.HIGH\n        risk_score = self._calculate_authz_risk_score(action, success, details)\n        \n        event = SecurityEvent(\n            timestamp=datetime.now(timezone.utc),\n            event_type=event_type,\n            level=level,\n            user_id=user_id,\n            username=username,\n            ip_address=ip_address,\n            resource=resource,\n            action=action,\n            success=success,\n            message=message,\n            details=details,\n            risk_score=risk_score\n        )\n        \n        await self.log_event(event)\n    \n    async def log_security_incident(\n        self,\n        event_type: SecurityEventType,\n        level: SecurityLevel,\n        message: str,\n        user_id: Optional[str] = None,\n        ip_address: Optional[str] = None,\n        details: Optional[Dict[str, Any]] = None\n    ) -> None:\n        \"\"\"Log security incident.\"\"\"\n        event = SecurityEvent(\n            timestamp=datetime.now(timezone.utc),\n            event_type=event_type,\n            level=level,\n            user_id=user_id,\n            ip_address=ip_address,\n            message=message,\n            details=details,\n            success=False,\n            risk_score=self._calculate_incident_risk_score(level, details)\n        )\n        \n        await self.log_event(event)\n    \n    def _determine_auth_level(self, event_type: SecurityEventType, success: bool) -> SecurityLevel:\n        \"\"\"Determine security level for authentication events.\"\"\"\n        if not success:\n            if event_type in [SecurityEventType.LOGIN_FAILED]:\n                return SecurityLevel.MEDIUM\n            elif event_type in [SecurityEventType.LOGIN_BLOCKED]:\n                return SecurityLevel.HIGH\n            else:\n                return SecurityLevel.LOW\n        else:\n            return SecurityLevel.LOW\n    \n    def _calculate_auth_risk_score(self,\n        event_type: SecurityEventType,\n        success: bool,\n        details: Optional[Dict[str, Any]] = None\n    ) -> int:\n        \"\"\"Calculate risk score for authentication events.\"\"\"\n        base_score = 0\n        \n        if not success:\n            base_score += 30\n        \n        if event_type == SecurityEventType.LOGIN_FAILED:\n            base_score += 20\n        elif event_type == SecurityEventType.LOGIN_BLOCKED:\n            base_score += 50\n        elif event_type == SecurityEventType.BRUTE_FORCE_DETECTED:\n            base_score += 70\n        \n        # Additional risk factors\n        if details:\n            if details.get('failed_attempts', 0) > 5:\n                base_score += 20\n            if details.get('suspicious_ip'):\n                base_score += 30\n            if details.get('unusual_location'):\n                base_score += 25\n        \n        return min(base_score, 100)  # Cap at 100\n    \n    def _calculate_authz_risk_score(self,\n        action: str,\n        success: bool,\n        details: Optional[Dict[str, Any]] = None\n    ) -> int:\n        \"\"\"Calculate risk score for authorization events.\"\"\"\n        base_score = 0\n        \n        if not success:\n            base_score += 40\n        \n        # Higher risk for sensitive actions\n        sensitive_actions = ['delete', 'admin', 'escalate', 'modify_permissions']\n        if any(sensitive in action.lower() for sensitive in sensitive_actions):\n            base_score += 30\n        \n        return min(base_score, 100)\n    \n    def _calculate_incident_risk_score(self,\n        level: SecurityLevel,\n        details: Optional[Dict[str, Any]] = None\n    ) -> int:\n        \"\"\"Calculate risk score for security incidents.\"\"\"\n        level_scores = {\n            SecurityLevel.LOW: 20,\n            SecurityLevel.MEDIUM: 50,\n            SecurityLevel.HIGH: 80,\n            SecurityLevel.CRITICAL: 100\n        }\n        \n        return level_scores.get(level, 50)\n    \n    async def _send_alert(self, event: SecurityEvent) -> None:\n        \"\"\"Send alert for critical security events.\"\"\"\n        if not self.alert_webhook:\n            return\n        \n        try:\n            import httpx\n            \n            alert_data = {\n                'timestamp': event.timestamp.isoformat(),\n                'event_type': event.event_type.value,\n                'level': event.level.value,\n                'message': event.message or f\"Security event: {event.event_type.value}\",\n                'user_id': event.user_id,\n                'ip_address': event.ip_address,\n                'risk_score': event.risk_score,\n                'details': event.details\n            }\n            \n            async with httpx.AsyncClient() as client:\n                response = await client.post(\n                    self.alert_webhook,\n                    json=alert_data,\n                    timeout=10\n                )\n                \n                if response.status_code != 200:\n                    self.logger.error(\n                        f\"Failed to send security alert: {response.status_code}\"\n                    )\n        \n        except Exception as e:\n            self.logger.error(f\"Error sending security alert: {str(e)}\")\n    \n    async def get_events(\n        self,\n        start_time: Optional[datetime] = None,\n        end_time: Optional[datetime] = None,\n        event_types: Optional[List[SecurityEventType]] = None,\n        user_id: Optional[str] = None,\n        ip_address: Optional[str] = None,\n        min_risk_score: Optional[int] = None,\n        limit: int = 100\n    ) -> List[SecurityEvent]:\n        \"\"\"Query security events (requires log parsing or database storage).\"\"\"\n        # This would require implementing log parsing or storing events in database\n        # For now, return empty list\n        return []\n    \n    async def generate_report(\n        self,\n        start_time: datetime,\n        end_time: datetime,\n        include_summary: bool = True,\n        include_top_users: bool = True,\n        include_top_ips: bool = True\n    ) -> Dict[str, Any]:\n        \"\"\"Generate security audit report.\"\"\"\n        events = await self.get_events(start_time=start_time, end_time=end_time)\n        \n        report = {\n            'period': {\n                'start': start_time.isoformat(),\n                'end': end_time.isoformat()\n            },\n            'total_events': len(events),\n            'events_by_type': {},\n            'events_by_level': {},\n            'high_risk_events': []\n        }\n        \n        # Process events for summary\n        for event in events:\n            # Count by type\n            event_type = event.event_type.value\n            report['events_by_type'][event_type] = report['events_by_type'].get(event_type, 0) + 1\n            \n            # Count by level\n            level = event.level.value\n            report['events_by_level'][level] = report['events_by_level'].get(level, 0) + 1\n            \n            # High risk events\n            if event.risk_score >= 70:\n                report['high_risk_events'].append(asdict(event))\n        \n        return report\n\n\n# Global audit logger instance\n_audit_logger: Optional[SecurityAuditLogger] = None\n\n\ndef get_audit_logger() -> SecurityAuditLogger:\n    \"\"\"Get global audit logger instance.\"\"\"\n    global _audit_logger\n    if _audit_logger is None:\n        _audit_logger = SecurityAuditLogger()\n    return _audit_logger\n\n\n# Convenience functions\nasync def log_login_success(user_id: str, username: str, ip_address: str, session_id: str, **kwargs):\n    \"\"\"Log successful login.\"\"\"\n    logger = get_audit_logger()\n    await logger.log_authentication(\n        SecurityEventType.LOGIN_SUCCESS,\n        user_id=user_id,\n        username=username,\n        ip_address=ip_address,\n        session_id=session_id,\n        success=True,\n        **kwargs\n    )\n\n\nasync def log_login_failed(identifier: str, ip_address: str, reason: str = None, **kwargs):\n    \"\"\"Log failed login attempt.\"\"\"\n    logger = get_audit_logger()\n    await logger.log_authentication(\n        SecurityEventType.LOGIN_FAILED,\n        ip_address=ip_address,\n        success=False,\n        message=f\"Login failed for {identifier}\",\n        details={'identifier': identifier, 'reason': reason},\n        **kwargs\n    )\n\n\nasync def log_access_denied(user_id: str, username: str, resource: str, action: str, **kwargs):\n    \"\"\"Log access denied event.\"\"\"\n    logger = get_audit_logger()\n    await logger.log_authorization(\n        SecurityEventType.ACCESS_DENIED,\n        user_id=user_id,\n        username=username,\n        resource=resource,\n        action=action,\n        success=False,\n        **kwargs\n    )\n\n\nasync def log_security_incident(incident_type: str, message: str, level: SecurityLevel = SecurityLevel.HIGH, **kwargs):\n    \"\"\"Log security incident.\"\"\"\n    logger = get_audit_logger()\n    await logger.log_security_incident(\n        SecurityEventType.SUSPICIOUS_ACTIVITY,\n        level=level,\n        message=f\"{incident_type}: {message}\",\n        **kwargs\n    )