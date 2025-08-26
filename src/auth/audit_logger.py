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
            except (ImportError, OSError, ConnectionRefusedError) as e:
                # Syslog not available - log warning but continue
                logger.warning(f"Syslog handler setup failed: {e}. Continuing without syslog.")
    
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
            'claude-tui-security: %(message)s'
        )
    
    async def log_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        try:
            # Convert event to dict
            event_dict = asdict(event)
            event_dict['timestamp'] = event.timestamp.isoformat()
            event_dict['event_type'] = event.event_type.value
            event_dict['level'] = event.level.value
            
            # Log as JSON for structured logging
            self.logger.info(json.dumps(event_dict, default=str))
            
            # Check for critical events that need immediate alerting
            if event.level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                await self._send_alert(event)
            
        except (json.JSONEncodeError, TypeError, ValueError) as e:
            # Handle serialization errors specifically
            self.logger.error(f"Failed to serialize security event: {e}")
        except (IOError, OSError) as e:
            # Handle file/network I/O errors
            self.logger.error(f"I/O error during security event logging: {e}")
        except Exception as e:
            # Last resort - never let audit logging break the application
            self.logger.error(f"Unexpected error logging security event: {e}", exc_info=True)
    
    async def log_authentication(self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log authentication event."""
        level = self._determine_auth_level(event_type, success)
        risk_score = self._calculate_auth_risk_score(event_type, success, details)
        
        event = SecurityEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            level=level,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            success=success,
            message=message,
            details=details,
            risk_score=risk_score
        )
        
        await self.log_event(event)
    
    async def log_authorization(
        self,
        event_type: SecurityEventType,
        user_id: str,
        username: str,
        resource: str,
        action: str,
        success: bool,
        ip_address: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log authorization event."""
        level = SecurityLevel.MEDIUM if success else SecurityLevel.HIGH
        risk_score = self._calculate_authz_risk_score(action, success, details)
        
        event = SecurityEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            level=level,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            resource=resource,
            action=action,
            success=success,
            message=message,
            details=details,
            risk_score=risk_score
        )
        
        await self.log_event(event)
    
    async def log_security_incident(
        self,
        event_type: SecurityEventType,
        level: SecurityLevel,
        message: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security incident."""
        event = SecurityEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            level=level,
            user_id=user_id,
            ip_address=ip_address,
            message=message,
            details=details,
            success=False,
            risk_score=self._calculate_incident_risk_score(level, details)
        )
        
        await self.log_event(event)
    
    def _determine_auth_level(self, event_type: SecurityEventType, success: bool) -> SecurityLevel:
        """Determine security level for authentication events."""
        if not success:
            if event_type in [SecurityEventType.LOGIN_FAILED]:
                return SecurityLevel.MEDIUM
            elif event_type in [SecurityEventType.LOGIN_BLOCKED]:
                return SecurityLevel.HIGH
            else:
                return SecurityLevel.LOW
        else:
            return SecurityLevel.LOW
    
    def _calculate_auth_risk_score(self,
        event_type: SecurityEventType,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Calculate risk score for authentication events."""
        base_score = 0
        
        if not success:
            base_score += 30
        
        if event_type == SecurityEventType.LOGIN_FAILED:
            base_score += 20
        elif event_type == SecurityEventType.LOGIN_BLOCKED:
            base_score += 50
        elif event_type == SecurityEventType.BRUTE_FORCE_DETECTED:
            base_score += 70
        
        # Additional risk factors
        if details:
            if details.get('failed_attempts', 0) > 5:
                base_score += 20
            if details.get('suspicious_ip'):
                base_score += 30
            if details.get('unusual_location'):
                base_score += 25
        
        return min(base_score, 100)  # Cap at 100
    
    def _calculate_authz_risk_score(self,
        action: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Calculate risk score for authorization events."""
        base_score = 0
        
        if not success:
            base_score += 40
        
        # Higher risk for sensitive actions
        sensitive_actions = ['delete', 'admin', 'escalate', 'modify_permissions']
        if any(sensitive in action.lower() for sensitive in sensitive_actions):
            base_score += 30
        
        return min(base_score, 100)
    
    def _calculate_incident_risk_score(self,
        level: SecurityLevel,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Calculate risk score for security incidents."""
        level_scores = {
            SecurityLevel.LOW: 20,
            SecurityLevel.MEDIUM: 50,
            SecurityLevel.HIGH: 80,
            SecurityLevel.CRITICAL: 100
        }
        
        return level_scores.get(level, 50)
    
    async def _send_alert(self, event: SecurityEvent) -> None:
        """Send alert for critical security events."""
        if not self.alert_webhook:
            return
        
        try:
            import httpx
            
            alert_data = {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'level': event.level.value,
                'message': event.message or f"Security event: {event.event_type.value}",
                'user_id': event.user_id,
                'ip_address': event.ip_address,
                'risk_score': event.risk_score,
                'details': event.details
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.alert_webhook,
                    json=alert_data,
                    timeout=10
                )
                
                if response.status_code != 200:
                    self.logger.error(
                        f"Failed to send security alert: {response.status_code}"
                    )
        
        except Exception as e:
            self.logger.error(f"Error sending security alert: {str(e)}")
    
    async def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        min_risk_score: Optional[int] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Query security events (requires log parsing or database storage)."""
        # This would require implementing log parsing or storing events in database
        # For now, return empty list
        return []
    
    async def generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
        include_summary: bool = True,
        include_top_users: bool = True,
        include_top_ips: bool = True
    ) -> Dict[str, Any]:
        """Generate security audit report."""
        events = await self.get_events(start_time=start_time, end_time=end_time)
        
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_events': len(events),
            'events_by_type': {},
            'events_by_level': {},
            'high_risk_events': []
        }
        
        # Process events for summary
        for event in events:
            # Count by type
            event_type = event.event_type.value
            report['events_by_type'][event_type] = report['events_by_type'].get(event_type, 0) + 1
            
            # Count by level
            level = event.level.value
            report['events_by_level'][level] = report['events_by_level'].get(level, 0) + 1
            
            # High risk events
            if event.risk_score >= 70:
                report['high_risk_events'].append(asdict(event))
        
        return report


# Global audit logger instance
_audit_logger: Optional[SecurityAuditLogger] = None


def get_audit_logger() -> SecurityAuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger()
    return _audit_logger


# Convenience functions
async def log_login_success(user_id: str, username: str, ip_address: str, session_id: str, **kwargs):
    """Log successful login."""
    logger = get_audit_logger()
    await logger.log_authentication(
        SecurityEventType.LOGIN_SUCCESS,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        session_id=session_id,
        success=True,
        **kwargs
    )


async def log_login_failed(identifier: str, ip_address: str, reason: str = None, **kwargs):
    """Log failed login attempt."""
    logger = get_audit_logger()
    await logger.log_authentication(
        SecurityEventType.LOGIN_FAILED,
        ip_address=ip_address,
        success=False,
        message=f"Login failed for {identifier}",
        details={'identifier': identifier, 'reason': reason},
        **kwargs
    )


async def log_access_denied(user_id: str, username: str, resource: str, action: str, **kwargs):
    """Log access denied event."""
    logger = get_audit_logger()
    await logger.log_authorization(
        SecurityEventType.ACCESS_DENIED,
        user_id=user_id,
        username=username,
        resource=resource,
        action=action,
        success=False,
        **kwargs
    )


async def log_security_incident(incident_type: str, message: str, level: SecurityLevel = SecurityLevel.HIGH, **kwargs):
    """Log security incident."""
    logger = get_audit_logger()
    await logger.log_security_incident(
        SecurityEventType.SUSPICIOUS_ACTIVITY,
        level=level,
        message=f"{incident_type}: {message}",
        **kwargs
    )