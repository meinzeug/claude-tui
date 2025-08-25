"""
Session Management Service with Redis for Claude-TIU

Provides secure session management with Redis backend,
device tracking, and suspicious activity detection.
"""

import hashlib
import secrets
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import logging
from ipaddress import ip_address, AddressValueError

from ..core.exceptions import AuthenticationError, SecurityError
from ..database.models import User, UserSession

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPICIOUS = "suspicious"


@dataclass
class SessionMetadata:
    """Session metadata structure"""
    session_id: str
    user_id: str
    user_agent: str
    ip_address: str
    device_fingerprint: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime = None
    last_activity: datetime = None
    expires_at: datetime = None
    status: SessionStatus = SessionStatus.ACTIVE
    device_trusted: bool = False
    login_method: str = "password"  # password, oauth, mfa
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc)
        if not self.last_activity:
            self.last_activity = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, SessionStatus):
                data[key] = value.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """Create from dictionary retrieved from Redis"""
        # Convert ISO strings back to datetime objects
        datetime_fields = ['created_at', 'last_activity', 'expires_at']
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        if 'status' in data:
            data['status'] = SessionStatus(data['status'])
        
        return cls(**data)


@dataclass
class DeviceInfo:
    """Device information for tracking"""
    device_id: str
    device_name: str
    device_type: str  # web, mobile, desktop
    os_info: str
    browser_info: str
    first_seen: datetime
    last_seen: datetime
    trusted: bool = False
    location_history: List[str] = None
    
    def __post_init__(self):
        if self.location_history is None:
            self.location_history = []


class SessionService:
    """
    Enhanced session management service with Redis backend.
    
    Features:
    - Session creation and validation
    - Device fingerprinting and tracking
    - Suspicious activity detection
    - Session cleanup and management
    - IP-based location tracking
    - Multi-device session management
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_ttl_hours: int = 24,
        max_sessions_per_user: int = 10,
        suspicious_threshold: int = 3,
        device_trust_days: int = 30
    ):
        self.redis_client = redis_client or redis.Redis(
            host='localhost', port=6379, db=1, decode_responses=True
        )
        self.session_ttl_hours = session_ttl_hours
        self.max_sessions_per_user = max_sessions_per_user
        self.suspicious_threshold = suspicious_threshold
        self.device_trust_days = device_trust_days
        
        # Redis key prefixes
        self.session_prefix = "session:"
        self.user_sessions_prefix = "user_sessions:"
        self.device_prefix = "device:"
        self.suspicious_prefix = "suspicious:"
        self.location_prefix = "location:"
        
        logger.info("Session service initialized with TTL %d hours", session_ttl_hours)
    
    async def create_session(
        self,
        user: User,
        ip_address: str,
        user_agent: str,
        device_fingerprint: Optional[str] = None,
        location: Optional[str] = None,
        login_method: str = "password"
    ) -> str:
        """
        Create a new user session with security checks.
        
        Args:
            user: User object
            ip_address: Client IP address
            user_agent: Client user agent
            device_fingerprint: Device fingerprint hash
            location: Client location (optional)
            login_method: Authentication method used
            
        Returns:
            Session ID
            
        Raises:
            SecurityError: If security checks fail
        """
        try:
            # Generate secure session ID
            session_id = self._generate_session_id()
            
            # Validate IP address
            if not self._is_valid_ip(ip_address):
                raise SecurityError("Invalid IP address")
            
            # Check for suspicious activity
            await self._check_suspicious_activity(str(user.id), ip_address)
            
            # Create session metadata
            metadata = SessionMetadata(
                session_id=session_id,
                user_id=str(user.id),
                user_agent=user_agent,
                ip_address=ip_address,
                device_fingerprint=device_fingerprint,
                location=location,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=self.session_ttl_hours),
                login_method=login_method
            )
            
            # Check if device is trusted
            if device_fingerprint:
                metadata.device_trusted = await self._is_device_trusted(str(user.id), device_fingerprint)
            
            # Store session
            await self._store_session(metadata)
            
            # Update user sessions list
            await self._add_user_session(str(user.id), session_id)
            
            # Update device tracking
            if device_fingerprint:
                await self._update_device_tracking(str(user.id), device_fingerprint, metadata)
            
            # Enforce session limits
            await self._enforce_session_limits(str(user.id))
            
            # Log session creation
            logger.info(
                "Session created for user %s from %s (method: %s)",
                user.username, ip_address, login_method
            )
            
            return session_id
            
        except Exception as e:
            logger.error("Session creation failed: %s", e)
            raise SecurityError(f"Session creation failed: {str(e)}")
    
    async def validate_session(self, session_id: str, ip_address: Optional[str] = None) -> Optional[SessionMetadata]:
        """
        Validate session and return metadata.
        
        Args:
            session_id: Session ID to validate
            ip_address: Current client IP (optional)
            
        Returns:
            SessionMetadata if valid, None otherwise
        """
        try:
            # Get session data
            session_key = f"{self.session_prefix}{session_id}"
            session_data = await self._redis_hgetall(session_key)
            
            if not session_data:
                return None
            
            # Parse metadata
            metadata = SessionMetadata.from_dict(session_data)
            
            # Check if expired
            if metadata.expires_at and datetime.now(timezone.utc) >= metadata.expires_at:
                await self._revoke_session(session_id, "expired")
                return None
            
            # Check if revoked
            if metadata.status != SessionStatus.ACTIVE:
                return None
            
            # IP address validation (optional - can be disabled for mobile apps)
            if ip_address and metadata.ip_address != ip_address:
                # Log potential session hijacking
                logger.warning(
                    "IP address mismatch for session %s: %s != %s",
                    session_id[:8], metadata.ip_address, ip_address
                )
                # Mark as suspicious but don't immediately revoke
                await self._mark_session_suspicious(session_id, "ip_mismatch")
            
            # Update last activity
            await self._update_session_activity(session_id)
            
            return metadata
            
        except Exception as e:
            logger.error("Session validation failed: %s", e)
            return None
    
    async def revoke_session(self, session_id: str, reason: str = "manual") -> bool:
        """
        Revoke a specific session.
        
        Args:
            session_id: Session ID to revoke
            reason: Revocation reason
            
        Returns:
            True if successful
        """
        return await self._revoke_session(session_id, reason)
    
    async def revoke_all_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """
        Revoke all sessions for a user.
        
        Args:
            user_id: User ID
            except_session: Session ID to keep active (optional)
            
        Returns:
            Number of sessions revoked
        """
        try:
            user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
            session_ids = await self._redis_smembers(user_sessions_key)
            
            revoked_count = 0
            for session_id in session_ids:
                if except_session and session_id == except_session:
                    continue
                
                if await self._revoke_session(session_id, "user_revoked"):
                    revoked_count += 1
            
            logger.info("Revoked %d sessions for user %s", revoked_count, user_id)
            return revoked_count
            
        except Exception as e:
            logger.error("Failed to revoke user sessions: %s", e)
            return 0
    
    async def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[SessionMetadata]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User ID
            active_only: Return only active sessions
            
        Returns:
            List of session metadata
        """
        try:
            user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
            session_ids = await self._redis_smembers(user_sessions_key)
            
            sessions = []
            for session_id in session_ids:
                session_key = f"{self.session_prefix}{session_id}"
                session_data = await self._redis_hgetall(session_key)
                
                if session_data:
                    metadata = SessionMetadata.from_dict(session_data)
                    
                    # Filter by status
                    if active_only and metadata.status != SessionStatus.ACTIVE:
                        continue
                    
                    sessions.append(metadata)
            
            # Sort by last activity
            sessions.sort(key=lambda x: x.last_activity, reverse=True)
            return sessions
            
        except Exception as e:
            logger.error("Failed to get user sessions: %s", e)
            return []
    
    async def get_user_devices(self, user_id: str) -> List[DeviceInfo]:
        """
        Get all known devices for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of device information
        """
        try:
            pattern = f"{self.device_prefix}{user_id}:*"
            device_keys = await self._redis_keys(pattern)
            
            devices = []
            for device_key in device_keys:
                device_data = await self._redis_hgetall(device_key)
                if device_data:
                    # Parse device data (simplified)
                    device_info = DeviceInfo(
                        device_id=device_data.get('device_id', ''),
                        device_name=device_data.get('device_name', 'Unknown'),
                        device_type=device_data.get('device_type', 'web'),
                        os_info=device_data.get('os_info', ''),
                        browser_info=device_data.get('browser_info', ''),
                        first_seen=datetime.fromisoformat(device_data.get('first_seen', datetime.now(timezone.utc).isoformat())),
                        last_seen=datetime.fromisoformat(device_data.get('last_seen', datetime.now(timezone.utc).isoformat())),
                        trusted=device_data.get('trusted', 'false') == 'true'
                    )
                    devices.append(device_info)
            
            return devices
            
        except Exception as e:
            logger.error("Failed to get user devices: %s", e)
            return []
    
    async def trust_device(self, user_id: str, device_fingerprint: str) -> bool:
        """
        Mark a device as trusted.
        
        Args:
            user_id: User ID
            device_fingerprint: Device fingerprint
            
        Returns:
            True if successful
        """
        try:
            device_key = f"{self.device_prefix}{user_id}:{device_fingerprint}"
            await self._redis_hset(device_key, "trusted", "true")
            await self._redis_hset(device_key, "trusted_at", datetime.now(timezone.utc).isoformat())
            
            # Set expiration for device trust
            await self._redis_expire(device_key, self.device_trust_days * 24 * 3600)
            
            logger.info("Device trusted for user %s", user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to trust device: %s", e)
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned
        """
        try:
            pattern = f"{self.session_prefix}*"
            session_keys = await self._redis_keys(pattern)
            
            cleaned_count = 0
            now = datetime.now(timezone.utc)
            
            for session_key in session_keys:
                session_data = await self._redis_hgetall(session_key)
                if session_data:
                    expires_at_str = session_data.get('expires_at')
                    if expires_at_str:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if now >= expires_at:
                            session_id = session_key.replace(self.session_prefix, '')
                            await self._revoke_session(session_id, "expired")
                            cleaned_count += 1
            
            logger.info("Cleaned up %d expired sessions", cleaned_count)
            return cleaned_count
            
        except Exception as e:
            logger.error("Session cleanup failed: %s", e)
            return 0
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format"""
        try:
            ip_address(ip)
            return True
        except AddressValueError:
            return False
    
    async def _check_suspicious_activity(self, user_id: str, ip_address: str) -> None:
        """Check for suspicious login activity"""
        try:
            # Check recent failed attempts from this IP
            suspicious_key = f"{self.suspicious_prefix}{ip_address}"
            attempts = await self._redis_get(suspicious_key)
            
            if attempts and int(attempts) >= self.suspicious_threshold:
                raise SecurityError("Too many suspicious attempts from this IP")
            
            # Check for rapid session creation
            user_sessions = await self.get_user_sessions(user_id, active_only=True)
            recent_sessions = [
                s for s in user_sessions
                if s.created_at > datetime.now(timezone.utc) - timedelta(minutes=5)
            ]
            
            if len(recent_sessions) >= 3:
                logger.warning("Rapid session creation detected for user %s", user_id)
                # Don't block, but log for monitoring
                
        except SecurityError:
            raise
        except Exception as e:
            logger.warning("Suspicious activity check failed: %s", e)
    
    async def _store_session(self, metadata: SessionMetadata) -> None:
        """Store session metadata in Redis"""
        session_key = f"{self.session_prefix}{metadata.session_id}"
        session_data = metadata.to_dict()
        
        await self._redis_hmset(session_key, session_data)
        await self._redis_expire(session_key, self.session_ttl_hours * 3600)
    
    async def _add_user_session(self, user_id: str, session_id: str) -> None:
        """Add session to user's session set"""
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        await self._redis_sadd(user_sessions_key, session_id)
        await self._redis_expire(user_sessions_key, self.session_ttl_hours * 3600)
    
    async def _revoke_session(self, session_id: str, reason: str) -> bool:
        """Revoke a session"""
        try:
            session_key = f"{self.session_prefix}{session_id}"
            session_data = await self._redis_hgetall(session_key)
            
            if not session_data:
                return False
            
            # Update status
            await self._redis_hset(session_key, "status", SessionStatus.REVOKED.value)
            await self._redis_hset(session_key, "revoked_at", datetime.now(timezone.utc).isoformat())
            await self._redis_hset(session_key, "revocation_reason", reason)
            
            # Remove from user sessions set
            user_id = session_data.get('user_id')
            if user_id:
                user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
                await self._redis_srem(user_sessions_key, session_id)
            
            logger.info("Session %s revoked: %s", session_id[:8], reason)
            return True
            
        except Exception as e:
            logger.error("Failed to revoke session: %s", e)
            return False
    
    async def _update_session_activity(self, session_id: str) -> None:
        """Update session last activity timestamp"""
        session_key = f"{self.session_prefix}{session_id}"
        await self._redis_hset(
            session_key,
            "last_activity",
            datetime.now(timezone.utc).isoformat()
        )
    
    async def _mark_session_suspicious(self, session_id: str, reason: str) -> None:
        """Mark session as suspicious"""
        session_key = f"{self.session_prefix}{session_id}"
        await self._redis_hset(session_key, "status", SessionStatus.SUSPICIOUS.value)
        await self._redis_hset(session_key, "suspicious_reason", reason)
        await self._redis_hset(session_key, "suspicious_at", datetime.now(timezone.utc).isoformat())
        
        logger.warning("Session %s marked suspicious: %s", session_id[:8], reason)
    
    async def _is_device_trusted(self, user_id: str, device_fingerprint: str) -> bool:
        """Check if device is trusted"""
        device_key = f"{self.device_prefix}{user_id}:{device_fingerprint}"
        trusted = await self._redis_hget(device_key, "trusted")
        return trusted == "true"
    
    async def _update_device_tracking(self, user_id: str, device_fingerprint: str, metadata: SessionMetadata) -> None:
        """Update device tracking information"""
        device_key = f"{self.device_prefix}{user_id}:{device_fingerprint}"
        
        # Check if device exists
        exists = await self._redis_exists(device_key)
        
        device_data = {
            "device_id": device_fingerprint,
            "last_seen": metadata.created_at.isoformat(),
            "last_ip": metadata.ip_address,
            "last_location": metadata.location or "",
            "user_agent": metadata.user_agent
        }
        
        if not exists:
            device_data["first_seen"] = metadata.created_at.isoformat()
            device_data["device_name"] = self._parse_device_name(metadata.user_agent)
            device_data["device_type"] = self._parse_device_type(metadata.user_agent)
            device_data["os_info"] = self._parse_os_info(metadata.user_agent)
            device_data["browser_info"] = self._parse_browser_info(metadata.user_agent)
        
        await self._redis_hmset(device_key, device_data)
        await self._redis_expire(device_key, self.device_trust_days * 24 * 3600)
    
    async def _enforce_session_limits(self, user_id: str) -> None:
        """Enforce maximum sessions per user"""
        try:
            sessions = await self.get_user_sessions(user_id, active_only=True)
            
            if len(sessions) > self.max_sessions_per_user:
                # Sort by last activity and revoke oldest sessions
                sessions.sort(key=lambda x: x.last_activity)
                sessions_to_revoke = sessions[:len(sessions) - self.max_sessions_per_user]
                
                for session in sessions_to_revoke:
                    await self._revoke_session(session.session_id, "session_limit_exceeded")
                
                logger.info(
                    "Revoked %d oldest sessions for user %s (limit: %d)",
                    len(sessions_to_revoke), user_id, self.max_sessions_per_user
                )
        except Exception as e:
            logger.error("Failed to enforce session limits: %s", e)
    
    def _parse_device_name(self, user_agent: str) -> str:
        """Parse device name from user agent"""
        # Simplified parsing - in production use a proper library
        if "iPhone" in user_agent:
            return "iPhone"
        elif "iPad" in user_agent:
            return "iPad"
        elif "Android" in user_agent:
            return "Android Device"
        elif "Windows" in user_agent:
            return "Windows PC"
        elif "Mac" in user_agent:
            return "Mac"
        else:
            return "Unknown Device"
    
    def _parse_device_type(self, user_agent: str) -> str:
        """Parse device type from user agent"""
        if any(mobile in user_agent for mobile in ["iPhone", "Android", "Mobile"]):
            return "mobile"
        elif "iPad" in user_agent:
            return "tablet"
        else:
            return "web"
    
    def _parse_os_info(self, user_agent: str) -> str:
        """Parse OS information from user agent"""
        # Simplified - use a proper library in production
        if "Windows NT" in user_agent:
            return "Windows"
        elif "Mac OS X" in user_agent:
            return "macOS"
        elif "iPhone OS" in user_agent or "iOS" in user_agent:
            return "iOS"
        elif "Android" in user_agent:
            return "Android"
        elif "Linux" in user_agent:
            return "Linux"
        else:
            return "Unknown"
    
    def _parse_browser_info(self, user_agent: str) -> str:
        """Parse browser information from user agent"""
        if "Chrome" in user_agent:
            return "Chrome"
        elif "Firefox" in user_agent:
            return "Firefox"
        elif "Safari" in user_agent and "Chrome" not in user_agent:
            return "Safari"
        elif "Edge" in user_agent:
            return "Edge"
        else:
            return "Unknown"
    
    # Redis helper methods
    async def _redis_hgetall(self, key: str) -> Dict[str, str]:
        """Get all hash fields"""
        try:
            return self.redis_client.hgetall(key)
        except Exception:
            return {}
    
    async def _redis_hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set hash fields"""
        try:
            return self.redis_client.hset(key, mapping=mapping)
        except Exception:
            return False
    
    async def _redis_hset(self, key: str, field: str, value: str) -> bool:
        """Set hash field"""
        try:
            return self.redis_client.hset(key, field, value)
        except Exception:
            return False
    
    async def _redis_hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field"""
        try:
            return self.redis_client.hget(key, field)
        except Exception:
            return None
    
    async def _redis_expire(self, key: str, ttl: int) -> bool:
        """Set key expiration"""
        try:
            return self.redis_client.expire(key, ttl)
        except Exception:
            return False
    
    async def _redis_exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return self.redis_client.exists(key) > 0
        except Exception:
            return False
    
    async def _redis_sadd(self, key: str, member: str) -> bool:
        """Add to set"""
        try:
            return self.redis_client.sadd(key, member)
        except Exception:
            return False
    
    async def _redis_srem(self, key: str, member: str) -> bool:
        """Remove from set"""
        try:
            return self.redis_client.srem(key, member)
        except Exception:
            return False
    
    async def _redis_smembers(self, key: str) -> List[str]:
        """Get set members"""
        try:
            return list(self.redis_client.smembers(key))
        except Exception:
            return []
    
    async def _redis_keys(self, pattern: str) -> List[str]:
        """Get keys by pattern"""
        try:
            return self.redis_client.keys(pattern)
        except Exception:
            return []
    
    async def _redis_get(self, key: str) -> Optional[str]:
        """Get key value"""
        try:
            return self.redis_client.get(key)
        except Exception:
            return None