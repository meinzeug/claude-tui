"""
Redis-based Session Management System.

Provides comprehensive session management with Redis backend,
timeout handling, concurrent session limits, and security features.
"""

import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
import redis
from pydantic import BaseModel, Field

from ..core.exceptions import AuthenticationError, ValidationError


class SessionData(BaseModel):
    """Session data model."""
    session_id: str = Field(..., description="Unique session ID")
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    role: str = Field(..., description="User role")
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str = Field(..., description="Client user agent")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    expires_at: datetime = Field(..., description="Session expiration time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    is_active: bool = Field(default=True, description="Session active status")


class SessionStats(BaseModel):
    """Session statistics."""
    total_sessions: int = Field(..., description="Total active sessions")
    user_sessions: int = Field(..., description="Sessions for specific user")
    expired_sessions: int = Field(..., description="Expired sessions")
    active_sessions: int = Field(..., description="Currently active sessions")


class SessionManager:
    """
    Redis-based session management system.
    
    Features:
    - Redis-based session storage
    - Session timeout management
    - Concurrent session limits
    - IP-based security checks
    - Activity tracking
    - Session cleanup
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_timeout_minutes: int = 480,  # 8 hours
        max_sessions_per_user: int = 5,
        cleanup_interval_minutes: int = 60,
        session_key_prefix: str = "session:",
        user_sessions_prefix: str = "user_sessions:",
        stats_key: str = "session_stats"
    ):
        self.redis_client = redis_client or redis.Redis(
            host='localhost', port=6379, db=1, decode_responses=True
        )
        self.session_timeout_minutes = session_timeout_minutes
        self.max_sessions_per_user = max_sessions_per_user
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.session_key_prefix = session_key_prefix
        self.user_sessions_prefix = user_sessions_prefix
        self.stats_key = stats_key
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID."""
        return secrets.token_urlsafe(32)
    
    def _session_key(self, session_id: str) -> str:
        """Get Redis key for session."""
        return f"{self.session_key_prefix}{session_id}"
    
    def _user_sessions_key(self, user_id: str) -> str:
        """Get Redis key for user sessions."""
        return f"{self.user_sessions_prefix}{user_id}"
    
    async def create_session(
        self,
        user_id: str,
        username: str,
        role: str,
        ip_address: str,
        user_agent: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """
        Create a new session.
        
        Args:
            user_id: User ID
            username: Username
            role: User role
            ip_address: Client IP address
            user_agent: Client user agent string
            metadata: Additional session metadata
            
        Returns:
            SessionData: Created session data
            
        Raises:
            AuthenticationError: If session creation fails
        """
        try:
            # Check concurrent session limit
            user_session_count = await self._get_user_session_count(user_id)
            if user_session_count >= self.max_sessions_per_user:
                # Remove oldest session
                await self._cleanup_oldest_user_session(user_id)
            
            # Generate session ID
            session_id = self._generate_session_id()
            
            # Calculate expiration
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(minutes=self.session_timeout_minutes)
            
            # Create session data
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
                username=username,
                role=role,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=now,
                last_activity=now,
                expires_at=expires_at,
                metadata=metadata or {},
                is_active=True
            )
            
            # Store session in Redis
            session_key = self._session_key(session_id)
            session_json = session_data.json()
            
            # Set session with expiration
            ttl_seconds = int(self.session_timeout_minutes * 60)
            await self._redis_setex(session_key, ttl_seconds, session_json)
            
            # Add to user sessions set
            user_sessions_key = self._user_sessions_key(user_id)
            await self._redis_sadd(user_sessions_key, session_id)
            await self._redis_expire(user_sessions_key, ttl_seconds)
            
            # Update stats
            await self._update_session_stats(1, 0)
            
            return session_data
            
        except Exception as e:
            raise AuthenticationError(f"Session creation failed: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            SessionData if found and valid, None otherwise
        """
        try:
            session_key = self._session_key(session_id)
            session_json = await self._redis_get(session_key)
            
            if not session_json:
                return None
            
            session_data = SessionData.parse_raw(session_json)
            
            # Check if session is expired
            if datetime.now(timezone.utc) >= session_data.expires_at:
                await self.invalidate_session(session_id)
                return None
            
            # Check if session is active
            if not session_data.is_active:
                return None
            
            return session_data
            
        except Exception:
            return None
    
    async def update_session_activity(
        self,
        session_id: str,
        ip_address: Optional[str] = None,
        extend_session: bool = True
    ) -> bool:
        """
        Update session last activity and optionally extend expiration.
        
        Args:
            session_id: Session ID
            ip_address: New IP address (for security check)
            extend_session: Whether to extend session expiration
            
        Returns:
            True if updated successfully
        """
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return False
            
            # Security check: IP address change
            if ip_address and ip_address != session_data.ip_address:
                # Log suspicious activity
                await self._log_security_event(
                    session_id,
                    "ip_change",
                    {
                        "old_ip": session_data.ip_address,
                        "new_ip": ip_address,
                        "user_id": session_data.user_id
                    }
                )
                # Optionally invalidate session for security
                # await self.invalidate_session(session_id)
                # return False
            
            # Update activity time
            now = datetime.now(timezone.utc)
            session_data.last_activity = now
            
            if ip_address:
                session_data.ip_address = ip_address
            
            # Extend expiration if requested
            if extend_session:
                session_data.expires_at = now + timedelta(minutes=self.session_timeout_minutes)
            
            # Save updated session
            session_key = self._session_key(session_id)
            session_json = session_data.json()
            
            if extend_session:
                ttl_seconds = int(self.session_timeout_minutes * 60)
                await self._redis_setex(session_key, ttl_seconds, session_json)
            else:
                await self._redis_set(session_key, session_json)
            
            return True
            
        except Exception:
            return False
    
    async def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a session.
        
        Args:
            session_id: Session ID to invalidate
            
        Returns:
            True if invalidated successfully
        """
        try:
            # Get session data first
            session_data = await self.get_session(session_id)
            if not session_data:
                return False
            
            # Remove from Redis
            session_key = self._session_key(session_id)
            await self._redis_delete(session_key)
            
            # Remove from user sessions set
            user_sessions_key = self._user_sessions_key(session_data.user_id)
            await self._redis_srem(user_sessions_key, session_id)
            
            # Update stats
            await self._update_session_stats(-1, 0)
            
            return True
            
        except Exception:
            return False
    
    async def invalidate_user_sessions(
        self,
        user_id: str,
        exclude_session_id: Optional[str] = None
    ) -> int:
        """
        Invalidate all sessions for a user.
        
        Args:
            user_id: User ID
            exclude_session_id: Session ID to exclude from invalidation
            
        Returns:
            Number of sessions invalidated
        """
        try:
            user_sessions_key = self._user_sessions_key(user_id)
            session_ids = await self._redis_smembers(user_sessions_key)
            
            invalidated_count = 0
            for session_id in session_ids:
                if exclude_session_id and session_id == exclude_session_id:
                    continue
                
                if await self.invalidate_session(session_id):
                    invalidated_count += 1
            
            return invalidated_count
            
        except Exception:
            return 0
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active sessions
        """
        try:
            user_sessions_key = self._user_sessions_key(user_id)
            session_ids = await self._redis_smembers(user_sessions_key)
            
            sessions = []
            for session_id in session_ids:
                session_data = await self.get_session(session_id)
                if session_data:
                    sessions.append(session_data)
            
            return sessions
            
        except Exception:
            return []
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Cleanup expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            # Get all session keys
            pattern = f"{self.session_key_prefix}*"
            session_keys = await self._redis_keys(pattern)
            
            now = datetime.now(timezone.utc)
            cleaned_count = 0
            
            for session_key in session_keys:
                try:
                    session_json = await self._redis_get(session_key)
                    if not session_json:
                        continue
                    
                    session_data = SessionData.parse_raw(session_json)
                    
                    # Check if expired
                    if now >= session_data.expires_at:
                        await self.invalidate_session(session_data.session_id)
                        cleaned_count += 1
                        
                except Exception:
                    # Remove corrupted session data
                    await self._redis_delete(session_key)
                    cleaned_count += 1
            
            # Update stats
            if cleaned_count > 0:
                await self._update_session_stats(-cleaned_count, cleaned_count)
            
            return cleaned_count
            
        except Exception:
            return 0
    
    async def get_session_stats(self) -> SessionStats:
        """
        Get session statistics.
        
        Returns:
            SessionStats: Current session statistics
        """
        try:
            # Get stats from Redis
            stats_json = await self._redis_get(self.stats_key)
            
            if stats_json:
                stats_data = json.loads(stats_json)
            else:
                stats_data = {
                    "total_sessions": 0,
                    "expired_sessions": 0
                }
            
            # Count active sessions
            pattern = f"{self.session_key_prefix}*"
            active_sessions = len(await self._redis_keys(pattern))
            
            return SessionStats(
                total_sessions=stats_data.get("total_sessions", 0),
                user_sessions=0,  # Will be calculated per user
                expired_sessions=stats_data.get("expired_sessions", 0),
                active_sessions=active_sessions
            )
            
        except Exception:
            return SessionStats(
                total_sessions=0,
                user_sessions=0,
                expired_sessions=0,
                active_sessions=0
            )
    
    async def _get_user_session_count(self, user_id: str) -> int:
        """Get number of active sessions for user."""
        try:
            user_sessions_key = self._user_sessions_key(user_id)
            return await self._redis_scard(user_sessions_key)
        except Exception:
            return 0
    
    async def _cleanup_oldest_user_session(self, user_id: str) -> bool:
        """Cleanup oldest session for user."""
        try:
            sessions = await self.get_user_sessions(user_id)
            if not sessions:
                return False
            
            # Sort by creation time and remove oldest
            sessions.sort(key=lambda s: s.created_at)
            oldest_session = sessions[0]
            
            return await self.invalidate_session(oldest_session.session_id)
            
        except Exception:
            return False
    
    async def _update_session_stats(self, session_delta: int, expired_delta: int):
        """Update session statistics."""
        try:
            stats_json = await self._redis_get(self.stats_key)
            
            if stats_json:
                stats_data = json.loads(stats_json)
            else:
                stats_data = {"total_sessions": 0, "expired_sessions": 0}
            
            stats_data["total_sessions"] = max(0, stats_data["total_sessions"] + session_delta)
            stats_data["expired_sessions"] = max(0, stats_data["expired_sessions"] + expired_delta)
            
            await self._redis_set(self.stats_key, json.dumps(stats_data))
            
        except Exception:
            pass
    
    async def _log_security_event(
        self,
        session_id: str,
        event_type: str,
        details: Dict[str, Any]
    ):
        """Log security event."""
        try:
            event_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "event_type": event_type,
                "details": details
            }
            
            # Store in Redis with expiration (30 days)
            event_key = f"security_event:{secrets.token_hex(8)}"
            await self._redis_setex(
                event_key,
                30 * 24 * 3600,  # 30 days
                json.dumps(event_data)
            )
            
        except Exception:
            pass
    
    # Redis helper methods
    async def _redis_set(self, key: str, value: str) -> bool:
        """Set key-value pair."""
        try:
            return self.redis_client.set(key, value)
        except Exception:
            return False
    
    async def _redis_setex(self, key: str, ttl: int, value: str) -> bool:
        """Set key-value pair with expiration."""
        try:
            return self.redis_client.setex(key, ttl, value)
        except Exception:
            return False
    
    async def _redis_get(self, key: str) -> Optional[str]:
        """Get value by key."""
        try:
            return self.redis_client.get(key)
        except Exception:
            return None
    
    async def _redis_delete(self, key: str) -> int:
        """Delete key."""
        try:
            return self.redis_client.delete(key)
        except Exception:
            return 0
    
    async def _redis_expire(self, key: str, ttl: int) -> bool:
        """Set key expiration."""
        try:
            return self.redis_client.expire(key, ttl)
        except Exception:
            return False
    
    async def _redis_keys(self, pattern: str) -> List[str]:
        """Get keys by pattern."""
        try:
            return self.redis_client.keys(pattern)
        except Exception:
            return []
    
    async def _redis_sadd(self, key: str, value: str) -> int:
        """Add to set."""
        try:
            return self.redis_client.sadd(key, value)
        except Exception:
            return 0
    
    async def _redis_srem(self, key: str, value: str) -> int:
        """Remove from set."""
        try:
            return self.redis_client.srem(key, value)
        except Exception:
            return 0
    
    async def _redis_smembers(self, key: str) -> List[str]:
        """Get set members."""
        try:
            return list(self.redis_client.smembers(key))
        except Exception:
            return []
    
    async def _redis_scard(self, key: str) -> int:
        """Get set cardinality."""
        try:
            return self.redis_client.scard(key)
        except Exception:
            return 0