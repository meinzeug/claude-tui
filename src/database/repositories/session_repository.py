"""
Session Repository Implementation

Provides user session management with:
- Session creation and validation
- Token management
- Session cleanup and security
- Activity tracking
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, and_
from sqlalchemy.sql import Select

from .base import BaseRepository, RepositoryError
from ..models import UserSession, User
from ...core.logger import get_logger

logger = get_logger(__name__)


class SessionRepository(BaseRepository[UserSession]):
    """User session repository for authentication tracking and management."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize session repository.
        
        Args:
            session: AsyncSession instance
        """
        super().__init__(session, UserSession)
    
    def _add_relationship_loading(self, query: Select) -> Select:
        """
        Add eager loading for session relationships.
        
        Args:
            query: SQLAlchemy query
            
        Returns:
            Query with relationship loading options
        """
        return query.options(
            selectinload(UserSession.user)
        )
    
    async def create_session(
        self,
        user_id: uuid.UUID,
        session_token: str,
        refresh_token: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        location: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> Optional[UserSession]:
        """
        Create new user session with security tracking.
        
        Args:
            user_id: User ID
            session_token: Unique session token
            refresh_token: Optional refresh token
            ip_address: Client IP address
            user_agent: Client user agent string
            location: Geographic location (optional)
            expires_at: Session expiration time
            
        Returns:
            Created UserSession instance
            
        Raises:
            RepositoryError: If creation fails
        """
        try:
            # Validate user exists
            user_exists = await self.session.scalar(
                select(User.id).where(User.id == user_id)
            )
            
            if not user_exists:
                raise RepositoryError(
                    "User not found for session creation",
                    "SESSION_USER_NOT_FOUND",
                    {"user_id": str(user_id)}
                )
            
            # Set default expiration if not provided (24 hours)
            if not expires_at:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
            
            # Check for existing session with same token
            existing_session = await self.session.scalar(
                select(UserSession.id).where(UserSession.session_token == session_token)
            )
            
            if existing_session:
                raise RepositoryError(
                    "Session token already exists",
                    "SESSION_TOKEN_EXISTS",
                    {"session_token": session_token[:8] + "..."}
                )
            
            # Create session
            user_session = UserSession(
                user_id=user_id,
                session_token=session_token,
                refresh_token=refresh_token,
                ip_address=ip_address,
                user_agent=user_agent,
                location=location,
                expires_at=expires_at,
                last_activity=datetime.now(timezone.utc)
            )
            
            self.session.add(user_session)
            await self.session.flush()
            await self.session.refresh(user_session)
            
            self.logger.info(
                f"Created session for user",
                extra={
                    "session_id": str(user_session.id),
                    "user_id": str(user_id),
                    "ip_address": ip_address,
                    "expires_at": expires_at.isoformat(),
                    "location": location
                }
            )
            
            return user_session
            
        except RepositoryError:
            await self.session.rollback()
            raise
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error creating session: {e}")
            raise RepositoryError(
                "Failed to create user session",
                "CREATE_SESSION_ERROR",
                {
                    "user_id": str(user_id),
                    "ip_address": ip_address,
                    "error": str(e)
                }
            )
    
    async def get_active_session(
        self, 
        session_token: str,
        update_activity: bool = True
    ) -> Optional[UserSession]:
        """
        Get active session by token with optional activity update.
        
        Args:
            session_token: Session token
            update_activity: Whether to update last activity timestamp
            
        Returns:
            UserSession instance if active, None otherwise
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            result = await self.session.execute(
                select(UserSession)
                .options(selectinload(UserSession.user))
                .where(
                    and_(
                        UserSession.session_token == session_token,
                        UserSession.is_active == True,
                        UserSession.expires_at > datetime.now(timezone.utc)
                    )
                )
            )
            
            user_session = result.scalar_one_or_none()
            
            if user_session and update_activity:
                # Update last activity
                user_session.last_activity = datetime.now(timezone.utc)
                await self.session.flush()
                
                self.logger.debug(
                    f"Updated session activity",
                    extra={
                        "session_id": str(user_session.id),
                        "user_id": str(user_session.user_id)
                    }
                )
            
            return user_session
            
        except Exception as e:
            self.logger.error(f"Error getting active session: {e}")
            raise RepositoryError(
                "Failed to retrieve active session",
                "GET_ACTIVE_SESSION_ERROR",
                {"session_token": session_token[:8] + "...", "error": str(e)}
            )
    
    async def get_user_sessions(
        self,
        user_id: uuid.UUID,
        active_only: bool = True,
        limit: int = 50
    ) -> List[UserSession]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User ID
            active_only: Whether to return only active sessions
            limit: Maximum number of sessions to return
            
        Returns:
            List of UserSession instances for the user
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            filters = {'user_id': user_id}
            
            if active_only:
                filters.update({
                    'is_active': True,
                    'expires_at__gt': datetime.now(timezone.utc)
                })
            
            sessions = await self.get_all(
                filters=filters,
                limit=limit,
                order_by='last_activity',
                order_desc=True,
                load_relationships=True
            )
            
            self.logger.debug(
                f"Retrieved {len(sessions)} sessions for user {user_id}",
                extra={
                    "user_id": str(user_id),
                    "active_only": active_only,
                    "session_count": len(sessions)
                }
            )
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"Error getting user sessions for {user_id}: {e}")
            raise RepositoryError(
                "Failed to retrieve user sessions",
                "GET_USER_SESSIONS_ERROR",
                {"user_id": str(user_id), "active_only": active_only, "error": str(e)}
            )
    
    async def invalidate_session(
        self,
        session_id: uuid.UUID,
        reason: Optional[str] = None
    ) -> bool:
        """
        Invalidate a specific session.
        
        Args:
            session_id: Session ID
            reason: Reason for invalidation (for logging)
            
        Returns:
            True if session was invalidated, False if not found
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            user_session = await self.get_by_id(session_id)
            if not user_session:
                self.logger.warning(f"Session {session_id} not found for invalidation")
                return False
            
            # Invalidate session
            user_session.is_active = False
            user_session.updated_at = datetime.now(timezone.utc)
            
            await self.session.flush()
            
            self.logger.info(
                f"Session invalidated",
                extra={
                    "session_id": str(session_id),
                    "user_id": str(user_session.user_id),
                    "reason": reason or "manual_invalidation"
                }
            )
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error invalidating session {session_id}: {e}")
            raise RepositoryError(
                "Failed to invalidate session",
                "INVALIDATE_SESSION_ERROR",
                {"session_id": str(session_id), "reason": reason, "error": str(e)}
            )
    
    async def invalidate_user_sessions(
        self,
        user_id: uuid.UUID,
        exclude_session_id: Optional[uuid.UUID] = None,
        reason: Optional[str] = None
    ) -> int:
        """
        Invalidate all sessions for a user.
        
        Args:
            user_id: User ID
            exclude_session_id: Session ID to exclude from invalidation
            reason: Reason for invalidation (for logging)
            
        Returns:
            Number of sessions invalidated
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            # Build update query
            query = (
                update(UserSession)
                .where(
                    and_(
                        UserSession.user_id == user_id,
                        UserSession.is_active == True
                    )
                )
                .values(
                    is_active=False,
                    updated_at=datetime.now(timezone.utc)
                )
            )
            
            # Exclude specific session if provided
            if exclude_session_id:
                query = query.where(UserSession.id != exclude_session_id)
            
            result = await self.session.execute(query)
            invalidated_count = result.rowcount
            
            await self.session.flush()
            
            self.logger.info(
                f"Invalidated {invalidated_count} sessions for user",
                extra={
                    "user_id": str(user_id),
                    "exclude_session_id": str(exclude_session_id) if exclude_session_id else None,
                    "reason": reason or "user_invalidation",
                    "invalidated_count": invalidated_count
                }
            )
            
            return invalidated_count
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error invalidating sessions for user {user_id}: {e}")
            raise RepositoryError(
                "Failed to invalidate user sessions",
                "INVALIDATE_USER_SESSIONS_ERROR",
                {
                    "user_id": str(user_id),
                    "exclude_session_id": str(exclude_session_id) if exclude_session_id else None,
                    "error": str(e)
                }
            )
    
    async def extend_session(
        self,
        session_id: uuid.UUID,
        extend_hours: int = 24
    ) -> bool:
        """
        Extend session expiration time.
        
        Args:
            session_id: Session ID
            extend_hours: Number of hours to extend
            
        Returns:
            True if session was extended, False if not found or inactive
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            user_session = await self.get_by_id(session_id)
            if not user_session or not user_session.is_active:
                self.logger.warning(f"Session {session_id} not found or inactive for extension")
                return False
            
            # Extend expiration
            new_expiration = datetime.now(timezone.utc) + timedelta(hours=extend_hours)
            user_session.expires_at = new_expiration
            user_session.last_activity = datetime.now(timezone.utc)
            user_session.updated_at = datetime.now(timezone.utc)
            
            await self.session.flush()
            
            self.logger.info(
                f"Session extended",
                extra={
                    "session_id": str(session_id),
                    "user_id": str(user_session.user_id),
                    "new_expiration": new_expiration.isoformat(),
                    "extend_hours": extend_hours
                }
            )
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error extending session {session_id}: {e}")
            raise RepositoryError(
                "Failed to extend session",
                "EXTEND_SESSION_ERROR",
                {"session_id": str(session_id), "extend_hours": extend_hours, "error": str(e)}
            )
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
            
        Raises:
            RepositoryError: If cleanup fails
        """
        try:
            now = datetime.now(timezone.utc)
            
            # Update expired sessions to inactive
            result = await self.session.execute(
                update(UserSession)
                .where(
                    and_(
                        UserSession.expires_at < now,
                        UserSession.is_active == True
                    )
                )
                .values(
                    is_active=False,
                    updated_at=now
                )
            )
            
            cleaned_count = result.rowcount
            await self.session.flush()
            
            self.logger.info(
                f"Cleaned up {cleaned_count} expired sessions",
                extra={"cleaned_count": cleaned_count, "cleanup_time": now.isoformat()}
            )
            
            return cleaned_count
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error cleaning up expired sessions: {e}")
            raise RepositoryError(
                "Failed to cleanup expired sessions",
                "CLEANUP_EXPIRED_SESSIONS_ERROR",
                {"error": str(e)}
            )
    
    async def get_session_statistics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get session statistics for monitoring.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with session statistics
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            now = datetime.now(timezone.utc)
            
            # Get all sessions in the period
            sessions = await self.get_all(
                filters={'created_at__gte': start_date},
                limit=100000  # Large limit for statistics
            )
            
            # Calculate statistics
            total_sessions = len(sessions)
            active_sessions = len([s for s in sessions if s.is_active and s.expires_at > now])
            expired_sessions = len([s for s in sessions if s.expires_at <= now])
            
            # User activity
            unique_users = len(set(s.user_id for s in sessions))
            
            # IP address tracking
            unique_ips = len(set(s.ip_address for s in sessions if s.ip_address))
            
            # Average session duration (for completed sessions)
            completed_sessions = [s for s in sessions if not s.is_active and s.last_activity]
            avg_duration = None
            
            if completed_sessions:
                durations = [
                    (s.last_activity - s.created_at).total_seconds()
                    for s in completed_sessions
                    if s.last_activity > s.created_at
                ]
                avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Daily session counts
            daily_counts = {}
            for session in sessions:
                day = session.created_at.date().isoformat()
                daily_counts[day] = daily_counts.get(day, 0) + 1
            
            statistics = {
                'period_days': days,
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'expired_sessions': expired_sessions,
                'unique_users': unique_users,
                'unique_ips': unique_ips,
                'average_duration_seconds': round(avg_duration, 2) if avg_duration else None,
                'daily_session_counts': daily_counts,
                'generated_at': now,
                'start_date': start_date,
                'end_date': now
            }
            
            self.logger.debug(
                f"Generated session statistics for {days} days",
                extra={
                    "days": days,
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "unique_users": unique_users
                }
            )
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error getting session statistics: {e}")
            raise RepositoryError(
                "Failed to get session statistics",
                "GET_SESSION_STATISTICS_ERROR",
                {"days": days, "error": str(e)}
            )
    
    async def get_suspicious_sessions(
        self,
        days: int = 7,
        max_sessions_per_user: int = 10,
        max_sessions_per_ip: int = 20
    ) -> Dict[str, List[UserSession]]:
        """
        Get potentially suspicious session activity.
        
        Args:
            days: Number of days to analyze
            max_sessions_per_user: Threshold for user session count
            max_sessions_per_ip: Threshold for IP session count
            
        Returns:
            Dictionary with suspicious session categories
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get recent sessions
            sessions = await self.get_all(
                filters={'created_at__gte': start_date},
                limit=10000,
                load_relationships=True
            )
            
            # Group by user and IP
            user_sessions = {}
            ip_sessions = {}
            
            for session in sessions:
                # Group by user
                user_id = str(session.user_id)
                if user_id not in user_sessions:
                    user_sessions[user_id] = []
                user_sessions[user_id].append(session)
                
                # Group by IP
                if session.ip_address:
                    if session.ip_address not in ip_sessions:
                        ip_sessions[session.ip_address] = []
                    ip_sessions[session.ip_address].append(session)
            
            # Find suspicious activity
            suspicious_users = [
                session for sessions_list in user_sessions.values()
                if len(sessions_list) > max_sessions_per_user
                for session in sessions_list
            ]
            
            suspicious_ips = [
                session for sessions_list in ip_sessions.values()
                if len(sessions_list) > max_sessions_per_ip
                for session in sessions_list
            ]
            
            # Find concurrent sessions from different locations
            concurrent_location_sessions = []
            for user_id, sessions_list in user_sessions.items():
                active_sessions = [
                    s for s in sessions_list
                    if s.is_active and s.expires_at > datetime.now(timezone.utc)
                ]
                
                if len(active_sessions) > 1:
                    # Check for different locations/IPs
                    locations = set(s.location for s in active_sessions if s.location)
                    ips = set(s.ip_address for s in active_sessions if s.ip_address)
                    
                    if len(locations) > 1 or len(ips) > 1:
                        concurrent_location_sessions.extend(active_sessions)
            
            suspicious_activity = {
                'high_user_activity': suspicious_users,
                'high_ip_activity': suspicious_ips,
                'concurrent_locations': concurrent_location_sessions
            }
            
            self.logger.debug(
                f"Identified suspicious session activity",
                extra={
                    "days": days,
                    "suspicious_users_count": len(suspicious_users),
                    "suspicious_ips_count": len(suspicious_ips),
                    "concurrent_locations_count": len(concurrent_location_sessions)
                }
            )
            
            return suspicious_activity
            
        except Exception as e:
            self.logger.error(f"Error getting suspicious sessions: {e}")
            raise RepositoryError(
                "Failed to get suspicious sessions",
                "GET_SUSPICIOUS_SESSIONS_ERROR",
                {"days": days, "error": str(e)}
            )
