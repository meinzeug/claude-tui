"""
Presence Manager - Real-time user presence and activity tracking
Manages user presence indicators, cursor positions, and activity status
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from .models import (
    CollaborationSession, Workspace, WorkspaceMember,
    ActivityFeed, ActivityType
)
from .sync_engine import UserPresence

logger = logging.getLogger(__name__)


class PresenceStatus(Enum):
    """User presence status"""
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class DetailedPresence:
    """Detailed user presence information"""
    user_id: str
    username: str
    full_name: str
    status: str
    current_file: Optional[str]
    current_line: int
    current_column: int
    selection_start: Dict[str, int]
    selection_end: Dict[str, int]
    is_editing: bool
    last_activity: str
    session_duration: int  # seconds
    color: str
    avatar_url: Optional[str]
    
    # Additional context
    active_files: List[str]
    recent_activities: List[Dict[str, Any]]
    typing_indicator: bool


class PresenceManager:
    """
    Manages real-time user presence and activity tracking.
    Provides comprehensive presence information and activity indicators.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize presence manager.
        
        Args:
            db_session: Database session for operations
        """
        self.db = db_session
        
        # Presence tracking
        self._user_presence: Dict[UUID, DetailedPresence] = {}
        self._workspace_presence: Dict[UUID, Set[UUID]] = {}  # workspace_id -> user_ids
        self._file_presence: Dict[str, Set[UUID]] = {}  # file_path -> user_ids
        
        # Activity tracking
        self._user_activities: Dict[UUID, List[Dict[str, Any]]] = {}
        self._typing_indicators: Dict[UUID, Dict[str, datetime]] = {}  # user_id -> file_path -> timestamp
        
        # Presence update callbacks
        self._presence_callbacks: List[callable] = []
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._update_task: Optional[asyncio.Task] = None
        
        logger.info("Presence manager initialized")
    
    async def start(self) -> None:
        """Start presence manager background tasks"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_inactive_presence())
        
        if self._update_task is None:
            self._update_task = asyncio.create_task(self._update_presence_status())
        
        logger.info("Presence manager started")
    
    async def stop(self) -> None:
        """Stop presence manager background tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        
        if self._update_task:
            self._update_task.cancel()
            self._update_task = None
        
        logger.info("Presence manager stopped")
    
    async def update_user_presence(
        self,
        workspace_id: UUID,
        user_id: UUID,
        presence_data: Dict[str, Any]
    ) -> None:
        """
        Update user presence information.
        
        Args:
            workspace_id: Workspace ID
            user_id: User ID
            presence_data: Presence update data
        """
        # Get or create presence record
        if user_id not in self._user_presence:
            await self._initialize_user_presence(workspace_id, user_id)
        
        presence = self._user_presence[user_id]
        
        # Update presence data
        if 'current_file' in presence_data:
            old_file = presence.current_file
            new_file = presence_data['current_file']
            
            # Update file presence tracking
            if old_file and old_file in self._file_presence:
                self._file_presence[old_file].discard(user_id)
            
            if new_file:
                if new_file not in self._file_presence:
                    self._file_presence[new_file] = set()
                self._file_presence[new_file].add(user_id)
            
            presence.current_file = new_file
        
        if 'cursor_position' in presence_data:
            cursor_pos = presence_data['cursor_position']
            presence.current_line = cursor_pos.get('line', 0)
            presence.current_column = cursor_pos.get('column', 0)
        
        if 'selection' in presence_data:
            selection = presence_data['selection']
            presence.selection_start = selection.get('start', {'line': 0, 'column': 0})
            presence.selection_end = selection.get('end', {'line': 0, 'column': 0})
        
        if 'is_editing' in presence_data:
            presence.is_editing = presence_data['is_editing']
        
        if 'status' in presence_data:
            presence.status = presence_data['status']
        
        # Update last activity
        presence.last_activity = datetime.now(timezone.utc).isoformat()
        
        # Update database session
        await self._update_collaboration_session(workspace_id, user_id, presence_data)
        
        # Trigger callbacks
        await self._notify_presence_change(workspace_id, user_id, presence)
        
        logger.debug(f"Updated presence for user {user_id} in workspace {workspace_id}")
    
    async def set_typing_indicator(
        self,
        workspace_id: UUID,
        user_id: UUID,
        file_path: str,
        is_typing: bool
    ) -> None:
        """
        Set typing indicator for user in file.
        
        Args:
            workspace_id: Workspace ID
            user_id: User ID
            file_path: File being edited
            is_typing: Whether user is typing
        """
        if user_id not in self._typing_indicators:
            self._typing_indicators[user_id] = {}
        
        if is_typing:
            self._typing_indicators[user_id][file_path] = datetime.now(timezone.utc)
        else:
            self._typing_indicators[user_id].pop(file_path, None)
        
        # Update presence
        if user_id in self._user_presence:
            self._user_presence[user_id].typing_indicator = is_typing
            
            # Notify other users
            await self._notify_typing_indicator(workspace_id, user_id, file_path, is_typing)
        
        logger.debug(f"Set typing indicator for user {user_id} in {file_path}: {is_typing}")
    
    async def get_workspace_presence(
        self,
        workspace_id: UUID,
        requester_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """
        Get presence information for all users in workspace.
        
        Args:
            workspace_id: Workspace ID
            requester_id: Optional user requesting the information
            
        Returns:
            List of user presence information
        """
        if workspace_id not in self._workspace_presence:
            return []
        
        user_ids = self._workspace_presence[workspace_id]
        presence_list = []
        
        for user_id in user_ids:
            if user_id in self._user_presence:
                presence = self._user_presence[user_id]
                
                # Filter sensitive information based on permissions
                presence_data = asdict(presence)
                
                # Add session information
                session = await self._get_active_session(workspace_id, user_id)
                if session:
                    presence_data['session_start'] = session.started_at.isoformat()
                    presence_data['last_heartbeat'] = session.last_heartbeat.isoformat()
                
                presence_list.append(presence_data)
        
        return presence_list
    
    async def get_file_presence(
        self,
        file_path: str,
        workspace_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """
        Get presence information for users working on specific file.
        
        Args:
            file_path: File path
            workspace_id: Optional workspace filter
            
        Returns:
            List of user presence for the file
        """
        if file_path not in self._file_presence:
            return []
        
        user_ids = self._file_presence[file_path]
        file_presence = []
        
        for user_id in user_ids:
            if user_id in self._user_presence:
                presence = self._user_presence[user_id]
                
                # Filter to current file activity
                file_presence_data = {
                    'user_id': presence.user_id,
                    'username': presence.username,
                    'full_name': presence.full_name,
                    'status': presence.status,
                    'current_line': presence.current_line,
                    'current_column': presence.current_column,
                    'selection_start': presence.selection_start,
                    'selection_end': presence.selection_end,
                    'is_editing': presence.is_editing,
                    'color': presence.color,
                    'typing_indicator': presence.typing_indicator,
                    'last_activity': presence.last_activity
                }
                
                file_presence.append(file_presence_data)
        
        return file_presence
    
    async def get_user_presence(
        self,
        workspace_id: UUID,
        user_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed presence information for specific user.
        
        Args:
            workspace_id: Workspace ID
            user_id: User ID
            
        Returns:
            User presence information or None
        """
        if user_id not in self._user_presence:
            return None
        
        presence = self._user_presence[user_id]
        presence_data = asdict(presence)
        
        # Add additional context
        session = await self._get_active_session(workspace_id, user_id)
        if session:
            presence_data['session_info'] = {
                'started_at': session.started_at.isoformat(),
                'last_heartbeat': session.last_heartbeat.isoformat(),
                'ip_address': session.ip_address,
                'user_agent': session.user_agent
            }
        
        # Add recent activities
        if user_id in self._user_activities:
            presence_data['recent_activities'] = self._user_activities[user_id][-10:]
        
        return presence_data
    
    async def track_user_activity(
        self,
        workspace_id: UUID,
        user_id: UUID,
        activity_type: str,
        activity_data: Dict[str, Any]
    ) -> None:
        """
        Track user activity for presence context.
        
        Args:
            workspace_id: Workspace ID
            user_id: User ID
            activity_type: Type of activity
            activity_data: Activity details
        """
        activity = {
            'type': activity_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data': activity_data
        }
        
        # Add to user activities
        if user_id not in self._user_activities:
            self._user_activities[user_id] = []
        
        self._user_activities[user_id].append(activity)
        
        # Keep only recent activities (last 100)
        if len(self._user_activities[user_id]) > 100:
            self._user_activities[user_id] = self._user_activities[user_id][-100:]
        
        # Update presence last activity
        if user_id in self._user_presence:
            self._user_presence[user_id].last_activity = activity['timestamp']
        
        # Create activity feed entry for significant activities
        if activity_type in ['file_created', 'file_modified', 'task_completed']:
            activity_feed = ActivityFeed(
                workspace_id=workspace_id,
                user_id=user_id,
                activity_type=activity_type,
                description=activity_data.get('description', f'User performed {activity_type}'),
                metadata=activity_data,
                is_public=True
            )
            self.db.add(activity_feed)
            self.db.commit()
        
        logger.debug(f"Tracked activity {activity_type} for user {user_id}")
    
    async def end_user_sessions(
        self,
        workspace_id: UUID,
        user_id: UUID
    ) -> None:
        """
        End all active sessions for user in workspace.
        
        Args:
            workspace_id: Workspace ID
            user_id: User ID
        """
        # End database sessions
        sessions = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.user_id == user_id,
                CollaborationSession.is_active == True
            )
        ).all()
        
        for session in sessions:
            session.is_active = False
            session.ended_at = datetime.now(timezone.utc)
        
        self.db.commit()
        
        # Clean up presence tracking
        if user_id in self._user_presence:
            presence = self._user_presence[user_id]
            
            # Remove from file presence
            if presence.current_file and presence.current_file in self._file_presence:
                self._file_presence[presence.current_file].discard(user_id)
            
            # Remove from workspace presence
            if workspace_id in self._workspace_presence:
                self._workspace_presence[workspace_id].discard(user_id)
            
            # Remove presence record
            del self._user_presence[user_id]
        
        # Clean up activities and typing indicators
        self._user_activities.pop(user_id, None)
        self._typing_indicators.pop(user_id, None)
        
        logger.info(f"Ended all sessions for user {user_id} in workspace {workspace_id}")
    
    async def get_presence_analytics(
        self,
        workspace_id: UUID,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get presence and activity analytics for workspace.
        
        Args:
            workspace_id: Workspace ID
            days: Number of days to analyze
            
        Returns:
            Analytics data
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Query sessions in time period
        sessions = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.started_at >= cutoff_date
            )
        ).all()
        
        # Query activities
        activities = self.db.query(ActivityFeed).filter(
            and_(
                ActivityFeed.workspace_id == workspace_id,
                ActivityFeed.created_at >= cutoff_date
            )
        ).all()
        
        # Calculate metrics
        total_sessions = len(sessions)
        unique_users = len(set(session.user_id for session in sessions))
        
        # Session duration analytics
        session_durations = []
        for session in sessions:
            if session.ended_at:
                duration = (session.ended_at - session.started_at).total_seconds()
                session_durations.append(duration)
        
        avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        # Activity analytics
        activity_by_type = {}
        activity_by_user = {}
        
        for activity in activities:
            activity_type = activity.activity_type
            activity_by_type[activity_type] = activity_by_type.get(activity_type, 0) + 1
            
            user_id = str(activity.user_id)
            activity_by_user[user_id] = activity_by_user.get(user_id, 0) + 1
        
        # Current presence
        current_presence = await self.get_workspace_presence(workspace_id)
        active_users = len([p for p in current_presence if p['status'] == PresenceStatus.ONLINE.value])
        
        return {
            'total_sessions': total_sessions,
            'unique_users': unique_users,
            'avg_session_duration_minutes': round(avg_session_duration / 60, 2),
            'total_activities': len(activities),
            'activity_by_type': activity_by_type,
            'most_active_users': dict(sorted(activity_by_user.items(), key=lambda x: x[1], reverse=True)[:10]),
            'current_active_users': active_users,
            'current_total_users': len(current_presence),
            'analysis_period_days': days
        }
    
    def add_presence_callback(self, callback: callable) -> None:
        """Add callback for presence changes"""
        self._presence_callbacks.append(callback)
    
    def remove_presence_callback(self, callback: callable) -> None:
        """Remove presence change callback"""
        if callback in self._presence_callbacks:
            self._presence_callbacks.remove(callback)
    
    # Private methods
    
    async def _initialize_user_presence(
        self,
        workspace_id: UUID,
        user_id: UUID
    ) -> None:
        """Initialize presence record for user"""
        # Get user info
        member = self.db.query(WorkspaceMember).join(
            WorkspaceMember.user
        ).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.is_active == True
            )
        ).first()
        
        if not member:
            return
        
        user = member.user
        
        # Create presence record
        presence = DetailedPresence(
            user_id=str(user_id),
            username=user.username,
            full_name=user.full_name or user.username,
            status=PresenceStatus.ONLINE.value,
            current_file=None,
            current_line=0,
            current_column=0,
            selection_start={'line': 0, 'column': 0},
            selection_end={'line': 0, 'column': 0},
            is_editing=False,
            last_activity=datetime.now(timezone.utc).isoformat(),
            session_duration=0,
            color=f"#{hash(str(user_id)) % 0xFFFFFF:06x}",
            avatar_url=None,
            active_files=[],
            recent_activities=[],
            typing_indicator=False
        )
        
        self._user_presence[user_id] = presence
        
        # Add to workspace presence
        if workspace_id not in self._workspace_presence:
            self._workspace_presence[workspace_id] = set()
        self._workspace_presence[workspace_id].add(user_id)
        
        logger.debug(f"Initialized presence for user {user_id}")
    
    async def _update_collaboration_session(
        self,
        workspace_id: UUID,
        user_id: UUID,
        presence_data: Dict[str, Any]
    ) -> None:
        """Update database collaboration session"""
        session = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.user_id == user_id,
                CollaborationSession.is_active == True
            )
        ).first()
        
        if session:
            if 'current_file' in presence_data:
                session.current_file = presence_data['current_file']
            
            if 'cursor_position' in presence_data:
                session.cursor_position = presence_data['cursor_position']
            
            if 'selection' in presence_data:
                session.selection_range = presence_data['selection']
            
            session.last_heartbeat = datetime.now(timezone.utc)
            self.db.commit()
    
    async def _get_active_session(
        self,
        workspace_id: UUID,
        user_id: UUID
    ) -> Optional[CollaborationSession]:
        """Get active collaboration session"""
        return self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.user_id == user_id,
                CollaborationSession.is_active == True
            )
        ).first()
    
    async def _notify_presence_change(
        self,
        workspace_id: UUID,
        user_id: UUID,
        presence: DetailedPresence
    ) -> None:
        """Notify callbacks of presence change"""
        for callback in self._presence_callbacks:
            try:
                await callback(workspace_id, user_id, asdict(presence))
            except Exception as e:
                logger.error(f"Error in presence callback: {e}")
    
    async def _notify_typing_indicator(
        self,
        workspace_id: UUID,
        user_id: UUID,
        file_path: str,
        is_typing: bool
    ) -> None:
        """Notify of typing indicator change"""
        # This would integrate with sync engine to broadcast typing indicators
        pass
    
    async def _cleanup_inactive_presence(self) -> None:
        """Background task to clean up inactive presence"""
        while True:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=5)
                
                # Find inactive presence records
                inactive_users = []
                for user_id, presence in self._user_presence.items():
                    last_activity = datetime.fromisoformat(presence.last_activity)
                    if last_activity < cutoff_time:
                        inactive_users.append(user_id)
                
                # Clean up inactive users
                for user_id in inactive_users:
                    presence = self._user_presence[user_id]
                    
                    # Update status to away or offline
                    if presence.status == PresenceStatus.ONLINE.value:
                        presence.status = PresenceStatus.AWAY.value
                    elif presence.status == PresenceStatus.AWAY.value:
                        # Remove completely after being away
                        await self._remove_user_presence(user_id)
                
                # Clean up typing indicators
                for user_id in list(self._typing_indicators.keys()):
                    user_typing = self._typing_indicators[user_id]
                    for file_path in list(user_typing.keys()):
                        if user_typing[file_path] < cutoff_time:
                            del user_typing[file_path]
                    
                    if not user_typing:
                        del self._typing_indicators[user_id]
                
                if inactive_users:
                    logger.debug(f"Cleaned up {len(inactive_users)} inactive presence records")
                
            except Exception as e:
                logger.error(f"Error in presence cleanup: {e}")
            
            await asyncio.sleep(60)  # Run every minute
    
    async def _update_presence_status(self) -> None:
        """Background task to update presence status"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                for user_id, presence in self._user_presence.items():
                    # Update session duration
                    if presence.status == PresenceStatus.ONLINE.value:
                        last_activity = datetime.fromisoformat(presence.last_activity)
                        presence.session_duration = int((current_time - last_activity).total_seconds())
                
            except Exception as e:
                logger.error(f"Error updating presence status: {e}")
            
            await asyncio.sleep(30)  # Run every 30 seconds
    
    async def _remove_user_presence(self, user_id: UUID) -> None:
        """Remove user from all presence tracking"""
        if user_id not in self._user_presence:
            return
        
        presence = self._user_presence[user_id]
        
        # Remove from file presence
        if presence.current_file and presence.current_file in self._file_presence:
            self._file_presence[presence.current_file].discard(user_id)
            
            # Clean up empty file presence
            if not self._file_presence[presence.current_file]:
                del self._file_presence[presence.current_file]
        
        # Remove from workspace presence
        for workspace_id in self._workspace_presence:
            self._workspace_presence[workspace_id].discard(user_id)
        
        # Remove presence record
        del self._user_presence[user_id]
        
        # Clean up activities and typing
        self._user_activities.pop(user_id, None)
        self._typing_indicators.pop(user_id, None)
        
        logger.debug(f"Removed presence for user {user_id}")