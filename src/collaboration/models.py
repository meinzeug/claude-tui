"""
Collaboration Data Models
Extended models for shared workspace management and team coordination
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ..database.models import Base, TimestampMixin, User


class WorkspaceType(Enum):
    """Workspace collaboration types"""
    PRIVATE = "private"
    TEAM = "team" 
    PUBLIC = "public"
    ORGANIZATION = "organization"


class WorkspaceState(Enum):
    """Workspace states"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    LOCKED = "locked"
    MAINTENANCE = "maintenance"


class MemberRole(Enum):
    """Member roles within workspace"""
    OWNER = "owner"
    ADMIN = "admin"
    MAINTAINER = "maintainer"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    GUEST = "guest"


class ConflictType(Enum):
    """Types of collaboration conflicts"""
    CONCURRENT_EDIT = "concurrent_edit"
    MERGE_CONFLICT = "merge_conflict"
    PERMISSION_CONFLICT = "permission_conflict"
    RESOURCE_CONFLICT = "resource_conflict"
    STATE_CONFLICT = "state_conflict"


class ConflictStatus(Enum):
    """Conflict resolution statuses"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    IGNORED = "ignored"


class ActivityType(Enum):
    """Activity types for team analytics"""
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    CONFLICT_RESOLVED = "conflict_resolved"
    MEMBER_JOINED = "member_joined"
    MEMBER_LEFT = "member_left"
    COMMENT_ADDED = "comment_added"
    REVIEW_SUBMITTED = "review_submitted"


class Workspace(Base, TimestampMixin):
    """Shared workspace for multi-developer collaboration"""
    __tablename__ = "workspaces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Workspace configuration
    workspace_type = Column(String(20), default=WorkspaceType.TEAM.value, nullable=False)
    state = Column(String(20), default=WorkspaceState.ACTIVE.value, nullable=False)
    
    # Ownership and access
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)
    
    # Collaboration settings
    max_members = Column(Integer, default=50)
    allow_public_join = Column(Boolean, default=False)
    require_approval = Column(Boolean, default=True)
    enable_real_time = Column(Boolean, default=True)
    enable_conflict_resolution = Column(Boolean, default=True)
    
    # Workspace metadata
    settings = Column(JSON, default=dict)
    last_activity = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    owner = relationship("User", foreign_keys=[owner_id])
    project = relationship("Project", foreign_keys=[project_id])
    members = relationship("WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan")
    sessions = relationship("CollaborationSession", back_populates="workspace", cascade="all, delete-orphan") 
    conflicts = relationship("ConflictResolution", back_populates="workspace", cascade="all, delete-orphan")
    activities = relationship("ActivityFeed", back_populates="workspace", cascade="all, delete-orphan")
    
    # Database constraints
    __table_args__ = (
        Index('ix_workspaces_owner_type', 'owner_id', 'workspace_type'),
        Index('ix_workspaces_state', 'state'),
        Index('ix_workspaces_last_activity', 'last_activity'),
    )
    
    def get_active_members_count(self) -> int:
        """Get count of active members"""
        return sum(1 for m in self.members if m.is_active)
    
    def can_add_member(self) -> bool:
        """Check if workspace can accept new members"""
        return (self.state == WorkspaceState.ACTIVE and 
                self.get_active_members_count() < self.max_members)


class WorkspaceMember(Base, TimestampMixin):
    """Workspace membership with role-based permissions"""
    __tablename__ = "workspace_members"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Member status
    role = Column(String(20), default=MemberRole.DEVELOPER.value, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Membership lifecycle
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Permission overrides (JSON format for granular permissions)
    custom_permissions = Column(JSON, default=dict)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="members")
    user = relationship("User", foreign_keys=[user_id])
    inviter = relationship("User", foreign_keys=[invited_by])
    
    # Database constraints
    __table_args__ = (
        UniqueConstraint('workspace_id', 'user_id', name='uq_workspace_member'),
        Index('ix_workspace_members_workspace_active', 'workspace_id', 'is_active'),
        Index('ix_workspace_members_user_id', 'user_id'),
    )
    
    def has_permission(self, permission: str) -> bool:
        """Check if member has specific permission"""
        # Check custom permissions first
        if permission in self.custom_permissions:
            return self.custom_permissions[permission]
        
        # Default role-based permissions
        role_permissions = {
            MemberRole.OWNER.value: ['all'],
            MemberRole.ADMIN.value: ['read', 'write', 'delete', 'manage_members', 'manage_settings'],
            MemberRole.MAINTAINER.value: ['read', 'write', 'delete', 'manage_conflicts'],
            MemberRole.DEVELOPER.value: ['read', 'write', 'create_tasks'],
            MemberRole.VIEWER.value: ['read'],
            MemberRole.GUEST.value: ['read']
        }
        
        permissions = role_permissions.get(self.role, [])
        return 'all' in permissions or permission in permissions


class CollaborationSession(Base, TimestampMixin):
    """Real-time collaboration sessions for presence tracking"""
    __tablename__ = "collaboration_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session metadata
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Presence data
    is_active = Column(Boolean, default=True)
    current_file = Column(String(500), nullable=True)
    cursor_position = Column(JSON, default=dict)  # Line, column info
    selection_range = Column(JSON, default=dict)  # Start/end selection
    
    # Session lifecycle
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_heartbeat = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="sessions")
    user = relationship("User", foreign_keys=[user_id])
    
    # Database constraints
    __table_args__ = (
        Index('ix_collaboration_sessions_workspace_active', 'workspace_id', 'is_active'),
        Index('ix_collaboration_sessions_user_id', 'user_id'),
        Index('ix_collaboration_sessions_heartbeat', 'last_heartbeat'),
    )
    
    def update_presence(self, file_path: str = None, cursor_pos: Dict = None, selection: Dict = None):
        """Update user presence information"""
        if file_path:
            self.current_file = file_path
        if cursor_pos:
            self.cursor_position = cursor_pos
        if selection:
            self.selection_range = selection
        self.last_heartbeat = datetime.now(timezone.utc)
    
    def is_session_active(self, timeout_minutes: int = 5) -> bool:
        """Check if session is still active based on heartbeat"""
        if not self.is_active or self.ended_at:
            return False
        
        timeout_delta = datetime.now(timezone.utc) - self.last_heartbeat
        return timeout_delta.total_seconds() < (timeout_minutes * 60)


class ConflictResolution(Base, TimestampMixin):
    """Conflict detection and resolution tracking"""
    __tablename__ = "conflict_resolutions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    
    # Conflict identification
    conflict_type = Column(String(30), nullable=False)
    status = Column(String(20), default=ConflictStatus.PENDING.value, nullable=False)
    
    # Involved parties
    affected_users = Column(JSON, default=list)  # List of user IDs
    affected_files = Column(JSON, default=list)  # List of file paths
    
    # Conflict details
    conflict_data = Column(JSON, default=dict)  # Detailed conflict information
    resolution_strategy = Column(String(50), nullable=True)  # Auto/manual resolution method
    
    # Resolution process
    detected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Auto-resolution attempts
    auto_resolution_attempts = Column(Integer, default=0)
    manual_intervention_required = Column(Boolean, default=False)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="conflicts")
    resolver = relationship("User", foreign_keys=[resolved_by])
    
    # Database constraints
    __table_args__ = (
        Index('ix_conflict_resolutions_workspace_status', 'workspace_id', 'status'),
        Index('ix_conflict_resolutions_type', 'conflict_type'),
        Index('ix_conflict_resolutions_detected_at', 'detected_at'),
    )
    
    def mark_resolved(self, resolved_by_id: UUID, notes: str = None, strategy: str = None):
        """Mark conflict as resolved"""
        self.status = ConflictStatus.RESOLVED.value
        self.resolved_at = datetime.now(timezone.utc)
        self.resolved_by = resolved_by_id
        if notes:
            self.resolution_notes = notes
        if strategy:
            self.resolution_strategy = strategy


class ActivityFeed(Base, TimestampMixin):
    """Team activity tracking for analytics and notifications"""
    __tablename__ = "activity_feed"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Activity details
    activity_type = Column(String(30), nullable=False)
    description = Column(Text, nullable=False)
    
    # Activity metadata
    target_resource = Column(String(100), nullable=True)  # File, task, etc.
    metadata = Column(JSON, default=dict)  # Additional activity data
    
    # Visibility and notifications
    is_public = Column(Boolean, default=True)
    notify_team = Column(Boolean, default=False)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="activities")
    user = relationship("User", foreign_keys=[user_id])
    
    # Database constraints
    __table_args__ = (
        Index('ix_activity_feed_workspace_type', 'workspace_id', 'activity_type'),
        Index('ix_activity_feed_user_id', 'user_id'),
        Index('ix_activity_feed_created_at', 'created_at'),
        Index('ix_activity_feed_public', 'is_public'),
    )


class TeamAnalytics(Base, TimestampMixin):
    """Team productivity and collaboration analytics"""
    __tablename__ = "team_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    
    # Analytics period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Team metrics
    active_members = Column(Integer, default=0)
    total_activities = Column(Integer, default=0)
    conflicts_resolved = Column(Integer, default=0)
    files_modified = Column(Integer, default=0)
    
    # Productivity metrics
    avg_session_duration = Column(Integer, default=0)  # Minutes
    collaboration_score = Column(Integer, default=0)  # 0-100 scale
    conflict_rate = Column(Integer, default=0)  # Conflicts per 100 activities
    
    # Detailed metrics (JSON format for flexibility)
    member_contributions = Column(JSON, default=dict)
    activity_breakdown = Column(JSON, default=dict)
    performance_trends = Column(JSON, default=dict)
    
    # Relationships
    workspace = relationship("Workspace", foreign_keys=[workspace_id])
    
    # Database constraints
    __table_args__ = (
        Index('ix_team_analytics_workspace_period', 'workspace_id', 'period_start', 'period_end'),
        UniqueConstraint('workspace_id', 'period_start', 'period_end', name='uq_team_analytics_period'),
    )


class WorkspacePermission(Base, TimestampMixin):
    """Custom workspace permissions system"""
    __tablename__ = "workspace_permissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    
    # Permission definition
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    resource_type = Column(String(50), nullable=False)  # file, task, setting, etc.
    action = Column(String(50), nullable=False)  # read, write, delete, etc.
    
    # Permission scope
    is_global = Column(Boolean, default=False)  # Applies to entire workspace
    resource_pattern = Column(String(500), nullable=True)  # Regex pattern for resources
    
    # Relationships
    workspace = relationship("Workspace", foreign_keys=[workspace_id])
    
    # Database constraints
    __table_args__ = (
        Index('ix_workspace_permissions_workspace_type', 'workspace_id', 'resource_type'),
        UniqueConstraint('workspace_id', 'name', name='uq_workspace_permission_name'),
    )


class WorkspaceRole(Base, TimestampMixin):
    """Custom roles for workspace members"""
    __tablename__ = "workspace_roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    
    # Role definition
    name = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    color = Column(String(7), default="#666666")  # Hex color for UI
    
    # Role configuration
    is_default = Column(Boolean, default=False)
    is_assignable = Column(Boolean, default=True)
    max_members = Column(Integer, nullable=True)  # Max users with this role
    
    # Permission mapping (JSON list of permission IDs or names)
    permissions = Column(JSON, default=list)
    
    # Relationships  
    workspace = relationship("Workspace", foreign_keys=[workspace_id])
    
    # Database constraints
    __table_args__ = (
        Index('ix_workspace_roles_workspace_name', 'workspace_id', 'name'),
        UniqueConstraint('workspace_id', 'name', name='uq_workspace_role_name'),
    )


class Comment(Base, TimestampMixin):
    """Comments and communication within workspace"""
    __tablename__ = "workspace_comments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Comment content
    content = Column(Text, nullable=False)
    content_type = Column(String(20), default="text")  # text, markdown, html
    
    # Comment context
    target_type = Column(String(30), nullable=False)  # file, task, conflict, general
    target_id = Column(String(100), nullable=True)  # Resource identifier
    line_number = Column(Integer, nullable=True)  # For file comments
    
    # Comment threading
    parent_id = Column(UUID(as_uuid=True), ForeignKey("workspace_comments.id"), nullable=True)
    thread_root_id = Column(UUID(as_uuid=True), nullable=True)  # Root comment of thread
    
    # Comment status
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    workspace = relationship("Workspace", foreign_keys=[workspace_id])
    user = relationship("User", foreign_keys=[user_id])
    resolver = relationship("User", foreign_keys=[resolved_by])
    parent = relationship("Comment", foreign_keys=[parent_id], remote_side=[id])
    
    # Database constraints
    __table_args__ = (
        Index('ix_workspace_comments_workspace_target', 'workspace_id', 'target_type', 'target_id'),
        Index('ix_workspace_comments_user_id', 'user_id'),
        Index('ix_workspace_comments_thread', 'thread_root_id'),
    )


class Notification(Base, TimestampMixin):
    """Notification system for team coordination"""
    __tablename__ = "workspace_notifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    recipient_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Notification content
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(30), nullable=False)
    
    # Notification metadata
    priority = Column(String(10), default="medium")  # low, medium, high, urgent
    action_url = Column(String(500), nullable=True)  # Link to relevant resource
    metadata = Column(JSON, default=dict)
    
    # Notification status
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime, nullable=True)
    is_dismissed = Column(Boolean, default=False)
    
    # Relationships
    workspace = relationship("Workspace", foreign_keys=[workspace_id])
    recipient = relationship("User", foreign_keys=[recipient_id])
    
    # Database constraints
    __table_args__ = (
        Index('ix_workspace_notifications_recipient_read', 'recipient_id', 'is_read'),
        Index('ix_workspace_notifications_workspace_type', 'workspace_id', 'notification_type'),
        Index('ix_workspace_notifications_created_at', 'created_at'),
    )