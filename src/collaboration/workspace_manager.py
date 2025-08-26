"""
Workspace Manager - Core workspace orchestration and management
Handles workspace lifecycle, member management, and coordination
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from .models import (
    Workspace, WorkspaceMember, WorkspaceType, WorkspaceState, 
    MemberRole, ActivityFeed, ActivityType, TeamAnalytics,
    WorkspacePermission, WorkspaceRole, Comment, Notification
)
from ..database.models import User, Project
from ..auth.rbac import rbac_manager
from .sync_engine import SynchronizationEngine
from .presence_manager import PresenceManager
from .analytics_engine import AnalyticsEngine

logger = logging.getLogger(__name__)


class WorkspaceManagerException(Exception):
    """Workspace management related errors"""
    pass


class WorkspaceManager:
    """
    Central workspace management system for multi-developer collaboration.
    Orchestrates workspace lifecycle, member management, permissions, and coordination.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize workspace manager.
        
        Args:
            db_session: Database session for operations
        """
        self.db = db_session
        self.sync_engine = SynchronizationEngine(db_session)
        self.presence_manager = PresenceManager(db_session)
        self.analytics_engine = AnalyticsEngine(db_session)
        
        # Cache for active workspaces and permissions
        self._workspace_cache: Dict[UUID, Dict[str, Any]] = {}
        self._permission_cache: Dict[UUID, Dict[str, Set[str]]] = {}
        
        logger.info("Workspace Manager initialized")
    
    async def create_workspace(
        self,
        name: str,
        owner_id: UUID,
        workspace_type: WorkspaceType = WorkspaceType.TEAM,
        description: str = "",
        project_id: Optional[UUID] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Workspace:
        """
        Create new collaborative workspace.
        
        Args:
            name: Workspace name
            owner_id: Workspace owner user ID
            workspace_type: Type of workspace collaboration
            description: Workspace description
            project_id: Associated project ID
            settings: Workspace configuration settings
            
        Returns:
            Created workspace instance
        """
        logger.info(f"Creating workspace '{name}' for owner {owner_id}")
        
        try:
            # Validate owner exists
            owner = self.db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise WorkspaceManagerException(f"Owner user {owner_id} not found")
            
            # Validate project if specified
            if project_id:
                project = self.db.query(Project).filter(Project.id == project_id).first()
                if not project:
                    raise WorkspaceManagerException(f"Project {project_id} not found")
                
                # Check if owner has access to project
                if project.owner_id != owner_id:
                    raise WorkspaceManagerException("Owner must have access to associated project")
            
            # Create workspace
            workspace = Workspace(
                name=name,
                description=description,
                workspace_type=workspace_type.value,
                owner_id=owner_id,
                project_id=project_id,
                settings=settings or {},
                state=WorkspaceState.ACTIVE.value
            )
            
            self.db.add(workspace)
            self.db.flush()  # Get workspace ID
            
            # Add owner as admin member
            owner_member = WorkspaceMember(
                workspace_id=workspace.id,
                user_id=owner_id,
                role=MemberRole.OWNER.value,
                joined_at=datetime.now(timezone.utc)
            )
            self.db.add(owner_member)
            
            # Create initial activity
            activity = ActivityFeed(
                workspace_id=workspace.id,
                user_id=owner_id,
                activity_type=ActivityType.MEMBER_JOINED.value,
                description=f"Workspace '{name}' created by {owner.username}",
                is_public=True,
                notify_team=True
            )
            self.db.add(activity)
            
            # Initialize default roles and permissions
            await self._setup_default_roles_and_permissions(workspace.id)
            
            # Initialize analytics
            await self.analytics_engine.initialize_workspace_analytics(workspace.id)
            
            self.db.commit()
            
            # Cache workspace
            self._cache_workspace(workspace)
            
            logger.info(f"Workspace '{name}' created successfully with ID {workspace.id}")
            return workspace
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create workspace '{name}': {e}")
            raise WorkspaceManagerException(f"Workspace creation failed: {e}") from e
    
    async def invite_member(
        self,
        workspace_id: UUID,
        user_id: UUID,
        inviter_id: UUID,
        role: MemberRole = MemberRole.DEVELOPER,
        custom_permissions: Optional[Dict[str, bool]] = None
    ) -> WorkspaceMember:
        """
        Invite user to workspace with specified role.
        
        Args:
            workspace_id: Workspace to invite to
            user_id: User to invite
            inviter_id: User performing the invitation
            role: Role to assign to new member
            custom_permissions: Optional custom permission overrides
            
        Returns:
            Created workspace member
        """
        logger.info(f"Inviting user {user_id} to workspace {workspace_id}")
        
        try:
            # Validate workspace and permissions
            workspace = await self._get_workspace_with_validation(workspace_id, inviter_id)
            if not workspace.can_add_member():
                raise WorkspaceManagerException("Workspace cannot accept new members")
            
            # Check inviter permissions
            if not await self._check_member_permission(workspace_id, inviter_id, "manage_members"):
                raise WorkspaceManagerException("Insufficient permissions to invite members")
            
            # Validate target user
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise WorkspaceManagerException(f"User {user_id} not found")
            
            # Check if already a member
            existing = self.db.query(WorkspaceMember).filter(
                and_(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.user_id == user_id,
                    WorkspaceMember.is_active == True
                )
            ).first()
            
            if existing:
                raise WorkspaceManagerException("User is already a member")
            
            # Create membership
            member = WorkspaceMember(
                workspace_id=workspace_id,
                user_id=user_id,
                role=role.value,
                invited_by=inviter_id,
                custom_permissions=custom_permissions or {}
            )
            
            self.db.add(member)
            
            # Create activity
            inviter = self.db.query(User).filter(User.id == inviter_id).first()
            activity = ActivityFeed(
                workspace_id=workspace_id,
                user_id=inviter_id,
                activity_type=ActivityType.MEMBER_JOINED.value,
                description=f"{inviter.username} invited {user.username} as {role.value}",
                metadata={"invited_user": str(user_id), "role": role.value},
                is_public=True,
                notify_team=True
            )
            self.db.add(activity)
            
            # Create notification for invited user
            notification = Notification(
                workspace_id=workspace_id,
                recipient_id=user_id,
                title=f"Invited to {workspace.name}",
                message=f"You've been invited to join workspace '{workspace.name}' as {role.value}",
                notification_type="workspace_invitation",
                priority="medium"
            )
            self.db.add(notification)
            
            self.db.commit()
            
            # Clear permission cache
            self._clear_permission_cache(workspace_id)
            
            logger.info(f"User {user_id} successfully invited to workspace {workspace_id}")
            return member
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to invite member: {e}")
            raise WorkspaceManagerException(f"Member invitation failed: {e}") from e
    
    async def update_member_role(
        self,
        workspace_id: UUID,
        member_user_id: UUID,
        new_role: MemberRole,
        updater_id: UUID,
        custom_permissions: Optional[Dict[str, bool]] = None
    ) -> WorkspaceMember:
        """
        Update workspace member role and permissions.
        
        Args:
            workspace_id: Workspace ID
            member_user_id: User whose role to update
            new_role: New role to assign
            updater_id: User performing the update
            custom_permissions: Optional permission overrides
            
        Returns:
            Updated workspace member
        """
        logger.info(f"Updating role for user {member_user_id} in workspace {workspace_id}")
        
        try:
            # Validate permissions
            if not await self._check_member_permission(workspace_id, updater_id, "manage_members"):
                raise WorkspaceManagerException("Insufficient permissions to update member roles")
            
            # Get member
            member = self.db.query(WorkspaceMember).filter(
                and_(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.user_id == member_user_id,
                    WorkspaceMember.is_active == True
                )
            ).first()
            
            if not member:
                raise WorkspaceManagerException("Member not found in workspace")
            
            # Prevent self-demotion from owner
            if (member.user_id == updater_id and 
                member.role == MemberRole.OWNER.value and 
                new_role != MemberRole.OWNER):
                raise WorkspaceManagerException("Owner cannot demote themselves")
            
            old_role = member.role
            member.role = new_role.value
            if custom_permissions:
                member.custom_permissions = custom_permissions
            
            # Create activity
            updater = self.db.query(User).filter(User.id == updater_id).first()
            user = self.db.query(User).filter(User.id == member_user_id).first()
            
            activity = ActivityFeed(
                workspace_id=workspace_id,
                user_id=updater_id,
                activity_type="role_updated",
                description=f"{updater.username} changed {user.username}'s role from {old_role} to {new_role.value}",
                metadata={"member_user": str(member_user_id), "old_role": old_role, "new_role": new_role.value},
                is_public=True
            )
            self.db.add(activity)
            
            self.db.commit()
            
            # Clear permission cache
            self._clear_permission_cache(workspace_id)
            
            logger.info(f"Role updated successfully for user {member_user_id}")
            return member
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update member role: {e}")
            raise WorkspaceManagerException(f"Role update failed: {e}") from e
    
    async def remove_member(
        self,
        workspace_id: UUID,
        member_user_id: UUID,
        remover_id: UUID,
        reason: str = ""
    ) -> bool:
        """
        Remove member from workspace.
        
        Args:
            workspace_id: Workspace ID
            member_user_id: User to remove
            remover_id: User performing the removal
            reason: Optional removal reason
            
        Returns:
            True if removal was successful
        """
        logger.info(f"Removing user {member_user_id} from workspace {workspace_id}")
        
        try:
            # Validate permissions
            if not await self._check_member_permission(workspace_id, remover_id, "manage_members"):
                raise WorkspaceManagerException("Insufficient permissions to remove members")
            
            # Get member
            member = self.db.query(WorkspaceMember).filter(
                and_(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.user_id == member_user_id,
                    WorkspaceMember.is_active == True
                )
            ).first()
            
            if not member:
                raise WorkspaceManagerException("Member not found in workspace")
            
            # Prevent removing the only owner
            if member.role == MemberRole.OWNER.value:
                owner_count = self.db.query(WorkspaceMember).filter(
                    and_(
                        WorkspaceMember.workspace_id == workspace_id,
                        WorkspaceMember.role == MemberRole.OWNER.value,
                        WorkspaceMember.is_active == True
                    )
                ).count()
                
                if owner_count <= 1:
                    raise WorkspaceManagerException("Cannot remove the only owner")
            
            # Deactivate member
            member.is_active = False
            
            # End any active collaboration sessions
            await self.presence_manager.end_user_sessions(workspace_id, member_user_id)
            
            # Create activity
            remover = self.db.query(User).filter(User.id == remover_id).first()
            user = self.db.query(User).filter(User.id == member_user_id).first()
            
            activity = ActivityFeed(
                workspace_id=workspace_id,
                user_id=remover_id,
                activity_type=ActivityType.MEMBER_LEFT.value,
                description=f"{remover.username} removed {user.username} from workspace",
                metadata={"removed_user": str(member_user_id), "reason": reason},
                is_public=True
            )
            self.db.add(activity)
            
            self.db.commit()
            
            # Clear caches
            self._clear_permission_cache(workspace_id)
            
            logger.info(f"User {member_user_id} successfully removed from workspace {workspace_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to remove member: {e}")
            return False
    
    async def get_workspace_members(
        self,
        workspace_id: UUID,
        requester_id: UUID,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get list of workspace members with their details.
        
        Args:
            workspace_id: Workspace ID
            requester_id: User requesting the information
            include_inactive: Include inactive members
            
        Returns:
            List of member details
        """
        # Validate access
        if not await self._check_member_permission(workspace_id, requester_id, "read"):
            raise WorkspaceManagerException("Insufficient permissions to view members")
        
        # Build query
        query = self.db.query(WorkspaceMember, User).join(User, WorkspaceMember.user_id == User.id)
        query = query.filter(WorkspaceMember.workspace_id == workspace_id)
        
        if not include_inactive:
            query = query.filter(WorkspaceMember.is_active == True)
        
        members = query.all()
        
        # Get presence information
        member_list = []
        for member, user in members:
            presence_info = await self.presence_manager.get_user_presence(workspace_id, user.id)
            
            member_info = {
                "id": str(user.id),
                "username": user.username,
                "full_name": user.full_name,
                "email": user.email,
                "role": member.role,
                "is_active": member.is_active,
                "joined_at": member.joined_at.isoformat(),
                "last_seen": member.last_seen.isoformat() if member.last_seen else None,
                "custom_permissions": member.custom_permissions,
                "presence": presence_info
            }
            member_list.append(member_info)
        
        return sorted(member_list, key=lambda m: m["joined_at"])
    
    async def get_workspace_details(
        self,
        workspace_id: UUID,
        requester_id: UUID
    ) -> Dict[str, Any]:
        """
        Get comprehensive workspace details.
        
        Args:
            workspace_id: Workspace ID
            requester_id: User requesting the information
            
        Returns:
            Workspace details with members, analytics, and activity
        """
        # Validate access
        workspace = await self._get_workspace_with_validation(workspace_id, requester_id)
        
        # Get members
        members = await self.get_workspace_members(workspace_id, requester_id)
        
        # Get recent activity
        recent_activities = self.db.query(ActivityFeed).filter(
            ActivityFeed.workspace_id == workspace_id
        ).order_by(ActivityFeed.created_at.desc()).limit(20).all()
        
        activities = [
            {
                "id": str(activity.id),
                "type": activity.activity_type,
                "description": activity.description,
                "user_id": str(activity.user_id),
                "created_at": activity.created_at.isoformat(),
                "metadata": activity.metadata
            }
            for activity in recent_activities
        ]
        
        # Get analytics summary
        analytics = await self.analytics_engine.get_workspace_summary(workspace_id)
        
        return {
            "id": str(workspace.id),
            "name": workspace.name,
            "description": workspace.description,
            "workspace_type": workspace.workspace_type,
            "state": workspace.state,
            "owner_id": str(workspace.owner_id),
            "project_id": str(workspace.project_id) if workspace.project_id else None,
            "created_at": workspace.created_at.isoformat(),
            "last_activity": workspace.last_activity.isoformat(),
            "settings": workspace.settings,
            "member_count": len([m for m in members if m["is_active"]]),
            "max_members": workspace.max_members,
            "members": members,
            "recent_activities": activities,
            "analytics": analytics
        }
    
    async def list_user_workspaces(
        self,
        user_id: UUID,
        workspace_type: Optional[WorkspaceType] = None,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all workspaces user is a member of.
        
        Args:
            user_id: User ID
            workspace_type: Optional filter by workspace type
            include_inactive: Include inactive workspaces
            
        Returns:
            List of workspace summaries
        """
        # Build query
        query = (self.db.query(Workspace, WorkspaceMember)
                .join(WorkspaceMember, Workspace.id == WorkspaceMember.workspace_id)
                .filter(WorkspaceMember.user_id == user_id))
        
        if not include_inactive:
            query = query.filter(
                and_(
                    WorkspaceMember.is_active == True,
                    Workspace.state == WorkspaceState.ACTIVE.value
                )
            )
        
        if workspace_type:
            query = query.filter(Workspace.workspace_type == workspace_type.value)
        
        results = query.all()
        
        workspaces = []
        for workspace, member in results:
            # Get member count
            member_count = (self.db.query(WorkspaceMember)
                          .filter(
                              and_(
                                  WorkspaceMember.workspace_id == workspace.id,
                                  WorkspaceMember.is_active == True
                              )
                          ).count())
            
            workspace_info = {
                "id": str(workspace.id),
                "name": workspace.name,
                "description": workspace.description,
                "workspace_type": workspace.workspace_type,
                "state": workspace.state,
                "member_role": member.role,
                "is_owner": member.user_id == workspace.owner_id,
                "member_count": member_count,
                "last_activity": workspace.last_activity.isoformat(),
                "joined_at": member.joined_at.isoformat()
            }
            workspaces.append(workspace_info)
        
        return sorted(workspaces, key=lambda w: w["last_activity"], reverse=True)
    
    async def _get_workspace_with_validation(
        self,
        workspace_id: UUID,
        user_id: UUID
    ) -> Workspace:
        """Get workspace and validate user access"""
        workspace = self.db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not workspace:
            raise WorkspaceManagerException(f"Workspace {workspace_id} not found")
        
        # Check if user is a member
        member = self.db.query(WorkspaceMember).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.is_active == True
            )
        ).first()
        
        if not member and workspace.workspace_type != WorkspaceType.PUBLIC.value:
            raise WorkspaceManagerException("Access denied to workspace")
        
        return workspace
    
    async def _check_member_permission(
        self,
        workspace_id: UUID,
        user_id: UUID,
        permission: str
    ) -> bool:
        """Check if member has specific permission in workspace"""
        # Check cache first
        cache_key = f"{workspace_id}_{user_id}"
        if cache_key in self._permission_cache:
            return permission in self._permission_cache[cache_key].get("permissions", set())
        
        # Get member
        member = self.db.query(WorkspaceMember).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.is_active == True
            )
        ).first()
        
        if not member:
            return False
        
        # Check permission
        has_permission = member.has_permission(permission)
        
        # Cache result
        if cache_key not in self._permission_cache:
            self._permission_cache[cache_key] = {"permissions": set()}
        if has_permission:
            self._permission_cache[cache_key]["permissions"].add(permission)
        
        return has_permission
    
    async def _setup_default_roles_and_permissions(self, workspace_id: UUID) -> None:
        """Setup default roles and permissions for new workspace"""
        # Default permissions
        default_permissions = [
            ("file.read", "file", "read", "Read files in workspace"),
            ("file.write", "file", "write", "Edit files in workspace"),
            ("file.create", "file", "create", "Create new files"),
            ("file.delete", "file", "delete", "Delete files"),
            ("task.read", "task", "read", "View tasks"),
            ("task.write", "task", "write", "Edit tasks"),
            ("task.create", "task", "create", "Create tasks"),
            ("member.read", "member", "read", "View members"),
            ("member.manage", "member", "manage", "Manage members"),
            ("settings.read", "settings", "read", "View settings"),
            ("settings.write", "settings", "write", "Modify settings")
        ]
        
        for name, resource, action, desc in default_permissions:
            permission = WorkspacePermission(
                workspace_id=workspace_id,
                name=name,
                description=desc,
                resource_type=resource,
                action=action,
                is_global=True
            )
            self.db.add(permission)
        
        # Default roles
        default_roles = [
            ("Developer", "Standard developer role", "#2196F3", 
             ["file.read", "file.write", "file.create", "task.read", "task.write", "task.create"]),
            ("Reviewer", "Code reviewer role", "#FF9800",
             ["file.read", "task.read", "member.read"]),
            ("Lead", "Team lead role", "#4CAF50", 
             ["file.read", "file.write", "file.create", "file.delete", "task.read", "task.write", 
              "task.create", "member.read", "member.manage"])
        ]
        
        for name, desc, color, perms in default_roles:
            role = WorkspaceRole(
                workspace_id=workspace_id,
                name=name,
                description=desc,
                color=color,
                permissions=perms,
                is_assignable=True
            )
            self.db.add(role)
    
    def _cache_workspace(self, workspace: Workspace) -> None:
        """Cache workspace data for quick access"""
        self._workspace_cache[workspace.id] = {
            "name": workspace.name,
            "state": workspace.state,
            "type": workspace.workspace_type,
            "owner_id": workspace.owner_id,
            "settings": workspace.settings,
            "cached_at": datetime.now(timezone.utc)
        }
    
    def _clear_permission_cache(self, workspace_id: UUID) -> None:
        """Clear permission cache for workspace"""
        keys_to_remove = [k for k in self._permission_cache.keys() if k.startswith(f"{workspace_id}_")]
        for key in keys_to_remove:
            del self._permission_cache[key]
    
    async def update_workspace_settings(
        self,
        workspace_id: UUID,
        user_id: UUID,
        settings: Dict[str, Any]
    ) -> Workspace:
        """Update workspace settings"""
        # Validate permissions
        if not await self._check_member_permission(workspace_id, user_id, "settings.write"):
            raise WorkspaceManagerException("Insufficient permissions to modify settings")
        
        workspace = self.db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not workspace:
            raise WorkspaceManagerException("Workspace not found")
        
        workspace.settings.update(settings)
        workspace.updated_at = datetime.now(timezone.utc)
        
        self.db.commit()
        
        # Clear cache
        if workspace_id in self._workspace_cache:
            del self._workspace_cache[workspace_id]
        
        return workspace