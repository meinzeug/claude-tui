"""
Collaboration System Package
Multi-developer support with real-time collaboration and conflict resolution
"""

from .models import (
    Workspace, 
    WorkspaceMember, 
    CollaborationSession, 
    ConflictResolution,
    TeamAnalytics,
    ActivityFeed,
    WorkspacePermission,
    WorkspaceRole
)
from .workspace_manager import WorkspaceManager
from .sync_engine import SynchronizationEngine
from .conflict_resolver import ConflictResolver
from .presence_manager import PresenceManager
from .team_coordinator import TeamCoordinator
from .analytics_engine import AnalyticsEngine

__all__ = [
    # Models
    'Workspace',
    'WorkspaceMember', 
    'CollaborationSession',
    'ConflictResolution',
    'TeamAnalytics',
    'ActivityFeed',
    'WorkspacePermission',
    'WorkspaceRole',
    
    # Core Components
    'WorkspaceManager',
    'SynchronizationEngine',
    'ConflictResolver',
    'PresenceManager', 
    'TeamCoordinator',
    'AnalyticsEngine'
]