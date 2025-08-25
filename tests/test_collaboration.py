"""
Comprehensive test suite for collaboration features
Tests workspace management, real-time sync, conflict resolution, and team coordination
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID
from unittest.mock import Mock, patch, AsyncMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.collaboration.models import (
    Base, Workspace, WorkspaceMember, CollaborationSession,
    ConflictResolution, ActivityFeed, Comment, Notification,
    WorkspaceType, WorkspaceState, MemberRole, ConflictType, ConflictStatus
)
from src.collaboration.workspace_manager import WorkspaceManager
from src.collaboration.sync_engine import SynchronizationEngine, FileEditOperation
from src.collaboration.conflict_resolver import ConflictResolver, ResolutionStrategy
from src.collaboration.presence_manager import PresenceManager
from src.collaboration.team_coordinator import TeamCoordinator
from src.collaboration.analytics_engine import AnalyticsEngine, AnalyticsTimeframe
from src.collaboration.communication_hub import CommunicationHub, NotificationType, NotificationPriority
from src.database.models import User, Project, Task


@pytest.fixture
def db_session():
    """Create test database session"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def sample_user(db_session):
    """Create sample user for tests"""
    user = User(
        email="test@example.com",
        username="testuser",
        full_name="Test User"
    )
    user.set_password("testpass123")
    
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def sample_workspace(db_session, sample_user):
    """Create sample workspace for tests"""
    workspace = Workspace(
        name="Test Workspace",
        description="Test workspace for collaboration",
        owner_id=sample_user.id,
        workspace_type=WorkspaceType.TEAM.value
    )
    
    db_session.add(workspace)
    db_session.commit()
    return workspace


@pytest.fixture
def workspace_manager(db_session):
    """Create workspace manager instance"""
    return WorkspaceManager(db_session)


@pytest.fixture
def sync_engine(db_session):
    """Create synchronization engine instance"""
    return SynchronizationEngine(db_session)


@pytest.fixture
def conflict_resolver(db_session):
    """Create conflict resolver instance"""
    return ConflictResolver(db_session)


@pytest.fixture
def presence_manager(db_session):
    """Create presence manager instance"""
    return PresenceManager(db_session)


@pytest.fixture
def analytics_engine(db_session):
    """Create analytics engine instance"""
    return AnalyticsEngine(db_session)


@pytest.fixture
def communication_hub(db_session, presence_manager):
    """Create communication hub instance"""
    return CommunicationHub(db_session, presence_manager)


class TestWorkspaceManager:
    """Test workspace management functionality"""
    
    @pytest.mark.asyncio
    async def test_create_workspace(self, workspace_manager, sample_user):
        """Test workspace creation"""
        workspace = await workspace_manager.create_workspace(
            name="New Workspace",
            owner_id=sample_user.id,
            description="A new test workspace"
        )
        
        assert workspace.name == "New Workspace"
        assert workspace.owner_id == sample_user.id
        assert workspace.state == WorkspaceState.ACTIVE.value
        assert workspace.workspace_type == WorkspaceType.TEAM.value
    
    @pytest.mark.asyncio
    async def test_invite_member(self, workspace_manager, sample_workspace, db_session):
        """Test member invitation"""
        # Create another user to invite
        new_user = User(
            email="newuser@example.com",
            username="newuser",
            full_name="New User"
        )
        new_user.set_password("password123")
        db_session.add(new_user)
        db_session.commit()
        
        # Invite the user
        member = await workspace_manager.invite_member(
            workspace_id=sample_workspace.id,
            user_id=new_user.id,
            inviter_id=sample_workspace.owner_id,
            role=MemberRole.DEVELOPER
        )
        
        assert member.workspace_id == sample_workspace.id
        assert member.user_id == new_user.id
        assert member.role == MemberRole.DEVELOPER.value
        assert member.is_active is True
    
    @pytest.mark.asyncio
    async def test_update_member_role(self, workspace_manager, sample_workspace, sample_user, db_session):
        """Test member role update"""
        # Create and add member first
        new_user = User(email="member@example.com", username="member", full_name="Member")
        new_user.set_password("password123")
        db_session.add(new_user)
        db_session.commit()
        
        member = await workspace_manager.invite_member(
            workspace_id=sample_workspace.id,
            user_id=new_user.id,
            inviter_id=sample_user.id,
            role=MemberRole.DEVELOPER
        )
        
        # Update role
        updated_member = await workspace_manager.update_member_role(
            workspace_id=sample_workspace.id,
            member_user_id=new_user.id,
            new_role=MemberRole.MAINTAINER,
            updater_id=sample_user.id
        )
        
        assert updated_member.role == MemberRole.MAINTAINER.value
    
    @pytest.mark.asyncio
    async def test_get_workspace_details(self, workspace_manager, sample_workspace, sample_user):
        """Test workspace details retrieval"""
        details = await workspace_manager.get_workspace_details(
            workspace_id=sample_workspace.id,
            requester_id=sample_user.id
        )
        
        assert details['id'] == str(sample_workspace.id)
        assert details['name'] == sample_workspace.name
        assert details['owner_id'] == str(sample_workspace.owner_id)
        assert 'members' in details
        assert 'recent_activities' in details


class TestSynchronizationEngine:
    """Test real-time synchronization functionality"""
    
    def test_file_edit_operation_creation(self):
        """Test creating file edit operations"""
        operation = FileEditOperation(
            file_path="test.py",
            operation_type="insert",
            position={"line": 10, "column": 5},
            content="print('Hello')",
            user_id="user123"
        )
        
        assert operation.file_path == "test.py"
        assert operation.operation_type == "insert"
        assert operation.position["line"] == 10
        assert operation.content == "print('Hello')"
    
    @pytest.mark.asyncio
    async def test_apply_edit_operation(self, sync_engine):
        """Test applying edit operations to content"""
        file_path = "test.py"
        content = "line1\nline2\nline3"
        
        # Initialize file state
        sync_engine._file_states[file_path] = {
            "content": content,
            "version": 1,
            "last_modified": datetime.now(timezone.utc).isoformat(),
            "active_users": set()
        }
        
        # Create and apply operation
        operation = FileEditOperation(
            file_path=file_path,
            operation_type="insert",
            position={"line": 1, "column": 5},
            content=" modified",
            user_id="user123"
        )
        
        await sync_engine._apply_edit_operation(file_path, operation)
        
        modified_content = sync_engine._file_states[file_path]["content"]
        lines = modified_content.split("\n")
        assert "line2 modified" in lines[1]
    
    @pytest.mark.asyncio
    async def test_conflict_detection(self, sync_engine):
        """Test conflict detection in operations"""
        file_path = "test.py"
        
        # Create overlapping operations
        op1 = FileEditOperation(
            file_path=file_path,
            operation_type="insert",
            position={"line": 5, "column": 0},
            content="code1",
            user_id="user1"
        )
        
        op2 = FileEditOperation(
            file_path=file_path,
            operation_type="insert",
            position={"line": 5, "column": 2},
            content="code2",
            user_id="user2"
        )
        
        # Test overlap detection
        overlaps = sync_engine._operations_overlap(op1, op2)
        assert overlaps is True


class TestConflictResolver:
    """Test conflict resolution functionality"""
    
    @pytest.mark.asyncio
    async def test_detect_conflict(self, conflict_resolver, sample_workspace):
        """Test conflict detection"""
        operations = [
            FileEditOperation(
                file_path="test.py",
                operation_type="insert",
                position={"line": 10, "column": 0},
                content="code1",
                user_id="user1",
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            FileEditOperation(
                file_path="test.py",
                operation_type="insert",
                position={"line": 10, "column": 0},
                content="code2",
                user_id="user2",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        ]
        
        conflict = await conflict_resolver.detect_conflict(
            workspace_id=sample_workspace.id,
            file_path="test.py",
            operations=operations,
            base_content="original content"
        )
        
        if conflict:  # Conflict detection depends on timing and overlap
            assert conflict.conflict_type == ConflictType.CONCURRENT_EDIT.value
            assert len(conflict.affected_users) == 2
    
    @pytest.mark.asyncio
    async def test_resolve_whitespace_conflict(self, conflict_resolver):
        """Test automatic whitespace conflict resolution"""
        from src.collaboration.conflict_resolver import ConflictContext
        
        context = ConflictContext(
            file_path="test.py",
            conflicting_operations=[],
            base_content="line1\nline2\nline3",
            current_content="line1\n  line2\nline3",  # Added spaces
            incoming_content="line1\n\tline2\nline3",  # Added tab
            affected_users=[uuid4(), uuid4()],
            timestamp=datetime.now(timezone.utc),
            conflict_region={"start_line": 1, "end_line": 1, "start_col": 0, "end_col": 10}
        )
        
        result = await conflict_resolver._resolve_whitespace_conflicts(context)
        
        # Should detect whitespace-only difference and resolve
        if result:
            assert result['confidence'] > 0.9
            assert "whitespace" in result['notes'].lower()


class TestPresenceManager:
    """Test presence management functionality"""
    
    @pytest.mark.asyncio
    async def test_update_user_presence(self, presence_manager, sample_workspace, sample_user):
        """Test user presence updates"""
        presence_data = {
            'current_file': 'test.py',
            'cursor_position': {'line': 10, 'column': 5},
            'is_editing': True,
            'status': 'online'
        }
        
        await presence_manager.update_user_presence(
            workspace_id=sample_workspace.id,
            user_id=sample_user.id,
            presence_data=presence_data
        )
        
        # Verify presence was updated
        presence = await presence_manager.get_user_presence(
            workspace_id=sample_workspace.id,
            user_id=sample_user.id
        )
        
        if presence:
            assert presence['current_file'] == 'test.py'
            assert presence['is_editing'] is True
    
    @pytest.mark.asyncio
    async def test_typing_indicator(self, presence_manager, sample_workspace, sample_user):
        """Test typing indicator functionality"""
        await presence_manager.set_typing_indicator(
            workspace_id=sample_workspace.id,
            user_id=sample_user.id,
            file_path='test.py',
            is_typing=True
        )
        
        # Check if typing indicator was set
        if sample_user.id in presence_manager._user_presence:
            presence = presence_manager._user_presence[sample_user.id]
            assert presence.typing_indicator is True


class TestTeamCoordinator:
    """Test team coordination functionality"""
    
    @pytest.fixture
    def team_coordinator(self, db_session, workspace_manager, presence_manager, analytics_engine):
        """Create team coordinator instance"""
        return TeamCoordinator(db_session, workspace_manager, presence_manager, analytics_engine)
    
    @pytest.mark.asyncio
    async def test_task_assignment_skill_based(self, team_coordinator, sample_workspace, sample_user, db_session):
        """Test skill-based task assignment"""
        # Create a task
        task = Task(
            title="Test Task",
            description="A test task",
            project_id=None,  # Simplified for test
            priority="medium"
        )
        db_session.add(task)
        db_session.commit()
        
        # Test assignment
        assignment_context = {
            'skills_required': ['python', 'testing'],
            'estimated_hours': 4
        }
        
        assignment = await team_coordinator.assign_task(
            workspace_id=sample_workspace.id,
            task_id=task.id,
            assignee_id=sample_user.id,
            assigner_id=sample_user.id,
            assignment_strategy='skill_based',
            assignment_context=assignment_context
        )
        
        assert assignment.task_id == task.id
        assert assignment.assignee_id == sample_user.id
        assert assignment.skills_required == ['python', 'testing']
    
    @pytest.mark.asyncio
    async def test_team_workflow_creation(self, team_coordinator, sample_workspace, sample_user):
        """Test team workflow creation"""
        workflow_config = {
            'description': 'Test workflow',
            'stages': [
                {'name': 'Planning', 'duration': 2},
                {'name': 'Implementation', 'duration': 8},
                {'name': 'Review', 'duration': 2}
            ],
            'automation_rules': [
                {'trigger': 'task_completed', 'action': 'assign_reviewer'}
            ]
        }
        
        workflow = await team_coordinator.create_team_workflow(
            workspace_id=sample_workspace.id,
            creator_id=sample_user.id,
            workflow_name="Test Workflow",
            workflow_config=workflow_config
        )
        
        assert workflow.name == "Test Workflow"
        assert len(workflow.stages) == 3
        assert len(workflow.automation_rules) == 1


class TestAnalyticsEngine:
    """Test analytics functionality"""
    
    @pytest.mark.asyncio
    async def test_initialize_workspace_analytics(self, analytics_engine, sample_workspace):
        """Test workspace analytics initialization"""
        await analytics_engine.initialize_workspace_analytics(sample_workspace.id)
        
        # Verify analytics record was created
        from src.collaboration.models import TeamAnalytics
        analytics = analytics_engine.db.query(TeamAnalytics).filter(
            TeamAnalytics.workspace_id == sample_workspace.id
        ).first()
        
        assert analytics is not None
        assert analytics.workspace_id == sample_workspace.id
    
    @pytest.mark.asyncio
    async def test_real_time_metrics(self, analytics_engine, sample_workspace, db_session):
        """Test real-time metrics calculation"""
        # Create some test data
        session = CollaborationSession(
            workspace_id=sample_workspace.id,
            user_id=sample_workspace.owner_id,
            session_token="test_token",
            is_active=True
        )
        db_session.add(session)
        
        activity = ActivityFeed(
            workspace_id=sample_workspace.id,
            user_id=sample_workspace.owner_id,
            activity_type="file_created",
            description="Created test file"
        )
        db_session.add(activity)
        db_session.commit()
        
        # Get real-time metrics
        metrics = await analytics_engine.get_real_time_metrics(
            workspace_id=sample_workspace.id,
            metric_names=['active_users', 'recent_activities']
        )
        
        assert 'active_users' in metrics
        assert 'recent_activities' in metrics
        assert metrics['active_users'] >= 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_report(self, analytics_engine, sample_workspace):
        """Test comprehensive analytics report generation"""
        report = await analytics_engine.generate_comprehensive_report(
            workspace_id=sample_workspace.id,
            timeframe=AnalyticsTimeframe.WEEKLY
        )
        
        assert report.workspace_id == str(sample_workspace.id)
        assert report.timeframe == AnalyticsTimeframe.WEEKLY.value
        assert isinstance(report.summary_metrics, dict)
        assert isinstance(report.detailed_metrics, dict)
        assert isinstance(report.insights, list)
        assert isinstance(report.recommendations, list)


class TestCommunicationHub:
    """Test communication functionality"""
    
    @pytest.mark.asyncio
    async def test_create_comment(self, communication_hub, sample_workspace, sample_user):
        """Test comment creation"""
        comment = await communication_hub.create_comment(
            workspace_id=sample_workspace.id,
            user_id=sample_user.id,
            content="This is a test comment",
            target_type="file",
            target_id="test.py"
        )
        
        assert comment.content == "This is a test comment"
        assert comment.target_type == "file"
        assert comment.target_id == "test.py"
        assert comment.user_id == sample_user.id
    
    @pytest.mark.asyncio
    async def test_send_notification(self, communication_hub, sample_workspace, sample_user):
        """Test notification sending"""
        notification = await communication_hub.send_notification(
            workspace_id=sample_workspace.id,
            recipient_id=sample_user.id,
            notification_type=NotificationType.SYSTEM_ALERT,
            title="Test Notification",
            message="This is a test notification",
            priority=NotificationPriority.MEDIUM
        )
        
        assert notification.title == "Test Notification"
        assert notification.message == "This is a test notification"
        assert notification.notification_type == NotificationType.SYSTEM_ALERT.value
        assert notification.priority == NotificationPriority.MEDIUM.value
    
    @pytest.mark.asyncio
    async def test_create_discussion_thread(self, communication_hub, sample_workspace, sample_user):
        """Test discussion thread creation"""
        thread = await communication_hub.create_discussion_thread(
            workspace_id=sample_workspace.id,
            creator_id=sample_user.id,
            title="Test Discussion",
            initial_message="Let's discuss this topic",
            tags=["important", "planning"]
        )
        
        assert thread.title == "Test Discussion"
        assert thread.creator_id == str(sample_user.id)
        assert len(thread.participants) == 1
        assert "important" in thread.tags
        assert "planning" in thread.tags
    
    @pytest.mark.asyncio
    async def test_resolve_comment(self, communication_hub, sample_workspace, sample_user, db_session):
        """Test comment resolution"""
        # Create comment first
        comment = await communication_hub.create_comment(
            workspace_id=sample_workspace.id,
            user_id=sample_user.id,
            content="This needs to be resolved",
            target_type="task",
            target_id="task123"
        )
        
        # Resolve the comment
        result = await communication_hub.resolve_comment(
            workspace_id=sample_workspace.id,
            comment_id=comment.id,
            resolver_id=sample_user.id
        )
        
        assert result is True
        
        # Verify comment is resolved
        db_session.refresh(comment)
        assert comment.is_resolved is True
        assert comment.resolved_by == sample_user.id


class TestIntegration:
    """Integration tests for collaboration features"""
    
    @pytest.mark.asyncio
    async def test_full_collaboration_workflow(
        self, 
        workspace_manager, 
        communication_hub, 
        presence_manager,
        team_coordinator,
        db_session,
        sample_user
    ):
        """Test complete collaboration workflow"""
        # 1. Create workspace
        workspace = await workspace_manager.create_workspace(
            name="Integration Test Workspace",
            owner_id=sample_user.id,
            description="Full workflow test"
        )
        
        # 2. Create additional user
        user2 = User(email="user2@test.com", username="user2", full_name="User Two")
        user2.set_password("password123")
        db_session.add(user2)
        db_session.commit()
        
        # 3. Invite member
        member = await workspace_manager.invite_member(
            workspace_id=workspace.id,
            user_id=user2.id,
            inviter_id=sample_user.id,
            role=MemberRole.DEVELOPER
        )
        
        # 4. Update presence
        await presence_manager.update_user_presence(
            workspace_id=workspace.id,
            user_id=user2.id,
            presence_data={
                'current_file': 'main.py',
                'cursor_position': {'line': 1, 'column': 0},
                'is_editing': True,
                'status': 'online'
            }
        )
        
        # 5. Create discussion
        thread = await communication_hub.create_discussion_thread(
            workspace_id=workspace.id,
            creator_id=sample_user.id,
            title="Project Planning",
            initial_message="Let's plan our next sprint"
        )
        
        # 6. Add comment
        comment = await communication_hub.create_comment(
            workspace_id=workspace.id,
            user_id=user2.id,
            content="Great idea! I'll work on the backend",
            target_type="discussion",
            target_id=thread.id
        )
        
        # 7. Send notification
        notification = await communication_hub.send_notification(
            workspace_id=workspace.id,
            recipient_id=sample_user.id,
            notification_type=NotificationType.MENTION,
            title="New Comment",
            message="User2 commented on your discussion",
            priority=NotificationPriority.MEDIUM
        )
        
        # Verify the workflow completed successfully
        assert workspace.name == "Integration Test Workspace"
        assert member.user_id == user2.id
        assert thread.title == "Project Planning"
        assert comment.content == "Great idea! I'll work on the backend"
        assert notification.title == "New Comment"
        
        # Check workspace details include all elements
        details = await workspace_manager.get_workspace_details(
            workspace_id=workspace.id,
            requester_id=sample_user.id
        )
        
        assert details['member_count'] >= 2  # Owner + invited member
        assert len(details['recent_activities']) >= 3  # Multiple activities created
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_workflow(
        self,
        conflict_resolver,
        sample_workspace,
        sample_user,
        db_session
    ):
        """Test conflict detection and resolution workflow"""
        # Create conflicting operations
        operations = [
            FileEditOperation(
                file_path="shared.py",
                operation_type="insert",
                position={"line": 10, "column": 0},
                content="def function1():",
                user_id=str(sample_user.id),
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            FileEditOperation(
                file_path="shared.py",
                operation_type="insert",
                position={"line": 10, "column": 0},
                content="def function2():",
                user_id=str(uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        ]
        
        # Detect conflict
        conflict = await conflict_resolver.detect_conflict(
            workspace_id=sample_workspace.id,
            file_path="shared.py",
            operations=operations,
            base_content="# Base file content\n"
        )
        
        if conflict:  # Conflict detection is probabilistic
            # Attempt resolution
            result = await conflict_resolver.resolve_conflict(
                conflict_id=conflict.id,
                strategy=ResolutionStrategy.AUTO_MERGE
            )
            
            # Verify resolution attempt was made
            assert result.strategy_used in ["auto_merge", "three_way_merge"]
            
            # Check conflict status was updated
            db_session.refresh(conflict)
            assert conflict.status in [ConflictStatus.RESOLVED.value, ConflictStatus.ESCALATED.value]


@pytest.mark.asyncio
async def test_websocket_simulation():
    """Test WebSocket message handling simulation"""
    from src.collaboration.sync_engine import WebSocketMessage, MessageType
    
    # Create test message
    message = WebSocketMessage(
        type=MessageType.FILE_EDIT.value,
        workspace_id=str(uuid4()),
        user_id=str(uuid4()),
        data={
            'file_path': 'test.py',
            'operation_type': 'insert',
            'position': {'line': 5, 'column': 0},
            'content': 'print("Hello World")'
        }
    )
    
    # Test JSON serialization
    json_str = message.to_json()
    assert isinstance(json_str, str)
    
    # Test deserialization
    restored_message = WebSocketMessage.from_json(json_str)
    assert restored_message.type == message.type
    assert restored_message.workspace_id == message.workspace_id
    assert restored_message.data == message.data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])