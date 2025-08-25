"""
Team Coordinator - Advanced team management and workflow coordination
Orchestrates team activities, task assignments, and collaborative workflows
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from .models import (
    Workspace, WorkspaceMember, MemberRole, ActivityFeed, ActivityType,
    Comment, Notification, TeamAnalytics
)
from ..database.models import User, Project, Task
from .workspace_manager import WorkspaceManager
from .presence_manager import PresenceManager
from .analytics_engine import AnalyticsEngine

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status for coordination"""
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    REVIEW_PENDING = "review_pending"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class CoordinationMode(Enum):
    """Team coordination modes"""
    AUTONOMOUS = "autonomous"  # Self-organizing teams
    HIERARCHICAL = "hierarchical"  # Traditional top-down
    COLLABORATIVE = "collaborative"  # Peer-to-peer coordination
    HYBRID = "hybrid"  # Mix of approaches


@dataclass
class TaskAssignment:
    """Task assignment with context"""
    task_id: UUID
    assignee_id: UUID
    assigner_id: UUID
    priority: str
    estimated_hours: Optional[int]
    deadline: Optional[datetime]
    skills_required: List[str]
    dependencies: List[UUID]
    assignment_reason: str


@dataclass
class TeamWorkflow:
    """Team workflow definition"""
    workflow_id: UUID
    name: str
    description: str
    stages: List[Dict[str, Any]]
    automation_rules: List[Dict[str, Any]]
    approval_requirements: Dict[str, Any]
    notification_settings: Dict[str, Any]


class TeamCoordinator:
    """
    Advanced team coordination system for managing collaborative workflows.
    Handles task assignment, team communication, and workflow orchestration.
    """
    
    def __init__(
        self,
        db_session: Session,
        workspace_manager: WorkspaceManager,
        presence_manager: PresenceManager,
        analytics_engine: AnalyticsEngine
    ):
        """
        Initialize team coordinator.
        
        Args:
            db_session: Database session
            workspace_manager: Workspace management system
            presence_manager: Presence tracking system
            analytics_engine: Analytics engine
        """
        self.db = db_session
        self.workspace_manager = workspace_manager
        self.presence_manager = presence_manager
        self.analytics_engine = analytics_engine
        
        # Coordination state
        self._active_workflows: Dict[UUID, TeamWorkflow] = {}
        self._task_assignments: Dict[UUID, TaskAssignment] = {}
        self._team_metrics: Dict[UUID, Dict[str, Any]] = {}
        
        # Smart assignment algorithms
        self._assignment_strategies = {
            'skill_based': self._assign_by_skills,
            'workload_balanced': self._assign_by_workload,
            'availability_based': self._assign_by_availability,
            'expertise_match': self._assign_by_expertise,
            'random': self._assign_randomly
        }
        
        # Workflow automation rules
        self._automation_rules = {
            'auto_assign_new_tasks': self._auto_assign_new_tasks,
            'auto_review_assignment': self._auto_review_assignment,
            'auto_escalate_blocked_tasks': self._auto_escalate_blocked_tasks,
            'auto_notify_deadlines': self._auto_notify_deadlines
        }
        
        logger.info("Team coordinator initialized")
    
    async def assign_task(
        self,
        workspace_id: UUID,
        task_id: UUID,
        assignee_id: Optional[UUID] = None,
        assigner_id: Optional[UUID] = None,
        assignment_strategy: str = 'skill_based',
        assignment_context: Optional[Dict[str, Any]] = None
    ) -> TaskAssignment:
        """
        Assign task to team member using intelligent assignment algorithms.
        
        Args:
            workspace_id: Workspace ID
            task_id: Task to assign
            assignee_id: Specific assignee (optional)
            assigner_id: User making the assignment
            assignment_strategy: Assignment algorithm to use
            assignment_context: Additional context for assignment
            
        Returns:
            TaskAssignment record
        """
        logger.info(f"Assigning task {task_id} in workspace {workspace_id}")
        
        # Get task details
        task = self.db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # If no specific assignee, use assignment strategy
        if not assignee_id:
            strategy_func = self._assignment_strategies.get(assignment_strategy)
            if not strategy_func:
                raise ValueError(f"Unknown assignment strategy: {assignment_strategy}")
            
            assignee_id = await strategy_func(workspace_id, task, assignment_context or {})
        
        # Validate assignee is workspace member
        member = self.db.query(WorkspaceMember).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == assignee_id,
                WorkspaceMember.is_active == True
            )
        ).first()
        
        if not member:
            raise ValueError(f"User {assignee_id} is not a member of workspace {workspace_id}")
        
        # Create assignment
        assignment = TaskAssignment(
            task_id=task_id,
            assignee_id=assignee_id,
            assigner_id=assigner_id or assignee_id,
            priority=task.priority,
            estimated_hours=assignment_context.get('estimated_hours'),
            deadline=assignment_context.get('deadline'),
            skills_required=assignment_context.get('skills_required', []),
            dependencies=assignment_context.get('dependencies', []),
            assignment_reason=f"Assigned using {assignment_strategy} strategy"
        )
        
        # Update task
        task.assigned_to = assignee_id
        task.status = 'assigned'
        
        # Store assignment
        self._task_assignments[task_id] = assignment
        
        # Create activity
        assignee = self.db.query(User).filter(User.id == assignee_id).first()
        activity = ActivityFeed(
            workspace_id=workspace_id,
            user_id=assigner_id or assignee_id,
            activity_type='task_assigned',
            description=f"Task '{task.title}' assigned to {assignee.username}",
            target_resource=str(task_id),
            metadata={
                'task_id': str(task_id),
                'assignee_id': str(assignee_id),
                'strategy': assignment_strategy
            },
            is_public=True,
            notify_team=True
        )
        self.db.add(activity)
        
        # Send notification to assignee
        notification = Notification(
            workspace_id=workspace_id,
            recipient_id=assignee_id,
            title=f"New task assigned: {task.title}",
            message=f"You have been assigned task '{task.title}' with priority {task.priority}",
            notification_type="task_assignment",
            priority="medium",
            metadata={'task_id': str(task_id)}
        )
        self.db.add(notification)
        
        self.db.commit()
        
        # Update team metrics
        await self._update_assignment_metrics(workspace_id, assignment)
        
        logger.info(f"Task {task_id} assigned to {assignee_id}")
        return assignment
    
    async def create_team_workflow(
        self,
        workspace_id: UUID,
        creator_id: UUID,
        workflow_name: str,
        workflow_config: Dict[str, Any]
    ) -> TeamWorkflow:
        """
        Create custom team workflow with automation rules.
        
        Args:
            workspace_id: Workspace ID
            creator_id: User creating the workflow
            workflow_name: Workflow name
            workflow_config: Workflow configuration
            
        Returns:
            Created workflow
        """
        logger.info(f"Creating team workflow '{workflow_name}' in workspace {workspace_id}")
        
        # Validate creator permissions
        if not await self.workspace_manager._check_member_permission(workspace_id, creator_id, "settings.write"):
            raise ValueError("Insufficient permissions to create workflows")
        
        workflow = TeamWorkflow(
            workflow_id=uuid4(),
            name=workflow_name,
            description=workflow_config.get('description', ''),
            stages=workflow_config.get('stages', []),
            automation_rules=workflow_config.get('automation_rules', []),
            approval_requirements=workflow_config.get('approval_requirements', {}),
            notification_settings=workflow_config.get('notification_settings', {})
        )
        
        self._active_workflows[workflow.workflow_id] = workflow
        
        # Create activity
        activity = ActivityFeed(
            workspace_id=workspace_id,
            user_id=creator_id,
            activity_type='workflow_created',
            description=f"Created workflow '{workflow_name}'",
            metadata={
                'workflow_id': str(workflow.workflow_id),
                'stages': len(workflow.stages),
                'automation_rules': len(workflow.automation_rules)
            },
            is_public=True
        )
        self.db.add(activity)
        self.db.commit()
        
        logger.info(f"Workflow '{workflow_name}' created with ID {workflow.workflow_id}")
        return workflow
    
    async def orchestrate_team_task(
        self,
        workspace_id: UUID,
        task_description: str,
        coordinator_id: UUID,
        coordination_mode: CoordinationMode = CoordinationMode.COLLABORATIVE,
        task_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate complex team task with multiple members.
        
        Args:
            workspace_id: Workspace ID
            task_description: Task description
            coordinator_id: User orchestrating the task
            coordination_mode: Coordination approach
            task_context: Additional context
            
        Returns:
            Orchestration result with assignments and timeline
        """
        logger.info(f"Orchestrating team task in workspace {workspace_id}")
        
        # Analyze task complexity and requirements
        task_analysis = await self._analyze_task_requirements(task_description, task_context)
        
        # Get available team members
        team_members = await self.workspace_manager.get_workspace_members(workspace_id, coordinator_id)
        available_members = [m for m in team_members if m['is_active']]
        
        # Create orchestration plan
        orchestration_plan = await self._create_orchestration_plan(
            task_analysis, available_members, coordination_mode
        )
        
        # Execute orchestration
        orchestration_result = await self._execute_orchestration(
            workspace_id, orchestration_plan, coordinator_id
        )
        
        # Create activity
        activity = ActivityFeed(
            workspace_id=workspace_id,
            user_id=coordinator_id,
            activity_type='team_orchestration',
            description=f"Orchestrated team task: {task_description[:100]}...",
            metadata={
                'coordination_mode': coordination_mode.value,
                'team_size': len(available_members),
                'subtasks_created': len(orchestration_result.get('subtasks', []))
            },
            is_public=True
        )
        self.db.add(activity)
        self.db.commit()
        
        return orchestration_result
    
    async def manage_team_communication(
        self,
        workspace_id: UUID,
        communication_type: str,
        participants: List[UUID],
        content: Dict[str, Any],
        initiator_id: UUID
    ) -> Dict[str, Any]:
        """
        Manage team communication and notifications.
        
        Args:
            workspace_id: Workspace ID
            communication_type: Type of communication
            participants: List of participant user IDs
            content: Communication content
            initiator_id: User initiating communication
            
        Returns:
            Communication result
        """
        logger.info(f"Managing team communication of type {communication_type}")
        
        if communication_type == 'announcement':
            return await self._handle_team_announcement(
                workspace_id, participants, content, initiator_id
            )
        elif communication_type == 'discussion':
            return await self._handle_team_discussion(
                workspace_id, participants, content, initiator_id
            )
        elif communication_type == 'urgent_notification':
            return await self._handle_urgent_notification(
                workspace_id, participants, content, initiator_id
            )
        elif communication_type == 'status_update':
            return await self._handle_status_update(
                workspace_id, participants, content, initiator_id
            )
        else:
            raise ValueError(f"Unknown communication type: {communication_type}")
    
    async def get_team_coordination_dashboard(
        self,
        workspace_id: UUID,
        requester_id: UUID
    ) -> Dict[str, Any]:
        """
        Get comprehensive team coordination dashboard.
        
        Args:
            workspace_id: Workspace ID
            requester_id: User requesting dashboard
            
        Returns:
            Dashboard data with team metrics and status
        """
        # Get team members and their status
        members = await self.workspace_manager.get_workspace_members(workspace_id, requester_id)
        presence_data = await self.presence_manager.get_workspace_presence(workspace_id)
        
        # Get active tasks and assignments
        active_tasks = self.db.query(Task).join(
            Task.project
        ).filter(
            and_(
                Task.project.has(owner_id=requester_id),  # Simplified - should check workspace
                Task.status.in_(['assigned', 'in_progress'])
            )
        ).all()
        
        # Get recent activities
        recent_activities = self.db.query(ActivityFeed).filter(
            ActivityFeed.workspace_id == workspace_id
        ).order_by(desc(ActivityFeed.created_at)).limit(20).all()
        
        # Calculate team metrics
        team_metrics = await self._calculate_team_metrics(workspace_id, members)
        
        # Get workflow status
        workflow_status = {
            wf_id: {
                'name': workflow.name,
                'stages': len(workflow.stages),
                'active': True
            }
            for wf_id, workflow in self._active_workflows.items()
        }
        
        return {
            'workspace_id': str(workspace_id),
            'team_summary': {
                'total_members': len(members),
                'active_members': len([m for m in members if m['is_active']]),
                'online_members': len([p for p in presence_data if p['status'] == 'online'])
            },
            'task_summary': {
                'total_active_tasks': len(active_tasks),
                'assigned_tasks': len([t for t in active_tasks if t.assigned_to]),
                'unassigned_tasks': len([t for t in active_tasks if not t.assigned_to]),
                'overdue_tasks': len([t for t in active_tasks if t.due_date and t.due_date < datetime.now(timezone.utc)])
            },
            'team_metrics': team_metrics,
            'recent_activities': [
                {
                    'id': str(activity.id),
                    'type': activity.activity_type,
                    'description': activity.description,
                    'user_id': str(activity.user_id),
                    'created_at': activity.created_at.isoformat()
                }
                for activity in recent_activities
            ],
            'active_workflows': workflow_status,
            'member_presence': presence_data
        }
    
    async def optimize_team_performance(
        self,
        workspace_id: UUID,
        optimization_goals: List[str],
        coordinator_id: UUID
    ) -> Dict[str, Any]:
        """
        Analyze and optimize team performance.
        
        Args:
            workspace_id: Workspace ID
            optimization_goals: Goals for optimization
            coordinator_id: User requesting optimization
            
        Returns:
            Optimization recommendations and actions
        """
        logger.info(f"Optimizing team performance for workspace {workspace_id}")
        
        # Analyze current team performance
        performance_analysis = await self._analyze_team_performance(workspace_id)
        
        # Generate optimization recommendations
        recommendations = await self._generate_optimization_recommendations(
            performance_analysis, optimization_goals
        )
        
        # Create optimization action plan
        action_plan = await self._create_optimization_plan(
            workspace_id, recommendations, coordinator_id
        )
        
        # Record optimization activity
        activity = ActivityFeed(
            workspace_id=workspace_id,
            user_id=coordinator_id,
            activity_type='performance_optimization',
            description="Team performance optimization analysis completed",
            metadata={
                'goals': optimization_goals,
                'recommendations_count': len(recommendations),
                'actions_planned': len(action_plan.get('actions', []))
            },
            is_public=True
        )
        self.db.add(activity)
        self.db.commit()
        
        return {
            'performance_analysis': performance_analysis,
            'recommendations': recommendations,
            'action_plan': action_plan,
            'optimization_goals': optimization_goals
        }
    
    # Assignment strategy implementations
    
    async def _assign_by_skills(
        self,
        workspace_id: UUID,
        task: Task,
        context: Dict[str, Any]
    ) -> UUID:
        """Assign task based on required skills"""
        required_skills = context.get('skills_required', [])
        
        # Get team members with skill information
        members = await self.workspace_manager.get_workspace_members(workspace_id, None)
        
        if not required_skills:
            # No specific skills required, use availability
            return await self._assign_by_availability(workspace_id, task, context)
        
        # Score members by skill match
        skill_scores = {}
        for member in members:
            if not member['is_active']:
                continue
            
            # This would integrate with skill tracking system
            # For now, use simple heuristic based on role
            member_skills = self._get_member_skills(member)
            skill_match = len(set(required_skills) & set(member_skills))
            skill_scores[UUID(member['id'])] = skill_match
        
        if not skill_scores:
            raise ValueError("No available members for task assignment")
        
        # Return member with highest skill match
        best_match = max(skill_scores.keys(), key=lambda k: skill_scores[k])
        return best_match
    
    async def _assign_by_workload(
        self,
        workspace_id: UUID,
        task: Task,
        context: Dict[str, Any]
    ) -> UUID:
        """Assign task based on current workload balance"""
        members = await self.workspace_manager.get_workspace_members(workspace_id, None)
        active_members = [m for m in members if m['is_active']]
        
        # Calculate current workload for each member
        workloads = {}
        for member in active_members:
            member_id = UUID(member['id'])
            
            # Count assigned tasks
            assigned_tasks = self.db.query(Task).filter(
                and_(
                    Task.assigned_to == member_id,
                    Task.status.in_(['assigned', 'in_progress'])
                )
            ).count()
            
            workloads[member_id] = assigned_tasks
        
        if not workloads:
            raise ValueError("No available members for task assignment")
        
        # Return member with lowest workload
        least_busy = min(workloads.keys(), key=lambda k: workloads[k])
        return least_busy
    
    async def _assign_by_availability(
        self,
        workspace_id: UUID,
        task: Task,
        context: Dict[str, Any]
    ) -> UUID:
        """Assign task based on member availability"""
        # Get current presence information
        presence_data = await self.presence_manager.get_workspace_presence(workspace_id)
        
        # Find online and available members
        available_members = [
            UUID(p['user_id']) for p in presence_data
            if p['status'] == 'online' and not p.get('is_busy', False)
        ]
        
        if not available_members:
            # Fallback to any active member
            members = await self.workspace_manager.get_workspace_members(workspace_id, None)
            available_members = [UUID(m['id']) for m in members if m['is_active']]
        
        if not available_members:
            raise ValueError("No available members for task assignment")
        
        # Return first available member (could be randomized)
        return available_members[0]
    
    async def _assign_by_expertise(
        self,
        workspace_id: UUID,
        task: Task,
        context: Dict[str, Any]
    ) -> UUID:
        """Assign task based on expertise level"""
        # This would integrate with expertise tracking
        # For now, delegate to skill-based assignment
        return await self._assign_by_skills(workspace_id, task, context)
    
    async def _assign_randomly(
        self,
        workspace_id: UUID,
        task: Task,
        context: Dict[str, Any]
    ) -> UUID:
        """Randomly assign task to available member"""
        import random
        
        members = await self.workspace_manager.get_workspace_members(workspace_id, None)
        available_members = [UUID(m['id']) for m in members if m['is_active']]
        
        if not available_members:
            raise ValueError("No available members for task assignment")
        
        return random.choice(available_members)
    
    # Helper methods
    
    def _get_member_skills(self, member: Dict[str, Any]) -> List[str]:
        """Get skills for team member (placeholder implementation)"""
        role = member.get('role', 'developer')
        
        skill_mapping = {
            'owner': ['leadership', 'project_management', 'technical'],
            'admin': ['administration', 'project_management', 'technical'],
            'maintainer': ['technical', 'code_review', 'debugging'],
            'developer': ['programming', 'debugging', 'testing'],
            'viewer': ['documentation', 'testing']
        }
        
        return skill_mapping.get(role, ['general'])
    
    async def _analyze_task_requirements(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze task to determine requirements"""
        # Simple analysis - in production this could use ML/AI
        return {
            'complexity': 'medium',
            'estimated_hours': context.get('estimated_hours', 4),
            'skills_required': context.get('skills_required', ['programming']),
            'team_size_needed': context.get('team_size', 1),
            'dependencies': context.get('dependencies', []),
            'priority': context.get('priority', 'medium')
        }
    
    async def _create_orchestration_plan(
        self,
        task_analysis: Dict[str, Any],
        available_members: List[Dict[str, Any]],
        coordination_mode: CoordinationMode
    ) -> Dict[str, Any]:
        """Create plan for task orchestration"""
        team_size = min(task_analysis['team_size_needed'], len(available_members))
        
        return {
            'coordination_mode': coordination_mode.value,
            'team_size': team_size,
            'selected_members': available_members[:team_size],
            'task_breakdown': [
                f"Subtask {i+1}" for i in range(team_size)
            ],
            'estimated_completion': datetime.now(timezone.utc) + timedelta(hours=task_analysis['estimated_hours'])
        }
    
    async def _execute_orchestration(
        self,
        workspace_id: UUID,
        plan: Dict[str, Any],
        coordinator_id: UUID
    ) -> Dict[str, Any]:
        """Execute orchestration plan"""
        # This would create actual subtasks and assignments
        return {
            'status': 'initiated',
            'subtasks': plan['task_breakdown'],
            'team_assigned': [m['id'] for m in plan['selected_members']],
            'estimated_completion': plan['estimated_completion'].isoformat()
        }
    
    async def _handle_team_announcement(
        self,
        workspace_id: UUID,
        participants: List[UUID],
        content: Dict[str, Any],
        initiator_id: UUID
    ) -> Dict[str, Any]:
        """Handle team announcement"""
        # Create notifications for all participants
        notifications = []
        for participant_id in participants:
            notification = Notification(
                workspace_id=workspace_id,
                recipient_id=participant_id,
                title=content.get('title', 'Team Announcement'),
                message=content.get('message', ''),
                notification_type='announcement',
                priority=content.get('priority', 'medium')
            )
            notifications.append(notification)
            self.db.add(notification)
        
        self.db.commit()
        
        return {
            'type': 'announcement',
            'recipients': len(participants),
            'notifications_sent': len(notifications)
        }
    
    async def _handle_team_discussion(
        self,
        workspace_id: UUID,
        participants: List[UUID],
        content: Dict[str, Any],
        initiator_id: UUID
    ) -> Dict[str, Any]:
        """Handle team discussion"""
        # Create comment/discussion thread
        comment = Comment(
            workspace_id=workspace_id,
            user_id=initiator_id,
            content=content.get('message', ''),
            target_type='discussion',
            target_id=str(uuid4())
        )
        self.db.add(comment)
        self.db.commit()
        
        return {
            'type': 'discussion',
            'comment_id': str(comment.id),
            'participants': len(participants)
        }
    
    async def _handle_urgent_notification(
        self,
        workspace_id: UUID,
        participants: List[UUID],
        content: Dict[str, Any],
        initiator_id: UUID
    ) -> Dict[str, Any]:
        """Handle urgent notification"""
        # Send high-priority notifications
        for participant_id in participants:
            notification = Notification(
                workspace_id=workspace_id,
                recipient_id=participant_id,
                title=f"URGENT: {content.get('title', 'Team Alert')}",
                message=content.get('message', ''),
                notification_type='urgent_alert',
                priority='urgent'
            )
            self.db.add(notification)
        
        self.db.commit()
        
        return {
            'type': 'urgent_notification',
            'recipients': len(participants),
            'priority': 'urgent'
        }
    
    async def _handle_status_update(
        self,
        workspace_id: UUID,
        participants: List[UUID],
        content: Dict[str, Any],
        initiator_id: UUID
    ) -> Dict[str, Any]:
        """Handle status update"""
        # Create activity for status update
        activity = ActivityFeed(
            workspace_id=workspace_id,
            user_id=initiator_id,
            activity_type='status_update',
            description=content.get('message', 'Status update'),
            is_public=True,
            notify_team=True
        )
        self.db.add(activity)
        self.db.commit()
        
        return {
            'type': 'status_update',
            'activity_id': str(activity.id),
            'visibility': 'team'
        }
    
    async def _calculate_team_metrics(
        self,
        workspace_id: UUID,
        members: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate team performance metrics"""
        # This would integrate with analytics engine
        return {
            'productivity_score': 85,
            'collaboration_index': 78,
            'task_completion_rate': 92,
            'average_response_time': 24,  # hours
            'conflict_resolution_rate': 95
        }
    
    async def _analyze_team_performance(self, workspace_id: UUID) -> Dict[str, Any]:
        """Analyze current team performance"""
        # Placeholder implementation
        return {
            'strengths': ['High collaboration', 'Fast response times'],
            'weaknesses': ['Uneven workload distribution', 'Communication gaps'],
            'bottlenecks': ['Code review process', 'Task assignment delays'],
            'efficiency_score': 82
        }
    
    async def _generate_optimization_recommendations(
        self,
        performance_analysis: Dict[str, Any],
        goals: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        return [
            {
                'type': 'workload_balancing',
                'description': 'Implement automatic workload balancing',
                'impact': 'high',
                'effort': 'medium'
            },
            {
                'type': 'communication_improvement',
                'description': 'Set up regular team sync meetings',
                'impact': 'medium',
                'effort': 'low'
            }
        ]
    
    async def _create_optimization_plan(
        self,
        workspace_id: UUID,
        recommendations: List[Dict[str, Any]],
        coordinator_id: UUID
    ) -> Dict[str, Any]:
        """Create actionable optimization plan"""
        return {
            'actions': [
                {
                    'action': rec['description'],
                    'priority': rec['impact'],
                    'timeline': '2 weeks',
                    'assigned_to': str(coordinator_id)
                }
                for rec in recommendations
            ],
            'success_metrics': ['Improved task completion rate', 'Reduced response time'],
            'review_date': (datetime.now(timezone.utc) + timedelta(weeks=4)).isoformat()
        }
    
    async def _update_assignment_metrics(
        self,
        workspace_id: UUID,
        assignment: TaskAssignment
    ) -> None:
        """Update team metrics after assignment"""
        if workspace_id not in self._team_metrics:
            self._team_metrics[workspace_id] = {
                'total_assignments': 0,
                'successful_assignments': 0,
                'assignment_strategies': {}
            }
        
        metrics = self._team_metrics[workspace_id]
        metrics['total_assignments'] += 1