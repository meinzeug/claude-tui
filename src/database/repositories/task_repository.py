"""
Task Repository Implementation

Provides task-specific database operations with:
- Project access control
- Task assignment and lifecycle management
- Due date and priority management
- Task search and filtering
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_, or_, func
from sqlalchemy.sql import Select

from .base import BaseRepository, RepositoryError
from .project_repository import ProjectRepository
from ..models import Task, Project, User
from ...core.logger import get_logger

logger = get_logger(__name__)


class TaskRepository(BaseRepository[Task]):
    """Task repository with project access control and lifecycle management."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize task repository.
        
        Args:
            session: AsyncSession instance
        """
        super().__init__(session, Task)
    
    def _add_relationship_loading(self, query: Select) -> Select:
        """
        Add eager loading for task relationships.
        
        Args:
            query: SQLAlchemy query
            
        Returns:
            Query with relationship loading options
        """
        return query.options(
            selectinload(Task.project).selectinload(Project.owner),
            selectinload(Task.assignee)
        )
    
    async def get_project_tasks(
        self,
        project_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        status: Optional[str] = None,
        assigned_to: Optional[uuid.UUID] = None,
        priority: Optional[str] = None,
        include_completed: bool = True
    ) -> List[Task]:
        """
        Get tasks for a project with access control and filtering.
        
        Args:
            project_id: Project ID
            user_id: User ID for access control (optional)
            status: Filter by task status
            assigned_to: Filter by assigned user
            priority: Filter by priority
            include_completed: Whether to include completed tasks
            
        Returns:
            List of Task instances
            
        Raises:
            RepositoryError: If operation fails or access is denied
        """
        try:
            # Check project access if user_id provided
            if user_id:
                project_repo = ProjectRepository(self.session)
                if not await project_repo.check_project_access(project_id, user_id):
                    self.logger.warning(
                        f"User denied access to tasks for project {project_id}",
                        extra={"project_id": str(project_id), "user_id": str(user_id)}
                    )
                    raise RepositoryError(
                        "Access denied to project tasks",
                        "PROJECT_ACCESS_DENIED",
                        {"project_id": str(project_id), "user_id": str(user_id)}
                    )
            
            # Build query with filters
            filters = {'project_id': project_id}
            
            if status:
                filters['status'] = status
            if assigned_to:
                filters['assigned_to'] = assigned_to
            if priority:
                filters['priority'] = priority
            
            if not include_completed:
                filters['status__in'] = ['pending', 'in_progress']
            
            tasks = await self.get_all(
                filters=filters,
                order_by='created_at',
                order_desc=True,
                load_relationships=True
            )
            
            self.logger.debug(
                f"Retrieved {len(tasks)} tasks for project {project_id}",
                extra={
                    "project_id": str(project_id),
                    "user_id": str(user_id) if user_id else None,
                    "filters": filters,
                    "task_count": len(tasks)
                }
            )
            
            return tasks
            
        except RepositoryError:
            raise
        except Exception as e:
            self.logger.error(f"Error getting tasks for project {project_id}: {e}")
            raise RepositoryError(
                "Failed to retrieve project tasks",
                "GET_PROJECT_TASKS_ERROR",
                {
                    "project_id": str(project_id),
                    "user_id": str(user_id) if user_id else None,
                    "error": str(e)
                }
            )
    
    async def get_user_tasks(
        self,
        user_id: uuid.UUID,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        due_soon: bool = False,
        overdue: bool = False,
        limit: int = 100
    ) -> List[Task]:
        """
        Get tasks assigned to user.
        
        Args:
            user_id: User ID
            status: Filter by task status
            priority: Filter by priority
            due_soon: Filter for tasks due within 7 days
            overdue: Filter for overdue tasks
            limit: Maximum number of tasks to return
            
        Returns:
            List of Task instances assigned to user
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            filters = {'assigned_to': user_id}
            
            if status:
                filters['status'] = status
            if priority:
                filters['priority'] = priority
            
            # Get tasks with basic filters first
            query = select(Task).options(
                selectinload(Task.project).selectinload(Project.owner),
                selectinload(Task.assignee)
            ).where(Task.assigned_to == user_id)
            
            # Apply additional filters
            if status:
                query = query.where(Task.status == status)
            if priority:
                query = query.where(Task.priority == priority)
            
            # Date-based filters
            now = datetime.now(timezone.utc)
            
            if due_soon:
                week_from_now = now + timedelta(days=7)
                query = query.where(
                    and_(
                        Task.due_date.is_not(None),
                        Task.due_date <= week_from_now,
                        Task.due_date >= now,
                        Task.status != 'completed'
                    )
                )
            
            if overdue:
                query = query.where(
                    and_(
                        Task.due_date.is_not(None),
                        Task.due_date < now,
                        Task.status != 'completed'
                    )
                )
            
            # Apply ordering and limit
            query = query.order_by(
                Task.due_date.asc().nulls_last(),
                Task.priority.desc(),
                Task.created_at.desc()
            ).limit(limit)
            
            result = await self.session.execute(query)
            tasks = result.scalars().all()
            
            self.logger.debug(
                f"Retrieved {len(tasks)} tasks for user {user_id}",
                extra={
                    "user_id": str(user_id),
                    "status": status,
                    "priority": priority,
                    "due_soon": due_soon,
                    "overdue": overdue,
                    "task_count": len(tasks)
                }
            )
            
            return list(tasks)
            
        except Exception as e:
            self.logger.error(f"Error getting tasks for user {user_id}: {e}")
            raise RepositoryError(
                "Failed to retrieve user tasks",
                "GET_USER_TASKS_ERROR",
                {
                    "user_id": str(user_id),
                    "status": status,
                    "priority": priority,
                    "error": str(e)
                }
            )
    
    async def create_task(
        self,
        project_id: uuid.UUID,
        title: str,
        description: Optional[str] = None,
        assigned_to: Optional[uuid.UUID] = None,
        due_date: Optional[datetime] = None,
        priority: str = "medium",
        creator_user_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> Optional[Task]:
        """
        Create task with project access validation.
        
        Args:
            project_id: Project ID
            title: Task title
            description: Task description
            assigned_to: User ID to assign task to
            due_date: Task due date
            priority: Task priority
            creator_user_id: User creating the task (for access control)
            **kwargs: Additional task fields
            
        Returns:
            Created Task instance
            
        Raises:
            RepositoryError: If creation fails or access is denied
        """
        try:
            # Check project access if creator specified
            if creator_user_id:
                project_repo = ProjectRepository(self.session)
                if not await project_repo.check_project_access(project_id, creator_user_id):
                    raise RepositoryError(
                        "Access denied to create task in project",
                        "PROJECT_ACCESS_DENIED",
                        {"project_id": str(project_id), "user_id": str(creator_user_id)}
                    )
            
            # Validate project exists
            project_exists = await self.session.scalar(
                select(Project.id).where(Project.id == project_id)
            )
            
            if not project_exists:
                raise RepositoryError(
                    "Project not found",
                    "PROJECT_NOT_FOUND",
                    {"project_id": str(project_id)}
                )
            
            # Validate assignee exists if specified
            if assigned_to:
                assignee_exists = await self.session.scalar(
                    select(User.id).where(User.id == assigned_to)
                )
                
                if not assignee_exists:
                    raise RepositoryError(
                        "Assigned user not found",
                        "ASSIGNEE_NOT_FOUND",
                        {"assigned_to": str(assigned_to)}
                    )
            
            # Create task
            task = Task(
                project_id=project_id,
                title=title,
                description=description,
                assigned_to=assigned_to,
                due_date=due_date,
                priority=priority,
                **kwargs
            )
            
            self.session.add(task)
            await self.session.flush()
            await self.session.refresh(task)
            
            self.logger.info(
                f"Created task: {title}",
                extra={
                    "task_id": str(task.id),
                    "project_id": str(project_id),
                    "assigned_to": str(assigned_to) if assigned_to else None,
                    "priority": priority,
                    "due_date": due_date.isoformat() if due_date else None
                }
            )
            
            return task
            
        except RepositoryError:
            await self.session.rollback()
            raise
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error creating task: {e}")
            raise RepositoryError(
                "Failed to create task",
                "CREATE_TASK_ERROR",
                {
                    "project_id": str(project_id),
                    "title": title,
                    "assigned_to": str(assigned_to) if assigned_to else None,
                    "error": str(e)
                }
            )
    
    async def assign_task(
        self,
        task_id: uuid.UUID,
        assigned_to: uuid.UUID,
        assigner_user_id: Optional[uuid.UUID] = None
    ) -> bool:
        """
        Assign task to user.
        
        Args:
            task_id: Task ID
            assigned_to: User ID to assign to
            assigner_user_id: User making the assignment (for access control)
            
        Returns:
            True if assigned successfully, False if task not found
            
        Raises:
            RepositoryError: If operation fails or access is denied
        """
        try:
            # Get task with project info
            task = await self.get_by_id(task_id, load_relationships=True)
            if not task:
                self.logger.warning(f"Task {task_id} not found for assignment")
                return False
            
            # Check project access if assigner specified
            if assigner_user_id:
                project_repo = ProjectRepository(self.session)
                if not await project_repo.check_project_access(task.project_id, assigner_user_id):
                    raise RepositoryError(
                        "Access denied to assign task",
                        "PROJECT_ACCESS_DENIED",
                        {"task_id": str(task_id), "user_id": str(assigner_user_id)}
                    )
            
            # Validate assignee exists
            assignee_exists = await self.session.scalar(
                select(User.id).where(User.id == assigned_to)
            )
            
            if not assignee_exists:
                raise RepositoryError(
                    "Assigned user not found",
                    "ASSIGNEE_NOT_FOUND",
                    {"assigned_to": str(assigned_to)}
                )
            
            # Assign task
            old_assignee = task.assigned_to
            task.assigned_to = assigned_to
            task.updated_at = datetime.now(timezone.utc)
            
            await self.session.flush()
            
            self.logger.info(
                f"Task assigned",
                extra={
                    "task_id": str(task_id),
                    "task_title": task.title,
                    "old_assignee": str(old_assignee) if old_assignee else None,
                    "new_assignee": str(assigned_to),
                    "assigner": str(assigner_user_id) if assigner_user_id else None
                }
            )
            
            return True
            
        except RepositoryError:
            await self.session.rollback()
            raise
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error assigning task {task_id}: {e}")
            raise RepositoryError(
                "Failed to assign task",
                "ASSIGN_TASK_ERROR",
                {
                    "task_id": str(task_id),
                    "assigned_to": str(assigned_to),
                    "error": str(e)
                }
            )
    
    async def update_task_status(
        self,
        task_id: uuid.UUID,
        status: str,
        user_id: Optional[uuid.UUID] = None
    ) -> bool:
        """
        Update task status.
        
        Args:
            task_id: Task ID
            status: New status
            user_id: User making the change (for access control)
            
        Returns:
            True if updated successfully, False if task not found
            
        Raises:
            RepositoryError: If operation fails or access is denied
        """
        try:
            # Get task with project info
            task = await self.get_by_id(task_id, load_relationships=True)
            if not task:
                self.logger.warning(f"Task {task_id} not found for status update")
                return False
            
            # Check access if user specified (assignee or project access)
            if user_id:
                has_access = (
                    task.assigned_to == user_id or  # Assignee can update
                    await ProjectRepository(self.session).check_project_access(
                        task.project_id, user_id
                    )  # Project members can update
                )
                
                if not has_access:
                    raise RepositoryError(
                        "Access denied to update task status",
                        "TASK_ACCESS_DENIED",
                        {"task_id": str(task_id), "user_id": str(user_id)}
                    )
            
            # Update status
            old_status = task.status
            task.status = status
            
            # Set completed timestamp if completing
            if status == 'completed' and old_status != 'completed':
                task.completed_at = datetime.now(timezone.utc)
            elif status != 'completed' and task.completed_at:
                task.completed_at = None
            
            task.updated_at = datetime.now(timezone.utc)
            
            await self.session.flush()
            
            self.logger.info(
                f"Task status updated",
                extra={
                    "task_id": str(task_id),
                    "task_title": task.title,
                    "old_status": old_status,
                    "new_status": status,
                    "user_id": str(user_id) if user_id else None
                }
            )
            
            return True
            
        except RepositoryError:
            await self.session.rollback()
            raise
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error updating task status {task_id}: {e}")
            raise RepositoryError(
                "Failed to update task status",
                "UPDATE_TASK_STATUS_ERROR",
                {
                    "task_id": str(task_id),
                    "status": status,
                    "user_id": str(user_id) if user_id else None,
                    "error": str(e)
                }
            )
    
    async def get_overdue_tasks(self, user_id: Optional[uuid.UUID] = None) -> List[Task]:
        """
        Get overdue tasks.
        
        Args:
            user_id: Filter by assigned user (optional)
            
        Returns:
            List of overdue Task instances
        """
        return await self.get_user_tasks(
            user_id=user_id,
            overdue=True,
            limit=1000
        ) if user_id else await self.get_all(
            filters={
                'due_date__lt': datetime.now(timezone.utc),
                'status__in': ['pending', 'in_progress']
            },
            order_by='due_date',
            load_relationships=True
        )
    
    async def get_tasks_due_soon(
        self,
        days: int = 7,
        user_id: Optional[uuid.UUID] = None
    ) -> List[Task]:
        """
        Get tasks due within specified days.
        
        Args:
            days: Number of days ahead to check
            user_id: Filter by assigned user (optional)
            
        Returns:
            List of Task instances due soon
        """
        return await self.get_user_tasks(
            user_id=user_id,
            due_soon=True,
            limit=1000
        ) if user_id else await self.get_all(
            filters={
                'due_date__lte': datetime.now(timezone.utc) + timedelta(days=days),
                'due_date__gte': datetime.now(timezone.utc),
                'status__in': ['pending', 'in_progress']
            },
            order_by='due_date',
            load_relationships=True
        )
    
    async def search_tasks(
        self,
        query: str,
        project_id: Optional[uuid.UUID] = None,
        user_id: Optional[uuid.UUID] = None,
        limit: int = 50
    ) -> List[Task]:
        """
        Search tasks by title or description.
        
        Args:
            query: Search query
            project_id: Filter by project (optional)
            user_id: Filter by assigned user or for access control
            limit: Maximum number of results
            
        Returns:
            List of matching Task instances
        """
        filters = {
            'title__like': query,
            'description__like': query
        }
        
        if project_id:
            filters['project_id'] = project_id
        
        if user_id:
            filters['assigned_to'] = user_id
        
        return await self.get_all(
            limit=limit,
            filters=filters,
            order_by='updated_at',
            order_desc=True,
            load_relationships=True
        )
    
    async def get_task_statistics(
        self,
        project_id: Optional[uuid.UUID] = None,
        user_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Get task statistics.
        
        Args:
            project_id: Filter by project (optional)
            user_id: Filter by assigned user (optional)
            
        Returns:
            Dictionary with task statistics
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            filters = {}
            
            if project_id:
                filters['project_id'] = project_id
            if user_id:
                filters['assigned_to'] = user_id
            
            # Get all tasks with filters
            tasks = await self.get_all(filters=filters, limit=10000)
            
            # Calculate statistics
            total_tasks = len(tasks)
            
            status_counts = {}
            priority_counts = {}
            
            completed_tasks = 0
            overdue_tasks = 0
            due_soon_tasks = 0
            
            now = datetime.now(timezone.utc)
            week_from_now = now + timedelta(days=7)
            
            for task in tasks:
                # Status counts
                status = task.status
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Priority counts
                priority = task.priority
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
                # Completion tracking
                if status == 'completed':
                    completed_tasks += 1
                
                # Due date tracking
                if task.due_date and status != 'completed':
                    if task.due_date < now:
                        overdue_tasks += 1
                    elif task.due_date <= week_from_now:
                        due_soon_tasks += 1
            
            completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            statistics = {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'overdue_tasks': overdue_tasks,
                'due_soon_tasks': due_soon_tasks,
                'completion_rate': round(completion_rate, 2),
                'status_counts': status_counts,
                'priority_counts': priority_counts,
                'project_id': str(project_id) if project_id else None,
                'user_id': str(user_id) if user_id else None,
                'generated_at': datetime.now(timezone.utc)
            }
            
            self.logger.debug(
                f"Generated task statistics",
                extra={
                    "project_id": str(project_id) if project_id else None,
                    "user_id": str(user_id) if user_id else None,
                    "statistics": statistics
                }
            )
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error getting task statistics: {e}")
            raise RepositoryError(
                "Failed to get task statistics",
                "GET_TASK_STATISTICS_ERROR",
                {
                    "project_id": str(project_id) if project_id else None,
                    "user_id": str(user_id) if user_id else None,
                    "error": str(e)
                }
            )
