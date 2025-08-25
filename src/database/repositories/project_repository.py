"""
Project Repository Implementation

Provides project-specific database operations with:
- Access control and ownership management
- Project lifecycle management
- Task relationships
- Collaboration features
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_, or_
from sqlalchemy.sql import Select

from .base import BaseRepository, RepositoryError
from ..models import Project, User, Task
from ...core.logger import get_logger

logger = get_logger(__name__)


class ProjectRepository(BaseRepository[Project]):
    """Project repository with access control and lifecycle management."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize project repository.
        
        Args:
            session: AsyncSession instance
        """
        super().__init__(session, Project)
    
    def _add_relationship_loading(self, query: Select) -> Select:
        """
        Add eager loading for project relationships.
        
        Args:
            query: SQLAlchemy query
            
        Returns:
            Query with relationship loading options
        """
        return query.options(
            selectinload(Project.owner),
            selectinload(Project.tasks).selectinload(Task.assignee)
        )
    
    async def get_user_projects(
        self, 
        user_id: uuid.UUID, 
        include_public: bool = True,
        include_archived: bool = False,
        project_type: Optional[str] = None
    ) -> List[Project]:
        """
        Get projects accessible to user.
        
        Args:
            user_id: User ID
            include_public: Whether to include public projects
            include_archived: Whether to include archived projects
            project_type: Filter by project type
            
        Returns:
            List of accessible Project instances
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            query = select(Project).options(
                selectinload(Project.owner),
                selectinload(Project.tasks)
            )
            
            # Access control filters
            if include_public:
                query = query.where(
                    or_(
                        Project.owner_id == user_id,
                        Project.is_public == True
                    )
                )
            else:
                query = query.where(Project.owner_id == user_id)
            
            # Archive filter
            if not include_archived:
                query = query.where(Project.is_archived == False)
            
            # Project type filter
            if project_type:
                query = query.where(Project.project_type == project_type)
            
            query = query.order_by(Project.created_at.desc())
            
            result = await self.session.execute(query)
            projects = result.scalars().all()
            
            self.logger.debug(
                f"Retrieved {len(projects)} projects for user {user_id}",
                extra={
                    "user_id": str(user_id),
                    "include_public": include_public,
                    "include_archived": include_archived,
                    "project_type": project_type
                }
            )
            
            return list(projects)
            
        except Exception as e:
            self.logger.error(f"Error getting projects for user {user_id}: {e}")
            raise RepositoryError(
                "Failed to retrieve user projects",
                "GET_USER_PROJECTS_ERROR",
                {
                    "user_id": str(user_id),
                    "include_public": include_public,
                    "project_type": project_type,
                    "error": str(e)
                }
            )
    
    async def create_project(
        self,
        owner_id: uuid.UUID,
        name: str,
        description: Optional[str] = None,
        project_type: str = "general",
        is_public: bool = False,
        **kwargs
    ) -> Optional[Project]:
        """
        Create project with ownership validation.
        
        Args:
            owner_id: Project owner user ID
            name: Project name
            description: Project description
            project_type: Type of project
            is_public: Whether project is public
            **kwargs: Additional project fields
            
        Returns:
            Created Project instance
            
        Raises:
            RepositoryError: If creation fails
        """
        try:
            # Validate owner exists
            owner_exists = await self.session.scalar(
                select(User.id).where(User.id == owner_id)
            )
            
            if not owner_exists:
                raise RepositoryError(
                    "Project owner not found",
                    "PROJECT_OWNER_NOT_FOUND",
                    {"owner_id": str(owner_id)}
                )
            
            # Create project
            project = Project(
                name=name,
                description=description,
                owner_id=owner_id,
                project_type=project_type,
                is_public=is_public,
                **kwargs
            )
            
            self.session.add(project)
            await self.session.flush()
            await self.session.refresh(project)
            
            self.logger.info(
                f"Created project: {name}",
                extra={
                    "project_id": str(project.id),
                    "owner_id": str(owner_id),
                    "project_type": project_type,
                    "is_public": is_public
                }
            )
            
            return project
            
        except RepositoryError:
            await self.session.rollback()
            raise
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error creating project: {e}")
            raise RepositoryError(
                "Failed to create project",
                "CREATE_PROJECT_ERROR",
                {
                    "name": name,
                    "owner_id": str(owner_id),
                    "project_type": project_type,
                    "error": str(e)
                }
            )
    
    async def check_project_access(
        self, 
        project_id: uuid.UUID, 
        user_id: uuid.UUID,
        require_ownership: bool = False
    ) -> bool:
        """
        Check if user has access to project.
        
        Args:
            project_id: Project ID
            user_id: User ID
            require_ownership: Whether to require ownership (vs just access)
            
        Returns:
            True if user has access, False otherwise
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            if require_ownership:
                # Check ownership only
                result = await self.session.execute(
                    select(Project.id).where(
                        and_(
                            Project.id == project_id,
                            Project.owner_id == user_id
                        )
                    )
                )
            else:
                # Check ownership or public access
                result = await self.session.execute(
                    select(Project.id).where(
                        and_(
                            Project.id == project_id,
                            or_(
                                Project.owner_id == user_id,
                                Project.is_public == True
                            )
                        )
                    )
                )
            
            has_access = result.scalar_one_or_none() is not None
            
            self.logger.debug(
                f"Access check for project {project_id}, user {user_id}: {has_access}",
                extra={
                    "project_id": str(project_id),
                    "user_id": str(user_id),
                    "require_ownership": require_ownership,
                    "has_access": has_access
                }
            )
            
            return has_access
            
        except Exception as e:
            self.logger.error(f"Error checking project access: {e}")
            raise RepositoryError(
                "Failed to check project access",
                "CHECK_PROJECT_ACCESS_ERROR",
                {
                    "project_id": str(project_id),
                    "user_id": str(user_id),
                    "require_ownership": require_ownership,
                    "error": str(e)
                }
            )
    
    async def get_project_with_tasks(
        self, 
        project_id: uuid.UUID, 
        user_id: Optional[uuid.UUID] = None
    ) -> Optional[Project]:
        """
        Get project with all tasks loaded.
        
        Args:
            project_id: Project ID
            user_id: User ID for access control (optional)
            
        Returns:
            Project instance with tasks or None if not found/accessible
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            # Check access if user_id provided
            if user_id and not await self.check_project_access(project_id, user_id):
                self.logger.warning(
                    f"User denied access to project {project_id}",
                    extra={"project_id": str(project_id), "user_id": str(user_id)}
                )
                return None
            
            # Get project with tasks
            result = await self.session.execute(
                select(Project)
                .options(
                    selectinload(Project.owner),
                    selectinload(Project.tasks).selectinload(Task.assignee)
                )
                .where(Project.id == project_id)
            )
            
            project = result.scalar_one_or_none()
            
            if project:
                self.logger.debug(
                    f"Retrieved project with {len(project.tasks)} tasks",
                    extra={
                        "project_id": str(project_id),
                        "task_count": len(project.tasks)
                    }
                )
            
            return project
            
        except Exception as e:
            self.logger.error(f"Error getting project with tasks: {e}")
            raise RepositoryError(
                "Failed to retrieve project with tasks",
                "GET_PROJECT_WITH_TASKS_ERROR",
                {
                    "project_id": str(project_id),
                    "user_id": str(user_id) if user_id else None,
                    "error": str(e)
                }
            )
    
    async def archive_project(self, project_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """
        Archive project (owner only).
        
        Args:
            project_id: Project ID
            user_id: User ID (must be owner)
            
        Returns:
            True if archived successfully, False if not found or no access
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            # Check ownership
            if not await self.check_project_access(project_id, user_id, require_ownership=True):
                self.logger.warning(
                    f"User {user_id} attempted to archive project {project_id} without ownership"
                )
                return False
            
            # Archive project
            project = await self.get_by_id(project_id)
            if not project:
                return False
            
            project.is_archived = True
            project.updated_at = datetime.now(timezone.utc)
            
            await self.session.flush()
            
            self.logger.info(
                f"Archived project",
                extra={
                    "project_id": str(project_id),
                    "user_id": str(user_id),
                    "project_name": project.name
                }
            )
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error archiving project {project_id}: {e}")
            raise RepositoryError(
                "Failed to archive project",
                "ARCHIVE_PROJECT_ERROR",
                {
                    "project_id": str(project_id),
                    "user_id": str(user_id),
                    "error": str(e)
                }
            )
    
    async def restore_project(self, project_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """
        Restore archived project (owner only).
        
        Args:
            project_id: Project ID
            user_id: User ID (must be owner)
            
        Returns:
            True if restored successfully, False if not found or no access
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            # Check ownership
            if not await self.check_project_access(project_id, user_id, require_ownership=True):
                self.logger.warning(
                    f"User {user_id} attempted to restore project {project_id} without ownership"
                )
                return False
            
            # Restore project
            project = await self.get_by_id(project_id)
            if not project:
                return False
            
            project.is_archived = False
            project.updated_at = datetime.now(timezone.utc)
            
            await self.session.flush()
            
            self.logger.info(
                f"Restored project",
                extra={
                    "project_id": str(project_id),
                    "user_id": str(user_id),
                    "project_name": project.name
                }
            )
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error restoring project {project_id}: {e}")
            raise RepositoryError(
                "Failed to restore project",
                "RESTORE_PROJECT_ERROR",
                {
                    "project_id": str(project_id),
                    "user_id": str(user_id),
                    "error": str(e)
                }
            )
    
    async def get_public_projects(
        self,
        limit: int = 50,
        project_type: Optional[str] = None
    ) -> List[Project]:
        """
        Get list of public projects.
        
        Args:
            limit: Maximum number of projects to return
            project_type: Filter by project type
            
        Returns:
            List of public Project instances
        """
        filters = {'is_public': True, 'is_archived': False}
        if project_type:
            filters['project_type'] = project_type
        
        return await self.get_all(
            limit=limit,
            filters=filters,
            order_by='created_at',
            order_desc=True,
            load_relationships=True
        )
    
    async def search_projects(
        self,
        query: str,
        user_id: Optional[uuid.UUID] = None,
        limit: int = 50
    ) -> List[Project]:
        """
        Search projects by name or description.
        
        Args:
            query: Search query
            user_id: User ID for access control (optional)
            limit: Maximum number of results
            
        Returns:
            List of matching Project instances
        """
        filters = {
            'name__like': query,
            'description__like': query,
            'is_archived': False
        }
        
        # Add access control if user specified
        if user_id:
            # This is a simplified approach - in practice you might want
            # to handle this with a more complex query
            projects = await self.get_user_projects(
                user_id=user_id,
                include_public=True,
                include_archived=False
            )
            
            # Filter by search terms
            return [
                p for p in projects
                if query.lower() in (p.name or '').lower() or
                   query.lower() in (p.description or '').lower()
            ][:limit]
        else:
            # Public projects only
            filters['is_public'] = True
            return await self.get_all(
                limit=limit,
                filters=filters,
                order_by='created_at',
                order_desc=True
            )
    
    async def get_project_statistics(self, project_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get project statistics.
        
        Args:
            project_id: Project ID
            
        Returns:
            Dictionary with project statistics
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            # Get project with tasks
            project = await self.get_project_with_tasks(project_id)
            if not project:
                raise RepositoryError(
                    "Project not found",
                    "PROJECT_NOT_FOUND",
                    {"project_id": str(project_id)}
                )
            
            # Calculate statistics
            tasks = project.tasks
            total_tasks = len(tasks)
            
            task_counts = {}
            for task in tasks:
                status = task.status
                task_counts[status] = task_counts.get(status, 0) + 1
            
            completed_tasks = task_counts.get('completed', 0)
            in_progress_tasks = task_counts.get('in_progress', 0)
            pending_tasks = task_counts.get('pending', 0)
            
            completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            statistics = {
                'project_id': str(project_id),
                'project_name': project.name,
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'in_progress_tasks': in_progress_tasks,
                'pending_tasks': pending_tasks,
                'completion_rate': round(completion_rate, 2),
                'task_counts_by_status': task_counts,
                'created_at': project.created_at,
                'updated_at': project.updated_at,
                'is_public': project.is_public,
                'is_archived': project.is_archived,
                'project_type': project.project_type
            }
            
            self.logger.debug(
                f"Generated statistics for project {project_id}",
                extra={"project_id": str(project_id), "statistics": statistics}
            )
            
            return statistics
            
        except RepositoryError:
            raise
        except Exception as e:
            self.logger.error(f"Error getting project statistics for {project_id}: {e}")
            raise RepositoryError(
                "Failed to get project statistics",
                "GET_PROJECT_STATISTICS_ERROR",
                {"project_id": str(project_id), "error": str(e)}
            )
