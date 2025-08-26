"""
Repository Factory for Dependency Injection

Provides centralized repository creation and management with:
- Singleton pattern for efficiency
- Dependency injection support
- Session management
- Easy access to all repositories
"""

from typing import Dict, Type, TypeVar, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository
from .user_repository import UserRepository
from .project_repository import ProjectRepository
from .task_repository import TaskRepository
from .audit_repository import AuditRepository
from .session_repository import SessionRepository
from ...core.logger import get_logger

T = TypeVar('T', bound=BaseRepository)
logger = get_logger(__name__)


class RepositoryFactory:
    """
    Factory for creating and managing repository instances.
    
    Provides centralized access to all repositories with session management
    and singleton pattern for efficiency.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository factory.
        
        Args:
            session: AsyncSession instance to use for all repositories
        """
        self.session = session
        self._repositories: Dict[str, BaseRepository] = {}
        logger.debug("Repository factory initialized")
    
    def _get_repository(self, repository_class: Type[T], key: str) -> T:
        """
        Get or create repository instance.
        
        Args:
            repository_class: Repository class to instantiate
            key: Cache key for the repository
            
        Returns:
            Repository instance
        """
        if key not in self._repositories:
            self._repositories[key] = repository_class(self.session)
            logger.debug(f"Created {key} repository instance")
        
        return self._repositories[key]
    
    def get_user_repository(self) -> UserRepository:
        """
        Get user repository instance.
        
        Returns:
            UserRepository instance
        """
        return self._get_repository(UserRepository, 'user')
    
    def get_project_repository(self) -> ProjectRepository:
        """
        Get project repository instance.
        
        Returns:
            ProjectRepository instance
        """
        return self._get_repository(ProjectRepository, 'project')
    
    def get_task_repository(self) -> TaskRepository:
        """
        Get task repository instance.
        
        Returns:
            TaskRepository instance
        """
        return self._get_repository(TaskRepository, 'task')
    
    def get_audit_repository(self) -> AuditRepository:
        """
        Get audit repository instance.
        
        Returns:
            AuditRepository instance
        """
        return self._get_repository(AuditRepository, 'audit')
    
    def get_session_repository(self) -> SessionRepository:
        """
        Get session repository instance.
        
        Returns:
            SessionRepository instance
        """
        return self._get_repository(SessionRepository, 'session')
    
    def get_repository(self, repository_class: Type[T]) -> T:
        """
        Get repository instance by class.
        
        Args:
            repository_class: Repository class
            
        Returns:
            Repository instance
            
        Raises:
            ValueError: If repository class is not supported
        """
        class_mapping = {
            UserRepository: 'user',
            ProjectRepository: 'project',
            TaskRepository: 'task',
            AuditRepository: 'audit',
            SessionRepository: 'session'
        }
        
        key = class_mapping.get(repository_class)
        if not key:
            raise ValueError(f"Unsupported repository class: {repository_class.__name__}")
        
        return self._get_repository(repository_class, key)
    
    def clear_cache(self) -> None:
        """
        Clear repository cache.
        
        Useful for testing or when session changes.
        """
        self._repositories.clear()
        logger.debug("Repository cache cleared")
    
    def get_all_repositories(self) -> Dict[str, BaseRepository]:
        """
        Get all cached repository instances.
        
        Returns:
            Dictionary of repository instances by name
        """
        return self._repositories.copy()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health checks on all repositories.
        
        Returns:
            Dictionary with health status of each repository
        """
        health_status = {}
        
        for name, repository in self._repositories.items():
            try:
                # Simple health check - try to count records
                await repository.count()
                health_status[name] = True
            except Exception as e:
                logger.error(f"Health check failed for {name} repository: {e}")
                health_status[name] = False
        
        return health_status
    
    def __str__(self) -> str:
        """String representation of factory."""
        cached_repos = list(self._repositories.keys())
        return f"RepositoryFactory(cached_repositories={cached_repos})"
    
    def __repr__(self) -> str:
        """Detailed representation of factory."""
        return (
            f"RepositoryFactory("
            f"session={self.session}, "
            f"cached_repositories={list(self._repositories.keys())}"
            f")"
        )


# Convenience functions for common repository access patterns
async def create_repository_factory(session: AsyncSession) -> RepositoryFactory:
    """
    Create and validate repository factory.
    
    Args:
        session: AsyncSession instance
        
    Returns:
        RepositoryFactory instance
        
    Raises:
        ValueError: If session is None or invalid
    """
    if session is None:
        raise ValueError("Session cannot be None")
    
    factory = RepositoryFactory(session)
    logger.info("Repository factory created and validated")
    
    return factory


def get_repository_classes() -> Dict[str, Type[BaseRepository]]:
    """
    Get mapping of repository names to classes.
    
    Returns:
        Dictionary mapping names to repository classes
    """
    return {
        'user': UserRepository,
        'project': ProjectRepository,
        'task': TaskRepository,
        'audit': AuditRepository,
        'session': SessionRepository
    }
