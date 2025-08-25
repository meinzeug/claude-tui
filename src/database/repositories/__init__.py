"""
Repository Pattern Implementation

This module provides the repository pattern implementation with:
- BaseRepository abstract class
- Concrete repository implementations
- Repository factory for dependency injection
- Async/await support throughout
"""

from .base import BaseRepository, RepositoryError
from .user_repository import UserRepository
from .project_repository import ProjectRepository  
from .task_repository import TaskRepository
from .audit_repository import AuditRepository
from .session_repository import SessionRepository
from .factory import RepositoryFactory

__all__ = [
    'BaseRepository',
    'RepositoryError',
    'UserRepository',
    'ProjectRepository',
    'TaskRepository', 
    'AuditRepository',
    'SessionRepository',
    'RepositoryFactory'
]