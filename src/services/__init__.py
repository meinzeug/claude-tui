"""
Service Layer Package for claude-tiu.

Provides business logic services with dependency injection, 
error handling, and coordination capabilities.

Services:
- BaseService: Foundation service class with DI
- AIService: AI Integration Management
- ProjectService: Project Operations
- TaskService: Task Management 
- ValidationService: Anti-Hallucination Validation
"""

from .base import BaseService
from .ai_service import AIService
from .project_service import ProjectService
from .task_service import TaskService
from .validation_service import ValidationService

__all__ = [
    'BaseService',
    'AIService', 
    'ProjectService',
    'TaskService',
    'ValidationService'
]