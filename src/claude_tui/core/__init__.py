"""Claude-TUI Core Module."""

# Make core classes available at package level
from .dependency_checker import DependencyChecker, get_dependency_checker
from .config_manager import ConfigManager
from .logger import get_logger, setup_logging
from .project_manager import ProjectManager
from .task_engine import TaskEngine
from .ai_interface import AIInterface

__all__ = [
    'DependencyChecker',
    'get_dependency_checker',
    'ConfigManager',
    'get_logger',
    'setup_logging',
    'ProjectManager',
    'TaskEngine',
    'AIInterface'
]