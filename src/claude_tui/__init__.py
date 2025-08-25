"""
Claude-TUI: Intelligent AI-powered Terminal User Interface.

A sophisticated tool that revolutionizes software development through
AI orchestration, continuous validation, and anti-hallucination mechanisms.
"""

__version__ = "0.1.0"
__author__ = "Claude TUI Team"
__email__ = "team@claude-tui.dev"
__description__ = "Intelligent AI-powered Terminal User Interface for advanced software development"

# Core imports - will be available when core modules are implemented
try:
    from core.project_manager import ProjectManager
    from core.task_engine import TaskEngine  
    from core.validator import ProgressValidator
    from core.ai_interface import AIInterface
    from ui.main_app import ClaudeTUIApp
except ImportError:
    # Graceful fallback for development
    ProjectManager = None
    TaskEngine = None
    ProgressValidator = None
    AIInterface = None
    ClaudeTUIApp = None

__all__ = [
    "ProjectManager",
    "TaskEngine", 
    "ProgressValidator",
    "AIInterface",
    "ClaudeTUIApp",
    "__version__",
]