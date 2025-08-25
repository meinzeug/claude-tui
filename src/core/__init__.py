"""
Core business logic modules for claude-tiu.

This package contains the essential components that orchestrate AI-powered
project management with anti-hallucination validation and real-time progress tracking.

Core Modules:
- project_manager: Central orchestration logic
- task_engine: Workflow management and execution
- validator: Anti-hallucination validation pipeline
- config_manager: Configuration and settings management
- ai_interface: Claude Code/Flow integration layer
- utils: Utility functions and helpers

Key Features:
- SOLID principles compliance
- Comprehensive error handling and logging
- Anti-hallucination validation pipeline
- Real-time progress monitoring with authenticity checks
- Intelligent AI service routing
- Resource management and coordination
"""

from .project_manager import ProjectManager, Project, create_simple_project, validate_project_quality
from .task_engine import TaskEngine, Task, Workflow, TaskStatus, create_task, execute_simple_workflow
from .validator import ProgressValidator, ValidationResult, Issue, validate_project_authenticity, quick_file_validation
from .config_manager import ConfigManager, ProjectConfig, load_project_config, save_project_config
from .ai_interface import AIInterface, AIContext, execute_simple_ai_task
from .utils import (
    SafeFileOperations, AsyncFileOperations, setup_logging, get_logger,
    error_handler, retry_on_failure, validate_path, format_file_size,
    ContextTimer, AsyncContextTimer, ensure_directory, get_project_root
)
from .types import (
    ProjectState, 
    ProgressReport, 
    DevelopmentResult,
    AITaskResult,
    ValidationReport,
    Priority,
    IssueType,
    Severity,
    ValidationType,
    ExecutionStrategy
)

__all__ = [
    # Core classes
    'ProjectManager',
    'Project', 
    'TaskEngine', 
    'Task',
    'Workflow',
    'TaskStatus',
    'ProgressValidator',
    'ValidationResult', 
    'Issue',
    'ConfigManager',
    'ProjectConfig',
    'AIInterface',
    'AIContext',
    
    # Utility classes
    'SafeFileOperations',
    'AsyncFileOperations',
    'ContextTimer',
    'AsyncContextTimer',
    
    # Data types and enums
    'ProjectState',
    'ProgressReport',
    'DevelopmentResult', 
    'AITaskResult',
    'ValidationReport',
    'Priority',
    'IssueType',
    'Severity',
    'ValidationType',
    'ExecutionStrategy',
    
    # Utility functions
    'setup_logging',
    'get_logger',
    'error_handler',
    'retry_on_failure',
    'validate_path',
    'format_file_size',
    'ensure_directory',
    'get_project_root',
    
    # Convenience functions
    'create_simple_project',
    'validate_project_quality',
    'create_task',
    'execute_simple_workflow',
    'validate_project_authenticity',
    'quick_file_validation',
    'load_project_config',
    'save_project_config',
    'execute_simple_ai_task'
]

__version__ = '1.0.0'
__author__ = 'Claude Code AI Development Team'
__description__ = 'Core business logic for claude-tiu AI-powered project management'

# Initialize default logging
_logger = setup_logging()
_logger.info("Claude-TIU core modules initialized successfully")

# Module-level configuration
DEFAULT_CONFIG = {
    'max_concurrent_tasks': 5,
    'enable_validation': True,
    'enable_monitoring': True,
    'log_level': 'INFO',
    'validation_level': 'strict',
    'auto_fix_enabled': True
}

def get_version() -> str:
    """Get package version."""
    return __version__

def get_core_components() -> dict:
    """Get information about core components."""
    return {
        'ProjectManager': 'Central orchestration for project lifecycle management',
        'TaskEngine': 'Advanced task scheduling and workflow execution',
        'ProgressValidator': 'Anti-hallucination validation with authenticity checking',
        'ConfigManager': 'Hierarchical configuration management',
        'AIInterface': 'Unified Claude Code/Flow integration with intelligent routing',
        'SafeFileOperations': 'Robust file system operations with error handling'
    }

def validate_system_requirements() -> dict:
    """Validate system requirements and capabilities."""
    import sys
    import platform
    
    requirements = {
        'python_version': {
            'current': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'required': '3.9+',
            'satisfied': sys.version_info >= (3, 9)
        },
        'platform': {
            'current': platform.system(),
            'supported': ['Linux', 'Darwin', 'Windows'],
            'satisfied': platform.system() in ['Linux', 'Darwin', 'Windows']
        },
        'memory': {
            'recommended_mb': 2048,
            'available': 'unknown'  # Would need psutil to check
        }
    }
    
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        requirements['memory']['available'] = memory_info.available // (1024 * 1024)
        requirements['memory']['satisfied'] = memory_info.available >= (2048 * 1024 * 1024)
    except ImportError:
        requirements['memory']['satisfied'] = True  # Assume OK if we can't check
    
    return requirements

def check_dependencies() -> dict:
    """Check availability of optional dependencies."""
    dependencies = {}
    
    # Check for optional performance dependencies
    for package in ['uvloop', 'orjson', 'msgpack', 'lz4', 'psutil']:
        try:
            __import__(package)
            dependencies[package] = {'available': True, 'version': 'unknown'}
        except ImportError:
            dependencies[package] = {'available': False, 'version': None}
    
    # Check for Claude CLI tools
    import subprocess
    try:
        result = subprocess.run(['claude', '--version'], 
                              capture_output=True, text=True, timeout=5)
        dependencies['claude_cli'] = {
            'available': result.returncode == 0,
            'version': result.stdout.strip() if result.returncode == 0 else None
        }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        dependencies['claude_cli'] = {'available': False, 'version': None}
    
    return dependencies

# Performance optimization: Import heavy modules only when needed
def _lazy_import_heavy_modules():
    """Lazy import of heavy modules for better startup performance."""
    global _heavy_modules_loaded
    if hasattr(_lazy_import_heavy_modules, '_loaded'):
        return
    
    try:
        # Import heavy optional dependencies
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        _logger.info("uvloop enabled for better async performance")
    except ImportError:
        pass
    
    _lazy_import_heavy_modules._loaded = True

# Initialization hooks for the core system
def initialize_core_system(config: dict = None) -> dict:
    """
    Initialize the complete core system with configuration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Initialization result summary
    """
    config = config or {}
    effective_config = {**DEFAULT_CONFIG, **config}
    
    _logger.info("Initializing claude-tiu core system")
    
    # System requirements check
    requirements_check = validate_system_requirements()
    if not all(req.get('satisfied', True) for req in requirements_check.values()):
        _logger.warning("Some system requirements not satisfied")
    
    # Dependencies check
    deps_check = check_dependencies()
    available_deps = sum(1 for dep in deps_check.values() if dep['available'])
    _logger.info(f"Optional dependencies: {available_deps}/{len(deps_check)} available")
    
    # Initialize core components with configuration
    try:
        project_manager = ProjectManager(
            enable_validation=effective_config['enable_validation'],
            max_concurrent_tasks=effective_config['max_concurrent_tasks']
        )
        
        task_engine = TaskEngine(
            max_concurrent_tasks=effective_config['max_concurrent_tasks'],
            enable_validation=effective_config['enable_validation'],
            enable_monitoring=effective_config['enable_monitoring']
        )
        
        validator = ProgressValidator() if effective_config['enable_validation'] else None
        
        ai_interface = AIInterface(enable_validation=effective_config['enable_validation'])
        
        _logger.info("Core system initialization completed successfully")
        
        return {
            'success': True,
            'components_initialized': ['ProjectManager', 'TaskEngine', 'ProgressValidator', 'AIInterface'],
            'configuration': effective_config,
            'system_requirements': requirements_check,
            'dependencies': deps_check,
            'performance_optimizations': {
                'uvloop': 'uvloop' in deps_check and deps_check['uvloop']['available'],
                'fast_json': 'orjson' in deps_check and deps_check['orjson']['available']
            }
        }
        
    except Exception as e:
        _logger.error(f"Core system initialization failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'system_requirements': requirements_check,
            'dependencies': deps_check
        }

# Auto-initialize when imported (can be disabled by setting environment variable)
import os
if os.getenv('CLAUDE_TIU_SKIP_AUTO_INIT') != '1':
    try:
        _lazy_import_heavy_modules()
    except Exception as e:
        _logger.warning(f"Heavy module initialization failed: {e}")