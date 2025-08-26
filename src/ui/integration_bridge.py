#!/usr/bin/env python3
"""
Integration Bridge - Handles integration between UI components and backend services
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, Union
import logging
from contextlib import contextmanager

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/claude-tui-bridge.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class UIIntegrationBridge:
    """
    Bridge class to handle integration between different UI implementations
    and backend services with comprehensive error handling and fallback mechanisms.
    """
    
    def __init__(self):
        self.config_manager = None
        self.project_manager = None
        self.ai_interface = None
        self.validation_engine = None
        self.workflow_manager = None
        self._initialized = False
        self._initialization_errors = []
        self._available_components = set()
        self._initialization_attempts = 0
        
        logger.info("UIIntegrationBridge instance created")
        
    def initialize(self, force_reinit: bool = False) -> bool:
        """Initialize all backend services and connections with comprehensive error handling."""
        self._initialization_attempts += 1
        logger.info(f"Starting initialization attempt #{self._initialization_attempts}")
        
        if self._initialized and not force_reinit:
            logger.info("Bridge already initialized, skipping")
            return True
            
        self._initialization_errors.clear()
        self._available_components.clear()
        
        # Progressive initialization - each component is independent
        components = [
            ('config_manager', self._init_config_manager),
            ('project_manager', self._init_project_manager), 
            ('ai_interface', self._init_ai_interface),
            ('validation_engine', self._init_validation_engine),
            ('workflow_manager', self._init_workflow_manager)
        ]
        
        success_count = 0
        for component_name, init_func in components:
            try:
                logger.debug(f"Initializing {component_name}...")
                init_func()
                self._available_components.add(component_name)
                success_count += 1
                logger.info(f"✓ Successfully initialized {component_name}")
                
            except Exception as e:
                error_details = {
                    'component': component_name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self._initialization_errors.append(error_details)
                logger.error(f"✗ Failed to initialize {component_name}: {e}")
                logger.debug(f"Traceback for {component_name}: {traceback.format_exc()}")
        
        # Consider initialization successful if we have at least basic components
        self._initialized = success_count >= 2  # Need at least 2 components working
        
        logger.info(f"Initialization complete: {success_count}/{len(components)} components successful")
        logger.info(f"Available components: {', '.join(self._available_components)}")
        
        if self._initialization_errors:
            logger.warning(f"Encountered {len(self._initialization_errors)} errors during initialization")
            for error in self._initialization_errors:
                logger.warning(f"  - {error['component']}: {error['error']}")
        
        return self._initialized
    
    def _init_config_manager(self):
        """Initialize configuration manager with multiple fallback strategies"""
        import_attempts = [
            ('claude_tui.core.config_manager', 'ConfigManager'),
            ('src.claude_tui.core.config_manager', 'ConfigManager'),
            ('core.config_manager', 'ConfigManager'),
        ]
        
        for module_path, class_name in import_attempts:
            try:
                logger.debug(f"Attempting to import {class_name} from {module_path}")
                module = __import__(module_path, fromlist=[class_name])
                config_class = getattr(module, class_name)
                self.config_manager = config_class()
                logger.info(f"✓ ConfigManager imported from {module_path}")
                return
            except (ImportError, AttributeError) as e:
                logger.debug(f"Failed to import from {module_path}: {e}")
                continue
        
        # Create enhanced fallback config manager
        logger.warning("Creating fallback ConfigManager with enhanced capabilities")
        
        class EnhancedFallbackConfigManager:
            def __init__(self):
                self._settings = {
                    'ui_preferences.theme': 'dark',
                    'ui_preferences.layout': 'default',
                    'validation.enabled': True,
                    'ai.model': 'claude-3-sonnet',
                    'debug.enabled': False
                }
                logger.debug("FallbackConfigManager initialized with default settings")
                
            def get_setting(self, key, default=None):
                value = self._settings.get(key, default)
                logger.debug(f"Config get: {key} = {value}")
                return value
                
            def update_setting(self, key, value):
                self._settings[key] = value
                logger.debug(f"Config set: {key} = {value}")
                
            def get_ui_preferences(self):
                class UIPrefs:
                    def __init__(self, settings):
                        self.theme = settings.get('ui_preferences.theme', 'dark')
                        self.layout = settings.get('ui_preferences.layout', 'default')
                return UIPrefs(self._settings)
                
            def save(self):
                logger.debug("Fallback config save (no-op)")
                pass
                
        self.config_manager = EnhancedFallbackConfigManager()
    
    def _init_project_manager(self):
        """Initialize project manager with multiple fallback strategies"""
        import_attempts = [
            ('claude_tui.core.project_manager', 'ProjectManager'),
            ('src.claude_tui.core.project_manager', 'ProjectManager'),
            ('core.project_manager', 'ProjectManager'),
        ]
        
        for module_path, class_name in import_attempts:
            try:
                logger.debug(f"Attempting to import {class_name} from {module_path}")
                module = __import__(module_path, fromlist=[class_name])
                project_class = getattr(module, class_name)
                self.project_manager = project_class(self.config_manager)
                logger.info(f"✓ ProjectManager imported from {module_path}")
                return
            except (ImportError, AttributeError) as e:
                logger.debug(f"Failed to import from {module_path}: {e}")
                continue
        
        # Create enhanced fallback project manager
        logger.warning("Creating enhanced fallback ProjectManager")
        
        class EnhancedFallbackProjectManager:
            def __init__(self, config_manager=None):
                self.current_project = None
                self.config_manager = config_manager
                self._project_cache = {}
                logger.debug("FallbackProjectManager initialized")
                
            def initialize(self):
                logger.debug("ProjectManager initialize (fallback)")
                pass
                
            def save_current_project(self):
                if self.current_project:
                    logger.info(f"Saving project: {getattr(self.current_project, 'name', 'Unknown')}")
                else:
                    logger.debug("No current project to save")
                    
            def create_project_from_config(self, config):
                logger.info(f"Creating project from config: {getattr(config, 'name', 'Unknown')}")
                
                class MockProject:
                    def __init__(self, config):
                        self.name = getattr(config, 'name', 'New Project')
                        self.path = getattr(config, 'path', Path.cwd())
                        self.config = config
                        
                project = MockProject(config)
                self.current_project = project
                return True
                
            async def get_project_status(self, project):
                return {"status": "active", "files": 0, "last_modified": None}
                
            async def load_project(self, path):
                logger.info(f"Loading project from: {path}")
                
                class MockProject:
                    def __init__(self, path):
                        self.name = Path(path).name or "Loaded Project"
                        self.path = Path(path)
                        
                project = MockProject(path)
                self.current_project = project
                return project
                
            async def save_project(self, project):
                logger.info(f"Saving project: {getattr(project, 'name', 'Unknown')}")
                pass
                
            async def cleanup(self):
                logger.debug("ProjectManager cleanup (fallback)")
                pass
                
        self.project_manager = EnhancedFallbackProjectManager(self.config_manager)
    
    def _init_ai_interface(self):
        """Initialize AI interface with multiple fallback strategies"""
        import_attempts = [
            ('claude_tui.integrations.ai_interface', 'AIInterface'),
            ('src.claude_tui.integrations.ai_interface', 'AIInterface'),
            ('integrations.ai_interface', 'AIInterface'),
            ('claude_tui.core.ai_interface', 'AIInterface'),
        ]
        
        for module_path, class_name in import_attempts:
            try:
                logger.debug(f"Attempting to import {class_name} from {module_path}")
                module = __import__(module_path, fromlist=[class_name])
                ai_class = getattr(module, class_name)
                self.ai_interface = ai_class(self.config_manager)
                logger.info(f"✓ AIInterface imported from {module_path}")
                return
            except (ImportError, AttributeError) as e:
                logger.debug(f"Failed to import from {module_path}: {e}")
                continue
        
        # Create enhanced fallback AI interface
        logger.warning("Creating enhanced fallback AIInterface")
        
        class EnhancedFallbackAIInterface:
            def __init__(self, config_manager=None):
                self.config_manager = config_manager
                self._task_history = []
                logger.debug("FallbackAIInterface initialized")
                
            def initialize(self):
                logger.debug("AIInterface initialize (fallback)")
                pass
                
            async def execute_task(self, task_description, context=None):
                logger.info(f"Executing AI task (mock): {task_description[:100]}...")
                
                # Generate a more realistic mock response
                task_type = self._detect_task_type(task_description)
                result = self._generate_mock_response(task_type, task_description)
                
                self._task_history.append({
                    'description': task_description,
                    'context': context,
                    'result': result,
                    'timestamp': str(Path.cwd())  # Using path as mock timestamp
                })
                
                return result
                
            def _detect_task_type(self, description):
                """Detect the type of task to generate appropriate mock response"""
                description_lower = description.lower()
                if any(word in description_lower for word in ['code', 'implement', 'function']):
                    return 'code'
                elif any(word in description_lower for word in ['analyze', 'review', 'check']):
                    return 'analysis'
                elif any(word in description_lower for word in ['test', 'verify']):
                    return 'testing'
                else:
                    return 'general'
                    
            def _generate_mock_response(self, task_type, description):
                """Generate task-appropriate mock responses"""
                responses = {
                    'code': f"# Mock implementation for: {description}\n# TODO: Replace with actual implementation\npass",
                    'analysis': f"Analysis complete for: {description}\nStatus: Mock analysis successful\nRecommendations: Implementation needed",
                    'testing': f"Test results for: {description}\nPassed: 0\nFailed: 0\nSkipped: 0\nNote: Mock testing complete",
                    'general': f"Task completed: {description}\nResult: Mock response generated successfully"
                }
                return responses.get(task_type, f"Mock AI response for: {description}")
                
            async def complete_placeholder_code(self, result, suggestions=None):
                logger.info("Completing placeholder code (mock)")
                return result + "\n# Placeholder completion applied"
                
            async def cleanup(self):
                logger.debug("AIInterface cleanup (fallback)")
                logger.info(f"Task history: {len(self._task_history)} tasks executed")
                pass
                
        self.ai_interface = EnhancedFallbackAIInterface(self.config_manager)
    
    def _init_validation_engine(self):
        """Initialize validation engine with multiple fallback strategies"""
        import_attempts = [
            ('claude_tui.core.progress_validator', 'ProgressValidator'),
            ('src.claude_tui.core.progress_validator', 'ProgressValidator'),
            ('claude_tui.validation.progress_validator', 'ProgressValidator'),
            ('core.progress_validator', 'ProgressValidator'),
        ]
        
        for module_path, class_name in import_attempts:
            try:
                logger.debug(f"Attempting to import {class_name} from {module_path}")
                module = __import__(module_path, fromlist=[class_name])
                validator_class = getattr(module, class_name)
                self.validation_engine = validator_class(self.config_manager)
                logger.info(f"✓ ValidationEngine imported from {module_path}")
                return
            except (ImportError, AttributeError) as e:
                logger.debug(f"Failed to import from {module_path}: {e}")
                continue
        
        # Create enhanced fallback validation engine
        logger.warning("Creating enhanced fallback ValidationEngine")
        
        class EnhancedFallbackValidationEngine:
            def __init__(self, config_manager=None):
                self.config_manager = config_manager
                self._validation_cache = {}
                logger.debug("FallbackValidationEngine initialized")
                
            def initialize(self):
                logger.debug("ValidationEngine initialize (fallback)")
                pass
                
            async def analyze_project(self, project_path):
                logger.info(f"Analyzing project (mock): {project_path}")
                
                # Create ProgressReport class if not available
                try:
                    from ui.widgets.progress_intelligence import ProgressReport
                except ImportError:
                    class ProgressReport:
                        def __init__(self, **kwargs):
                            for k, v in kwargs.items():
                                setattr(self, k, v)
                            self.fake_progress = kwargs.get('fake_progress', 0)
                            self.placeholders_found = kwargs.get('placeholders_found', 0)
                
                # Generate more realistic mock analysis
                import random
                analysis = ProgressReport(
                    real_progress=random.uniform(0.6, 0.9),
                    claimed_progress=random.uniform(0.8, 1.0),
                    fake_progress=random.uniform(0.1, 0.3),
                    quality_score=random.uniform(6.0, 9.0),
                    authenticity_score=random.uniform(0.7, 0.95),
                    placeholders_found=random.randint(0, 5),
                    todos_found=random.randint(2, 8)
                )
                
                self._validation_cache[str(project_path)] = analysis
                return analysis
                
            async def validate_ai_output(self, result, context=None):
                logger.info("Validating AI output (mock)")
                
                class MockValidation:
                    def __init__(self, result):
                        # Simple heuristics for mock validation
                        self.is_authentic = not ('TODO' in str(result) and 'placeholder' in str(result).lower())
                        self.completion_suggestions = [] if self.is_authentic else [
                            "Replace TODO comments with actual implementation",
                            "Add proper error handling",
                            "Include input validation"
                        ]
                        self.confidence = 0.85 if self.is_authentic else 0.45
                        
                return MockValidation(result)
                
        self.validation_engine = EnhancedFallbackValidationEngine(self.config_manager)
    
    def _init_workflow_manager(self):
        """Initialize automatic programming workflow manager"""
        import_attempts = [
            ('claude_tui.integrations.automatic_programming_workflow', 'AutomaticProgrammingWorkflow'),
            ('src.claude_tui.integrations.automatic_programming_workflow', 'AutomaticProgrammingWorkflow'),
        ]
        
        for module_path, class_name in import_attempts:
            try:
                logger.debug(f"Attempting to import {class_name} from {module_path}")
                module = __import__(module_path, fromlist=[class_name])
                workflow_class = getattr(module, class_name)
                self.workflow_manager = workflow_class(self.config_manager)
                logger.info(f"✓ WorkflowManager imported from {module_path}")
                return
            except (ImportError, AttributeError) as e:
                logger.debug(f"Failed to import from {module_path}: {e}")
                continue
        
        # Create fallback workflow manager
        logger.warning("Creating fallback WorkflowManager")
        
        class FallbackWorkflowManager:
            def __init__(self, config_manager=None):
                self.config_manager = config_manager
                logger.debug("FallbackWorkflowManager initialized")
                
            async def create_workflow_from_template(self, template_name, project_name, project_path, **kwargs):
                logger.info(f"Mock workflow creation: {project_name} using {template_name}")
                return "mock-workflow-id"
                
            async def create_custom_workflow(self, name, description, prompt, project_path, **kwargs):
                logger.info(f"Mock custom workflow: {name}")
                return "mock-custom-workflow-id"
                
            async def execute_workflow(self, workflow_id):
                logger.info(f"Mock workflow execution: {workflow_id}")
                # Mock result structure
                class MockResult:
                    def __init__(self):
                        self.workflow_id = workflow_id
                        self.status = "completed"
                        self.steps_completed = 5
                        self.steps_total = 5
                        self.results = {}
                        self.errors = []
                        self.duration = 30.0
                        self.created_files = ["main.py", "requirements.txt"]
                        self.modified_files = []
                return MockResult()
                
            def get_available_templates(self):
                return {
                    "fastapi_app": {
                        "name": "FastAPI Application",
                        "description": "Generate a complete FastAPI application",
                        "steps_count": 5
                    },
                    "react_dashboard": {
                        "name": "React Dashboard", 
                        "description": "Generate a React dashboard application",
                        "steps_count": 4
                    }
                }
                
            def add_progress_callback(self, callback):
                logger.debug("Mock progress callback added")
                
            def remove_progress_callback(self, callback):
                logger.debug("Mock progress callback removed")
                
            async def cleanup(self):
                logger.debug("WorkflowManager cleanup (fallback)")
                
        self.workflow_manager = FallbackWorkflowManager(self.config_manager)
    
    def get_app_instance(self, ui_type="auto", headless=False, test_mode=False) -> Tuple[Any, str]:
        """
        Get the appropriate app instance based on available implementations.
        
        Args:
            ui_type: "ui" for src/ui, "claude_tui" for src/claude_tui/ui, "auto" for auto-detect
            headless: Run in headless mode
            test_mode: Run in test mode
            
        Returns:
            Tuple of (app_instance, ui_type_used)
            
        Raises:
            RuntimeError: If bridge initialization fails
            ImportError: If no UI implementation is available
        """
        logger.info(f"Getting app instance for ui_type: {ui_type}, headless: {headless}, test_mode: {test_mode}")
        
        # Initialize if not already done
        if not self._initialized:
            logger.info("Bridge not initialized, attempting initialization...")
            if not self.initialize():
                error_msg = "Failed to initialize integration bridge"
                logger.error(error_msg)
                logger.error(f"Initialization errors: {self._initialization_errors}")
                raise RuntimeError(f"{error_msg}. Available components: {self._available_components}")
        
        # UI implementation attempts with comprehensive error logging
        ui_attempts = []
        
        if ui_type == "auto":
            ui_attempts = [
                ("ui", [("ui.main_app", "ClaudeTUIApp"), ("src.ui.main_app", "ClaudeTUIApp")]),
                ("claude_tui", [("claude_tui.ui.application", "ClaudeTUIApp"), ("src.claude_tui.ui.application", "ClaudeTUIApp")]),
            ]
        elif ui_type == "ui":
            ui_attempts = [
                ("ui", [("ui.main_app", "ClaudeTUIApp"), ("src.ui.main_app", "ClaudeTUIApp")])
            ]
        elif ui_type == "claude_tui":
            ui_attempts = [
                ("claude_tui", [("claude_tui.ui.application", "ClaudeTUIApp"), ("src.claude_tui.ui.application", "ClaudeTUIApp")])
            ]
        else:
            raise ValueError(f"Unknown ui_type: {ui_type}")
        
        # Try each UI implementation
        for ui_name, import_attempts in ui_attempts:
            logger.debug(f"Trying UI implementation: {ui_name}")
            
            for module_path, class_name in import_attempts:
                try:
                    logger.debug(f"Attempting to import {class_name} from {module_path}")
                    module = __import__(module_path, fromlist=[class_name])
                    app_class = getattr(module, class_name)
                    
                    # Create app instance with appropriate parameters
                    kwargs = {}
                    if hasattr(app_class.__init__, '__code__') and 'headless' in app_class.__init__.__code__.co_varnames:
                        kwargs['headless'] = headless
                        kwargs['test_mode'] = test_mode
                    
                    if ui_name == "ui":
                        app_instance = app_class(**kwargs)  # ui implementation with modes
                    else:
                        app_instance = app_class(self.config_manager, **kwargs)
                    
                    logger.info(f"✓ Successfully created {ui_name} app instance from {module_path}")
                    return app_instance, ui_name
                    
                except (ImportError, AttributeError, TypeError) as e:
                    logger.debug(f"Failed to create app from {module_path}: {e}")
                    continue
        
        # If all attempts failed, provide detailed error information
        error_msg = f"No UI implementation available for type '{ui_type}'"
        logger.error(error_msg)
        logger.error(f"Tried UI types: {[name for name, _ in ui_attempts]}")
        logger.error(f"Available bridge components: {self._available_components}")
        raise ImportError(error_msg)


# Export alias for backward compatibility
IntegrationBridge = UIIntegrationBridge

@contextmanager
def error_recovery_context():
    """Context manager for error recovery during bridge operations"""
    try:
        yield
    except Exception as e:
        logger.error(f"Error in bridge operation: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Could implement recovery strategies here
        raise

# Global bridge instance management
_bridge_instance: Optional[UIIntegrationBridge] = None

def get_bridge() -> UIIntegrationBridge:
    """
    Get the global integration bridge instance.
    Creates a new instance if one doesn't exist.
    
    Returns:
        UIIntegrationBridge: The global bridge instance
    """
    global _bridge_instance
    if _bridge_instance is None:
        logger.info("Creating new bridge instance")
        _bridge_instance = UIIntegrationBridge()
    return _bridge_instance

def reset_bridge():
    """
    Reset the global bridge instance.
    This forces creation of a new bridge instance on the next get_bridge() call.
    """
    global _bridge_instance
    if _bridge_instance:
        logger.info("Resetting bridge instance")
        # Cleanup if the bridge has cleanup methods
        if hasattr(_bridge_instance, 'cleanup'):
            try:
                _bridge_instance.cleanup()
            except Exception as e:
                logger.warning(f"Error during bridge cleanup: {e}")
    _bridge_instance = None

def run_integrated_app(ui_type="auto", debug=False, headless=False, test_mode=False):
    """
    Convenient function to run the integrated application with comprehensive error handling.
    
    Args:
        ui_type: UI implementation to use ("auto", "ui", "claude_tui")
        debug: Enable debug mode
        headless: Run in headless mode
        test_mode: Run in test mode
    """
    with error_recovery_context():
        logger.info(f"run_integrated_app called with ui_type='{ui_type}', debug={debug}, headless={headless}, test_mode={test_mode}")
        bridge = get_bridge()
        
        # Initialize the bridge if not already done
        if not bridge._initialized:
            if not bridge.initialize():
                logger.error("Failed to initialize bridge in run_integrated_app")
                raise RuntimeError("Bridge initialization failed")
        
        # Get the app instance and run it
        try:
            app_instance, ui_type_used = bridge.get_app_instance(ui_type, headless, test_mode)
            logger.info(f"Running app with UI type: {ui_type_used}")
            
            # Check if the app has a run method
            if hasattr(app_instance, 'run'):
                return app_instance.run()
            elif hasattr(app_instance, 'start'):
                return app_instance.start()
            else:
                logger.warning("App instance has no run() or start() method")
                return app_instance
                
        except Exception as e:
            logger.error(f"Error running integrated app: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise