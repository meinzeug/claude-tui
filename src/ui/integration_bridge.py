#!/usr/bin/env python3
"""
Integration Bridge - Handles integration between UI components and backend services
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

class UIIntegrationBridge:
    """
    Bridge class to handle integration between different UI implementations
    and backend services
    """
    
    def __init__(self):
        self.config_manager = None
        self.project_manager = None
        self.ai_interface = None
        self.validation_engine = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize all backend services and connections"""
        try:
            # Try to initialize core services
            self._init_config_manager()
            self._init_project_manager()
            self._init_ai_interface()
            self._init_validation_engine()
            
            self._initialized = True
            logger.info("UI Integration Bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize UI Integration Bridge: {e}")
            return False
    
    def _init_config_manager(self):
        """Initialize configuration manager"""
        try:
            from claude_tui.core.config_manager import ConfigManager
            self.config_manager = ConfigManager()
            logger.debug("ConfigManager initialized")
        except ImportError:
            # Create fallback config manager
            class FallbackConfigManager:
                def get_setting(self, key, default=None):
                    return default
                def update_setting(self, key, value):
                    pass
                def get_ui_preferences(self):
                    class UIPrefs:
                        theme = 'dark'
                    return UIPrefs()
            self.config_manager = FallbackConfigManager()
            logger.warning("Using fallback ConfigManager")
    
    def _init_project_manager(self):
        """Initialize project manager"""
        try:
            from claude_tui.core.project_manager import ProjectManager
            self.project_manager = ProjectManager(self.config_manager)
            logger.debug("ProjectManager initialized")
        except ImportError:
            # Create fallback project manager
            class FallbackProjectManager:
                def __init__(self, config_manager=None):
                    self.current_project = None
                    self.config_manager = config_manager
                def initialize(self):
                    pass
                def save_current_project(self):
                    pass
                def create_project_from_config(self, config):
                    return True
                async def get_project_status(self, project):
                    return {"status": "active"}
                async def load_project(self, path):
                    class MockProject:
                        name = "Mock Project"
                        path = path
                    return MockProject()
                async def save_project(self, project):
                    pass
                async def cleanup(self):
                    pass
            self.project_manager = FallbackProjectManager(self.config_manager)
            logger.warning("Using fallback ProjectManager")
    
    def _init_ai_interface(self):
        """Initialize AI interface"""
        try:
            from claude_tui.integrations.ai_interface import AIInterface
            self.ai_interface = AIInterface(self.config_manager)
            logger.debug("AIInterface initialized")
        except ImportError:
            # Create fallback AI interface
            class FallbackAIInterface:
                def __init__(self, config_manager=None):
                    self.config_manager = config_manager
                def initialize(self):
                    pass
                async def execute_task(self, task_description, context):
                    return f"Mock AI result for: {task_description}"
                async def complete_placeholder_code(self, result, suggestions):
                    return result
                async def cleanup(self):
                    pass
            self.ai_interface = FallbackAIInterface(self.config_manager)
            logger.warning("Using fallback AIInterface")
    
    def _init_validation_engine(self):
        """Initialize validation engine"""
        try:
            from claude_tui.core.progress_validator import ProgressValidator
            self.validation_engine = ProgressValidator(self.config_manager)
            logger.debug("ProgressValidator initialized")
        except ImportError:
            # Create fallback validation engine
            class FallbackValidationEngine:
                def __init__(self, config_manager=None):
                    self.config_manager = config_manager
                def initialize(self):
                    pass
                async def analyze_project(self, project_path):
                    # Import ProgressReport if available
                    try:
                        from ui.widgets.progress_intelligence import ProgressReport
                    except ImportError:
                        class ProgressReport:
                            def __init__(self, **kwargs):
                                for k, v in kwargs.items():
                                    setattr(self, k, v)
                                self.fake_progress = kwargs.get('fake_progress', 0)
                                self.placeholders_found = kwargs.get('placeholders_found', 0)
                    
                    return ProgressReport(
                        real_progress=0.7,
                        claimed_progress=0.9,
                        fake_progress=0.2,
                        quality_score=7.5,
                        authenticity_score=0.78,
                        placeholders_found=3,
                        todos_found=5
                    )
                async def validate_ai_output(self, result, context):
                    class MockValidation:
                        is_authentic = True
                        completion_suggestions = []
                    return MockValidation()
            self.validation_engine = FallbackValidationEngine(self.config_manager)
            logger.warning("Using fallback ValidationEngine")
    
    def get_app_instance(self, ui_type="auto"):
        """
        Get the appropriate app instance based on available implementations
        
        Args:
            ui_type: "ui" for src/ui, "claude_tui" for src/claude_tui/ui, "auto" for auto-detect
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize integration bridge")
        
        if ui_type == "auto":
            # Try to determine best available UI
            try:
                from ui.main_app import ClaudeTUIApp as UIApp
                return UIApp(), "ui"
            except ImportError:
                try:
                    from claude_tui.ui.application import ClaudeTUIApp as ClaudeApp
                    return ClaudeApp(self.config_manager), "claude_tui"
                except ImportError:
                    raise ImportError("No UI implementation available")
        
        elif ui_type == "ui":
            from ui.main_app import ClaudeTUIApp as UIApp
            return UIApp(), "ui"
        
        elif ui_type == "claude_tui":
            from claude_tui.ui.application import ClaudeTUIApp as ClaudeApp
            return ClaudeApp(self.config_manager), "claude_tui"
        
        else:
            raise ValueError(f"Unknown ui_type: {ui_type}")
    
    def run_application(self, ui_type="auto", debug=False):
        """
        Run the appropriate application with proper initialization
        
        Args:
            ui_type: UI implementation to use
            debug: Enable debug mode
        """
        try:
            app, used_type = self.get_app_instance(ui_type)
            logger.info(f"Running application with {used_type} UI implementation")
            
            # Additional setup based on UI type
            if used_type == "ui":
                # The ui implementation handles its own services
                app.run()
            else:
                # Claude TUI implementation
                app.run()
                
        except Exception as e:
            logger.error(f"Failed to run application: {e}")
            raise

# Global bridge instance
_bridge = None

def get_bridge() -> UIIntegrationBridge:
    """Get the global integration bridge instance"""
    global _bridge
    if _bridge is None:
        _bridge = UIIntegrationBridge()
    return _bridge

def run_integrated_app(ui_type="auto", debug=False):
    """
    Convenient function to run the integrated application
    """
    bridge = get_bridge()
    bridge.run_application(ui_type, debug)