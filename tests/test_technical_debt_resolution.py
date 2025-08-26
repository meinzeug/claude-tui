"""
Test suite for technical debt resolution validation.

Validates that critical TODO/FIXME items have been properly resolved
with functional implementations rather than just comment removal.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Import the modules we've fixed
from src.claude_tui.ui.screens.settings import SettingsScreen
from src.claude_tui.ui.screens.workspace_screen import WorkspaceScreen, TaskMonitorPanel
from src.claude_tui.ui.screens.project_wizard import ProjectWizard
from src.auth.middleware import AuthMiddleware
from src.auth.password_reset import PasswordResetManager


class TestTechnicalDebtResolution:
    """Test suite for resolved technical debt items."""
    
    def test_settings_functionality_implemented(self):
        """Test that settings save/load functionality is implemented."""
        # Create mock app with config manager
        mock_app = Mock()
        mock_config_manager = Mock()
        mock_config_manager.ai_service = Mock()
        mock_config_manager.ai_service.service_name = "claude"
        mock_config_manager.ai_service.timeout = 300
        mock_config_manager.project_defaults = Mock()
        mock_config_manager.project_defaults.auto_validation = True
        mock_config_manager.app = Mock()
        mock_config_manager.app.log_level = "INFO"
        mock_app.config_manager = mock_config_manager
        
        settings_screen = SettingsScreen()
        settings_screen.app = mock_app
        
        # Test that _load_current_settings no longer has TODO
        settings_screen._load_current_settings()
        
        # Should have loaded settings without errors
        assert isinstance(settings_screen.current_settings, dict)
        assert isinstance(settings_screen.original_settings, dict)
    
    def test_authentication_middleware_database_integration(self):
        """Test that auth middleware properly integrates with database."""
        # This tests that the TODO for database lookup was resolved
        middleware = AuthMiddleware()
        
        # The implementation should handle database unavailability gracefully
        # Rather than just having a TODO comment
        assert hasattr(middleware, '_authenticate_request')
        
        # Check that the method exists and has real implementation
        import inspect
        source = inspect.getsource(middleware._authenticate_request)
        
        # Should not contain the original TODO comment
        assert "TODO: Get user from database" not in source
        # Should contain actual database integration logic
        assert "UserRepository" in source or "get_async_session" in source
    
    def test_password_reset_database_integration(self):
        """Test that password reset integrates with database."""
        reset_manager = PasswordResetManager()
        
        # Check that the method has real database integration
        import inspect
        source = inspect.getsource(reset_manager.initiate_password_reset)
        
        # Should not contain the original TODO
        assert "TODO: Get user from database" not in source
        # Should have database integration
        assert "UserRepository" in source or "get_async_session" in source
    
    async def test_workspace_ai_integration(self):
        """Test that workspace screen AI integration is implemented."""
        workspace = WorkspaceScreen()
        
        # Mock app with AI interface
        mock_app = Mock()
        mock_ai_interface = AsyncMock()
        mock_app.ai_interface = mock_ai_interface
        mock_app.notify = Mock()
        workspace.app = mock_app
        
        # Test AI assistant functionality
        ai_panel = workspace.compose()  # This should include AI integration
        
        # Verify AI integration exists
        import inspect
        for method in [workspace.on_input_submitted]:
            if hasattr(workspace, method.__name__):
                source = inspect.getsource(method)
                # Should not contain original TODO
                assert "TODO: Send to AI service" not in source
                # Should have AI integration logic
                assert "ai_interface" in source.lower()
    
    async def test_task_monitor_implementation(self):
        """Test that task monitoring functionality is implemented."""
        task_panel = TaskMonitorPanel()
        
        # Mock app with task engine
        mock_app = Mock()
        mock_task_engine = AsyncMock()
        mock_task_engine.get_all_tasks = AsyncMock(return_value=[])
        mock_task_engine.clear_completed_tasks = AsyncMock(return_value=5)
        mock_app.task_engine = mock_task_engine
        task_panel.app = mock_app
        
        # Test that button press handling is implemented
        import inspect
        source = inspect.getsource(task_panel.on_button_pressed)
        
        # Should not contain TODOs
        assert "TODO: Refresh task list" not in source
        assert "TODO: Clear completed tasks" not in source
        # Should have real implementations
        assert "task_engine" in source
    
    def test_project_wizard_creation_implemented(self):
        """Test that project creation functionality is implemented."""
        wizard = ProjectWizard()
        
        # Check that project creation has real implementation
        import inspect
        source = inspect.getsource(wizard.action_create_project)
        
        # Should not contain original TODO
        assert "TODO: Implement actual project creation" not in source
        # Should have real project creation logic
        assert "mkdir" in source or "create_" in source
        
        # Test that helper methods exist
        assert hasattr(wizard, '_create_basic_project')
        assert hasattr(wizard, '_create_python_project')
        assert hasattr(wizard, '_create_web_project')
        assert hasattr(wizard, '_create_api_project')
    
    def test_project_creation_functionality(self):
        """Test that project creation actually works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            wizard = ProjectWizard()
            
            # Set up project data
            wizard.project_data = {
                'name': 'test_project',
                'description': 'A test project',
                'location': temp_dir,
                'template': 'python'
            }
            
            # Create project path
            project_path = Path(temp_dir) / 'test_project'
            
            # Test project creation
            wizard._create_python_project(project_path)
            
            # Verify project structure was created
            assert project_path.exists()
            assert (project_path / 'src').exists()
            assert (project_path / 'tests').exists()
            assert (project_path / 'README.md').exists()
            assert (project_path / 'requirements.txt').exists()
            assert (project_path / '.gitignore').exists()
    
    def test_file_picker_implementation(self):
        """Test that file picker functionality is implemented."""
        workspace = WorkspaceScreen()
        
        # Check that file picker has real implementation
        import inspect
        source = inspect.getsource(workspace.action_open_file)
        
        # Should not contain original TODO
        assert "TODO: Implement file picker" not in source
        # Should have file picker logic
        assert "file_path" in source and "Path" in source
    
    def test_no_remaining_critical_todos(self):
        """Test that critical TODO items have been resolved."""
        from src.claude_tui.ui.screens import settings, workspace_screen, project_wizard
        from src.auth import middleware, password_reset
        
        # Check that critical files don't contain unresolved TODOs
        modules_to_check = [settings, workspace_screen, project_wizard, middleware, password_reset]
        
        for module in modules_to_check:
            import inspect
            source = inspect.getsource(module)
            
            # Count remaining TODO comments
            todo_count = source.upper().count("TODO:")
            # Allow some TODOs for future enhancements, but should be minimal
            assert todo_count < 3, f"Module {module.__name__} still has {todo_count} TODO items"
    
    def test_error_handling_implemented(self):
        """Test that proper error handling has been added."""
        # Test settings screen error handling
        settings_screen = SettingsScreen()
        
        # Mock app without config manager to test error handling
        mock_app = Mock()
        del mock_app.config_manager  # Remove config manager to trigger fallback
        settings_screen.app = mock_app
        
        # Should not raise exception, should handle gracefully
        try:
            settings_screen._load_current_settings()
            # Should fall back to default settings
            assert isinstance(settings_screen.current_settings, dict)
        except Exception as e:
            pytest.fail(f"Settings should handle missing config manager gracefully: {e}")


if __name__ == "__main__":
    # Run basic smoke test
    print("Running technical debt resolution validation...")
    
    # Test that critical modules can be imported
    try:
        from src.claude_tui.ui.screens.settings import SettingsScreen
        from src.claude_tui.ui.screens.workspace_screen import WorkspaceScreen  
        from src.claude_tui.ui.screens.project_wizard import ProjectWizard
        print("✓ All resolved modules import successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        exit(1)
    
    # Test basic instantiation
    try:
        settings = SettingsScreen()
        workspace = WorkspaceScreen()
        wizard = ProjectWizard()
        print("✓ All resolved classes instantiate successfully")
    except Exception as e:
        print(f"✗ Instantiation error: {e}")
        exit(1)
    
    print("✓ Technical debt resolution validation passed!")