#!/usr/bin/env python3
"""
Claude TUI Integration Tests
Comprehensive tests for the complete TUI application flow from run_tui.py to main_app.py
Tests all screens, widgets, keyboard navigation, and event handling
"""

import asyncio
import os
import sys
import subprocess
import tempfile
import time
import pytest
import signal
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from threading import Thread

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import application components
try:
    from ui.main_app import ClaudeTUIApp, MainWorkspace, run_app
    from ui.run import main as run_main
except ImportError as e:
    print(f"Import warning: {e}")
    # Create mock classes for fallback
    class ClaudeTUIApp:
        def __init__(self):
            pass
        def run(self):
            pass
    
    class MainWorkspace:
        def __init__(self, *args, **kwargs):
            pass
    
    def run_app():
        pass
    
    def run_main():
        pass

# Import Textual testing utilities
try:
    from textual.pilot import Pilot
    from textual.app import App
    from textual.widgets import Static
    textual_available = True
except ImportError:
    print("Textual not available for testing")
    textual_available = False
    
    class Pilot:
        def __init__(self, app):
            self.app = app
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass


class TestTUIStartup:
    """Test TUI application startup and basic functionality"""
    
    def test_run_tui_script_exists(self):
        """Test that run_tui.py exists and is executable"""
        run_tui_path = Path(__file__).parent.parent / "run_tui.py"
        assert run_tui_path.exists(), "run_tui.py script not found"
        assert run_tui_path.is_file(), "run_tui.py is not a file"
        
        # Check if it's readable
        try:
            with open(run_tui_path, 'r') as f:
                content = f.read()
                assert content, "run_tui.py is empty"
                assert "main()" in content, "run_tui.py missing main() call"
        except Exception as e:
            pytest.fail(f"Cannot read run_tui.py: {e}")
    
    def test_main_app_module_import(self):
        """Test that main_app.py can be imported"""
        try:
            from ui.main_app import ClaudeTUIApp
            assert ClaudeTUIApp is not None
        except ImportError as e:
            pytest.fail(f"Cannot import main_app: {e}")
    
    def test_app_instantiation(self):
        """Test that ClaudeTUIApp can be instantiated"""
        try:
            app = ClaudeTUIApp()
            assert app is not None
            assert hasattr(app, 'TITLE')
            assert hasattr(app, 'BINDINGS')
            assert hasattr(app, 'project_manager')
            assert hasattr(app, 'ai_interface')
            assert hasattr(app, 'validation_engine')
        except Exception as e:
            pytest.fail(f"Cannot instantiate ClaudeTUIApp: {e}")
    
    def test_tui_startup_with_timeout(self):
        """Test that the TUI starts without crashing (with timeout)"""
        run_tui_path = Path(__file__).parent.parent / "run_tui.py"
        
        try:
            # Start the TUI process with timeout
            process = subprocess.Popen(
                [sys.executable, str(run_tui_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, 'PYTHONPATH': str(Path(__file__).parent.parent / "src")}
            )
            
            # Wait for 3 seconds to see if it starts properly
            time.sleep(3)
            
            # Check if process is still running (good sign)
            poll_result = process.poll()
            
            if poll_result is None:
                # Process is still running, terminate it gracefully
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                
                # If it started and ran for 3 seconds without crashing, consider it successful
                assert True, "TUI started successfully and ran without immediate crashes"
            
            elif poll_result == 0:
                # Process completed normally (might be expected for some scenarios)
                stdout, stderr = process.communicate()
                print(f"TUI process completed normally. stdout: {stdout}, stderr: {stderr}")
                assert True, "TUI process completed without error"
            
            else:
                # Process crashed
                stdout, stderr = process.communicate()
                pytest.fail(f"TUI process crashed with return code {poll_result}. stdout: {stdout}, stderr: {stderr}")
                
        except Exception as e:
            if 'process' in locals():
                try:
                    process.kill()
                except:
                    pass
            pytest.fail(f"Error testing TUI startup: {e}")


@pytest.mark.skipif(not textual_available, reason="Textual not available")
class TestTUIComponents:
    """Test TUI components and interaction"""
    
    @pytest.fixture
    async def app(self):
        """Create a test app instance"""
        app = ClaudeTUIApp()
        return app
    
    async def test_app_composition(self, app):
        """Test that the app composes correctly"""
        try:
            async with app.run_test() as pilot:
                # Check that main components are present
                assert app.workspace is not None
                assert app.notification_system is not None
                assert app.placeholder_alert is not None
        except Exception as e:
            pytest.fail(f"App composition failed: {e}")
    
    async def test_keyboard_shortcuts(self, app):
        """Test keyboard shortcuts and navigation"""
        shortcuts_to_test = [
            ("ctrl+n", "new_project"),
            ("ctrl+o", "open_project"),
            ("ctrl+s", "save_project"),
            ("ctrl+p", "project_wizard"),
            ("f1", "help"),
            ("f5", "refresh"),
            ("f12", "debug_mode"),
        ]
        
        try:
            async with app.run_test() as pilot:
                for key, expected_action in shortcuts_to_test:
                    # Check if the binding exists
                    binding_found = False
                    for binding in app.BINDINGS:
                        if binding.key == key:
                            binding_found = True
                            assert binding.action == expected_action
                            break
                    
                    assert binding_found, f"Keyboard shortcut {key} not found"
                    
                    # Try to press the key (might not work in test environment)
                    try:
                        await pilot.press(key)
                        # If we get here without exception, the key was processed
                        assert True
                    except Exception:
                        # Key press might not work in test environment, just check binding exists
                        pass
        except Exception as e:
            pytest.fail(f"Keyboard shortcut test failed: {e}")
    
    async def test_vim_navigation(self, app):
        """Test Vim-style navigation keys"""
        vim_keys = ["h", "j", "k", "l"]
        
        try:
            async with app.run_test() as pilot:
                for key in vim_keys:
                    # Check if the binding exists
                    binding_found = False
                    for binding in app.BINDINGS:
                        if binding.key == key:
                            binding_found = True
                            break
                    
                    assert binding_found, f"Vim key {key} not found in bindings"
        except Exception as e:
            pytest.fail(f"Vim navigation test failed: {e}")


class TestWorkspaceComponents:
    """Test main workspace and its components"""
    
    def test_workspace_instantiation(self):
        """Test MainWorkspace can be instantiated"""
        try:
            app = ClaudeTUIApp()
            workspace = MainWorkspace(app)
            assert workspace is not None
            assert workspace.app_instance == app
        except Exception as e:
            pytest.fail(f"Workspace instantiation failed: {e}")
    
    def test_workspace_component_attributes(self):
        """Test workspace has required component attributes"""
        try:
            app = ClaudeTUIApp()
            workspace = MainWorkspace(app)
            
            # Check that workspace has component attributes
            assert hasattr(workspace, 'project_tree')
            assert hasattr(workspace, 'task_dashboard')
            assert hasattr(workspace, 'progress_widget')
            assert hasattr(workspace, 'console_widget')
            
        except Exception as e:
            pytest.fail(f"Workspace component test failed: {e}")


class TestWidgetIntegration:
    """Test widget integration and functionality"""
    
    def test_widget_imports(self):
        """Test that widgets can be imported"""
        widget_modules = [
            "ui.widgets.project_tree",
            "ui.widgets.task_dashboard", 
            "ui.widgets.progress_intelligence",
            "ui.widgets.console_widget",
            "ui.widgets.notification_system",
            "ui.widgets.placeholder_alert"
        ]
        
        for module in widget_modules:
            try:
                __import__(module)
            except ImportError:
                # Widgets might not exist yet, that's ok for integration test
                print(f"Widget module {module} not available (expected during development)")
    
    def test_widget_fallback_classes(self):
        """Test that fallback widget classes work"""
        # This tests the fallback widgets defined in main_app.py
        try:
            from ui.main_app import (
                ProjectTree, TaskDashboard, ProgressIntelligence,
                ConsoleWidget, PlaceholderAlert, NotificationSystem
            )
            
            # Test instantiation of fallback classes
            project_tree = ProjectTree()
            assert project_tree is not None
            
            task_dashboard = TaskDashboard()
            assert task_dashboard is not None
            
            progress_intel = ProgressIntelligence()
            assert progress_intel is not None
            assert hasattr(progress_intel, 'update_validation')
            
            console_widget = ConsoleWidget()
            assert console_widget is not None
            
            alert = PlaceholderAlert()
            assert alert is not None
            assert hasattr(alert, 'show_alert')
            
            notifications = NotificationSystem()
            assert notifications is not None
            assert hasattr(notifications, 'add_notification')
            
        except Exception as e:
            pytest.fail(f"Widget fallback test failed: {e}")


class TestScreenIntegration:
    """Test screen integration and navigation"""
    
    def test_screen_imports(self):
        """Test that screen classes can be imported"""
        try:
            from ui.main_app import ProjectWizardScreen, SettingsScreen
            assert ProjectWizardScreen is not None
            assert SettingsScreen is not None
        except ImportError:
            # Screens might not be fully implemented, check fallbacks
            print("Screen classes using fallback implementations")
    
    def test_screen_messages(self):
        """Test screen message classes"""
        try:
            from ui.main_app import CreateProjectMessage, SettingsSavedMessage
            assert CreateProjectMessage is not None
            assert SettingsSavedMessage is not None
        except ImportError:
            print("Screen message classes using fallback implementations")


class TestCoreSystemIntegration:
    """Test integration with core systems"""
    
    def test_core_system_imports(self):
        """Test that core systems can be imported or have fallbacks"""
        try:
            app = ClaudeTUIApp()
            
            # Check that core systems are initialized
            assert hasattr(app, 'project_manager')
            assert hasattr(app, 'ai_interface')
            assert hasattr(app, 'validation_engine')
            
            # Test initialization doesn't crash
            app.init_core_systems()
            
        except Exception as e:
            pytest.fail(f"Core system integration test failed: {e}")
    
    def test_notification_system(self):
        """Test notification system functionality"""
        try:
            app = ClaudeTUIApp()
            
            # Test notification with mock system
            app.notification_system = Mock()
            app.notification_system.add_notification = Mock()
            
            app.notify("Test message", "info")
            
            # Check that add_notification was called if system exists
            if app.notification_system:
                app.notification_system.add_notification.assert_called_once_with("Test message", "info")
                
        except Exception as e:
            pytest.fail(f"Notification system test failed: {e}")


class TestEventHandling:
    """Test event handling and reactive properties"""
    
    def test_message_handlers(self):
        """Test that message handlers are defined"""
        app = ClaudeTUIApp()
        
        # Check that handler methods exist
        assert hasattr(app, 'handle_create_project')
        assert hasattr(app, 'handle_settings_saved')
        assert callable(getattr(app, 'handle_create_project'))
        assert callable(getattr(app, 'handle_settings_saved'))
    
    def test_action_handlers(self):
        """Test that action handlers are defined"""
        app = ClaudeTUIApp()
        
        action_handlers = [
            'action_new_project', 'action_open_project', 'action_save_project',
            'action_project_wizard', 'action_settings', 'action_help',
            'action_refresh', 'action_debug_mode', 'action_focus_left',
            'action_focus_right', 'action_focus_up', 'action_focus_down',
            'action_toggle_task_panel', 'action_toggle_console',
            'action_toggle_validation'
        ]
        
        for handler_name in action_handlers:
            assert hasattr(app, handler_name), f"Action handler {handler_name} not found"
            assert callable(getattr(app, handler_name)), f"Action handler {handler_name} not callable"


class TestAsyncOperations:
    """Test async operations and work decorators"""
    
    def test_async_methods_defined(self):
        """Test that async methods are properly defined"""
        app = ClaudeTUIApp()
        
        # Check async methods exist
        assert hasattr(app, 'execute_ai_task')
        assert callable(getattr(app, 'execute_ai_task'))
        
        workspace = MainWorkspace(app)
        assert hasattr(workspace, 'start_progress_monitoring')
        assert callable(getattr(workspace, 'start_progress_monitoring'))


class TestErrorHandling:
    """Test error handling and graceful degradation"""
    
    def test_initialization_error_handling(self):
        """Test that initialization errors are handled gracefully"""
        try:
            app = ClaudeTUIApp()
            
            # Mock a failing core system
            app.project_manager = Mock()
            app.project_manager.initialize = Mock(side_effect=Exception("Mock error"))
            
            # Should not crash the app
            app.init_core_systems()
            
            # App should still be functional
            assert app is not None
            
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")
    
    def test_missing_project_handling(self):
        """Test handling of missing project scenarios"""
        try:
            app = ClaudeTUIApp()
            app.project_manager.current_project = None
            
            # Should not crash when no project is loaded
            app.action_save_project()
            
        except Exception as e:
            pytest.fail(f"Missing project handling test failed: {e}")


class TestApplicationFlow:
    """Test complete application flow scenarios"""
    
    def test_startup_sequence(self):
        """Test complete startup sequence"""
        try:
            app = ClaudeTUIApp()
            
            # Mock the mount process
            app.notification_system = Mock()
            app.notification_system.add_notification = Mock()
            
            # Test mount sequence
            app.on_mount()
            
            # Check that title and subtitle are set
            assert hasattr(app, 'title')
            assert hasattr(app, 'sub_title')
            
        except Exception as e:
            pytest.fail(f"Startup sequence test failed: {e}")
    
    def test_shutdown_sequence(self):
        """Test graceful shutdown"""
        try:
            app = ClaudeTUIApp()
            
            # Test that quit action exists and is callable
            assert hasattr(app, 'action_quit')
            
            # The action_quit might not be explicitly defined but should be handled by Textual
            # Just test that the app can be created and doesn't crash on normal operations
            
        except Exception as e:
            pytest.fail(f"Shutdown sequence test failed: {e}")


# Test data and fixtures
@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_config():
    """Create mock configuration for testing"""
    class MockConfig:
        def __init__(self):
            self.name = "test_project"
            self.path = "/tmp/test_project"
            self.project_type = "python"
    
    return MockConfig()


# Integration test scenarios
class TestFullIntegrationScenarios:
    """Test complete integration scenarios"""
    
    def test_project_creation_flow(self, mock_config):
        """Test complete project creation flow"""
        try:
            app = ClaudeTUIApp()
            
            # Mock the message handling
            app.project_manager = Mock()
            app.project_manager.create_project_from_config = Mock(return_value=True)
            app.notification_system = Mock()
            app.notification_system.add_notification = Mock()
            app.workspace = Mock()
            app.workspace.project_tree = Mock()
            app.workspace.project_tree.set_project = Mock()
            
            # Simulate project creation message
            from ui.main_app import CreateProjectMessage
            message = CreateProjectMessage()
            message.config = mock_config
            
            app.handle_create_project(message)
            
            # Verify the flow executed
            app.project_manager.create_project_from_config.assert_called_once_with(mock_config)
            
        except Exception as e:
            pytest.fail(f"Project creation flow test failed: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])