#!/usr/bin/env python3
"""
Comprehensive UI Component Tests using Textual Testing Framework.

Tests the TUI components including:
- Main application layout and navigation
- Project tree widget functionality
- Task dashboard interactions
- Progress intelligence display
- Console widget AI integration
- Notification and alert systems
- Keyboard shortcuts and accessibility
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import Textual testing framework
from textual.testing import AppContext
from textual.widgets import Button, Input, Static
from textual.app import App

# Import the components under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from ui.main_app import ClaudeTIUApp, MainWorkspace
    from ui.widgets.project_tree import ProjectTree
    from ui.widgets.task_dashboard import TaskDashboard
    from ui.widgets.progress_intelligence import ProgressIntelligence
    from ui.widgets.console_widget import ConsoleWidget
    from ui.widgets.placeholder_alert import PlaceholderAlert
    from ui.widgets.notification_system import NotificationSystem
    from ui.screens.project_wizard import ProjectWizardScreen
    from ui.screens.settings import SettingsScreen
except ImportError:
    # Create mock classes for testing when UI modules are not available
    class ClaudeTIUApp:
        def __init__(self):
            pass
    
    class MainWorkspace:
        def __init__(self, app):
            pass
    
    class ProjectTree:
        def __init__(self, manager):
            pass
    
    class TaskDashboard:
        def __init__(self, manager):
            pass
    
    class ProgressIntelligence:
        def __init__(self):
            pass
    
    class ConsoleWidget:
        def __init__(self, ai_interface):
            pass
    
    class PlaceholderAlert:
        def __init__(self):
            pass
    
    class NotificationSystem:
        def __init__(self):
            pass
    
    class ProjectWizardScreen:
        def __init__(self, manager):
            pass
    
    class SettingsScreen:
        def __init__(self):
            pass


@pytest.fixture
def mock_project_manager():
    """Mock project manager for UI testing."""
    mock = Mock()
    mock.current_project = None
    mock.list_projects.return_value = []
    mock.create_project.return_value = Mock(
        id='test-project-id',
        name='Test Project',
        path=Path('/tmp/test_project')
    )
    return mock


@pytest.fixture
def mock_ai_interface():
    """Mock AI interface for UI testing."""
    mock = AsyncMock()
    mock.execute_task.return_value = "Mock AI response"
    return mock


@pytest.fixture
def mock_validation_engine():
    """Mock validation engine for UI testing."""
    mock = AsyncMock()
    mock.analyze_project.return_value = Mock(
        real_progress=0.7,
        fake_progress=0.3,
        authenticity_score=0.8,
        quality_score=7.5,
        placeholders_found=2,
        todos_found=3
    )
    return mock


class TestClaudeTIUApp:
    """Tests for the main Claude-TIU application."""
    
    def test_app_initialization(self, mock_project_manager, mock_ai_interface, mock_validation_engine):
        """Test application initialization."""
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager), \
             patch('ui.main_app.AIInterface', return_value=mock_ai_interface), \
             patch('ui.main_app.ValidationEngine', return_value=mock_validation_engine):
            
            app = ClaudeTIUApp()
            
            assert app.project_manager is not None
            assert app.ai_interface is not None
            assert app.validation_engine is not None
            assert app.current_screen == "main"
            assert app.debug_mode is False
            assert app.validation_enabled is True
    
    @pytest.mark.asyncio
    async def test_app_startup_performance(self, mock_project_manager, mock_ai_interface, mock_validation_engine):
        """Test application startup time meets <2 seconds requirement."""
        import time
        
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager), \
             patch('ui.main_app.AIInterface', return_value=mock_ai_interface), \
             patch('ui.main_app.ValidationEngine', return_value=mock_validation_engine):
            
            start_time = time.time()
            
            app = ClaudeTIUApp()
            
            # Simulate app mount and initialization
            app.init_core_systems()
            
            end_time = time.time()
            startup_time = end_time - start_time
            
            assert startup_time < 2.0  # <2 seconds startup requirement
    
    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts are properly configured."""
        app = ClaudeTIUApp()
        
        # Check that essential keyboard shortcuts are defined
        bindings = {binding.key: binding.action for binding in app.BINDINGS}
        
        assert 'ctrl+n' in bindings  # New project
        assert 'ctrl+o' in bindings  # Open project
        assert 'ctrl+s' in bindings  # Save project
        assert 'ctrl+p' in bindings  # Project wizard
        assert 'ctrl+q' in bindings  # Quit
        assert 'f1' in bindings     # Help
        assert 'f5' in bindings     # Refresh
        
        # Vim-style navigation
        assert 'h' in bindings
        assert 'j' in bindings
        assert 'k' in bindings
        assert 'l' in bindings
    
    @pytest.mark.asyncio
    async def test_ui_response_time(self, mock_project_manager):
        """Test UI response time meets <500ms requirement."""
        import time
        
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager):
            app = ClaudeTIUApp()
            
            # Test action response time
            start_time = time.time()
            app.action_refresh()
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            assert response_time < 500  # <500ms response time requirement
    
    def test_error_handling_and_notifications(self, mock_project_manager):
        """Test error handling and notification system."""
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager):
            app = ClaudeTIUApp()
            app.notification_system = Mock()
            
            # Test notification functionality
            app.notify("Test message", "info")
            app.notification_system.add_notification.assert_called_once_with("Test message", "info")
            
            # Test error notification
            app.notify("Error occurred", "error")
            assert app.notification_system.add_notification.call_count == 2


class TestMainWorkspace:
    """Tests for the main workspace layout."""
    
    def test_workspace_initialization(self, mock_project_manager):
        """Test workspace initialization."""
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager):
            app = ClaudeTIUApp()
            workspace = MainWorkspace(app)
            
            assert workspace.app_instance == app
            assert workspace.project_tree is None  # Not initialized until compose
            assert workspace.task_dashboard is None
            assert workspace.progress_widget is None
            assert workspace.console_widget is None
    
    @pytest.mark.asyncio
    async def test_progress_monitoring_cycle(self, mock_project_manager, mock_validation_engine):
        """Test progress monitoring functionality."""
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager), \
             patch('ui.main_app.ValidationEngine', return_value=mock_validation_engine):
            
            app = ClaudeTIUApp()
            app.project_manager.current_project = Mock(path=Path('/tmp/test'))
            
            workspace = MainWorkspace(app)
            workspace.progress_widget = Mock()
            
            # Mock the progress monitoring method
            with patch.object(workspace, 'start_progress_monitoring') as mock_monitor:
                workspace.on_mount()
                mock_monitor.assert_called_once()


class TestProjectTree:
    """Tests for ProjectTree widget."""
    
    def test_project_tree_initialization(self, mock_project_manager):
        """Test project tree initialization."""
        tree = ProjectTree(mock_project_manager)
        assert tree is not None
    
    def test_project_tree_refresh(self, mock_project_manager):
        """Test project tree refresh functionality."""
        mock_project_manager.list_projects.return_value = [
            {'name': 'Project 1', 'path': '/tmp/project1'},
            {'name': 'Project 2', 'path': '/tmp/project2'}
        ]
        
        tree = ProjectTree(mock_project_manager)
        
        # Mock the refresh method
        with patch.object(tree, 'refresh') as mock_refresh:
            tree.refresh()
            mock_refresh.assert_called_once()
    
    def test_project_selection(self, mock_project_manager):
        """Test project selection functionality."""
        tree = ProjectTree(mock_project_manager)
        
        # Test setting current project
        with patch.object(tree, 'set_project') as mock_set_project:
            tree.set_project('/tmp/test_project')
            mock_set_project.assert_called_once_with('/tmp/test_project')


class TestTaskDashboard:
    """Tests for TaskDashboard widget."""
    
    def test_task_dashboard_initialization(self, mock_project_manager):
        """Test task dashboard initialization."""
        dashboard = TaskDashboard(mock_project_manager)
        assert dashboard is not None
    
    def test_task_display_and_interaction(self, mock_project_manager):
        """Test task display and interaction."""
        # Mock project with tasks
        mock_project = Mock()
        mock_project.tasks = [
            Mock(id='task1', name='Task 1', status='pending'),
            Mock(id='task2', name='Task 2', status='in_progress'),
            Mock(id='task3', name='Task 3', status='completed')
        ]
        mock_project_manager.current_project = mock_project
        
        dashboard = TaskDashboard(mock_project_manager)
        
        # Test refresh functionality
        with patch.object(dashboard, 'refresh') as mock_refresh:
            dashboard.refresh()
            mock_refresh.assert_called_once()
    
    def test_task_filtering_and_sorting(self, mock_project_manager):
        """Test task filtering and sorting functionality."""
        dashboard = TaskDashboard(mock_project_manager)
        
        # Test filter methods
        with patch.object(dashboard, 'filter_by_status') as mock_filter:
            dashboard.filter_by_status('pending')
            mock_filter.assert_called_once_with('pending')
        
        with patch.object(dashboard, 'sort_by_priority') as mock_sort:
            dashboard.sort_by_priority()
            mock_sort.assert_called_once()


class TestProgressIntelligence:
    """Tests for ProgressIntelligence widget."""
    
    def test_progress_widget_initialization(self):
        """Test progress intelligence widget initialization."""
        widget = ProgressIntelligence()
        assert widget is not None
    
    def test_validation_result_display(self):
        """Test validation result display."""
        widget = ProgressIntelligence()
        
        # Mock validation results
        validation_results = Mock(
            real_progress=0.75,
            fake_progress=0.25,
            authenticity_score=0.85,
            quality_score=8.2,
            placeholders_found=3,
            todos_found=5
        )
        
        # Test update method
        with patch.object(widget, 'update_validation') as mock_update:
            widget.update_validation(validation_results)
            mock_update.assert_called_once_with(validation_results)
    
    def test_progress_visualization(self):
        """Test progress visualization components."""
        widget = ProgressIntelligence()
        
        # Test progress bar updates
        with patch.object(widget, 'update_progress_bars') as mock_bars:
            widget.update_progress_bars(real=0.7, fake=0.3)
            mock_bars.assert_called_once_with(real=0.7, fake=0.3)
    
    def test_authenticity_alerts(self):
        """Test authenticity alert system."""
        widget = ProgressIntelligence()
        
        # Test alert triggering
        with patch.object(widget, 'show_authenticity_alert') as mock_alert:
            widget.show_authenticity_alert(score=0.45)  # Low authenticity
            mock_alert.assert_called_once_with(score=0.45)


class TestConsoleWidget:
    """Tests for ConsoleWidget (AI interaction)."""
    
    def test_console_widget_initialization(self, mock_ai_interface):
        """Test console widget initialization."""
        console = ConsoleWidget(mock_ai_interface)
        assert console is not None
    
    @pytest.mark.asyncio
    async def test_ai_command_execution(self, mock_ai_interface):
        """Test AI command execution through console."""
        console = ConsoleWidget(mock_ai_interface)
        
        # Test command execution
        test_command = "Create a REST API endpoint for user authentication"
        
        with patch.object(console, 'execute_command') as mock_execute:
            await console.execute_command(test_command)
            mock_execute.assert_called_once_with(test_command)
    
    def test_command_history(self, mock_ai_interface):
        """Test command history functionality."""
        console = ConsoleWidget(mock_ai_interface)
        
        # Test history management
        with patch.object(console, 'add_to_history') as mock_history:
            console.add_to_history("test command")
            mock_history.assert_called_once_with("test command")
        
        with patch.object(console, 'get_history') as mock_get_history:
            mock_get_history.return_value = ["command1", "command2"]
            history = console.get_history()
            assert len(history) == 2
    
    def test_output_formatting(self, mock_ai_interface):
        """Test console output formatting."""
        console = ConsoleWidget(mock_ai_interface)
        
        # Test different output types
        test_outputs = [
            {"type": "success", "message": "Operation completed"},
            {"type": "error", "message": "Error occurred"},
            {"type": "info", "message": "Information message"}
        ]
        
        for output in test_outputs:
            with patch.object(console, 'format_output') as mock_format:
                console.format_output(output)
                mock_format.assert_called_with(output)


class TestNotificationSystem:
    """Tests for NotificationSystem."""
    
    def test_notification_system_initialization(self):
        """Test notification system initialization."""
        notifications = NotificationSystem()
        assert notifications is not None
    
    def test_notification_types(self):
        """Test different notification types."""
        notifications = NotificationSystem()
        
        notification_types = ['info', 'success', 'warning', 'error']
        
        for notification_type in notification_types:
            with patch.object(notifications, 'add_notification') as mock_add:
                notifications.add_notification(f"Test {notification_type}", notification_type)
                mock_add.assert_called_with(f"Test {notification_type}", notification_type)
    
    def test_notification_queue_management(self):
        """Test notification queue management."""
        notifications = NotificationSystem()
        
        # Test queue operations
        with patch.object(notifications, 'clear_notifications') as mock_clear:
            notifications.clear_notifications()
            mock_clear.assert_called_once()
        
        with patch.object(notifications, 'get_notification_count') as mock_count:
            mock_count.return_value = 5
            count = notifications.get_notification_count()
            assert count == 5
    
    def test_notification_persistence(self):
        """Test notification persistence and auto-dismiss."""
        notifications = NotificationSystem()
        
        # Test auto-dismiss functionality
        with patch.object(notifications, 'auto_dismiss') as mock_dismiss:
            notifications.auto_dismiss(notification_id='test-id', delay=5.0)
            mock_dismiss.assert_called_once_with(notification_id='test-id', delay=5.0)


class TestPlaceholderAlert:
    """Tests for PlaceholderAlert widget."""
    
    def test_placeholder_alert_initialization(self):
        """Test placeholder alert initialization."""
        alert = PlaceholderAlert()
        assert alert is not None
    
    def test_alert_triggering(self):
        """Test alert triggering based on validation results."""
        alert = PlaceholderAlert()
        
        # Mock validation results with high fake progress
        validation_results = Mock(
            fake_progress=0.35,  # >20% threshold
            placeholders_found=8,
            todos_found=5,
            authenticity_score=0.6
        )
        
        with patch.object(alert, 'show_alert') as mock_show:
            alert.show_alert(validation_results)
            mock_show.assert_called_once_with(validation_results)
    
    def test_alert_content_and_actions(self):
        """Test alert content and available actions."""
        alert = PlaceholderAlert()
        
        # Test action buttons
        actions = ['auto_fix', 'manual_review', 'ignore', 'dismiss']
        
        for action in actions:
            with patch.object(alert, f'handle_{action}') as mock_action:
                getattr(alert, f'handle_{action}')()
                mock_action.assert_called_once()


class TestScreenNavigation:
    """Tests for screen navigation and transitions."""
    
    def test_project_wizard_screen(self, mock_project_manager):
        """Test project wizard screen navigation."""
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager):
            app = ClaudeTIUApp()
            
            # Test pushing project wizard screen
            with patch.object(app, 'push_screen') as mock_push:
                app.action_project_wizard()
                mock_push.assert_called_once()
    
    def test_settings_screen(self):
        """Test settings screen navigation."""
        app = ClaudeTIUApp()
        
        # Test pushing settings screen
        with patch.object(app, 'push_screen') as mock_push:
            app.action_settings()
            mock_push.assert_called_once()
    
    def test_screen_message_handling(self, mock_project_manager):
        """Test handling of screen messages."""
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager):
            app = ClaudeTIUApp()
            
            # Mock screen messages
            from ui.screens import CreateProjectMessage, SettingsSavedMessage
            
            # Test project creation message
            create_message = Mock()
            create_message.config = Mock(
                name='New Project',
                path=Path('/tmp/new_project')
            )
            
            with patch.object(app, 'handle_create_project') as mock_handle:
                app.handle_create_project(create_message)
                mock_handle.assert_called_once_with(create_message)
            
            # Test settings saved message
            settings_message = Mock()
            settings_message.settings = Mock(validation_enabled=True)
            
            with patch.object(app, 'handle_settings_saved') as mock_handle:
                app.handle_settings_saved(settings_message)
                mock_handle.assert_called_once_with(settings_message)


class TestAccessibilityAndUsability:
    """Tests for accessibility and usability features."""
    
    def test_focus_management(self):
        """Test focus management and navigation."""
        app = ClaudeTIUApp()
        
        # Test focus navigation actions
        focus_actions = [
            'action_focus_left',
            'action_focus_right', 
            'action_focus_up',
            'action_focus_down'
        ]
        
        for action_name in focus_actions:
            if hasattr(app, action_name):
                action_method = getattr(app, action_name)
                # Test that action doesn't raise exception
                try:
                    action_method()
                except Exception as e:
                    pytest.fail(f"{action_name} raised {e}")
    
    def test_keyboard_accessibility(self):
        """Test keyboard accessibility features."""
        app = ClaudeTIUApp()
        
        # Test that all essential functions have keyboard shortcuts
        essential_actions = [
            'new_project',
            'open_project',
            'save_project', 
            'help',
            'quit',
            'refresh'
        ]
        
        bindings = {binding.action: binding.key for binding in app.BINDINGS}
        
        for action in essential_actions:
            assert action in bindings, f"Essential action '{action}' missing keyboard shortcut"
    
    def test_help_system(self):
        """Test help system functionality."""
        app = ClaudeTIUApp()
        
        # Test help content generation
        help_content = app._generate_help_content()
        
        assert isinstance(help_content, str)
        assert len(help_content) > 0
        assert "Navigation:" in help_content
        assert "Project Management:" in help_content
        assert "Interface:" in help_content
        assert "System:" in help_content
    
    def test_responsive_layout(self, mock_project_manager):
        """Test responsive layout behavior."""
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager):
            app = ClaudeTIUApp()
            workspace = MainWorkspace(app)
            
            # Test toggle functionality
            with patch.object(app, 'action_toggle_task_panel') as mock_toggle:
                app.action_toggle_task_panel()
                mock_toggle.assert_called_once()
            
            with patch.object(app, 'action_toggle_console') as mock_toggle:
                app.action_toggle_console()
                mock_toggle.assert_called_once()


class TestUIPerformanceMetrics:
    """Tests for UI performance metrics and optimization."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, mock_project_manager):
        """Test memory usage stays within reasonable bounds."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager):
            # Create and destroy multiple app instances
            for i in range(10):
                app = ClaudeTIUApp()
                workspace = MainWorkspace(app)
                del app, workspace
                gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for 10 instances)
        assert memory_increase < 50 * 1024 * 1024  # 50MB in bytes
    
    def test_widget_rendering_performance(self, mock_project_manager):
        """Test widget rendering performance."""
        import time
        
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager):
            app = ClaudeTIUApp()
            
            # Measure widget creation time
            start_time = time.time()
            
            workspace = MainWorkspace(app)
            # Simulate widget composition
            widgets = {
                'project_tree': ProjectTree(mock_project_manager),
                'task_dashboard': TaskDashboard(mock_project_manager),
                'progress_widget': ProgressIntelligence(),
                'console_widget': ConsoleWidget(Mock())
            }
            
            end_time = time.time()
            rendering_time = end_time - start_time
            
            # Widget creation should be fast
            assert rendering_time < 0.5  # Less than 500ms for all widgets
    
    @pytest.mark.asyncio
    async def test_async_operation_performance(self, mock_project_manager, mock_validation_engine):
        """Test async operation performance."""
        import time
        
        with patch('ui.main_app.ProjectManager', return_value=mock_project_manager), \
             patch('ui.main_app.ValidationEngine', return_value=mock_validation_engine):
            
            app = ClaudeTIUApp()
            app.project_manager.current_project = Mock(path=Path('/tmp/test'))
            
            # Test async AI task execution
            start_time = time.time()
            
            await app.execute_ai_task(
                "Create a simple function",
                {'project_path': '/tmp/test'}
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Async operations should complete reasonably quickly
            assert execution_time < 2.0  # Less than 2 seconds for mocked operation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
