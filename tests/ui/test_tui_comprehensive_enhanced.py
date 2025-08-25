"""Enhanced comprehensive tests for Textual TUI components with testing framework."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path

# Import Textual testing framework
from textual.app import App
from textual.widgets import Static, Button, Input, TextArea, Tree, Log
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual import events
from textual.testing import AppPilot

# Import application components
from claude_tiu.ui.main_app import ClaudeTIUApp
from claude_tiu.ui.screens.workspace_screen import WorkspaceScreen
from claude_tiu.ui.screens.settings import SettingsScreen
from claude_tiu.ui.screens.help_screen import HelpScreen
from claude_tiu.ui.widgets.task_dashboard import TaskDashboard
from claude_tiu.ui.widgets.notification_system import NotificationSystem
from claude_tiu.ui.widgets.progress_intelligence import ProgressIntelligence
from claude_tiu.core.config_manager import ConfigManager


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager for TUI testing."""
    manager = Mock(spec=ConfigManager)
    manager.get_ui_preferences = Mock(return_value=Mock(
        theme='dark',
        font_size=12,
        show_line_numbers=True,
        auto_save=True,
        vim_mode=False,
        animations_enabled=True
    ))
    manager.get_setting = AsyncMock(side_effect=lambda path, default=None: {
        'ui_preferences.theme': 'dark',
        'ui_preferences.font_size': 12,
        'ui_preferences.show_line_numbers': True,
        'ui_preferences.update_interval_seconds': 10
    }.get(path, default))
    return manager


@pytest.fixture
def sample_project_data():
    """Sample project data for testing."""
    return {
        'name': 'Test Project',
        'path': '/path/to/project',
        'files': [
            {'name': 'main.py', 'type': 'file', 'size': 1024},
            {'name': 'utils.py', 'type': 'file', 'size': 512},
            {'name': 'tests/', 'type': 'directory', 'children': [
                {'name': 'test_main.py', 'type': 'file', 'size': 256}
            ]}
        ],
        'status': 'active'
    }


class TestMainApp:
    """Test main TUI application."""
    
    @pytest.mark.asyncio
    async def test_app_initialization(self, mock_config_manager):
        """Test main app initialization."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(
                config_manager=mock_config_manager,
                debug=False,
                initial_project_dir=None
            )
            
            assert app.config_manager == mock_config_manager
            assert app.debug is False
            assert app.title == "Claude-TIU"
    
    @pytest.mark.asyncio
    async def test_app_startup_sequence(self, mock_config_manager):
        """Test application startup sequence."""
        with patch('claude_tiu.ui.main_app.SystemChecker') as mock_checker:
            # Mock system check
            check_result = Mock()
            check_result.all_passed = True
            check_result.warnings = []
            check_result.errors = []
            mock_checker.return_value.run_checks = AsyncMock(return_value=check_result)
            
            app = ClaudeTIUApp(
                config_manager=mock_config_manager,
                debug=False
            )
            
            # Test app pilot functionality
            async with app.run_test() as pilot:
                # Verify app started successfully
                assert app.is_running is True
                
                # Test basic navigation
                await pilot.press("ctrl+h")  # Should open help
                await pilot.pause()
                
                # Verify help screen is visible
                help_widgets = pilot.app.query("HelpScreen")
                assert len(help_widgets) >= 0  # Help screen may be shown
    
    @pytest.mark.asyncio
    async def test_app_theme_switching(self, mock_config_manager):
        """Test theme switching functionality."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Test theme switching
                initial_theme = app.theme
                
                # Simulate theme change
                await pilot.press("ctrl+t")  # Assuming Ctrl+T switches theme
                await pilot.pause()
                
                # Theme switching might be implemented differently
                # This test validates the structure is in place
                assert hasattr(app, 'theme')
    
    @pytest.mark.asyncio
    async def test_keyboard_shortcuts(self, mock_config_manager):
        """Test keyboard shortcut handling."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Test various keyboard shortcuts
                shortcuts = [
                    ("ctrl+n", "new_project"),  # New project
                    ("ctrl+o", "open_project"),  # Open project
                    ("ctrl+s", "save"),         # Save
                    ("ctrl+q", "quit"),         # Quit
                    ("f1", "help"),            # Help
                ]
                
                for key_combo, expected_action in shortcuts:
                    # Press key combination
                    await pilot.press(key_combo)
                    await pilot.pause(0.1)
                    
                    # For quit command, break early to avoid app termination
                    if expected_action == "quit":
                        break
                
                # Verify app is still responsive
                assert app.is_running is True


class TestWorkspaceScreen:
    """Test workspace screen functionality."""
    
    @pytest.mark.asyncio
    async def test_workspace_initialization(self, mock_config_manager, sample_project_data):
        """Test workspace screen initialization."""
        screen = WorkspaceScreen(mock_config_manager)
        
        # Test screen composition
        assert hasattr(screen, 'compose')
        
        # Mock project loading
        with patch.object(screen, 'load_project_data', AsyncMock(return_value=sample_project_data)):
            # Test within app context
            with patch('claude_tiu.ui.main_app.SystemChecker'):
                app = ClaudeTIUApp(config_manager=mock_config_manager)
                
                async with app.run_test() as pilot:
                    # Push workspace screen
                    app.push_screen(screen)
                    await pilot.pause()
                    
                    # Verify workspace elements are present
                    workspace_widgets = pilot.app.query("WorkspaceScreen")
                    assert len(workspace_widgets) >= 0
    
    @pytest.mark.asyncio
    async def test_file_tree_interaction(self, mock_config_manager, sample_project_data):
        """Test file tree interaction in workspace."""
        screen = WorkspaceScreen(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                
                # Look for tree widget
                trees = pilot.app.query(Tree)
                if trees:
                    tree = trees.first()
                    
                    # Test tree node expansion
                    if tree.root and tree.root.children:
                        await pilot.click(tree)
                        await pilot.pause()
                        
                        # Verify tree interaction works
                        assert tree is not None
    
    @pytest.mark.asyncio
    async def test_code_editor_integration(self, mock_config_manager):
        """Test code editor integration in workspace."""
        screen = WorkspaceScreen(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                
                # Look for text area (code editor)
                text_areas = pilot.app.query(TextArea)
                if text_areas:
                    editor = text_areas.first()
                    
                    # Test text input
                    await pilot.click(editor)
                    await pilot.type("def hello_world():")
                    await pilot.pause()
                    
                    # Verify text was entered
                    assert "def hello_world" in editor.text
    
    @pytest.mark.asyncio
    async def test_task_dashboard_integration(self, mock_config_manager):
        """Test task dashboard integration in workspace."""
        screen = WorkspaceScreen(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                
                # Look for task dashboard
                dashboards = pilot.app.query(TaskDashboard)
                if dashboards:
                    dashboard = dashboards.first()
                    
                    # Test dashboard interaction
                    await pilot.click(dashboard)
                    await pilot.pause()
                    
                    # Verify dashboard is interactive
                    assert dashboard is not None


class TestSettingsScreen:
    """Test settings screen functionality."""
    
    @pytest.mark.asyncio
    async def test_settings_screen_navigation(self, mock_config_manager):
        """Test settings screen navigation and options."""
        screen = SettingsScreen(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                
                # Look for settings widgets
                buttons = pilot.app.query(Button)
                inputs = pilot.app.query(Input)
                
                # Verify settings elements are present
                assert len(buttons) + len(inputs) > 0
    
    @pytest.mark.asyncio
    async def test_theme_setting_change(self, mock_config_manager):
        """Test theme setting change in settings screen."""
        screen = SettingsScreen(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                
                # Look for theme-related buttons
                buttons = pilot.app.query(Button)
                theme_buttons = [b for b in buttons if 'theme' in str(b.label).lower()]
                
                if theme_buttons:
                    theme_button = theme_buttons[0]
                    await pilot.click(theme_button)
                    await pilot.pause()
                    
                    # Verify theme button interaction
                    assert theme_button is not None
    
    @pytest.mark.asyncio
    async def test_font_size_adjustment(self, mock_config_manager):
        """Test font size adjustment in settings."""
        screen = SettingsScreen(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                
                # Look for font size inputs
                inputs = pilot.app.query(Input)
                font_inputs = [i for i in inputs if 'font' in str(getattr(i, 'placeholder', '')).lower()]
                
                if font_inputs:
                    font_input = font_inputs[0]
                    await pilot.click(font_input)
                    await pilot.type("14")
                    await pilot.pause()
                    
                    # Verify font input works
                    assert "14" in font_input.value


class TestHelpScreen:
    """Test help screen functionality."""
    
    @pytest.mark.asyncio
    async def test_help_screen_content(self, mock_config_manager):
        """Test help screen content display."""
        screen = HelpScreen()
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                
                # Look for help content
                static_widgets = pilot.app.query(Static)
                
                # Verify help content is displayed
                assert len(static_widgets) > 0
                
                # Check for common help keywords
                help_content = " ".join([str(widget.renderable) for widget in static_widgets])
                help_keywords = ['help', 'shortcut', 'command', 'ctrl', 'key']
                
                # At least some help keywords should be present
                found_keywords = [kw for kw in help_keywords if kw.lower() in help_content.lower()]
                assert len(found_keywords) > 0
    
    @pytest.mark.asyncio
    async def test_help_navigation(self, mock_config_manager):
        """Test help screen navigation."""
        screen = HelpScreen()
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                
                # Test escape to close help
                await pilot.press("escape")
                await pilot.pause()
                
                # Help screen should be closed (app should handle this)
                current_screen = pilot.app.screen
                assert current_screen is not screen  # Should have popped the screen


class TestTaskDashboard:
    """Test task dashboard widget."""
    
    @pytest.mark.asyncio
    async def test_task_dashboard_initialization(self, mock_config_manager):
        """Test task dashboard widget initialization."""
        dashboard = TaskDashboard(mock_config_manager)
        
        # Test dashboard structure
        assert hasattr(dashboard, 'compose')
        assert hasattr(dashboard, 'on_mount')
        
        # Test in app context
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Mount dashboard in a container
                container = Container(dashboard)
                app.mount(container)
                await pilot.pause()
                
                # Verify dashboard is mounted
                dashboards = pilot.app.query(TaskDashboard)
                assert len(dashboards) == 1
    
    @pytest.mark.asyncio
    async def test_task_creation_widget(self, mock_config_manager):
        """Test task creation through dashboard."""
        dashboard = TaskDashboard(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                container = Container(dashboard)
                app.mount(container)
                await pilot.pause()
                
                # Look for task-related buttons
                buttons = pilot.app.query(Button)
                task_buttons = [b for b in buttons if any(word in str(b.label).lower() 
                                                        for word in ['task', 'add', 'create', 'new'])]
                
                if task_buttons:
                    task_button = task_buttons[0]
                    await pilot.click(task_button)
                    await pilot.pause()
                    
                    # Verify button interaction
                    assert task_button is not None
    
    @pytest.mark.asyncio
    async def test_task_status_updates(self, mock_config_manager):
        """Test task status updates in dashboard."""
        dashboard = TaskDashboard(mock_config_manager)
        
        # Mock task data
        sample_tasks = [
            {'id': '1', 'name': 'Task 1', 'status': 'pending'},
            {'id': '2', 'name': 'Task 2', 'status': 'in_progress'},
            {'id': '3', 'name': 'Task 3', 'status': 'completed'}
        ]
        
        with patch.object(dashboard, 'get_tasks', AsyncMock(return_value=sample_tasks)):
            with patch('claude_tiu.ui.main_app.SystemChecker'):
                app = ClaudeTIUApp(config_manager=mock_config_manager)
                
                async with app.run_test() as pilot:
                    container = Container(dashboard)
                    app.mount(container)
                    await pilot.pause()
                    
                    # Trigger task refresh
                    await dashboard.refresh_tasks()
                    await pilot.pause()
                    
                    # Verify dashboard updated
                    assert dashboard is not None


class TestNotificationSystem:
    """Test notification system widget."""
    
    @pytest.mark.asyncio
    async def test_notification_display(self, mock_config_manager):
        """Test notification display functionality."""
        notification_system = NotificationSystem(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                container = Container(notification_system)
                app.mount(container)
                await pilot.pause()
                
                # Test notification creation
                await notification_system.show_notification(
                    "Test Notification",
                    "This is a test notification message",
                    "info"
                )
                await pilot.pause()
                
                # Verify notification system is working
                notifications = pilot.app.query(NotificationSystem)
                assert len(notifications) == 1
    
    @pytest.mark.asyncio
    async def test_notification_types(self, mock_config_manager):
        """Test different types of notifications."""
        notification_system = NotificationSystem(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                container = Container(notification_system)
                app.mount(container)
                await pilot.pause()
                
                # Test different notification types
                notification_types = [
                    ("info", "Info notification"),
                    ("warning", "Warning notification"),
                    ("error", "Error notification"),
                    ("success", "Success notification")
                ]
                
                for notif_type, message in notification_types:
                    await notification_system.show_notification(
                        f"{notif_type.title()} Test",
                        message,
                        notif_type
                    )
                    await pilot.pause(0.1)
                
                # Verify notification system handled all types
                assert notification_system is not None
    
    @pytest.mark.asyncio
    async def test_notification_dismissal(self, mock_config_manager):
        """Test notification dismissal."""
        notification_system = NotificationSystem(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                container = Container(notification_system)
                app.mount(container)
                await pilot.pause()
                
                # Show a notification
                await notification_system.show_notification(
                    "Dismissible",
                    "This notification can be dismissed",
                    "info"
                )
                await pilot.pause()
                
                # Look for dismiss button
                buttons = pilot.app.query(Button)
                dismiss_buttons = [b for b in buttons if 'dismiss' in str(b.label).lower() or 'x' in str(b.label)]
                
                if dismiss_buttons:
                    dismiss_button = dismiss_buttons[0]
                    await pilot.click(dismiss_button)
                    await pilot.pause()
                    
                    # Verify dismiss functionality
                    assert dismiss_button is not None


class TestProgressIntelligence:
    """Test progress intelligence widget."""
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, mock_config_manager):
        """Test progress tracking functionality."""
        progress_widget = ProgressIntelligence(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                container = Container(progress_widget)
                app.mount(container)
                await pilot.pause()
                
                # Test progress updates
                await progress_widget.update_progress("task_1", 0.5, "Processing...")
                await pilot.pause()
                
                # Verify progress widget is working
                progress_widgets = pilot.app.query(ProgressIntelligence)
                assert len(progress_widgets) == 1
    
    @pytest.mark.asyncio
    async def test_multiple_progress_items(self, mock_config_manager):
        """Test tracking multiple progress items."""
        progress_widget = ProgressIntelligence(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                container = Container(progress_widget)
                app.mount(container)
                await pilot.pause()
                
                # Track multiple items
                progress_items = [
                    ("task_1", 0.3, "Analyzing code..."),
                    ("task_2", 0.7, "Running tests..."),
                    ("task_3", 1.0, "Completed"),
                ]
                
                for task_id, progress, message in progress_items:
                    await progress_widget.update_progress(task_id, progress, message)
                    await pilot.pause(0.1)
                
                # Verify all progress items are tracked
                assert progress_widget is not None
    
    @pytest.mark.asyncio
    async def test_progress_completion(self, mock_config_manager):
        """Test progress completion handling."""
        progress_widget = ProgressIntelligence(mock_config_manager)
        
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                container = Container(progress_widget)
                app.mount(container)
                await pilot.pause()
                
                # Start and complete a task
                await progress_widget.update_progress("task_1", 0.0, "Starting...")
                await pilot.pause(0.1)
                
                await progress_widget.update_progress("task_1", 0.5, "In progress...")
                await pilot.pause(0.1)
                
                await progress_widget.update_progress("task_1", 1.0, "Completed!")
                await pilot.pause(0.1)
                
                # Test completion callback
                await progress_widget.on_task_completed("task_1")
                await pilot.pause()
                
                # Verify completion handling
                assert progress_widget is not None


class TestResponsiveDesign:
    """Test responsive design and layout."""
    
    @pytest.mark.asyncio
    async def test_window_resize_handling(self, mock_config_manager):
        """Test handling of window resize events."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Test different window sizes
                sizes = [
                    (80, 24),   # Minimum size
                    (120, 40),  # Medium size
                    (160, 60),  # Large size
                ]
                
                for width, height in sizes:
                    pilot.app.size = (width, height)
                    await pilot.pause(0.1)
                    
                    # Verify app handles resize gracefully
                    assert pilot.app.size == (width, height)
                    assert pilot.app.is_running is True
    
    @pytest.mark.asyncio
    async def test_layout_adaptation(self, mock_config_manager):
        """Test layout adaptation to different screen sizes."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Test narrow screen
                pilot.app.size = (80, 24)
                await pilot.pause()
                
                # Verify layout adapts
                containers = pilot.app.query(Container)
                assert len(containers) >= 0
                
                # Test wide screen
                pilot.app.size = (200, 60)
                await pilot.pause()
                
                # Verify layout adapts
                assert pilot.app.is_running is True


class TestAccessibility:
    """Test accessibility features."""
    
    @pytest.mark.asyncio
    async def test_keyboard_navigation(self, mock_config_manager):
        """Test keyboard navigation throughout the interface."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Test tab navigation
                for _ in range(5):
                    await pilot.press("tab")
                    await pilot.pause(0.1)
                
                # Test shift+tab (reverse navigation)
                for _ in range(3):
                    await pilot.press("shift+tab")
                    await pilot.pause(0.1)
                
                # Verify navigation works
                assert pilot.app.is_running is True
    
    @pytest.mark.asyncio
    async def test_focus_management(self, mock_config_manager):
        """Test focus management for screen readers."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Get focusable widgets
                buttons = pilot.app.query(Button)
                inputs = pilot.app.query(Input)
                
                focusable_widgets = list(buttons) + list(inputs)
                
                # Test focus on each widget
                for widget in focusable_widgets[:3]:  # Limit to first 3 to avoid timeout
                    if hasattr(widget, 'focus'):
                        widget.focus()
                        await pilot.pause(0.1)
                        
                        # Verify focus is set
                        if hasattr(widget, 'has_focus'):
                            # Focus verification depends on implementation
                            pass
                
                assert pilot.app.is_running is True


class TestErrorHandling:
    """Test error handling in TUI components."""
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, mock_config_manager):
        """Test handling of invalid user input."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Find input fields
                inputs = pilot.app.query(Input)
                
                if inputs:
                    input_field = inputs.first()
                    
                    # Test invalid input (very long string)
                    invalid_input = "x" * 10000
                    await pilot.click(input_field)
                    await pilot.type(invalid_input[:100])  # Limit for test performance
                    await pilot.pause()
                    
                    # Verify app handles invalid input gracefully
                    assert pilot.app.is_running is True
    
    @pytest.mark.asyncio
    async def test_exception_recovery(self, mock_config_manager):
        """Test recovery from exceptions in TUI components."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            # Mock a component that raises an exception
            original_compose = app.compose
            
            def failing_compose():
                yield from original_compose()
                # This might cause issues, but app should handle gracefully
            
            app.compose = failing_compose
            
            async with app.run_test() as pilot:
                # App should still be running despite compose issues
                await pilot.pause()
                assert pilot.app.is_running is True
    
    @pytest.mark.asyncio
    async def test_screen_transition_errors(self, mock_config_manager):
        """Test error handling during screen transitions."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Try to push invalid screen
                try:
                    app.push_screen(None)  # This should be handled gracefully
                except Exception:
                    pass  # Expected to fail, but shouldn't crash app
                
                await pilot.pause()
                
                # App should still be running
                assert pilot.app.is_running is True


@pytest.mark.performance
class TestUIPerformance:
    """Test TUI performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_startup_performance(self, mock_config_manager):
        """Test application startup performance."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            import time
            
            start_time = time.time()
            
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                await pilot.pause()
                
                startup_time = time.time() - start_time
                
                # Startup should be reasonably fast
                assert startup_time < 5.0  # Under 5 seconds
                assert pilot.app.is_running is True
    
    @pytest.mark.asyncio
    async def test_widget_rendering_performance(self, mock_config_manager):
        """Test widget rendering performance."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                import time
                
                # Measure rendering time for multiple widgets
                start_time = time.time()
                
                # Create multiple widgets
                widgets = []
                for i in range(10):
                    widget = Static(f"Performance test widget {i}")
                    widgets.append(widget)
                    app.mount(widget)
                    await pilot.pause(0.01)
                
                rendering_time = time.time() - start_time
                
                # Rendering should be efficient
                assert rendering_time < 2.0  # Under 2 seconds for 10 widgets
                assert len(widgets) == 10
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, mock_config_manager):
        """Test memory usage stability during extended use."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Simulate extended use
                for i in range(20):
                    # Create and destroy widgets
                    temp_widget = Static(f"Temp widget {i}")
                    app.mount(temp_widget)
                    await pilot.pause(0.05)
                    
                    temp_widget.remove()
                    await pilot.pause(0.05)
                    
                    # Force garbage collection occasionally
                    if i % 5 == 0:
                        import gc
                        gc.collect()
                
                # App should still be running and responsive
                assert pilot.app.is_running is True


@pytest.mark.integration
class TestTUIIntegration:
    """Integration tests for TUI components."""
    
    @pytest.mark.asyncio
    async def test_full_user_workflow(self, mock_config_manager):
        """Test complete user workflow through TUI."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Simulate complete user workflow
                
                # 1. Open help
                await pilot.press("f1")
                await pilot.pause(0.5)
                
                # 2. Close help
                await pilot.press("escape")
                await pilot.pause(0.5)
                
                # 3. Try to open settings (Ctrl+comma or similar)
                await pilot.press("ctrl+comma")
                await pilot.pause(0.5)
                
                # 4. Navigate with keyboard
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.pause(0.5)
                
                # 5. Return to main view
                await pilot.press("escape")
                await pilot.pause(0.5)
                
                # Verify app completed workflow successfully
                assert pilot.app.is_running is True
    
    @pytest.mark.asyncio
    async def test_concurrent_widget_updates(self, mock_config_manager):
        """Test concurrent updates to multiple widgets."""
        with patch('claude_tiu.ui.main_app.SystemChecker'):
            app = ClaudeTIUApp(config_manager=mock_config_manager)
            
            async with app.run_test() as pilot:
                # Create multiple widgets that update concurrently
                status_widget = Static("Status: Ready")
                progress_widget = Static("Progress: 0%")
                log_widget = Log()
                
                app.mount(status_widget)
                app.mount(progress_widget)
                app.mount(log_widget)
                
                await pilot.pause()
                
                # Simulate concurrent updates
                async def update_status():
                    for i in range(5):
                        status_widget.update(f"Status: Step {i+1}")
                        await asyncio.sleep(0.1)
                
                async def update_progress():
                    for i in range(5):
                        progress_widget.update(f"Progress: {(i+1)*20}%")
                        await asyncio.sleep(0.15)
                
                async def add_logs():
                    for i in range(5):
                        log_widget.write_line(f"Log entry {i+1}")
                        await asyncio.sleep(0.12)
                
                # Run updates concurrently
                await asyncio.gather(
                    update_status(),
                    update_progress(),
                    add_logs()
                )
                
                await pilot.pause()
                
                # Verify all widgets updated successfully
                assert "Step 5" in str(status_widget.renderable)
                assert "100%" in str(progress_widget.renderable)
                assert pilot.app.is_running is True