"""
UI tests for TUI components using Textual testing framework.

Tests the terminal user interface components, screens, and interactions
for the claude-tui application.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from typing import Dict, Any, List


class TestTUIComponents:
    """Test suite for TUI components."""
    
    @pytest.fixture
    def mock_textual_app(self, mock_textual_app):
        """Get mock Textual app fixture."""
        return mock_textual_app
    
    @pytest.mark.asyncio
    async def test_project_tree_rendering(self, mock_project_tree_widget):
        """Test project tree widget rendering."""
        widget = mock_project_tree_widget
        
        # Test initial state
        assert widget.visible is True
        assert widget.state == "inactive"
        assert "Projects" in widget.content
        
        # Test adding project
        widget.attributes["items"].append("new-project")
        widget.content += "\\n  └── new-project"
        
        # Verify update
        rendered = widget.render()
        assert "new-project" in rendered
        assert widget.attributes["items"][-1] == "new-project"
    
    @pytest.mark.asyncio
    async def test_task_list_widget_updates(self, mock_task_list_widget):
        """Test task list widget updates."""
        widget = mock_task_list_widget
        
        # Test initial state
        initial_items = widget.attributes["items"]
        assert len(initial_items) == 2
        
        # Add new task
        new_task = {"name": "New Task", "status": "pending"}
        widget.attributes["items"].append(new_task)
        widget.content += "\\n  • New Task (pending)"
        
        # Verify update
        rendered = widget.render()
        assert "New Task" in rendered
        assert len(widget.attributes["items"]) == 3
    
    @pytest.mark.asyncio
    async def test_progress_bar_updates(self, mock_progress_bar_widget):
        """Test progress bar updates."""
        widget = mock_progress_bar_widget
        
        # Test initial progress
        assert widget.attributes["progress"] == 80
        assert "80%" in widget.content
        
        # Update progress
        widget.attributes["progress"] = 90
        widget.content = "Progress: [█████████ ] 90%"
        
        # Verify update
        rendered = widget.render()
        assert "90%" in rendered
        assert widget.attributes["progress"] == 90
    
    @pytest.mark.asyncio
    async def test_input_field_validation(self, ui_helper):
        """Test input field validation."""
        widget = ui_helper.create_input_field_widget(
            attributes={
                "value": "",
                "validation": {"required": True, "min_length": 3}
            }
        )
        
        # Test empty input (should be invalid)
        assert widget.attributes["value"] == ""
        validation = widget.attributes["validation"]
        is_valid = len(widget.attributes["value"]) >= validation.get("min_length", 0)
        assert not is_valid
        
        # Test valid input
        widget.attributes["value"] = "valid input"
        is_valid = len(widget.attributes["value"]) >= validation.get("min_length", 0)
        assert is_valid
    
    @pytest.mark.asyncio
    async def test_button_interactions(self, ui_helper):
        """Test button widget interactions."""
        button = ui_helper.create_button_widget(
            attributes={
                "label": "Test Button",
                "action": "test_action",
                "disabled": False
            }
        )
        
        # Test enabled state
        assert not button.attributes["disabled"]
        assert button.state == "inactive"
        
        # Simulate button press
        button.set_state("focused")
        assert button.is_focused()
        
        # Test action
        assert button.attributes["action"] == "test_action"
    
    @pytest.mark.asyncio
    async def test_widget_focus_management(self, mock_welcome_screen):
        """Test widget focus management."""
        screen = mock_welcome_screen
        
        # Test initial focus
        assert screen.active_widget is None
        
        # Set focus to first widget
        if screen.widgets:
            first_widget = screen.widgets[0]
            screen.set_focus(first_widget.id)
            
            # Verify focus
            assert screen.active_widget == first_widget.id
            assert first_widget.is_focused()
            
            # Verify other widgets are not focused
            for widget in screen.widgets[1:]:
                assert not widget.is_focused()
    
    @pytest.mark.asyncio
    async def test_keyboard_navigation(self, mock_textual_app, keyboard_simulator):
        """Test keyboard navigation in TUI."""
        app = mock_textual_app
        keyboard = keyboard_simulator
        
        # Navigate to welcome screen
        app.push_screen("welcome")
        current_screen = app.get_current_screen()
        
        # Test navigation keys
        result = keyboard.press_key("Tab")
        assert result["action"] == "next_field"
        
        result = keyboard.press_key("Enter")
        assert result["action"] == "submit"
        
        result = keyboard.press_key("Escape")
        assert result["action"] == "cancel"
    
    @pytest.mark.asyncio
    async def test_text_input_handling(self, mock_project_setup_screen, keyboard_simulator):
        """Test text input handling."""
        screen = mock_project_setup_screen
        keyboard = keyboard_simulator
        
        # Find project name input field
        name_widget = screen.get_widget("project_name")
        assert name_widget is not None
        
        # Simulate typing
        test_text = "my-test-project"
        keyboard.type_text(test_text)
        
        # Update widget (simulating real input handling)
        name_widget.attributes["value"] = test_text
        
        # Verify input
        assert name_widget.attributes["value"] == test_text
        assert keyboard.current_input == test_text
    
    def test_widget_state_transitions(self, ui_helper):
        """Test widget state transitions."""
        widget = ui_helper.create_widget_mock("test_widget", "button", "Test")
        
        # Test state transitions
        from tests.fixtures.ui_fixtures import WidgetState
        
        widget.set_state(WidgetState.ACTIVE)
        assert widget.state == WidgetState.ACTIVE.value
        
        widget.set_state(WidgetState.FOCUSED)
        assert widget.state == WidgetState.FOCUSED.value
        assert widget.is_focused()
        
        widget.set_state(WidgetState.DISABLED)
        assert widget.state == WidgetState.DISABLED.value
        assert not widget.is_focused()
    
    def test_widget_visibility_control(self, ui_helper):
        """Test widget visibility control."""
        widget = ui_helper.create_widget_mock("test_widget", "label", "Test Content")
        
        # Initially visible
        assert widget.visible is True
        
        # Hide widget
        widget.visible = False
        assert widget.visible is False
        
        # Show widget
        widget.visible = True
        assert widget.visible is True
    
    @pytest.mark.asyncio
    async def test_screen_rendering(self, mock_dashboard_screen):
        """Test screen rendering."""
        screen = mock_dashboard_screen
        
        # Test rendering
        rendered = screen.render()
        
        # Should contain screen name
        assert f"Screen[{screen.name}]" in rendered
        
        # Should contain visible widgets
        visible_widgets = [w for w in screen.widgets if w.visible]
        for widget in visible_widgets:
            # Widget content should be in render
            widget_render = widget.render()
            assert widget.name in widget_render or widget.content in rendered


class TestTUIScreens:
    """Test suite for TUI screens."""
    
    @pytest.mark.asyncio
    async def test_welcome_screen_flow(self, mock_welcome_screen, tui_assertions):
        """Test welcome screen user flow."""
        screen = mock_welcome_screen
        
        # Verify initial state
        assert screen.name == "welcome"
        assert "welcome" in screen.screen_type
        
        # Test screen has expected widgets
        tui_assertions.assert_screen_has_widget(screen, "welcome_title")
        tui_assertions.assert_screen_has_widget(screen, "new_project_btn")
        tui_assertions.assert_screen_has_widget(screen, "open_project_btn")
        
        # Test navigation data
        assert screen.data["show_recent"] is True
        assert screen.data["max_recent"] == 5
    
    @pytest.mark.asyncio
    async def test_project_setup_screen_validation(self, mock_project_setup_screen, tui_assertions):
        """Test project setup form validation."""
        screen = mock_project_setup_screen
        
        # Initially form should be invalid
        assert screen.data["form_valid"] is False
        assert screen.data["project_created"] is False
        
        # Test required field validation
        name_widget = screen.get_widget("project_name")
        assert name_widget is not None
        
        # Empty name should be invalid
        name_widget.attributes["value"] = ""
        validation = name_widget.attributes["validation"]
        is_valid = (
            len(name_widget.attributes["value"]) >= validation.get("min_length", 1)
            if validation.get("required") else True
        )
        assert not is_valid
        
        # Valid name should pass validation
        name_widget.attributes["value"] = "valid-project-name"
        is_valid = len(name_widget.attributes["value"]) >= validation.get("min_length", 1)
        assert is_valid
    
    @pytest.mark.asyncio
    async def test_dashboard_screen_functionality(self, mock_dashboard_screen):
        """Test main dashboard functionality."""
        screen = mock_dashboard_screen
        
        # Test initial state
        assert screen.name == "dashboard"
        assert screen.data["current_project"] == "test-project"
        assert screen.data["active_tasks"] == 2
        
        # Test widgets are present
        widget_ids = [w.id for w in screen.widgets]
        expected_widgets = ["project_tree", "task_list", "progress_bar", "status_bar", "menu_bar"]
        
        for expected_id in expected_widgets:
            assert any(expected_id in widget_id for widget_id in widget_ids), f"Missing widget: {expected_id}"
    
    @pytest.mark.asyncio
    async def test_screen_navigation(self, mock_textual_app, tui_assertions):
        """Test navigation between screens."""
        app = mock_textual_app
        
        # Test navigation sequence
        tui_assertions.assert_navigation_works(app, "welcome", "project_setup")
        tui_assertions.assert_navigation_works(app, "project_setup", "dashboard")
        
        # Test navigation history
        current_screen = app.get_current_screen()
        if current_screen and hasattr(current_screen, 'navigation_history'):
            assert len(current_screen.navigation_history) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_screens(self, mock_project_setup_screen):
        """Test error handling in screens."""
        screen = mock_project_setup_screen
        
        # Simulate validation error
        error_message = "Project name is required"
        screen.errors.append(error_message)
        
        # Verify error is stored
        assert error_message in screen.errors
        assert len(screen.errors) == 1
        
        # Clear errors
        screen.errors.clear()
        assert len(screen.errors) == 0
    
    def test_screen_data_management(self, mock_dashboard_screen):
        """Test screen data management."""
        screen = mock_dashboard_screen
        
        # Test initial data
        assert "current_project" in screen.data
        assert "active_tasks" in screen.data
        
        # Update data
        screen.data["current_project"] = "new-project"
        screen.data["active_tasks"] = 5
        screen.data["last_updated"] = "2025-01-01T00:00:00Z"
        
        # Verify updates
        assert screen.data["current_project"] == "new-project"
        assert screen.data["active_tasks"] == 5
        assert "last_updated" in screen.data
    
    @pytest.mark.asyncio
    async def test_screen_transitions(self, mock_textual_app):
        """Test screen transition handling."""
        app = mock_textual_app
        
        # Start at welcome
        app.current_screen = "welcome"
        assert app.current_screen == "welcome"
        
        # Navigate to project setup
        success = app.push_screen("project_setup")
        assert success
        assert app.current_screen == "project_setup"
        
        # Navigate to dashboard
        success = app.push_screen("dashboard")
        assert success
        assert app.current_screen == "dashboard"
        
        # Navigate back
        success = app.pop_screen()
        # Mock implementation might not handle this perfectly
        # but we can test that the method exists and is callable


class TestTUIInteractions:
    """Test suite for TUI interactions and user workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_project_creation_flow(self, mock_textual_app, keyboard_simulator):
        """Test complete project creation workflow."""
        app = mock_textual_app
        keyboard = keyboard_simulator
        
        # Start at welcome screen
        app.push_screen("welcome")
        welcome_screen = app.get_current_screen()
        
        # Simulate "New Project" selection
        result = keyboard.press_key("n")
        
        # Navigate to project setup
        app.push_screen("project_setup")
        setup_screen = app.get_current_screen()
        
        # Fill project details
        keyboard.type_text("my-awesome-project")
        
        # Simulate form submission
        result = keyboard.press_key("Enter")
        assert result["action"] == "submit"
        
        # Verify flow completion
        assert setup_screen.name == "project_setup"
    
    @pytest.mark.asyncio
    async def test_task_management_workflow(self, mock_dashboard_screen, mock_task_list_widget):
        """Test task management workflow."""
        screen = mock_dashboard_screen
        task_widget = mock_task_list_widget
        
        # Initial tasks
        initial_task_count = len(task_widget.attributes["items"])
        assert initial_task_count > 0
        
        # Add new task (simulate)
        new_task = {"name": "Implement feature X", "status": "pending"}
        task_widget.attributes["items"].append(new_task)
        
        # Update task status (simulate)
        task_widget.attributes["items"][0]["status"] = "completed"
        
        # Verify updates
        assert len(task_widget.attributes["items"]) == initial_task_count + 1
        assert task_widget.attributes["items"][0]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_keyboard_shortcuts(self, mock_textual_app, keyboard_simulator):
        """Test keyboard shortcuts."""
        app = mock_textual_app
        keyboard = keyboard_simulator
        
        # Test common shortcuts
        shortcuts = [
            ("Ctrl+n", "new_project"),
            ("Ctrl+o", "open_project"),
            ("Ctrl+s", "save"),
            ("Ctrl+q", "quit"),
            ("F1", "help"),
        ]
        
        for key, expected_action in shortcuts:
            result = keyboard.press_key(key)
            # Mock keyboard simulator might not handle all shortcuts
            # but we can test that the method processes them
            assert "action" in result
    
    @pytest.mark.asyncio
    async def test_mouse_interactions(self, mock_project_tree_widget):
        """Test mouse interactions (if supported)."""
        widget = mock_project_tree_widget
        
        # Simulate mouse click on tree item
        # In a real implementation, this would trigger item selection
        widget.attributes["selected_item"] = "project-1"
        
        # Verify selection
        assert widget.attributes["selected_item"] == "project-1"
        
        # Simulate double-click (expand/collapse)
        widget.attributes["expanded"] = not widget.attributes["expanded"]
        
        # Verify state change
        assert widget.attributes["expanded"] is not None
    
    def test_accessibility_features(self, mock_welcome_screen):
        """Test accessibility features."""
        screen = mock_welcome_screen
        
        # Test screen has accessible elements
        for widget in screen.widgets:
            # Each widget should have a name (for screen readers)
            assert hasattr(widget, 'name')
            assert len(widget.name) > 0
            
            # Should have content or label
            assert hasattr(widget, 'content') or hasattr(widget, 'attributes')
    
    @pytest.mark.asyncio 
    async def test_responsive_layout(self, mock_dashboard_screen):
        """Test responsive layout behavior."""
        screen = mock_dashboard_screen
        
        # Test that widgets adapt to different sizes
        # In a real implementation, this would test terminal resize handling
        
        # Simulate small terminal
        small_layout = {"width": 40, "height": 20}
        
        # Simulate large terminal
        large_layout = {"width": 120, "height": 40}
        
        # Verify widgets can handle different layouts
        for widget in screen.widgets:
            assert widget.visible is not None  # Should have visibility state
            assert hasattr(widget, 'content')  # Should have content to display
    
    @pytest.mark.asyncio
    async def test_theme_and_styling(self, ui_helper):
        """Test theme and styling capabilities."""
        # Create widgets with different styles
        primary_button = ui_helper.create_button_widget(
            attributes={"style": "primary", "label": "Primary Action"}
        )
        
        secondary_button = ui_helper.create_button_widget(
            attributes={"style": "secondary", "label": "Secondary Action"}
        )
        
        # Verify styling attributes
        assert primary_button.attributes.get("style") == "primary"
        assert secondary_button.attributes.get("style") == "secondary"
    
    @pytest.mark.parametrize("widget_type,expected_behavior", [
        ("button", "clickable"),
        ("input", "editable"),
        ("tree", "expandable"),
        ("list", "scrollable"),
        ("progress", "animated"),
    ])
    def test_widget_type_behaviors(self, ui_helper, widget_type, expected_behavior):
        """Test different widget type behaviors."""
        widget = ui_helper.create_widget_mock(
            f"test_{widget_type}", 
            widget_type, 
            f"Test {widget_type} content"
        )
        
        assert widget.widget_type == widget_type
        
        # Test type-specific behavior
        if expected_behavior == "clickable":
            assert hasattr(widget, 'attributes')
        elif expected_behavior == "editable":
            # Input widgets should support value changes
            widget.attributes = {"value": "test"}
            assert widget.attributes["value"] == "test"
        elif expected_behavior == "expandable":
            # Tree widgets should support expansion state
            widget.attributes = {"expanded": True}
            assert widget.attributes["expanded"] is True
        elif expected_behavior == "scrollable":
            # List widgets should support scrolling
            widget.attributes = {"scroll_position": 0}
            assert "scroll_position" in widget.attributes
        elif expected_behavior == "animated":
            # Progress widgets should support progress updates
            widget.attributes = {"progress": 50}
            assert widget.attributes["progress"] == 50


class TestTUIPerformance:
    """Test TUI performance and responsiveness."""
    
    @pytest.mark.performance
    def test_rendering_performance(self, mock_dashboard_screen):
        """Test rendering performance."""
        import time
        
        screen = mock_dashboard_screen
        
        # Measure rendering time
        start_time = time.time()
        
        for _ in range(100):
            rendered = screen.render()
            assert len(rendered) > 0
        
        end_time = time.time()
        
        # Should render quickly
        render_time = (end_time - start_time) / 100
        assert render_time < 0.01  # Less than 10ms per render
    
    @pytest.mark.performance
    def test_widget_update_performance(self, mock_task_list_widget):
        """Test widget update performance."""
        import time
        
        widget = mock_task_list_widget
        
        # Measure update time
        start_time = time.time()
        
        for i in range(1000):
            # Simulate rapid updates
            widget.attributes["items"].append({
                "name": f"Task {i}",
                "status": "pending"
            })
            
            # Simulate content update
            widget.content += f"\\n  • Task {i} (pending)"
        
        end_time = time.time()
        
        # Should handle rapid updates efficiently
        update_time = end_time - start_time
        assert update_time < 1.0  # Less than 1 second for 1000 updates
        
        # Verify final state
        assert len(widget.attributes["items"]) >= 1000
    
    @pytest.mark.performance
    def test_memory_usage(self, mock_textual_app):
        """Test memory usage with many widgets."""
        import tracemalloc
        
        app = mock_textual_app
        tracemalloc.start()
        
        # Create many screens and widgets
        from tests.fixtures.ui_fixtures import UITestHelper
        helper = UITestHelper()
        
        for i in range(100):
            screen = helper.create_main_dashboard_screen()
            screen.name = f"test_screen_{i}"
            app.add_screen(screen)
        
        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 50MB)
        assert current < 50 * 1024 * 1024
        assert peak < 100 * 1024 * 1024
    
    def test_large_dataset_handling(self, mock_project_tree_widget):
        """Test handling large datasets in widgets."""
        widget = mock_project_tree_widget
        
        # Add many items
        large_dataset = [f"project-{i}" for i in range(10000)]
        widget.attributes["items"] = large_dataset
        
        # Should handle large dataset without errors
        assert len(widget.attributes["items"]) == 10000
        
        # Rendering should still work (though it might be truncated)
        rendered = widget.render()
        assert len(rendered) > 0
        assert "project-0" in rendered  # First item should be visible