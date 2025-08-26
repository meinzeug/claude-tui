"""
UI-related test fixtures for TUI testing.
"""

import pytest
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from unittest.mock import Mock, AsyncMock


class WidgetState(Enum):
    """Widget state values."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    FOCUSED = "focused"
    DISABLED = "disabled"
    ERROR = "error"


class ScreenType(Enum):
    """Available screen types."""
    WELCOME = "welcome"
    PROJECT_SETUP = "project_setup"
    MAIN_DASHBOARD = "main_dashboard"
    PROJECT_BROWSER = "project_browser"
    TASK_MANAGER = "task_manager"
    CODE_EDITOR = "code_editor"
    SETTINGS = "settings"


@dataclass
class MockWidget:
    """Mock widget for testing TUI components."""
    id: str
    name: str
    widget_type: str
    state: str
    visible: bool
    content: str
    attributes: Dict[str, Any]
    
    def render(self) -> str:
        """Mock render method."""
        return f"[{self.widget_type}:{self.name}] {self.content}"
    
    def set_state(self, state: WidgetState):
        """Set widget state."""
        self.state = state.value
    
    def is_focused(self) -> bool:
        """Check if widget is focused."""
        return self.state == WidgetState.FOCUSED.value


@dataclass 
class MockScreen:
    """Mock screen for testing TUI screens."""
    name: str
    screen_type: str
    widgets: List[MockWidget]
    active_widget: Optional[str]
    errors: List[str]
    navigation_history: List[str]
    data: Dict[str, Any]
    
    def add_widget(self, widget: MockWidget):
        """Add widget to screen."""
        self.widgets.append(widget)
    
    def get_widget(self, widget_id: str) -> Optional[MockWidget]:
        """Get widget by ID."""
        for widget in self.widgets:
            if widget.id == widget_id:
                return widget
        return None
    
    def set_focus(self, widget_id: str):
        """Set focus to widget."""
        for widget in self.widgets:
            if widget.id == widget_id:
                widget.set_state(WidgetState.FOCUSED)
                self.active_widget = widget_id
            else:
                widget.set_state(WidgetState.INACTIVE)
    
    def render(self) -> str:
        """Mock screen rendering."""
        widget_renders = [w.render() for w in self.widgets if w.visible]
        return f"Screen[{self.name}]:\\n" + "\\n".join(widget_renders)


class UITestHelper:
    """Helper for creating UI test fixtures."""
    
    @staticmethod
    def create_project_tree_widget(**overrides) -> MockWidget:
        """Create project tree widget."""
        defaults = {
            "id": "project_tree",
            "name": "Project Tree",
            "widget_type": "tree",
            "state": WidgetState.INACTIVE.value,
            "visible": True,
            "content": "📁 Projects\\n  ├── project-1\\n  └── project-2",
            "attributes": {
                "expanded": True,
                "selected_item": None,
                "items": ["project-1", "project-2"]
            }
        }
        defaults.update(overrides)
        return MockWidget(**defaults)
    
    @staticmethod
    def create_task_list_widget(**overrides) -> MockWidget:
        """Create task list widget."""
        defaults = {
            "id": "task_list", 
            "name": "Task List",
            "widget_type": "list",
            "state": WidgetState.INACTIVE.value,
            "visible": True,
            "content": "📋 Tasks\\n  • Task 1 (pending)\\n  • Task 2 (in progress)",
            "attributes": {
                "selected_index": 0,
                "items": [
                    {"name": "Task 1", "status": "pending"},
                    {"name": "Task 2", "status": "in_progress"}
                ]
            }
        }
        defaults.update(overrides)
        return MockWidget(**defaults)
    
    @staticmethod
    def create_progress_bar_widget(**overrides) -> MockWidget:
        """Create progress bar widget."""
        defaults = {
            "id": "progress_bar",
            "name": "Progress Bar", 
            "widget_type": "progress",
            "state": WidgetState.INACTIVE.value,
            "visible": True,
            "content": "Progress: [████████  ] 80%",
            "attributes": {
                "progress": 80,
                "max_value": 100,
                "show_percentage": True
            }
        }
        defaults.update(overrides)
        return MockWidget(**defaults)
    
    @staticmethod
    def create_input_field_widget(**overrides) -> MockWidget:
        """Create input field widget."""
        defaults = {
            "id": "input_field",
            "name": "Input Field",
            "widget_type": "input",
            "state": WidgetState.INACTIVE.value,
            "visible": True,
            "content": "Enter project name: [test-project]",
            "attributes": {
                "value": "test-project",
                "placeholder": "Enter project name",
                "validation": {"required": True, "min_length": 1}
            }
        }
        defaults.update(overrides)
        return MockWidget(**defaults)
    
    @staticmethod
    def create_button_widget(**overrides) -> MockWidget:
        """Create button widget."""
        defaults = {
            "id": "submit_button",
            "name": "Submit Button",
            "widget_type": "button", 
            "state": WidgetState.INACTIVE.value,
            "visible": True,
            "content": "[Create Project]",
            "attributes": {
                "label": "Create Project",
                "action": "submit_form",
                "disabled": False
            }
        }
        defaults.update(overrides)
        return MockWidget(**defaults)
    
    @staticmethod
    def create_welcome_screen() -> MockScreen:
        """Create welcome screen."""
        widgets = [
            UITestHelper.create_widget_mock("welcome_title", "title", "Welcome to claude-tui"),
            UITestHelper.create_widget_mock("new_project_btn", "button", "New Project"),
            UITestHelper.create_widget_mock("open_project_btn", "button", "Open Project"),
            UITestHelper.create_widget_mock("recent_projects", "list", "Recent Projects")
        ]
        
        return MockScreen(
            name="welcome",
            screen_type=ScreenType.WELCOME.value,
            widgets=widgets,
            active_widget=None,
            errors=[],
            navigation_history=[],
            data={"show_recent": True, "max_recent": 5}
        )
    
    @staticmethod
    def create_project_setup_screen() -> MockScreen:
        """Create project setup screen."""
        widgets = [
            UITestHelper.create_input_field_widget(
                id="project_name",
                attributes={
                    "value": "",
                    "placeholder": "Enter project name",
                    "validation": {"required": True, "min_length": 1}
                }
            ),
            UITestHelper.create_widget_mock("template_select", "select", "Select Template"),
            UITestHelper.create_input_field_widget(
                id="project_description", 
                content="Description: []",
                attributes={"value": "", "placeholder": "Project description"}
            ),
            UITestHelper.create_button_widget(
                id="create_btn",
                attributes={"label": "Create", "action": "create_project"}
            ),
            UITestHelper.create_button_widget(
                id="cancel_btn", 
                attributes={"label": "Cancel", "action": "cancel"}
            )
        ]
        
        return MockScreen(
            name="project_setup",
            screen_type=ScreenType.PROJECT_SETUP.value,
            widgets=widgets,
            active_widget="project_name",
            errors=[],
            navigation_history=["welcome"],
            data={"form_valid": False, "project_created": False}
        )
    
    @staticmethod
    def create_main_dashboard_screen() -> MockScreen:
        """Create main dashboard screen."""
        widgets = [
            UITestHelper.create_project_tree_widget(),
            UITestHelper.create_task_list_widget(),
            UITestHelper.create_progress_bar_widget(),
            UITestHelper.create_widget_mock("status_bar", "status", "Ready"),
            UITestHelper.create_widget_mock("menu_bar", "menu", "File | Edit | View | Tools")
        ]
        
        return MockScreen(
            name="dashboard",
            screen_type=ScreenType.MAIN_DASHBOARD.value,
            widgets=widgets,
            active_widget="project_tree", 
            errors=[],
            navigation_history=["welcome", "project_setup"],
            data={"current_project": "test-project", "active_tasks": 2}
        )
    
    @staticmethod
    def create_widget_mock(widget_id: str, widget_type: str, content: str, **attributes) -> MockWidget:
        """Create a generic widget mock."""
        return MockWidget(
            id=widget_id,
            name=widget_id.replace("_", " ").title(),
            widget_type=widget_type,
            state=WidgetState.INACTIVE.value,
            visible=True,
            content=content,
            attributes=attributes
        )


class KeyboardSimulator:
    """Simulate keyboard interactions for testing."""
    
    def __init__(self):
        self.key_sequence = []
        self.current_input = ""
    
    def press_key(self, key: str):
        """Simulate key press."""
        self.key_sequence.append(key)
        
        if key == "Enter":
            return {"action": "submit", "input": self.current_input}
        elif key == "Escape":
            return {"action": "cancel", "input": self.current_input}
        elif key == "Tab":
            return {"action": "next_field"}
        elif key.startswith("Shift+"):
            return {"action": "shift_action", "key": key[6:]}
        elif len(key) == 1:  # Regular character
            self.current_input += key
            return {"action": "input", "input": self.current_input}
        else:
            return {"action": "special_key", "key": key}
    
    def type_text(self, text: str):
        """Simulate typing text."""
        for char in text:
            self.press_key(char)
        self.current_input = text
    
    def clear_input(self):
        """Clear current input."""
        self.current_input = ""
        self.key_sequence.append("Ctrl+A")
        self.key_sequence.append("Delete")


class MockTextualApp:
    """Mock Textual app for testing."""
    
    def __init__(self):
        self.screens = {}
        self.current_screen = None
        self.keyboard = KeyboardSimulator()
        self.running = False
        self.exit_code = 0
    
    def add_screen(self, screen: MockScreen):
        """Add screen to app."""
        self.screens[screen.name] = screen
    
    def push_screen(self, screen_name: str):
        """Navigate to screen."""
        if screen_name in self.screens:
            self.current_screen = screen_name
            return True
        return False
    
    def pop_screen(self):
        """Go back to previous screen."""
        if self.current_screen:
            screen = self.screens[self.current_screen]
            if screen.navigation_history:
                previous = screen.navigation_history[-1]
                if previous in self.screens:
                    self.current_screen = previous
                    return True
        return False
    
    def get_current_screen(self) -> Optional[MockScreen]:
        """Get current screen."""
        if self.current_screen:
            return self.screens.get(self.current_screen)
        return None
    
    def simulate_key_press(self, key: str):
        """Simulate key press on current screen."""
        return self.keyboard.press_key(key)
    
    def simulate_text_input(self, text: str):
        """Simulate text input."""
        self.keyboard.type_text(text)


@pytest.fixture
def ui_helper():
    """Provide UI test helper."""
    return UITestHelper


@pytest.fixture
def mock_project_tree_widget():
    """Create mock project tree widget."""
    return UITestHelper.create_project_tree_widget()


@pytest.fixture
def mock_task_list_widget():
    """Create mock task list widget."""
    return UITestHelper.create_task_list_widget()


@pytest.fixture
def mock_progress_bar_widget():
    """Create mock progress bar widget."""
    return UITestHelper.create_progress_bar_widget()


@pytest.fixture
def mock_welcome_screen():
    """Create mock welcome screen."""
    return UITestHelper.create_welcome_screen()


@pytest.fixture
def mock_project_setup_screen():
    """Create mock project setup screen.""" 
    return UITestHelper.create_project_setup_screen()


@pytest.fixture
def mock_dashboard_screen():
    """Create mock dashboard screen."""
    return UITestHelper.create_main_dashboard_screen()


@pytest.fixture
def keyboard_simulator():
    """Provide keyboard simulator."""
    return KeyboardSimulator()


@pytest.fixture
def mock_textual_app():
    """Create mock Textual app with screens."""
    app = MockTextualApp()
    
    # Add standard screens
    app.add_screen(UITestHelper.create_welcome_screen())
    app.add_screen(UITestHelper.create_project_setup_screen())
    app.add_screen(UITestHelper.create_main_dashboard_screen())
    
    app.current_screen = "welcome"
    return app


@pytest.fixture
def screen_navigation_sequence():
    """Provide sequence of screen navigation actions."""
    return [
        {"action": "navigate", "to": "welcome"},
        {"action": "press_key", "key": "n"},  # New project
        {"action": "navigate", "to": "project_setup"},
        {"action": "type_text", "field": "project_name", "text": "my-test-project"},
        {"action": "press_key", "key": "Tab"},  # Move to next field
        {"action": "select_option", "field": "template", "value": "python"},
        {"action": "press_key", "key": "Enter"},  # Submit form
        {"action": "navigate", "to": "dashboard"}
    ]


class TUITestAssertions:
    """Common assertions for TUI testing."""
    
    @staticmethod
    def assert_widget_visible(widget: MockWidget):
        """Assert widget is visible."""
        assert widget.visible, f"Widget {widget.name} should be visible"
    
    @staticmethod
    def assert_widget_focused(widget: MockWidget):
        """Assert widget is focused."""
        assert widget.is_focused(), f"Widget {widget.name} should be focused"
    
    @staticmethod
    def assert_screen_has_widget(screen: MockScreen, widget_id: str):
        """Assert screen contains widget."""
        widget = screen.get_widget(widget_id)
        assert widget is not None, f"Screen {screen.name} should contain widget {widget_id}"
    
    @staticmethod
    def assert_navigation_works(app: MockTextualApp, from_screen: str, to_screen: str):
        """Assert navigation between screens works."""
        app.current_screen = from_screen
        success = app.push_screen(to_screen)
        assert success, f"Navigation from {from_screen} to {to_screen} should succeed"
        assert app.current_screen == to_screen, f"Current screen should be {to_screen}"
    
    @staticmethod
    def assert_form_validation(screen: MockScreen, expected_errors: List[str]):
        """Assert form validation errors."""
        for error in expected_errors:
            assert error in screen.errors, f"Expected error '{error}' not found in screen errors"
    
    @staticmethod
    def assert_keyboard_handling(app: MockTextualApp, key: str, expected_action: str):
        """Assert keyboard input handling."""
        result = app.simulate_key_press(key)
        assert result["action"] == expected_action, f"Key {key} should trigger action {expected_action}"


@pytest.fixture
def tui_assertions():
    """Provide TUI test assertions."""
    return TUITestAssertions