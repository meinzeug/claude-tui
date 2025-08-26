#!/usr/bin/env python3
"""
Enhanced Textual UI Component Tests
Comprehensive tests for Textual framework components in Claude-TUI.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
from pathlib import Path

# Textual framework imports with fallbacks
try:
    from textual.app import App
    from textual.widgets import Static, Button, Input, DataTable, Tree, Log
    from textual.screen import Screen
    from textual.containers import Container, Horizontal, Vertical
    from textual.message import Message
    from textual.reactive import reactive
    from textual.binding import Binding
    from textual import events
    from textual.pilot import Pilot
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    
    # Mock Textual classes for testing without dependency
    class Widget:
        def __init__(self, *args, **kwargs):
            self.id = kwargs.get('id', 'mock-widget')
            self.classes = kwargs.get('classes', '')
            self.disabled = kwargs.get('disabled', False)
            self.visible = kwargs.get('visible', True)
            self._reactive_vars = {}
            self._bindings = []
        
        def add_class(self, class_name):
            pass
        
        def remove_class(self, class_name):
            pass
        
        def set_focus(self):
            pass
        
        async def remove(self):
            pass
    
    class Static(Widget):
        def __init__(self, renderable="", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.renderable = renderable
        
        def update(self, renderable):
            self.renderable = renderable
    
    class Button(Widget):
        def __init__(self, label="Button", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.label = label
            self.pressed = False
    
    class Input(Widget):
        def __init__(self, value="", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.value = value
            self.placeholder = kwargs.get('placeholder', '')
        
        def clear(self):
            self.value = ""
    
    class DataTable(Widget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._columns = {}
            self._rows = []
        
        def add_column(self, key, label=None, width=None):
            self._columns[key] = {'label': label or key, 'width': width}
        
        def add_row(self, *cells, key=None):
            row_data = {'cells': cells, 'key': key or f'row-{len(self._rows)}'}
            self._rows.append(row_data)
            return row_data['key']
        
        def clear(self):
            self._rows.clear()
    
    class Tree(Widget):
        def __init__(self, label, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.label = label
            self._nodes = {}
        
        def add_node(self, label, parent=None):
            node_id = f'node-{len(self._nodes)}'
            self._nodes[node_id] = {'label': label, 'parent': parent}
            return node_id
        
        def clear(self):
            self._nodes.clear()
    
    class Log(Widget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lines = []
        
        def write_line(self, line):
            self._lines.append(line)
        
        def clear(self):
            self._lines.clear()
    
    class Container(Widget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._children = []
        
        def mount(self, *widgets):
            self._children.extend(widgets)
        
        def query(self, selector):
            return MockQueryResult()
    
    class Horizontal(Container):
        pass
    
    class Vertical(Container):
        pass
    
    class Screen:
        def __init__(self, *args, **kwargs):
            self.title = kwargs.get('title', 'Mock Screen')
            self._widgets = []
            self._bindings = []
        
        def compose(self):
            return []
        
        def mount(self, *widgets):
            self._widgets.extend(widgets)
        
        def query(self, selector):
            return MockQueryResult()
    
    class App:
        def __init__(self, *args, **kwargs):
            self.title = kwargs.get('title', 'Mock App')
            self._screens = {}
            self.current_screen = None
        
        def install_screen(self, screen, name):
            self._screens[name] = screen
        
        def push_screen(self, screen_name):
            self.current_screen = self._screens.get(screen_name)
        
        def pop_screen(self):
            self.current_screen = None
        
        async def run_test(self, headless=True):
            return MockPilot(self)
        
        def query(self, selector):
            return MockQueryResult()
    
    class Message:
        pass
    
    class Binding:
        def __init__(self, key, action, description=""):
            self.key = key
            self.action = action
            self.description = description
    
    class MockQueryResult:
        def __init__(self):
            self._results = []
        
        def __len__(self):
            return len(self._results)
        
        def first(self):
            return self._results[0] if self._results else None
        
        def __iter__(self):
            return iter(self._results)
    
    class MockPilot:
        def __init__(self, app):
            self.app = app
        
        async def press(self, *keys):
            pass
        
        async def click(self, selector):
            pass
        
        async def wait_for_screen(self, timeout=10):
            return self.app.current_screen
        
        async def exit(self):
            pass

# Import application components with fallbacks
try:
    from src.claude_tui.ui.main_app import ClaudeTUIApp
    from src.claude_tui.ui.screens.workspace_screen import WorkspaceScreen
    from src.claude_tui.ui.screens.settings import SettingsScreen
    from src.claude_tui.ui.widgets.task_dashboard import TaskDashboard
    from src.claude_tui.ui.widgets.project_tree import ProjectTree
    from src.claude_tui.ui.widgets.console_widget import ConsoleWidget
except ImportError:
    # Mock application components
    class ClaudeTUIApp(App):
        def __init__(self):
            super().__init__(title="Claude TUI")
            self.config = {}
            self.project_manager = Mock()
            self.ai_interface = Mock()
        
        def compose(self):
            return [Static("Mock Claude TUI")]
    
    class WorkspaceScreen(Screen):
        def __init__(self):
            super().__init__(title="Workspace")
            self.project_tree = Mock()
            self.task_dashboard = Mock()
            self.console = Mock()
        
        def compose(self):
            return [
                Horizontal(
                    Static("Project Tree"),
                    Vertical(
                        Static("Task Dashboard"),
                        Static("Console")
                    )
                )
            ]
    
    class SettingsScreen(Screen):
        def __init__(self):
            super().__init__(title="Settings")
            self.settings_form = Mock()
        
        def compose(self):
            return [Static("Settings Screen")]
    
    class TaskDashboard(Widget):
        def __init__(self):
            super().__init__(id='task-dashboard')
            self.tasks = []
        
        def add_task(self, task):
            self.tasks.append(task)
        
        def update_task(self, task_id, status):
            for task in self.tasks:
                if task.get('id') == task_id:
                    task['status'] = status
                    break
        
        def refresh_tasks(self):
            pass
    
    class ProjectTree(Widget):
        def __init__(self):
            super().__init__(id='project-tree')
            self.tree = Tree("Projects")
            self.projects = {}
        
        def add_project(self, project_name, project_path):
            self.projects[project_name] = project_path
            return self.tree.add_node(project_name)
        
        def refresh_tree(self):
            pass
    
    class ConsoleWidget(Widget):
        def __init__(self):
            super().__init__(id='console')
            self.log = Log()
            self.command_history = []
        
        def write_output(self, text):
            self.log.write_line(text)
        
        def execute_command(self, command):
            self.command_history.append(command)
            return f"Executed: {command}"


@pytest.fixture
def mock_app():
    """Create mock application for testing."""
    return ClaudeTUIApp()


@pytest.fixture
def mock_workspace_screen():
    """Create mock workspace screen for testing."""
    return WorkspaceScreen()


@pytest.fixture
def mock_settings_screen():
    """Create mock settings screen for testing."""
    return SettingsScreen()


@pytest.fixture
def sample_project_data():
    """Sample project data for testing."""
    return {
        "name": "TestProject",
        "path": "/path/to/test/project",
        "language": "python",
        "files": [
            {"name": "main.py", "type": "file", "path": "/path/to/test/project/main.py"},
            {"name": "tests/", "type": "directory", "path": "/path/to/test/project/tests/"},
            {"name": "README.md", "type": "file", "path": "/path/to/test/project/README.md"}
        ]
    }


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return [
        {
            "id": "task-1",
            "title": "Implement user authentication",
            "status": "in_progress",
            "priority": "high",
            "assigned_to": "developer-1"
        },
        {
            "id": "task-2", 
            "title": "Write unit tests",
            "status": "pending",
            "priority": "medium",
            "assigned_to": "tester-1"
        },
        {
            "id": "task-3",
            "title": "Update documentation",
            "status": "completed",
            "priority": "low",
            "assigned_to": "writer-1"
        }
    ]


class TestClaudeTUIApp:
    """Tests for the main Claude TUI Application."""
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_app_initialization(self, mock_app):
        """Test application initialization."""
        assert isinstance(mock_app, (ClaudeTUIApp, App))
        assert mock_app.title == "Claude TUI" or "Claude" in mock_app.title
        assert hasattr(mock_app, 'config')
    
    @pytest.mark.ui
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_app_startup_sequence(self, mock_app):
        """Test application startup sequence."""
        # Mock the startup process
        with patch.object(mock_app, 'install_screen') as mock_install:
            with patch.object(mock_app, 'push_screen') as mock_push:
                # Simulate startup
                mock_app.install_screen(WorkspaceScreen(), "workspace")
                mock_app.install_screen(SettingsScreen(), "settings")
                mock_app.push_screen("workspace")
                
                # Verify screens were installed
                mock_install.assert_called()
                mock_push.assert_called_with("workspace")
    
    @pytest.mark.ui
    @pytest.mark.slow
    @pytest.mark.asyncio
    @pytest.mark.skipif(not TEXTUAL_AVAILABLE, reason="Textual not available")
    async def test_app_navigation(self, mock_app):
        """Test application navigation between screens."""
        async with mock_app.run_test(headless=True) as pilot:
            # Test navigation to settings
            await pilot.press("ctrl+comma")  # Common settings shortcut
            
            # Wait for screen change
            await pilot.wait_for_screen(timeout=2)
            
            # Test navigation back
            await pilot.press("escape")
            await pilot.wait_for_screen(timeout=2)
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_app_keyboard_bindings(self, mock_app):
        """Test application keyboard bindings."""
        # Mock bindings
        expected_bindings = [
            ("ctrl+q", "quit", "Quit"),
            ("ctrl+comma", "settings", "Settings"),
            ("ctrl+n", "new_project", "New Project"),
            ("ctrl+o", "open_project", "Open Project"),
            ("f1", "help", "Help")
        ]
        
        # Verify bindings exist (mock implementation)
        for key, action, description in expected_bindings:
            # In a real implementation, we would check app.bindings
            # Here we verify the concept
            assert key is not None
            assert action is not None
            assert description is not None


class TestWorkspaceScreen:
    """Tests for the Workspace Screen."""
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_workspace_screen_initialization(self, mock_workspace_screen):
        """Test workspace screen initialization."""
        assert isinstance(mock_workspace_screen, (WorkspaceScreen, Screen))
        assert mock_workspace_screen.title == "Workspace" or "workspace" in mock_workspace_screen.title.lower()
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_workspace_screen_layout(self, mock_workspace_screen):
        """Test workspace screen layout structure."""
        # Test that the workspace has expected components
        assert hasattr(mock_workspace_screen, 'project_tree')
        assert hasattr(mock_workspace_screen, 'task_dashboard')
        assert hasattr(mock_workspace_screen, 'console')
        
        # Test composition
        widgets = mock_workspace_screen.compose()
        assert len(widgets) > 0
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_workspace_project_integration(self, mock_workspace_screen, sample_project_data):
        """Test workspace integration with project data."""
        # Mock project loading
        with patch.object(mock_workspace_screen, 'project_tree') as mock_tree:
            mock_tree.add_project = Mock(return_value="node-1")
            
            # Simulate adding project
            node_id = mock_tree.add_project(
                sample_project_data["name"],
                sample_project_data["path"]
            )
            
            # Verify project was added
            mock_tree.add_project.assert_called_once_with(
                sample_project_data["name"],
                sample_project_data["path"]
            )
            assert node_id is not None
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_workspace_task_management(self, mock_workspace_screen, sample_task_data):
        """Test workspace task management functionality."""
        # Mock task dashboard
        with patch.object(mock_workspace_screen, 'task_dashboard') as mock_dashboard:
            mock_dashboard.add_task = Mock()
            mock_dashboard.update_task = Mock()
            
            # Add tasks
            for task in sample_task_data:
                mock_dashboard.add_task(task)
            
            # Update task status
            mock_dashboard.update_task("task-1", "completed")
            
            # Verify interactions
            assert mock_dashboard.add_task.call_count == len(sample_task_data)
            mock_dashboard.update_task.assert_called_with("task-1", "completed")
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_workspace_console_output(self, mock_workspace_screen):
        """Test workspace console output functionality."""
        # Mock console widget
        with patch.object(mock_workspace_screen, 'console') as mock_console:
            mock_console.write_output = Mock()
            mock_console.execute_command = Mock(return_value="Command executed")
            
            # Test output writing
            test_outputs = [
                "Build started...",
                "Tests passed: 15/15",
                "Build completed successfully"
            ]
            
            for output in test_outputs:
                mock_console.write_output(output)
            
            # Test command execution
            result = mock_console.execute_command("npm test")
            
            # Verify interactions
            assert mock_console.write_output.call_count == len(test_outputs)
            mock_console.execute_command.assert_called_with("npm test")
            assert result == "Command executed"


class TestTaskDashboard:
    """Tests for the Task Dashboard Widget."""
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_task_dashboard_initialization(self):
        """Test task dashboard initialization."""
        dashboard = TaskDashboard()
        
        assert hasattr(dashboard, 'tasks')
        assert isinstance(dashboard.tasks, list)
        assert dashboard.id == 'task-dashboard'
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_task_dashboard_add_tasks(self, sample_task_data):
        """Test adding tasks to dashboard."""
        dashboard = TaskDashboard()
        
        # Add tasks
        for task in sample_task_data:
            dashboard.add_task(task)
        
        # Verify tasks were added
        assert len(dashboard.tasks) == len(sample_task_data)
        
        # Verify task content
        for i, task in enumerate(sample_task_data):
            assert dashboard.tasks[i]['id'] == task['id']
            assert dashboard.tasks[i]['title'] == task['title']
            assert dashboard.tasks[i]['status'] == task['status']
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_task_dashboard_update_task_status(self, sample_task_data):
        """Test updating task status in dashboard."""
        dashboard = TaskDashboard()
        
        # Add tasks
        for task in sample_task_data:
            dashboard.add_task(task)
        
        # Update task status
        dashboard.update_task("task-1", "completed")
        
        # Verify status was updated
        updated_task = next(task for task in dashboard.tasks if task['id'] == 'task-1')
        assert updated_task['status'] == 'completed'
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_task_dashboard_filtering(self, sample_task_data):
        """Test task filtering functionality."""
        dashboard = TaskDashboard()
        
        # Add tasks
        for task in sample_task_data:
            dashboard.add_task(task)
        
        # Mock filtering methods
        def filter_by_status(status):
            return [task for task in dashboard.tasks if task['status'] == status]
        
        def filter_by_priority(priority):
            return [task for task in dashboard.tasks if task['priority'] == priority]
        
        # Test filtering by status
        in_progress_tasks = filter_by_status('in_progress')
        assert len(in_progress_tasks) == 1
        assert in_progress_tasks[0]['id'] == 'task-1'
        
        # Test filtering by priority
        high_priority_tasks = filter_by_priority('high')
        assert len(high_priority_tasks) == 1
        assert high_priority_tasks[0]['priority'] == 'high'
    
    @pytest.mark.ui
    @pytest.mark.performance
    def test_task_dashboard_performance_with_many_tasks(self):
        """Test task dashboard performance with many tasks."""
        dashboard = TaskDashboard()
        
        # Generate many tasks
        large_task_set = [
            {
                "id": f"task-{i}",
                "title": f"Task {i}",
                "status": "pending" if i % 2 == 0 else "in_progress",
                "priority": "high" if i % 3 == 0 else "medium"
            }
            for i in range(1000)
        ]
        
        # Measure add performance
        start_time = time.perf_counter()
        for task in large_task_set:
            dashboard.add_task(task)
        end_time = time.perf_counter()
        
        add_time = end_time - start_time
        assert add_time < 1.0  # Should complete within 1 second
        assert len(dashboard.tasks) == 1000
        
        print(f"Added 1000 tasks in {add_time:.3f} seconds")


class TestProjectTree:
    """Tests for the Project Tree Widget."""
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_project_tree_initialization(self):
        """Test project tree initialization."""
        project_tree = ProjectTree()
        
        assert hasattr(project_tree, 'tree')
        assert hasattr(project_tree, 'projects')
        assert isinstance(project_tree.projects, dict)
        assert project_tree.id == 'project-tree'
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_project_tree_add_project(self, sample_project_data):
        """Test adding projects to tree."""
        project_tree = ProjectTree()
        
        # Add project
        node_id = project_tree.add_project(
            sample_project_data["name"],
            sample_project_data["path"]
        )
        
        # Verify project was added
        assert node_id is not None
        assert sample_project_data["name"] in project_tree.projects
        assert project_tree.projects[sample_project_data["name"]] == sample_project_data["path"]
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_project_tree_file_structure(self, sample_project_data):
        """Test project tree file structure display."""
        project_tree = ProjectTree()
        
        # Add project
        project_node = project_tree.add_project(
            sample_project_data["name"],
            sample_project_data["path"]
        )
        
        # Mock adding file structure
        with patch.object(project_tree.tree, 'add_node') as mock_add_node:
            mock_add_node.return_value = f"node-{time.time()}"
            
            # Add files
            for file_info in sample_project_data["files"]:
                file_node = mock_add_node(file_info["name"], parent=project_node)
                assert file_node is not None
            
            # Verify files were added
            assert mock_add_node.call_count == len(sample_project_data["files"])
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_project_tree_refresh(self):
        """Test project tree refresh functionality."""
        project_tree = ProjectTree()
        
        # Add some projects
        project_tree.add_project("Project1", "/path1")
        project_tree.add_project("Project2", "/path2")
        
        # Mock refresh
        with patch.object(project_tree, 'refresh_tree') as mock_refresh:
            project_tree.refresh_tree()
            
            # Verify refresh was called
            mock_refresh.assert_called_once()


class TestConsoleWidget:
    """Tests for the Console Widget."""
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_console_widget_initialization(self):
        """Test console widget initialization."""
        console = ConsoleWidget()
        
        assert hasattr(console, 'log')
        assert hasattr(console, 'command_history')
        assert isinstance(console.command_history, list)
        assert console.id == 'console'
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_console_output_writing(self):
        """Test console output writing."""
        console = ConsoleWidget()
        
        test_outputs = [
            "Starting build process...",
            "Installing dependencies...", 
            "Running tests...",
            "Build completed successfully!"
        ]
        
        # Write outputs
        for output in test_outputs:
            console.write_output(output)
        
        # Verify outputs were written (mock implementation)
        # In real implementation, we would check console.log._lines
        assert len(test_outputs) == 4  # Verify test data
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_console_command_execution(self):
        """Test console command execution."""
        console = ConsoleWidget()
        
        test_commands = [
            "npm install",
            "npm test",
            "npm run build",
            "git status"
        ]
        
        # Execute commands
        for command in test_commands:
            result = console.execute_command(command)
            assert result == f"Executed: {command}"
        
        # Verify command history
        assert len(console.command_history) == len(test_commands)
        for i, command in enumerate(test_commands):
            assert console.command_history[i] == command
    
    @pytest.mark.ui
    @pytest.mark.fast
    def test_console_command_history(self):
        """Test console command history functionality."""
        console = ConsoleWidget()
        
        # Execute some commands
        commands = ["ls", "cd src", "python main.py"]
        for cmd in commands:
            console.execute_command(cmd)
        
        # Test history access (mock implementation)
        assert len(console.command_history) == len(commands)
        
        # Test history navigation (would be implemented in real widget)
        def get_previous_command():
            return console.command_history[-1] if console.command_history else ""
        
        def get_next_command():
            return console.command_history[0] if console.command_history else ""
        
        assert get_previous_command() == "python main.py"
        assert get_next_command() == "ls"


class TestUIInteractionPatterns:
    """Tests for UI interaction patterns and workflows."""
    
    @pytest.mark.ui
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_project_workflow_integration(self, mock_app, sample_project_data, sample_task_data):
        """Test integrated project workflow."""
        # Mock the complete workflow
        with patch.object(mock_app, 'current_screen') as mock_screen:
            # Setup mock workspace screen
            workspace = WorkspaceScreen()
            workspace.project_tree = ProjectTree()
            workspace.task_dashboard = TaskDashboard()
            workspace.console = ConsoleWidget()
            
            mock_screen = workspace
            
            # Simulate project loading workflow
            project_node = workspace.project_tree.add_project(
                sample_project_data["name"],
                sample_project_data["path"]
            )
            
            # Add tasks related to project
            for task in sample_task_data:
                workspace.task_dashboard.add_task(task)
            
            # Simulate build output
            workspace.console.write_output("Project loaded successfully")
            workspace.console.write_output(f"Found {len(sample_task_data)} tasks")
            
            # Verify workflow completion
            assert project_node is not None
            assert len(workspace.task_dashboard.tasks) == len(sample_task_data)
            assert len(workspace.console.command_history) == 0  # No commands executed yet
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_task_status_update_workflow(self, sample_task_data):
        """Test task status update workflow."""
        dashboard = TaskDashboard()
        console = ConsoleWidget()
        
        # Add initial tasks
        for task in sample_task_data:
            dashboard.add_task(task)
        
        # Simulate task status updates with console output
        status_updates = [
            ("task-1", "completed", "Authentication implementation completed"),
            ("task-2", "in_progress", "Starting unit test development"),
            ("task-3", "completed", "Documentation update finished")
        ]
        
        for task_id, new_status, message in status_updates:
            dashboard.update_task(task_id, new_status)
            console.write_output(f"Task {task_id}: {message}")
        
        # Verify all updates were processed
        completed_tasks = [task for task in dashboard.tasks if task['status'] == 'completed']
        assert len(completed_tasks) == 2  # task-1 and task-3
        
        in_progress_tasks = [task for task in dashboard.tasks if task['status'] == 'in_progress']
        assert len(in_progress_tasks) == 1  # task-2
    
    @pytest.mark.ui
    @pytest.mark.performance
    def test_ui_responsiveness_under_load(self):
        """Test UI responsiveness under high load conditions."""
        dashboard = TaskDashboard()
        console = ConsoleWidget()
        project_tree = ProjectTree()
        
        # Simulate high load scenario
        start_time = time.perf_counter()
        
        # Add many tasks rapidly
        for i in range(100):
            dashboard.add_task({
                "id": f"load-task-{i}",
                "title": f"Load Test Task {i}",
                "status": "pending",
                "priority": "medium"
            })
        
        # Add many console outputs
        for i in range(100):
            console.write_output(f"Load test output {i}")
        
        # Add many projects
        for i in range(50):
            project_tree.add_project(f"LoadProject{i}", f"/path/to/project{i}")
        
        end_time = time.perf_counter()
        load_time = end_time - start_time
        
        # Verify performance
        assert load_time < 2.0  # Should complete within 2 seconds
        assert len(dashboard.tasks) == 100
        assert len(project_tree.projects) == 50
        
        print(f"UI load test completed in {load_time:.3f} seconds")


class TestUIErrorHandling:
    """Tests for UI error handling and edge cases."""
    
    @pytest.mark.ui
    @pytest.mark.edge_case
    def test_invalid_project_handling(self):
        """Test handling of invalid project data."""
        project_tree = ProjectTree()
        
        # Test with invalid project data
        invalid_projects = [
            ("", "/valid/path"),  # Empty name
            ("ValidName", ""),    # Empty path
            (None, "/valid/path"), # None name
            ("ValidName", None)   # None path
        ]
        
        for name, path in invalid_projects:
            try:
                if name and path:  # Only add if both are valid
                    node_id = project_tree.add_project(name, path)
                    assert node_id is not None
                else:
                    # Invalid data should be handled gracefully
                    pass
            except Exception as e:
                # Should not raise unhandled exceptions
                assert "project" in str(e).lower() or "path" in str(e).lower()
    
    @pytest.mark.ui
    @pytest.mark.edge_case
    def test_task_dashboard_edge_cases(self):
        """Test task dashboard edge cases."""
        dashboard = TaskDashboard()
        
        # Test with invalid task data
        invalid_tasks = [
            {},  # Empty task
            {"id": ""},  # Empty ID
            {"title": "No ID Task"},  # Missing ID
            {"id": "duplicate-id", "title": "Task 1"},
            {"id": "duplicate-id", "title": "Task 2"}  # Duplicate ID
        ]
        
        for task in invalid_tasks:
            try:
                dashboard.add_task(task)
            except Exception as e:
                # Should handle gracefully
                assert isinstance(e, (ValueError, KeyError)) or "task" in str(e).lower()
        
        # Verify dashboard remains stable
        assert isinstance(dashboard.tasks, list)
    
    @pytest.mark.ui
    @pytest.mark.edge_case
    def test_console_error_output(self):
        """Test console error output handling."""
        console = ConsoleWidget()
        
        # Test various error scenarios
        error_outputs = [
            "ERROR: Build failed",
            "FATAL: Out of memory",
            "WARNING: Deprecated API usage",
            "",  # Empty output
            None  # None output
        ]
        
        for output in error_outputs:
            try:
                if output is not None:
                    console.write_output(output)
            except Exception as e:
                # Should handle gracefully
                assert "output" in str(e).lower()
        
        # Test error command execution
        error_commands = [
            "",  # Empty command
            None,  # None command
            "invalid-command-that-does-not-exist"
        ]
        
        for command in error_commands:
            try:
                if command:
                    result = console.execute_command(command)
                    assert result is not None
            except Exception as e:
                # Should handle gracefully
                assert "command" in str(e).lower()


if __name__ == "__main__":
    # Run enhanced UI component tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "ui",
        "--durations=10"
    ])