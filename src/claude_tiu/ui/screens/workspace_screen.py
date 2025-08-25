"""
Workspace Screen - Main development workspace interface.

Provides the primary workspace for project development including file tree,
editor integration, AI assistant panel, and real-time task monitoring.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static, Button, Input, TextArea, Tree, DirectoryTree, 
    LoadingIndicator, Label, ProgressBar, Tabs, TabbedContent, TabPane
)
from textual.reactive import reactive
from textual.message import Message
from textual.screen import ModalScreen
from textual.binding import Binding

logger = logging.getLogger(__name__)


class FileEditor(TextArea):
    """Enhanced text editor for code editing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_file: Optional[Path] = None
        self.is_modified = False
        
    def load_file(self, file_path: Path) -> bool:
        """Load a file into the editor."""
        try:
            if file_path.exists() and file_path.is_file():
                content = file_path.read_text(encoding='utf-8')
                self.text = content
                self.current_file = file_path
                self.is_modified = False
                return True
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
        return False
    
    def save_file(self) -> bool:
        """Save current content to file."""
        try:
            if self.current_file:
                self.current_file.write_text(self.text, encoding='utf-8')
                self.is_modified = False
                return True
        except Exception as e:
            logger.error(f"Failed to save file {self.current_file}: {e}")
        return False
    
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Mark file as modified when content changes."""
        self.is_modified = True


class AIAssistantPanel(Container):
    """AI assistant panel for code generation and analysis."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversation_history: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the AI assistant panel."""
        with Vertical():
            yield Label("🤖 AI Assistant", classes="panel-title")
            yield ScrollableContainer(
                Static("Welcome! I'm here to help with your development tasks.", 
                       id="ai_conversation", classes="conversation"),
                id="ai_history"
            )
            with Horizontal(classes="input-row"):
                yield Input(placeholder="Ask me anything about your code...", 
                           id="ai_input", classes="ai-input")
                yield Button("Send", id="ai_send", variant="primary")
    
    def add_message(self, message: str, sender: str = "user") -> None:
        """Add a message to the conversation."""
        timestamp = f"[dim]{sender}[/dim]"
        conversation = self.query_one("#ai_conversation", Static)
        
        current_text = conversation.renderable
        if current_text:
            new_text = f"{current_text}\n\n{timestamp}: {message}"
        else:
            new_text = f"{timestamp}: {message}"
        
        conversation.update(new_text)
        
        # Auto-scroll to bottom
        history_container = self.query_one("#ai_history", ScrollableContainer)
        history_container.scroll_end()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle AI assistant button presses."""
        if event.button.id == "ai_send":
            input_widget = self.query_one("#ai_input", Input)
            message = input_widget.value.strip()
            
            if message:
                self.add_message(message, "user")
                input_widget.value = ""
                
                # TODO: Send to AI service
                self.add_message("I received your message and I'm processing it...", "assistant")


class TaskMonitorPanel(Container):
    """Panel for monitoring active tasks and their progress."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the task monitor panel."""
        with Vertical():
            yield Label("📋 Active Tasks", classes="panel-title")
            yield ScrollableContainer(
                Static("No active tasks", id="task_list", classes="task-list"),
                id="task_container"
            )
            with Horizontal(classes="button-row"):
                yield Button("Refresh", id="refresh_tasks", variant="default")
                yield Button("Clear Completed", id="clear_completed", variant="default")
    
    def update_tasks(self, tasks: Dict[str, Dict[str, Any]]) -> None:
        """Update the task display."""
        self.active_tasks = tasks
        
        if not tasks:
            task_display = "No active tasks"
        else:
            task_lines = []
            for task_id, task_info in tasks.items():
                status = task_info.get('status', 'unknown')
                name = task_info.get('name', 'Unnamed Task')
                progress = task_info.get('progress', 0)
                
                status_emoji = {
                    'pending': '⏳',
                    'running': '🔄',
                    'completed': '✅',
                    'failed': '❌'
                }.get(status, '❓')
                
                task_lines.append(f"{status_emoji} {name} ({progress}%)")
            
            task_display = "\n".join(task_lines)
        
        task_list = self.query_one("#task_list", Static)
        task_list.update(task_display)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle task monitor button presses."""
        if event.button.id == "refresh_tasks":
            # TODO: Refresh task list from task engine
            pass
        elif event.button.id == "clear_completed":
            # TODO: Clear completed tasks
            pass


class WorkspaceScreen(Container):
    """
    Main workspace screen providing development environment.
    
    Features:
    - File browser and editor
    - AI assistant integration  
    - Task monitoring
    - Real-time project status
    """
    
    BINDINGS = [
        Binding("ctrl+s", "save_file", "Save File"),
        Binding("ctrl+o", "open_file", "Open File"),
        Binding("ctrl+n", "new_file", "New File"),
        Binding("f5", "refresh", "Refresh"),
    ]
    
    # Reactive attributes
    current_project = reactive(None)
    current_file = reactive(None)
    ai_available = reactive(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.project_path: Optional[Path] = None
        
    def compose(self) -> ComposeResult:
        """Compose the workspace layout."""
        with Horizontal():
            # Left panel - File browser
            with Vertical(classes="left-panel"):
                yield Label("📁 Project Files", classes="panel-title")
                yield DirectoryTree("./", id="file_tree")
                
                with Horizontal(classes="button-row"):
                    yield Button("📄 New", id="new_file_btn", variant="default")
                    yield Button("📂 Open", id="open_file_btn", variant="default")
            
            # Center panel - Editor
            with Vertical(classes="center-panel"):
                with Horizontal(classes="editor-header"):
                    yield Label("📝 Editor", classes="panel-title") 
                    yield Button("💾 Save", id="save_file_btn", variant="primary")
                
                yield FileEditor(
                    placeholder="Select a file to edit or create a new one...",
                    id="main_editor",
                    classes="main-editor"
                )
            
            # Right panel - AI Assistant and Tasks
            with Vertical(classes="right-panel"):
                with TabbedContent(initial="ai"):
                    with TabPane("AI Assistant", id="ai"):
                        yield AIAssistantPanel(id="ai_panel")
                    
                    with TabPane("Tasks", id="tasks"):
                        yield TaskMonitorPanel(id="task_panel")
    
    def on_mount(self) -> None:
        """Initialize the workspace screen."""
        # Set up initial state
        self.update_project_status()
        logger.info("Workspace screen mounted")
    
    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection from directory tree."""
        file_path = event.path
        
        if file_path.is_file():
            editor = self.query_one("#main_editor", FileEditor)
            
            if editor.load_file(file_path):
                self.current_file = file_path
                self.app.notify(f"Opened: {file_path.name}", severity="success")
            else:
                self.app.notify(f"Failed to open: {file_path.name}", severity="error")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle workspace button presses."""
        if event.button.id == "new_file_btn":
            self.action_new_file()
        elif event.button.id == "open_file_btn":
            self.action_open_file()
        elif event.button.id == "save_file_btn":
            self.action_save_file()
    
    def action_save_file(self) -> None:
        """Save the current file."""
        editor = self.query_one("#main_editor", FileEditor)
        
        if editor.current_file:
            if editor.save_file():
                self.app.notify(f"Saved: {editor.current_file.name}", severity="success")
            else:
                self.app.notify("Save failed", severity="error")
        else:
            self.app.notify("No file to save", severity="warning")
    
    def action_open_file(self) -> None:
        """Open a file dialog."""
        # TODO: Implement file picker
        self.app.notify("File picker not implemented yet", severity="info")
    
    def action_new_file(self) -> None:
        """Create a new file."""
        editor = self.query_one("#main_editor", FileEditor)
        editor.text = ""
        editor.current_file = None
        editor.is_modified = True
        self.app.notify("New file created", severity="success")
    
    def action_refresh(self) -> None:
        """Refresh the workspace."""
        file_tree = self.query_one("#file_tree", DirectoryTree)
        file_tree.reload()
        
        task_panel = self.query_one("#task_panel", TaskMonitorPanel)
        # TODO: Refresh tasks from engine
        
        self.app.notify("Workspace refreshed", severity="success")
    
    def update_project_status(self) -> None:
        """Update project status information."""
        try:
            # Update current project path
            self.project_path = Path.cwd()
            
            # Update file tree to current directory
            file_tree = self.query_one("#file_tree", DirectoryTree)
            file_tree.path = str(self.project_path)
            
        except Exception as e:
            logger.error(f"Failed to update project status: {e}")
    
    def set_ai_availability(self, available: bool) -> None:
        """Update AI availability status."""
        self.ai_available = available
        
        if available:
            ai_panel = self.query_one("#ai_panel", AIAssistantPanel)
            ai_panel.add_message("AI services are now available!", "system")
        else:
            # Could disable AI panel or show offline message
            pass
    
    def update_task_status(self, tasks: Dict[str, Dict[str, Any]]) -> None:
        """Update task status display."""
        task_panel = self.query_one("#task_panel", TaskMonitorPanel)
        task_panel.update_tasks(tasks)
    
    def load_project(self, project_path: Path) -> None:
        """Load a project into the workspace."""
        try:
            self.project_path = project_path
            self.current_project = project_path.name
            
            # Update file tree
            file_tree = self.query_one("#file_tree", DirectoryTree)
            file_tree.path = str(project_path)
            
            self.app.notify(f"Loaded project: {project_path.name}", severity="success")
            
        except Exception as e:
            logger.error(f"Failed to load project {project_path}: {e}")
            self.app.notify(f"Failed to load project: {e}", severity="error")