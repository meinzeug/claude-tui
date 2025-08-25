#!/usr/bin/env python3
"""
Project Tree Widget - File system navigation with real-time updates
and validation status indicators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

from textual import on, work
from textual.containers import Vertical
from textual.widgets import DirectoryTree, Static, Label, Button
from textual.message import Message
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.tree import Tree
from rich.console import Console


class ProjectFileTree(DirectoryTree):
    """Enhanced directory tree with validation status"""
    
    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.validation_status: Dict[str, str] = {}  # file_path -> status
        self.file_icons = {
            '.py': '🐍',
            '.js': '🟨',
            '.ts': '🔷',
            '.html': '🌐',
            '.css': '🎨',
            '.json': '📄',
            '.md': '📝',
            '.yml': '⚙️',
            '.yaml': '⚙️',
            '.txt': '📄',
            '.log': '📋',
        }
        self.status_icons = {
            'validated': '✅',
            'placeholder': '⚠️',
            'error': '❌',
            'pending': '🔄',
            'unknown': '❓',
        }
    
    def render_tree_label(self, node, path: Path, is_dir: bool) -> Text:
        """Render tree label with validation status indicators"""
        text = Text()
        
        # Add file type icon
        if not is_dir:
            icon = self.file_icons.get(path.suffix.lower(), '📄')
            text.append(f"{icon} ")
        else:
            text.append("📁 ")
        
        # Add filename
        text.append(path.name)
        
        # Add validation status
        file_status = self.validation_status.get(str(path), 'unknown')
        status_icon = self.status_icons.get(file_status, '❓')
        text.append(f" {status_icon}")
        
        return text
    
    def update_file_status(self, file_path: str, status: str) -> None:
        """Update validation status for a file"""
        self.validation_status[file_path] = status
        self.refresh()
    
    def update_validation_batch(self, status_dict: Dict[str, str]) -> None:
        """Batch update validation statuses"""
        self.validation_status.update(status_dict)
        self.refresh()


class ProjectTree(Vertical):
    """Project tree widget with enhanced functionality"""
    
    current_project_path: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, project_manager) -> None:
        super().__init__()
        self.project_manager = project_manager
        self.tree_widget: Optional[ProjectFileTree] = None
        self.status_label: Optional[Label] = None
        self.refresh_button: Optional[Button] = None
        
    def compose(self):
        """Compose the project tree widget"""
        yield Static("📁 Project Explorer", classes="header")
        
        # Project status
        self.status_label = Label("No project loaded", classes="status")
        yield self.status_label
        
        # Refresh button
        self.refresh_button = Button("🔄 Refresh", id="refresh_tree")
        yield self.refresh_button
        
        # Directory tree (will be added dynamically)
        yield Static("Select a project to view files", id="tree_placeholder")
    
    def watch_current_project_path(self, path: Optional[str]) -> None:
        """React to project path changes"""
        if path:
            self.load_project_tree(Path(path))
            if self.status_label:
                self.status_label.update(f"Project: {Path(path).name}")
        else:
            self.clear_tree()
            if self.status_label:
                self.status_label.update("No project loaded")
    
    def load_project_tree(self, project_path: Path) -> None:
        """Load project directory tree"""
        try:
            # Remove placeholder
            placeholder = self.query_one("#tree_placeholder", Static)
            placeholder.remove()
            
            # Create new tree widget
            self.tree_widget = ProjectFileTree(project_path)
            self.mount(self.tree_widget)
            
            # Start validation status monitoring
            self.start_status_monitoring()
            
        except Exception as e:
            if self.status_label:
                self.status_label.update(f"Error loading project: {e}")
    
    def clear_tree(self) -> None:
        """Clear the current tree"""
        if self.tree_widget:
            self.tree_widget.remove()
            self.tree_widget = None
        
        # Add placeholder back
        placeholder = Static("Select a project to view files", id="tree_placeholder")
        self.mount(placeholder)
    
    @work(exclusive=True)
    async def start_status_monitoring(self) -> None:
        """Start monitoring file validation status"""
        if not self.tree_widget or not self.current_project_path:
            return
            
        project_path = Path(self.current_project_path)
        
        while self.tree_widget and self.current_project_path:
            try:
                # Get validation status for all files
                status_updates = await self._get_file_statuses(project_path)
                
                if self.tree_widget:
                    self.tree_widget.update_validation_batch(status_updates)
                    
                await asyncio.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                # Log error but continue monitoring
                await asyncio.sleep(30)  # Wait longer on errors
    
    async def _get_file_statuses(self, project_path: Path) -> Dict[str, str]:
        """Get validation status for all project files"""
        status_dict = {}
        
        # Walk through all files in project
        for file_path in project_path.rglob('*'):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                # Determine validation status
                status = await self._check_file_status(file_path)
                status_dict[str(file_path)] = status
        
        return status_dict
    
    async def _check_file_status(self, file_path: Path) -> str:
        """Check validation status for a single file"""
        try:
            # Check if file is a code file that needs validation
            if file_path.suffix in ['.py', '.js', '.ts', '.jsx', '.tsx']:
                # Read file content
                content = file_path.read_text(encoding='utf-8')
                
                # Basic placeholder detection
                placeholder_patterns = [
                    'TODO:', 'FIXME:', 'XXX:', 'HACK:',
                    'placeholder', 'dummy', 'mock',
                    'NotImplemented', 'pass  # implement',
                    'console.log("test")',
                ]
                
                for pattern in placeholder_patterns:
                    if pattern.lower() in content.lower():
                        return 'placeholder'
                
                # Check if file is empty or minimal
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                if len(lines) < 5:  # Very short files might be stubs
                    return 'pending'
                
                return 'validated'
            else:
                return 'unknown'  # Non-code files
                
        except Exception:
            return 'error'
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored in validation"""
        ignore_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            '.vscode',
            '.idea',
            '.DS_Store',
            '*.pyc',
            '*.log',
        ]
        
        path_str = str(file_path)
        for pattern in ignore_patterns:
            if pattern.replace('*', '') in path_str:
                return True
        
        return False
    
    @on(Button.Pressed, "#refresh_tree")
    def refresh_tree(self) -> None:
        """Handle refresh button press"""
        self.refresh()
        if self.current_project_path:
            self.watch_current_project_path(self.current_project_path)
    
    @on(DirectoryTree.FileSelected)
    def file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection"""
        # Emit custom message for file selection
        self.post_message(FileSelectedMessage(str(event.path)))
    
    def refresh(self) -> None:
        """Refresh the project tree"""
        if self.tree_widget:
            self.tree_widget.reload()
        
    def set_project(self, project_path: str) -> None:
        """Set the current project path"""
        self.current_project_path = project_path


class FileSelectedMessage(Message):
    """Message sent when a file is selected in the tree"""
    
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path


class TreeRefreshedMessage(Message):
    """Message sent when tree is refreshed"""
    
    def __init__(self) -> None:
        super().__init__()