"""
File Picker Screen for TUI Application.

Provides file and directory selection functionality with filtering support.
"""

import os
from pathlib import Path
from typing import Callable, List, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button, DirectoryTree, Footer, Header, Input, Label, Static
)
from textual.binding import Binding


class FilePickerScreen(ModalScreen[Path]):
    """
    Modal screen for file and directory selection.
    
    Provides a file browser with filtering and selection capabilities.
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select", "Select"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]
    
    def __init__(
        self,
        title: str = "Select File",
        file_types: Optional[List[str]] = None,
        directories_only: bool = False,
        callback: Optional[Callable[[Path], None]] = None,
        initial_path: Optional[Path] = None
    ):
        """
        Initialize file picker.
        
        Args:
            title: Dialog title
            file_types: List of allowed file extensions (e.g., ['.py', '.json'])
            directories_only: If True, only allow directory selection
            callback: Callback function to call with selected path
            initial_path: Initial directory to show
        """
        super().__init__()
        self.title = title
        self.file_types = file_types or []
        self.directories_only = directories_only
        self.callback = callback
        self.initial_path = initial_path or Path.cwd()
        self.selected_path: Optional[Path] = None
    
    def compose(self) -> ComposeResult:
        """Compose the file picker layout."""
        with Container(id="file_picker_container"):
            yield Header()
            
            with Vertical():
                yield Label(self.title, id="picker_title")
                
                with Horizontal():
                    yield Label("Path:", classes="path_label")
                    yield Input(
                        value=str(self.initial_path),
                        placeholder="Enter path...",
                        id="path_input"
                    )
                    yield Button("Go", id="go_btn", variant="primary")
                
                yield DirectoryTree(str(self.initial_path), id="file_tree")
                
                with Horizontal(id="selection_info"):
                    yield Label("Selected: None", id="selected_label")
                
                with Horizontal(id="button_container"):
                    yield Button("Select", id="select_btn", variant="success")
                    yield Button("Cancel", id="cancel_btn", variant="error")
            
            yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the file picker when mounted."""
        tree = self.query_one("#file_tree", DirectoryTree)
        tree.show_root = True
        tree.show_guides = True
        
        # Update selection info
        self._update_selection_info()
    
    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle directory selection."""
        if self.directories_only:
            self.selected_path = event.path
            self._update_selection_info()
    
    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection."""
        if not self.directories_only:
            file_path = event.path
            
            # Check file type filter
            if self.file_types:
                if not any(str(file_path).endswith(ext) for ext in self.file_types):
                    self.notify(f"Invalid file type. Allowed: {', '.join(self.file_types)}", severity="warning")
                    return
            
            self.selected_path = file_path
            self._update_selection_info()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle path input submission."""
        if event.input.id == "path_input":
            self._navigate_to_path(event.value)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "go_btn":
            path_input = self.query_one("#path_input", Input)
            self._navigate_to_path(path_input.value)
        
        elif event.button.id == "select_btn":
            self.action_select()
        
        elif event.button.id == "cancel_btn":
            self.action_cancel()
    
    def action_select(self) -> None:
        """Select the current file/directory."""
        if not self.selected_path:
            # If no specific selection, use current tree path
            tree = self.query_one("#file_tree", DirectoryTree)
            self.selected_path = Path(tree.path)
        
        if self.selected_path and self.selected_path.exists():
            # Validate selection
            if self.directories_only and not self.selected_path.is_dir():
                self.notify("Please select a directory", severity="warning")
                return
            
            if not self.directories_only and self.selected_path.is_dir():
                self.notify("Please select a file", severity="warning")
                return
            
            # Call callback if provided
            if self.callback:
                self.callback(self.selected_path)
            
            # Dismiss modal with result
            self.dismiss(self.selected_path)
        else:
            self.notify("Please select a valid file or directory", severity="warning")
    
    def action_cancel(self) -> None:
        """Cancel file selection."""
        self.dismiss()
    
    def _navigate_to_path(self, path_str: str) -> None:
        """Navigate to specified path."""
        try:
            path = Path(path_str).expanduser().resolve()
            
            if path.exists() and path.is_dir():
                tree = self.query_one("#file_tree", DirectoryTree)
                tree.path = str(path)
                
                path_input = self.query_one("#path_input", Input)
                path_input.value = str(path)
                
                self.notify(f"Navigated to: {path}", severity="info")
            else:
                self.notify(f"Path does not exist or is not a directory: {path}", severity="error")
        
        except Exception as e:
            self.notify(f"Invalid path: {e}", severity="error")
    
    def _update_selection_info(self) -> None:
        """Update the selection information display."""
        selected_label = self.query_one("#selected_label", Label)
        
        if self.selected_path:
            display_path = str(self.selected_path)
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            selected_label.update(f"Selected: {display_path}")
        else:
            selected_label.update("Selected: None")