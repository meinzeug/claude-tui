"""
Clone Project Dialog for TUI Application.

Provides Git repository cloning functionality with progress tracking.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Footer, Header, Input, Label, LoadingIndicator, ProgressBar, Static
)
from textual.binding import Binding
from textual.worker import Worker


class CloneProjectDialog(ModalScreen[Dict[str, Any]]):
    """
    Modal dialog for cloning Git repositories.
    
    Provides URL input, destination selection, and clone progress tracking.
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]
    
    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize clone dialog.
        
        Args:
            callback: Callback function to call with clone result
        """
        super().__init__()
        self.callback = callback
        self.clone_worker: Optional[Worker] = None
        self.is_cloning = False
    
    def compose(self) -> ComposeResult:
        """Compose the clone dialog layout."""
        with Container(id="clone_dialog_container"):
            yield Header()
            
            with Vertical():
                yield Label("Clone Git Repository", id="dialog_title")
                
                with Vertical(id="input_section"):
                    yield Label("Repository URL:", classes="input_label")
                    yield Input(
                        placeholder="https://github.com/user/repo.git",
                        id="repo_url_input"
                    )
                    
                    yield Label("Destination Directory:", classes="input_label")
                    with Horizontal():
                        yield Input(
                            value=str(Path.cwd()),
                            placeholder="Destination path...",
                            id="dest_path_input"
                        )
                        yield Button("Browse", id="browse_btn", variant="default")
                    
                    yield Label("Project Name (optional):", classes="input_label")
                    yield Input(
                        placeholder="Leave empty to use repo name",
                        id="project_name_input"
                    )
                
                with Vertical(id="progress_section"):
                    yield Label("", id="status_label")
                    yield ProgressBar(id="clone_progress", show_eta=False)
                    yield LoadingIndicator(id="loading_indicator")
                
                with Horizontal(id="button_container"):
                    yield Button("Clone", id="clone_btn", variant="success")
                    yield Button("Cancel", id="cancel_btn", variant="error")
            
            yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the dialog when mounted."""
        # Hide progress section initially
        progress_section = self.query_one("#progress_section")
        progress_section.display = False
        
        loading_indicator = self.query_one("#loading_indicator", LoadingIndicator)
        loading_indicator.display = False
        
        # Focus on URL input
        url_input = self.query_one("#repo_url_input", Input)
        url_input.focus()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "repo_url_input":
            # Move focus to destination path
            dest_input = self.query_one("#dest_path_input", Input)
            dest_input.focus()
        elif event.input.id == "dest_path_input":
            # Move focus to project name
            name_input = self.query_one("#project_name_input", Input)
            name_input.focus()
        elif event.input.id == "project_name_input":
            # Start clone
            if not self.is_cloning:
                self._start_clone()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "browse_btn":
            self._show_directory_picker()
        
        elif event.button.id == "clone_btn":
            if not self.is_cloning:
                self._start_clone()
            else:
                self.notify("Clone in progress...", severity="info")
        
        elif event.button.id == "cancel_btn":
            self.action_cancel()
    
    def action_cancel(self) -> None:
        """Cancel the clone operation."""
        if self.is_cloning and self.clone_worker:
            self.clone_worker.cancel()
            self.notify("Clone cancelled", severity="warning")
        
        self.dismiss({"success": False, "cancelled": True})
    
    def _show_directory_picker(self) -> None:
        """Show directory picker for destination."""
        from .file_picker import FilePickerScreen
        
        def on_directory_selected(directory_path: Path) -> None:
            dest_input = self.query_one("#dest_path_input", Input)
            dest_input.value = str(directory_path)
        
        file_picker = FilePickerScreen(
            title="Select Destination Directory",
            directories_only=True,
            callback=on_directory_selected,
            initial_path=Path.cwd()
        )
        
        self.app.push_screen(file_picker)
    
    def _start_clone(self) -> None:
        """Start the git clone operation."""
        # Get input values
        repo_url = self.query_one("#repo_url_input", Input).value.strip()
        dest_path = self.query_one("#dest_path_input", Input).value.strip()
        project_name = self.query_one("#project_name_input", Input).value.strip()
        
        # Validate inputs
        if not repo_url:
            self.notify("Please enter a repository URL", severity="error")
            return
        
        if not dest_path:
            self.notify("Please enter a destination path", severity="error")
            return
        
        # Extract repo name from URL if project name not provided
        if not project_name:
            if repo_url.endswith('.git'):
                project_name = repo_url.split('/')[-1][:-4]  # Remove .git
            else:
                project_name = repo_url.split('/')[-1]
        
        # Validate destination directory exists
        dest_dir = Path(dest_path)
        if not dest_dir.exists():
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.notify(f"Cannot create destination directory: {e}", severity="error")
                return
        
        # Show progress section
        progress_section = self.query_one("#progress_section")
        progress_section.display = True
        
        loading_indicator = self.query_one("#loading_indicator", LoadingIndicator)
        loading_indicator.display = True
        
        # Update status
        status_label = self.query_one("#status_label", Label)
        status_label.update("Initializing clone...")
        
        # Disable inputs and clone button
        self.is_cloning = True
        self._set_inputs_enabled(False)
        
        clone_btn = self.query_one("#clone_btn", Button)
        clone_btn.label = "Cloning..."
        clone_btn.disabled = True
        
        # Start clone worker
        self.clone_worker = self.run_worker(
            self._clone_repository(repo_url, dest_dir, project_name),
            exclusive=True
        )
    
    async def _clone_repository(
        self,
        repo_url: str,
        dest_dir: Path,
        project_name: str
    ) -> Dict[str, Any]:
        """
        Clone the repository using git.
        
        Args:
            repo_url: Git repository URL
            dest_dir: Destination directory
            project_name: Project name/directory name
            
        Returns:
            Clone result dictionary
        """
        project_path = dest_dir / project_name
        
        try:
            # Update status
            await self._update_clone_status("Cloning repository...", 20)
            
            # Check if directory already exists
            if project_path.exists():
                return {
                    "success": False,
                    "error": f"Directory '{project_name}' already exists in destination"
                }
            
            # Run git clone command
            cmd = ["git", "clone", repo_url, str(project_path)]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Monitor progress
            await self._update_clone_status("Downloading files...", 50)
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                await self._update_clone_status("Clone completed successfully!", 100)
                
                # Brief success display
                await asyncio.sleep(1)
                
                result = {
                    "success": True,
                    "name": project_name,
                    "path": str(project_path),
                    "url": repo_url
                }
                
                # Call callback
                if self.callback:
                    self.callback(result)
                
                # Dismiss dialog
                self.dismiss(result)
                
                return result
            else:
                error_msg = stderr.decode() if stderr else "Unknown git error"
                return {
                    "success": False,
                    "error": f"Git clone failed: {error_msg}"
                }
        
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Git is not installed or not in PATH"
            }
        
        except asyncio.CancelledError:
            # Cleanup partial clone
            if project_path.exists():
                try:
                    import shutil
                    shutil.rmtree(project_path)
                except Exception:
                    pass
            
            return {
                "success": False,
                "error": "Clone was cancelled"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Clone failed: {str(e)}"
            }
        
        finally:
            # Reset UI state
            self.is_cloning = False
            self._set_inputs_enabled(True)
            
            clone_btn = self.query_one("#clone_btn", Button)
            clone_btn.label = "Clone"
            clone_btn.disabled = False
            
            loading_indicator = self.query_one("#loading_indicator", LoadingIndicator)
            loading_indicator.display = False
    
    async def _update_clone_status(self, message: str, progress: int) -> None:
        """Update clone status and progress."""
        status_label = self.query_one("#status_label", Label)
        status_label.update(message)
        
        progress_bar = self.query_one("#clone_progress", ProgressBar)
        progress_bar.update(progress=progress)
        
        # Allow UI to update
        await asyncio.sleep(0.1)
    
    def _set_inputs_enabled(self, enabled: bool) -> None:
        """Enable or disable input fields."""
        inputs = [
            "#repo_url_input",
            "#dest_path_input", 
            "#project_name_input"
        ]
        
        for input_id in inputs:
            input_widget = self.query_one(input_id, Input)
            input_widget.disabled = not enabled
        
        browse_btn = self.query_one("#browse_btn", Button)
        browse_btn.disabled = not enabled
    
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes."""
        if event.worker == self.clone_worker:
            if event.state == Worker.State.SUCCESS:
                # Worker completed successfully
                pass
            elif event.state == Worker.State.ERROR:
                # Handle worker error
                self.notify("Clone operation failed", severity="error")
                
                result = {
                    "success": False,
                    "error": "Clone operation encountered an error"
                }
                
                if self.callback:
                    self.callback(result)
                
                self.dismiss(result)
            elif event.state == Worker.State.CANCELLED:
                # Handle worker cancellation
                self.notify("Clone operation was cancelled", severity="warning")