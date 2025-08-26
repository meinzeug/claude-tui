"""
Claude-TUI User Interface Module.

Provides the Textual-based Terminal User Interface components.
"""

from src.claude_tui.ui.application import ClaudeTUIApp

# Import available screens
try:
    from src.claude_tui.ui.screens.workspace_screen import WorkspaceScreen
    from src.claude_tui.ui.screens.project_wizard import ProjectWizard
    from src.claude_tui.ui.screens.settings import SettingsScreen
    from src.claude_tui.ui.screens.help_screen import HelpScreen
except ImportError:
    # Fallback for missing screens
    WorkspaceScreen = None
    ProjectWizard = None
    SettingsScreen = None
    HelpScreen = None

__all__ = [
    "ClaudeTUIApp",
]