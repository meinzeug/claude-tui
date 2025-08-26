"""Claude-TUI UI Widgets Module.

Provides reusable UI components built on Textual.
"""

from textual.widgets import (
    Button, Input, TextArea, ListView, Tree, 
    Static, Label, Header, Footer, DataTable,
    ProgressBar, Log, DirectoryTree
)
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen

# Custom widgets
from .text_input import TextInput
from .code_editor import CodeEditor
from .file_tree import FileTree
from .command_palette import CommandPalette
from .status_bar import StatusBar
from .task_dashboard import TaskDashboard, TaskInfo

__all__ = [
    'Button', 'Input', 'TextArea', 'ListView', 'Tree', 
    'Static', 'Label', 'Header', 'Footer', 'DataTable',
    'ProgressBar', 'Log', 'DirectoryTree', 'Container', 
    'Horizontal', 'Vertical', 'Screen',
    'TextInput', 'CodeEditor', 'FileTree', 'CommandPalette', 'StatusBar',
    'TaskDashboard', 'TaskInfo'
]