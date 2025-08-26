"""Interface definitions for Claude-TUI components."""

from .ui_interfaces import (
    UIComponentInterface,
    EditorInterface, 
    TreeInterface,
    PaletteInterface,
    StatusInterface
)

from .service_interfaces import (
    ConfigInterface,
    AIInterface,
    ProjectInterface,
    ValidationInterface
)

__all__ = [
    'UIComponentInterface',
    'EditorInterface',
    'TreeInterface', 
    'PaletteInterface',
    'StatusInterface',
    'ConfigInterface',
    'AIInterface',
    'ProjectInterface',
    'ValidationInterface'
]