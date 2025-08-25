#!/usr/bin/env python3
"""
UI Module - Claude-TUI User Interface Components
"""

try:
    from .main_app import ClaudeTUIApp, MainWorkspace, run_app
    _main_app_available = True
except ImportError:
    _main_app_available = False
    ClaudeTUIApp = None
    MainWorkspace = None
    run_app = None

# Try to import screens and widgets, but don't fail if they don't exist
try:
    from .screens import *
except ImportError:
    pass

try:
    from .widgets import *
except ImportError:
    pass

__all__ = []

if _main_app_available:
    __all__.extend([
        # Main Application
        'ClaudeTUIApp',
        'MainWorkspace', 
        'run_app',
    ])