"""
Error Handler Bridge Module.

This module provides a bridge to the main error handler located in src/core/error_handler.py
to maintain compatibility with the existing import structure.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src/core
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from core.error_handler import *
    from core.error_handler import ErrorHandler as CoreErrorHandler, get_error_handler as core_get_error_handler
    
    # Re-export with consistent interface
    def get_error_handler():
        return core_get_error_handler()
    
    # Re-export commonly used functions
    __all__ = ['get_error_handler', 'handle_errors', 'error_context', 'ErrorHandler']
    
except ImportError as e:
    # Fallback implementation
    def get_error_handler():
        return None
        
    def handle_errors(**kwargs):
        def decorator(func):
            return func
        return decorator
        
    def error_context(*args, **kwargs):
        from contextlib import nullcontext
        return nullcontext()
        
    class ErrorHandler:
        def __init__(self):
            pass
            
        def handle(self, error):
            print(f"Error: {error}")