"""
Claude-TIU CLI Commands Module.

This module contains all CLI command implementations organized by functionality:
- core_commands: Basic project management (init, build, test, deploy)
- ai_commands: AI integration features (generate, review, fix, optimize)
- workspace_commands: Workspace and template management
- integration_commands: External integrations (GitHub, CI/CD, etc.)
"""

from .core_commands import core_commands
from .ai_commands import ai_commands
from .workspace_commands import workspace_commands  
from .integration_commands import integration_commands

__all__ = [
    "core_commands",
    "ai_commands", 
    "workspace_commands",
    "integration_commands"
]