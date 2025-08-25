"""
Claude-TIU Integration Layer

This module provides comprehensive integration capabilities for external tools
including Claude Code, Claude Flow, Git operations, and file system management.
"""

from .claude_code import ClaudeCodeClient, ClaudeCodeResult, ClaudeCodeError
from .claude_flow import ClaudeFlowOrchestrator, WorkflowResult, SwarmManager
from .git_manager import GitManager, GitOperationResult, GitError
from .file_system import FileSystemManager, FileOperationResult, FileSystemError

__all__ = [
    # Claude Code Integration
    'ClaudeCodeClient',
    'ClaudeCodeResult', 
    'ClaudeCodeError',
    
    # Claude Flow Integration
    'ClaudeFlowOrchestrator',
    'WorkflowResult',
    'SwarmManager',
    
    # Git Integration
    'GitManager',
    'GitOperationResult',
    'GitError',
    
    # File System Integration
    'FileSystemManager',
    'FileOperationResult',
    'FileSystemError',
]

__version__ = "1.0.0"
__author__ = "Claude-TIU Integration Team"