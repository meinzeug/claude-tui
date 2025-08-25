"""
Claude-TUI Integrations Module.

Provides integration interfaces with external AI services and tools.
"""

from claude_tui.integrations.ai_interface import AIInterface
from claude_tui.integrations.claude_code_client import ClaudeCodeClient
from claude_tui.integrations.claude_flow_client import ClaudeFlowClient
from claude_tui.integrations.git_integration import GitIntegration

__all__ = [
    "AIInterface",
    "ClaudeCodeClient",
    "ClaudeFlowClient",
    "GitIntegration"
]