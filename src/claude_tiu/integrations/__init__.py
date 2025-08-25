"""
Claude-TIU Integrations Module.

Provides integration interfaces with external AI services and tools.
"""

from claude_tiu.integrations.ai_interface import AIInterface
from claude_tiu.integrations.claude_code_client import ClaudeCodeClient
from claude_tiu.integrations.claude_flow_client import ClaudeFlowClient
from claude_tiu.integrations.git_integration import GitIntegration

__all__ = [
    "AIInterface",
    "ClaudeCodeClient",
    "ClaudeFlowClient",
    "GitIntegration"
]