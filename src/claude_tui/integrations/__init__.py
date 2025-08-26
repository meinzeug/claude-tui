"""
Claude-TUI Integrations Module.

Provides integration interfaces with external AI services and tools.
"""

from src.claude_tui.integrations.ai_interface import AIInterface
from src.claude_tui.integrations.claude_code_client import ClaudeCodeClient
from src.claude_tui.integrations.claude_flow_client import ClaudeFlowClient
from src.claude_tui.integrations.anti_hallucination_integration import AntiHallucinationIntegration
from src.claude_tui.integrations.integration_manager import IntegrationManager
# from src.claude_tui.integrations.git_integration import GitIntegration  # Module not implemented yet

__all__ = [
    "AIInterface",
    "ClaudeCodeClient",
    "ClaudeFlowClient",
    "AntiHallucinationIntegration",
    "IntegrationManager",
    # "GitIntegration"  # Commented out until implemented
]