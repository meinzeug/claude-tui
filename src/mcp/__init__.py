#!/usr/bin/env python3
"""
MCP Integration Module
Provides integration with claude-flow MCP services
"""

from .server import MCPServerClient, SwarmCoordinator, HooksIntegration
from .endpoints import app as mcp_api_app

__all__ = [
    "MCPServerClient",
    "SwarmCoordinator", 
    "HooksIntegration",
    "mcp_api_app"
]

__version__ = "1.0.0"