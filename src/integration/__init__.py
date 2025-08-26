#!/usr/bin/env python3
"""
Integration Module
Bridges Claude-TUI with Claude-Flow and MCP services
"""

from .bridge import IntegrationBridge, BridgeConfig, IntegrationEvent, EventBus

__all__ = [
    "IntegrationBridge",
    "BridgeConfig", 
    "IntegrationEvent",
    "EventBus"
]

__version__ = "1.0.0"