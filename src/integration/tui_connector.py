#!/usr/bin/env python3
"""
TUI Connector
Connects the Textual UI with the MCP integration bridge
"""

import asyncio
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from textual.message import Message
from textual.widget import Widget

from .bridge import IntegrationBridge, BridgeConfig, IntegrationEvent
from ..mcp.server import SwarmCoordinator, MCPServerClient

class TUIBridgeMessage(Message):
    """Message for TUI-Bridge communication"""
    
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        super().__init__()

class TUIConnector:
    """Connects Textual UI with MCP integration bridge"""
    
    def __init__(self, bridge: IntegrationBridge):
        self.bridge = bridge
        self.ui_callbacks: Dict[str, Callable] = {}
        self.bridge.event_bus.subscribe("swarm_status_change", self._forward_to_ui)
        self.bridge.event_bus.subscribe("agent_error", self._forward_to_ui)
        self.bridge.event_bus.subscribe("performance_alert", self._forward_to_ui)
    
    def register_ui_callback(self, event_type: str, callback: Callable):
        """Register a UI callback for specific events"""
        self.ui_callbacks[event_type] = callback
    
    async def _forward_to_ui(self, event: IntegrationEvent):
        """Forward bridge events to UI"""
        if event.event_type in self.ui_callbacks:
            callback = self.ui_callbacks[event.event_type]
            await callback(event)
    
    async def send_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send command from UI to bridge"""
        return await self.bridge.execute_swarm_command(command, params or {})
    
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        return await self.send_command("status")
    
    async def initialize_swarm(self, topology: str = "mesh", max_agents: int = 5) -> bool:
        """Initialize swarm from UI"""
        result = await self.send_command("init", {
            "topology": topology,
            "max_agents": max_agents
        })
        return result.get("success", False)
    
    async def spawn_agent(self, agent_type: str, config: Dict[str, Any] = None) -> bool:
        """Spawn agent from UI"""
        result = await self.send_command("spawn", {
            "agent_type": agent_type,
            "config": config or {}
        })
        return result.get("success", False)
    
    async def orchestrate_task(self, description: str, agents: list = None) -> Optional[str]:
        """Orchestrate task from UI"""
        result = await self.send_command("orchestrate", {
            "description": description,
            "agents": agents or []
        })
        return result.get("task_id")

class SwarmControlWidget(Widget):
    """Widget for controlling swarm operations from TUI"""
    
    def __init__(self, connector: TUIConnector, **kwargs):
        super().__init__(**kwargs)
        self.connector = connector
        self.swarm_status = {}
        
        # Register for status updates
        self.connector.register_ui_callback("swarm_status_change", self.update_status)
        self.connector.register_ui_callback("agent_error", self.handle_agent_error)
    
    async def update_status(self, event: IntegrationEvent):
        """Update widget when swarm status changes"""
        self.swarm_status = event.data.get("status", {})
        self.refresh()
    
    async def handle_agent_error(self, event: IntegrationEvent):
        """Handle agent errors in UI"""
        agent_name = event.data.get("agent_name")
        error = event.data.get("error")
        # Could show error notification in UI
        pass
    
    def render(self) -> str:
        """Render swarm control widget"""
        if not self.swarm_status:
            return "Swarm not initialized"
        
        active_agents = self.swarm_status.get("active_agents", 0)
        topology = self.swarm_status.get("topology", "unknown")
        
        return f"Swarm Status: {active_agents} agents ({topology} topology)"

class MCPConnectionManager:
    """Manages MCP connection for the TUI"""
    
    def __init__(self):
        self.bridge: Optional[IntegrationBridge] = None
        self.connector: Optional[TUIConnector] = None
        self.connected = False
    
    async def connect(self, config: BridgeConfig = None) -> bool:
        """Connect to MCP services"""
        try:
            # Initialize bridge
            self.bridge = IntegrationBridge(config or BridgeConfig())
            success = await self.bridge.start()
            
            if success:
                # Create connector
                self.connector = TUIConnector(self.bridge)
                self.connected = True
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Failed to connect to MCP services: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP services"""
        if self.bridge:
            await self.bridge.stop()
            self.bridge = None
            self.connector = None
            self.connected = False
    
    def get_connector(self) -> Optional[TUIConnector]:
        """Get the TUI connector"""
        return self.connector if self.connected else None
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test MCP connection"""
        if not self.connector:
            return {"status": "disconnected", "error": "No active connection"}
        
        try:
            status = await self.connector.get_swarm_status()
            return {"status": "connected", "swarm_status": status}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Integration with existing TUI application
class MCPIntegratedTUIApp:
    """TUI Application with MCP integration"""
    
    def __init__(self):
        self.connection_manager = MCPConnectionManager()
        self.connector: Optional[TUIConnector] = None
    
    async def startup(self):
        """Initialize MCP integration on TUI startup"""
        config = BridgeConfig(
            mcp_host="localhost",
            mcp_port=3000,
            monitoring_enabled=True,
            hooks_enabled=True
        )
        
        success = await self.connection_manager.connect(config)
        if success:
            self.connector = self.connection_manager.get_connector()
            print("MCP integration initialized successfully")
            
            # Test initial connection
            test_result = await self.connection_manager.test_connection()
            print(f"Connection test: {test_result}")
        else:
            print("Failed to initialize MCP integration")
    
    async def shutdown(self):
        """Cleanup MCP integration on TUI shutdown"""
        await self.connection_manager.disconnect()
        print("MCP integration disconnected")
    
    async def execute_swarm_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute swarm operation from TUI"""
        if not self.connector:
            return {"error": "MCP integration not available"}
        
        if operation == "init":
            success = await self.connector.initialize_swarm(
                topology=kwargs.get("topology", "mesh"),
                max_agents=kwargs.get("max_agents", 5)
            )
            return {"success": success}
            
        elif operation == "spawn":
            success = await self.connector.spawn_agent(
                agent_type=kwargs.get("agent_type"),
                config=kwargs.get("config", {})
            )
            return {"success": success}
            
        elif operation == "orchestrate":
            task_id = await self.connector.orchestrate_task(
                description=kwargs.get("description"),
                agents=kwargs.get("agents", [])
            )
            return {"task_id": task_id}
            
        elif operation == "status":
            status = await self.connector.get_swarm_status()
            return {"status": status}
        
        else:
            return {"error": f"Unknown operation: {operation}"}

# Convenience function to add MCP integration to existing TUI
def add_mcp_integration(tui_app, config: BridgeConfig = None):
    """Add MCP integration to an existing TUI application"""
    
    async def setup_mcp():
        """Setup MCP integration"""
        manager = MCPConnectionManager()
        success = await manager.connect(config or BridgeConfig())
        
        if success:
            connector = manager.get_connector()
            # Add connector as attribute to TUI app
            setattr(tui_app, "mcp_connector", connector)
            setattr(tui_app, "mcp_manager", manager)
            return True
        return False
    
    async def cleanup_mcp():
        """Cleanup MCP integration"""
        if hasattr(tui_app, "mcp_manager"):
            await tui_app.mcp_manager.disconnect()
    
    # Add methods to TUI app
    setattr(tui_app, "setup_mcp_integration", setup_mcp)
    setattr(tui_app, "cleanup_mcp_integration", cleanup_mcp)
    
    return tui_app

if __name__ == "__main__":
    # Test the integration
    async def test_integration():
        app = MCPIntegratedTUIApp()
        await app.startup()
        
        # Test operations
        result = await app.execute_swarm_operation("init", topology="mesh", max_agents=3)
        print(f"Init result: {result}")
        
        result = await app.execute_swarm_operation("spawn", agent_type="coder")
        print(f"Spawn result: {result}")
        
        result = await app.execute_swarm_operation("status")
        print(f"Status result: {result}")
        
        await app.shutdown()
    
    asyncio.run(test_integration())