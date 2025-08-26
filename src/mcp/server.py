#!/usr/bin/env python3
"""
MCP Server Integration Layer
Handles communication between Claude-TUI and claude-flow MCP services
"""

import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MCPRequest:
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class MCPResponse:
    result: Optional[Dict[str, Any]]
    error: Optional[Dict[str, Any]]
    id: Optional[str] = None
    timestamp: Optional[datetime] = None

class MCPServerClient:
    """Client for communicating with claude-flow MCP server"""
    
    def __init__(self, host: str = "localhost", port: int = 3000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.connection_pool_size = 10
        self.timeout = aiohttp.ClientTimeout(total=30)
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.connection_pool_size)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_method(self, method: str, params: Dict[str, Any] = None) -> MCPResponse:
        """Call an MCP method on the server"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        request = MCPRequest(
            method=method,
            params=params or {},
            id=f"req_{datetime.now().timestamp()}",
            timestamp=datetime.now()
        )
        
        try:
            request_data = {
                "jsonrpc": "2.0",
                "method": request.method,
                "params": request.params,
                "id": request.id
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp",
                json=request_data
            ) as response:
                response_data = await response.json()
                
                return MCPResponse(
                    result=response_data.get("result"),
                    error=response_data.get("error"),
                    id=response_data.get("id"),
                    timestamp=datetime.now()
                )
                
        except aiohttp.ClientError as e:
            logger.error(f"MCP client error: {e}")
            return MCPResponse(
                result=None,
                error={"code": -32603, "message": f"Client error: {str(e)}"},
                id=request.id,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return MCPResponse(
                result=None,
                error={"code": -32603, "message": f"Internal error: {str(e)}"},
                id=request.id,
                timestamp=datetime.now()
            )

class SwarmCoordinator:
    """Coordinates swarm operations through MCP server"""
    
    def __init__(self, mcp_client: MCPServerClient):
        self.mcp_client = mcp_client
        self.db_path = Path.cwd() / ".swarm" / "coordination.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize coordination database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS swarm_state (
                    id INTEGER PRIMARY KEY,
                    topology TEXT,
                    agents TEXT,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    id INTEGER PRIMARY KEY,
                    agent_name TEXT,
                    task_id TEXT,
                    metrics TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def initialize_swarm(self, topology: str = "mesh", max_agents: int = 5) -> bool:
        """Initialize swarm with specified topology"""
        response = await self.mcp_client.call_method("swarm_init", {
            "topology": topology,
            "maxAgents": max_agents
        })
        
        if response.error:
            logger.error(f"Swarm initialization failed: {response.error}")
            return False
        
        # Store swarm state
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO swarm_state (topology, agents, status) VALUES (?, ?, ?)",
                (topology, json.dumps([]), "initialized")
            )
        
        logger.info(f"Swarm initialized with {topology} topology")
        return True
    
    async def spawn_agent(self, agent_type: str, config: Dict[str, Any] = None) -> bool:
        """Spawn a new agent in the swarm"""
        response = await self.mcp_client.call_method("agent_spawn", {
            "type": agent_type,
            "config": config or {}
        })
        
        if response.error:
            logger.error(f"Agent spawn failed: {response.error}")
            return False
        
        logger.info(f"Agent {agent_type} spawned successfully")
        return True
    
    async def orchestrate_task(self, task_description: str, agents: List[str] = None) -> Optional[str]:
        """Orchestrate a task across the swarm"""
        response = await self.mcp_client.call_method("task_orchestrate", {
            "description": task_description,
            "agents": agents or [],
            "parallel": True
        })
        
        if response.error:
            logger.error(f"Task orchestration failed: {response.error}")
            return None
        
        task_id = response.result.get("task_id") if response.result else None
        logger.info(f"Task orchestrated with ID: {task_id}")
        return task_id
    
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        response = await self.mcp_client.call_method("swarm_status", {})
        
        if response.error:
            logger.error(f"Failed to get swarm status: {response.error}")
            return {"status": "error", "message": str(response.error)}
        
        return response.result or {"status": "unknown"}
    
    async def get_agent_metrics(self, agent_name: str = None) -> List[Dict[str, Any]]:
        """Get metrics for agents"""
        response = await self.mcp_client.call_method("agent_metrics", {
            "agent": agent_name
        })
        
        if response.error:
            logger.warning(f"Failed to get agent metrics: {response.error}")
            return []
        
        return response.result.get("metrics", []) if response.result else []

class HooksIntegration:
    """Integration with claude-flow hooks system"""
    
    def __init__(self, mcp_client: MCPServerClient):
        self.mcp_client = mcp_client
    
    async def execute_hook(self, hook_type: str, params: Dict[str, Any]) -> bool:
        """Execute a claude-flow hook"""
        response = await self.mcp_client.call_method("execute_hook", {
            "hook": hook_type,
            "params": params
        })
        
        if response.error:
            logger.error(f"Hook execution failed: {response.error}")
            return False
        
        return True
    
    async def register_hook_handler(self, hook_type: str, callback_url: str) -> bool:
        """Register a callback for hook events"""
        response = await self.mcp_client.call_method("register_hook_handler", {
            "hook": hook_type,
            "callback": callback_url
        })
        
        if response.error:
            logger.error(f"Hook registration failed: {response.error}")
            return False
        
        return True

async def test_mcp_connection():
    """Test MCP server connectivity"""
    async with MCPServerClient() as client:
        coordinator = SwarmCoordinator(client)
        
        # Test basic connectivity
        status = await coordinator.get_swarm_status()
        print(f"Swarm Status: {status}")
        
        # Test swarm initialization
        success = await coordinator.initialize_swarm("mesh", 3)
        print(f"Swarm Initialization: {'Success' if success else 'Failed'}")
        
        # Test agent spawning
        success = await coordinator.spawn_agent("coder", {"language": "python"})
        print(f"Agent Spawn: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())