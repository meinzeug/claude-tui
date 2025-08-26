#!/usr/bin/env python3
"""
MCP API Endpoints
FastAPI endpoints for MCP server integration
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import json
from datetime import datetime

from .server import MCPServerClient, SwarmCoordinator, HooksIntegration

# Pydantic models for request/response
class SwarmInitRequest(BaseModel):
    topology: str = Field(default="mesh", description="Swarm topology type")
    max_agents: int = Field(default=5, description="Maximum number of agents")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class AgentSpawnRequest(BaseModel):
    agent_type: str = Field(description="Type of agent to spawn")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority: Optional[int] = Field(default=1)

class TaskOrchestrationRequest(BaseModel):
    description: str = Field(description="Task description")
    agents: Optional[List[str]] = Field(default_factory=list)
    parallel: bool = Field(default=True)
    timeout: Optional[int] = Field(default=300)

class HookExecutionRequest(BaseModel):
    hook_type: str = Field(description="Type of hook to execute")
    params: Dict[str, Any] = Field(default_factory=dict)

class MCPResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# FastAPI app
app = FastAPI(
    title="Claude-TUI MCP Integration API",
    description="API for MCP server integration with Claude-TUI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MCP client (will be initialized on startup)
mcp_client: Optional[MCPServerClient] = None
swarm_coordinator: Optional[SwarmCoordinator] = None
hooks_integration: Optional[HooksIntegration] = None

async def get_mcp_client() -> MCPServerClient:
    """Dependency to get MCP client"""
    global mcp_client
    if not mcp_client:
        mcp_client = MCPServerClient()
        await mcp_client.__aenter__()
    return mcp_client

async def get_swarm_coordinator() -> SwarmCoordinator:
    """Dependency to get swarm coordinator"""
    global swarm_coordinator
    if not swarm_coordinator:
        client = await get_mcp_client()
        swarm_coordinator = SwarmCoordinator(client)
    return swarm_coordinator

async def get_hooks_integration() -> HooksIntegration:
    """Dependency to get hooks integration"""
    global hooks_integration
    if not hooks_integration:
        client = await get_mcp_client()
        hooks_integration = HooksIntegration(client)
    return hooks_integration

@app.on_event("startup")
async def startup_event():
    """Initialize MCP connections on startup"""
    try:
        await get_mcp_client()
        await get_swarm_coordinator()
        await get_hooks_integration()
        print("MCP integration initialized successfully")
    except Exception as e:
        print(f"Failed to initialize MCP integration: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global mcp_client
    if mcp_client:
        await mcp_client.__aexit__(None, None, None)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        coordinator = await get_swarm_coordinator()
        status = await coordinator.get_swarm_status()
        return {"status": "healthy", "mcp_status": status}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Swarm management endpoints
@app.post("/api/swarm/init", response_model=MCPResponse)
async def init_swarm(
    request: SwarmInitRequest,
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator)
):
    """Initialize a new swarm"""
    try:
        success = await coordinator.initialize_swarm(
            topology=request.topology,
            max_agents=request.max_agents
        )
        
        return MCPResponse(
            success=success,
            data={"topology": request.topology, "max_agents": request.max_agents}
        )
    except Exception as e:
        return MCPResponse(success=False, error=str(e))

@app.post("/api/swarm/spawn", response_model=MCPResponse)
async def spawn_agent(
    request: AgentSpawnRequest,
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator)
):
    """Spawn a new agent in the swarm"""
    try:
        success = await coordinator.spawn_agent(
            agent_type=request.agent_type,
            config=request.config
        )
        
        return MCPResponse(
            success=success,
            data={"agent_type": request.agent_type, "config": request.config}
        )
    except Exception as e:
        return MCPResponse(success=False, error=str(e))

@app.post("/api/swarm/orchestrate", response_model=MCPResponse)
async def orchestrate_task(
    request: TaskOrchestrationRequest,
    background_tasks: BackgroundTasks,
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator)
):
    """Orchestrate a task across the swarm"""
    try:
        task_id = await coordinator.orchestrate_task(
            task_description=request.description,
            agents=request.agents
        )
        
        if task_id:
            return MCPResponse(
                success=True,
                data={"task_id": task_id, "description": request.description}
            )
        else:
            return MCPResponse(success=False, error="Task orchestration failed")
            
    except Exception as e:
        return MCPResponse(success=False, error=str(e))

@app.get("/api/swarm/status", response_model=MCPResponse)
async def get_swarm_status(
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator)
):
    """Get current swarm status"""
    try:
        status = await coordinator.get_swarm_status()
        return MCPResponse(success=True, data=status)
    except Exception as e:
        return MCPResponse(success=False, error=str(e))

@app.get("/api/swarm/metrics", response_model=MCPResponse)
async def get_swarm_metrics(
    agent_name: Optional[str] = None,
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator)
):
    """Get swarm and agent metrics"""
    try:
        metrics = await coordinator.get_agent_metrics(agent_name)
        return MCPResponse(success=True, data={"metrics": metrics})
    except Exception as e:
        return MCPResponse(success=False, error=str(e))

# Hooks integration endpoints
@app.post("/api/hooks/execute", response_model=MCPResponse)
async def execute_hook(
    request: HookExecutionRequest,
    hooks: HooksIntegration = Depends(get_hooks_integration)
):
    """Execute a claude-flow hook"""
    try:
        success = await hooks.execute_hook(
            hook_type=request.hook_type,
            params=request.params
        )
        
        return MCPResponse(
            success=success,
            data={"hook_type": request.hook_type, "params": request.params}
        )
    except Exception as e:
        return MCPResponse(success=False, error=str(e))

@app.post("/api/hooks/register", response_model=MCPResponse)
async def register_hook_handler(
    hook_type: str,
    callback_url: str,
    hooks: HooksIntegration = Depends(get_hooks_integration)
):
    """Register a hook handler callback"""
    try:
        success = await hooks.register_hook_handler(hook_type, callback_url)
        
        return MCPResponse(
            success=success,
            data={"hook_type": hook_type, "callback_url": callback_url}
        )
    except Exception as e:
        return MCPResponse(success=False, error=str(e))

# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.websocket("/ws/swarm")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time swarm updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and send periodic updates
            coordinator = await get_swarm_coordinator()
            status = await coordinator.get_swarm_status()
            
            await websocket.send_json({
                "type": "status_update",
                "data": status,
                "timestamp": datetime.now().isoformat()
            })
            
            await asyncio.sleep(10)  # Send updates every 10 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)