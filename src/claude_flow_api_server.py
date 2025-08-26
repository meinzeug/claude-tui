#!/usr/bin/env python3
"""
Claude Flow API Server
Provides REST API endpoints for Claude Flow orchestration
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests
class SwarmInitRequest(BaseModel):
    topology: str = "mesh"
    maxAgents: int = 5
    description: Optional[str] = None

class AgentSpawnRequest(BaseModel):
    type: str
    name: Optional[str] = None
    capabilities: Optional[list] = None

class WorkflowExecuteRequest(BaseModel):
    workflow: str
    params: Dict[str, Any] = {}
    agents: Optional[list] = None

class TaskOrchestrateRequest(BaseModel):
    task: str
    agents: list
    priority: str = "medium"

# Initialize FastAPI app
app = FastAPI(
    title="Claude Flow Orchestrator API",
    description="REST API for Claude Flow AI Agent Orchestration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
swarm_state = {
    "active_swarms": [],
    "agents": {},
    "tasks": {},
    "workflows": {}
}

def run_claude_flow_command(command: str, args: list = None) -> Dict[str, Any]:
    """Execute Claude Flow command and return result"""
    try:
        cmd = ["npx", "claude-flow@alpha"] + ([command] + (args or []))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/home/tekkadmin/claude-tui"
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "returncode": -1
        }

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "uptime": datetime.now().isoformat(),
        "service": "Claude Flow Orchestrator",
        "version": "2.0.0"
    }

# Status endpoint
@app.get("/api/status")
async def get_status():
    """Get system status"""
    status_result = run_claude_flow_command("status")
    return {
        "system_status": status_result.get("stdout", ""),
        "active_swarms": len(swarm_state["active_swarms"]),
        "active_agents": len(swarm_state["agents"]),
        "pending_tasks": len(swarm_state["tasks"])
    }

# Swarm endpoints
@app.post("/api/swarm/init")
async def initialize_swarm(request: SwarmInitRequest):
    """Initialize a new swarm"""
    swarm_id = f"swarm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Use Claude Flow swarm command
    result = run_claude_flow_command("swarm", [request.description or "Initialize swarm"])
    
    swarm_info = {
        "id": swarm_id,
        "topology": request.topology,
        "maxAgents": request.maxAgents,
        "status": "initialized",
        "created_at": datetime.now().isoformat(),
        "agents": []
    }
    swarm_state["active_swarms"].append(swarm_info)
    
    if result["success"]:
        
        return {
            "success": True,
            "swarm_id": swarm_id,
            "swarm_info": swarm_info,
            "claude_flow_output": result["stdout"]
        }
    else:
        raise HTTPException(status_code=500, detail=f"Swarm initialization failed: {result.get('stderr', 'Unknown error')}")

# Agent endpoints
@app.post("/api/agent/spawn")
async def spawn_agent(request: AgentSpawnRequest):
    """Spawn a new agent"""
    agent_id = f"agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Use Claude Flow agent spawn command
    result = run_claude_flow_command("agent", ["spawn", request.type])
    
    agent_info = {
        "id": agent_id,
        "type": request.type,
        "name": request.name or agent_id,
        "capabilities": request.capabilities or [],
        "status": "active",
        "created_at": datetime.now().isoformat()
    }
    swarm_state["agents"][agent_id] = agent_info
    
    if result["success"]:
        
        return {
            "success": True,
            "agent_id": agent_id,
            "agent_info": agent_info,
            "claude_flow_output": result["stdout"]
        }
    else:
        raise HTTPException(status_code=500, detail=f"Agent spawn failed: {result.get('stderr', 'Unknown error')}")

# Workflow endpoints
@app.post("/api/workflow/execute")
async def execute_workflow(request: WorkflowExecuteRequest, background_tasks: BackgroundTasks):
    """Execute a workflow"""
    workflow_id = f"workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Store workflow info
    workflow_info = {
        "id": workflow_id,
        "workflow": request.workflow,
        "params": request.params,
        "agents": request.agents or [],
        "status": "running",
        "created_at": datetime.now().isoformat()
    }
    swarm_state["workflows"][workflow_id] = workflow_info
    
    # Execute workflow using SPARC
    def execute_sparc_workflow():
        result = run_claude_flow_command("sparc", [request.workflow])
        workflow_info["status"] = "completed" if result["success"] else "failed"
        workflow_info["output"] = result["stdout"]
        workflow_info["completed_at"] = datetime.now().isoformat()
    
    background_tasks.add_task(execute_sparc_workflow)
    
    return {
        "success": True,
        "workflow_id": workflow_id,
        "workflow_info": workflow_info,
        "message": "Workflow execution started"
    }

# Task orchestration endpoints
@app.post("/api/task/orchestrate")
async def orchestrate_task(request: TaskOrchestrateRequest, background_tasks: BackgroundTasks):
    """Orchestrate a task across agents"""
    task_id = f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Store task info
    task_info = {
        "id": task_id,
        "task": request.task,
        "agents": request.agents,
        "priority": request.priority,
        "status": "orchestrating",
        "created_at": datetime.now().isoformat()
    }
    swarm_state["tasks"][task_id] = task_info
    
    # Execute task orchestration
    def execute_task_orchestration():
        # Create agents first
        for agent_type in request.agents:
            run_claude_flow_command("agent", ["spawn", agent_type])
        
        # Execute task using swarm
        result = run_claude_flow_command("swarm", [request.task])
        task_info["status"] = "completed" if result["success"] else "failed"
        task_info["output"] = result["stdout"]
        task_info["completed_at"] = datetime.now().isoformat()
    
    background_tasks.add_task(execute_task_orchestration)
    
    return {
        "success": True,
        "task_id": task_id,
        "task_info": task_info,
        "message": "Task orchestration started"
    }

# Get workflow status
@app.get("/api/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow execution status"""
    if workflow_id not in swarm_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return swarm_state["workflows"][workflow_id]

# Get task status
@app.get("/api/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Get task orchestration status"""
    if task_id not in swarm_state["tasks"]:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return swarm_state["tasks"][task_id]

# List active agents
@app.get("/api/agents")
async def list_agents():
    """List all active agents"""
    return {
        "agents": list(swarm_state["agents"].values()),
        "count": len(swarm_state["agents"])
    }

# List active swarms
@app.get("/api/swarms")
async def list_swarms():
    """List all active swarms"""
    return {
        "swarms": swarm_state["active_swarms"],
        "count": len(swarm_state["active_swarms"])
    }

if __name__ == "__main__":
    # Start the API server
    port = int(os.environ.get("CLAUDE_FLOW_API_PORT", 3001))
    uvicorn.run(
        "claude_flow_api_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True
    )