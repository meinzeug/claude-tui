"""
Claude Flow Client - REST API client for Claude Flow orchestration.

Provides async HTTP operations, swarm management, and workflow orchestration
for complex multi-agent AI tasks.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import aiohttp

from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.models.project import Project
from claude_tiu.models.ai_models import (
    WorkflowRequest, WorkflowResult, SwarmConfig, AgentConfig, 
    TaskOrchestrationRequest, OrchestrationResult
)

logger = logging.getLogger(__name__)


class ClaudeFlowClient:
    """
    REST API client for Claude Flow orchestration.
    
    Handles swarm initialization, agent management, and workflow execution
    through Claude Flow's REST API with comprehensive error handling and retry logic.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the Claude Flow client.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        
        # HTTP client configuration
        self.base_url = "http://localhost:3000"  # Default Claude Flow URL
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Runtime state
        self._active_swarms: Dict[str, str] = {}  # local_id -> remote_id
        self._active_agents: Dict[str, str] = {}  # local_id -> remote_id
        self._request_count = 0
        
        logger.info("Claude Flow client initialized")
    
    async def initialize(self) -> None:
        """
        Initialize the Claude Flow client.
        """
        try:
            # Load configuration
            flow_config = await self.config_manager.get_setting('claude_flow', {})
            self.base_url = flow_config.get('endpoint_url', self.base_url)
            
            timeout_seconds = flow_config.get('timeout', 300)
            self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'Claude-TIU/0.1.0',
                    'Content-Type': 'application/json'
                }
            )
            
            # Test connection
            await self._test_connection()
            
            logger.info(f"Claude Flow client initialized with endpoint: {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude Flow client: {e}")
            raise
    
    async def initialize_swarm(
        self,
        config: SwarmConfig,
        project: Optional[Project] = None
    ) -> str:
        """
        Initialize a new AI agent swarm.
        
        Args:
            config: Swarm configuration
            project: Associated project (optional)
            
        Returns:
            str: Swarm ID
        """
        logger.info(f"Initializing swarm with topology: {config.topology}")
        
        try:
            payload = {
                "topology": config.topology,
                "maxAgents": config.max_agents,
                "strategy": config.strategy,
                "project_context": {
                    "name": project.name if project else None,
                    "path": str(project.path) if project else None
                }
            }
            
            response = await self._make_request(
                'POST', '/api/swarm/init', 
                json=payload
            )
            
            swarm_id = response.get('swarm_id')
            if not swarm_id:
                raise ValueError("No swarm ID returned from Claude Flow")
            
            # Store swarm mapping
            local_swarm_id = f"swarm_{len(self._active_swarms)}"
            self._active_swarms[local_swarm_id] = swarm_id
            
            logger.info(f"Swarm initialized successfully: {swarm_id}")
            return local_swarm_id
            
        except Exception as e:
            logger.error(f"Failed to initialize swarm: {e}")
            raise
    
    async def spawn_agent(
        self,
        swarm_id: str,
        agent_config: AgentConfig,
        project: Optional[Project] = None
    ) -> str:
        """
        Spawn a new agent in the specified swarm.
        
        Args:
            swarm_id: Local swarm ID
            agent_config: Agent configuration
            project: Associated project (optional)
            
        Returns:
            str: Agent ID
        """
        logger.info(f"Spawning agent: {agent_config.type}")
        
        try:
            # Get remote swarm ID
            remote_swarm_id = self._active_swarms.get(swarm_id)
            if not remote_swarm_id:
                raise ValueError(f"Swarm {swarm_id} not found")
            
            payload = {
                "type": agent_config.type,
                "capabilities": agent_config.capabilities,
                "name": agent_config.name,
                "swarmId": remote_swarm_id
            }
            
            response = await self._make_request(
                'POST', '/api/agent/spawn',
                json=payload
            )
            
            agent_id = response.get('agent_id')
            if not agent_id:
                raise ValueError("No agent ID returned from Claude Flow")
            
            # Store agent mapping
            local_agent_id = f"agent_{len(self._active_agents)}"
            self._active_agents[local_agent_id] = agent_id
            
            logger.info(f"Agent spawned successfully: {agent_id}")
            return local_agent_id
            
        except Exception as e:
            logger.error(f"Failed to spawn agent: {e}")
            raise
    
    async def execute_workflow(
        self,
        workflow_request: WorkflowRequest,
        project: Optional[Project] = None
    ) -> WorkflowResult:
        """
        Execute a workflow using Claude Flow.
        
        Args:
            workflow_request: Workflow execution request
            project: Associated project (optional)
            
        Returns:
            WorkflowResult: Workflow execution result
        """
        logger.info(f"Executing workflow: {workflow_request.workflow_name}")
        
        start_time = datetime.now()
        
        try:\n            payload = {\n                "workflow": workflow_request.workflow_name,\n                "parameters": workflow_request.parameters,\n                "variables": workflow_request.variables,\n                "project_context": {\n                    "name": project.name if project else None,\n                    "path": str(project.path) if project else None\n                }\n            }\n            \n            response = await self._make_request(\n                'POST', '/api/workflow/execute',\n                json=payload\n            )\n            \n            # Parse workflow result\n            execution_time = (datetime.now() - start_time).total_seconds()\n            \n            result = WorkflowResult(\n                workflow_id=response.get('workflow_id', workflow_request.workflow_name),\n                success=response.get('success', False),\n                output=response.get('output', ''),\n                execution_time=execution_time,\n                agent_count=response.get('agent_count', 0),\n                task_count=response.get('task_count', 0)\n            )\n            \n            if not result.success:\n                result.error_message = response.get('error', 'Workflow execution failed')\n            \n            logger.info(\n                f"Workflow '{workflow_request.workflow_name}' completed in {execution_time:.2f}s "\n                f"(success: {result.success})\"\n            )\n            \n            return result\n            \n        except Exception as e:\n            execution_time = (datetime.now() - start_time).total_seconds()\n            logger.error(f\"Workflow execution failed: {e}\")\n            \n            return WorkflowResult(\n                workflow_id=workflow_request.workflow_name,\n                success=False,\n                output=\"\",\n                execution_time=execution_time,\n                error_message=str(e)\n            )\n    \n    async def orchestrate_task(\n        self,\n        swarm_id: str,\n        request: TaskOrchestrationRequest,\n        project: Optional[Project] = None\n    ) -> OrchestrationResult:\n        \"\"\"\n        Orchestrate a task across the swarm.\n        \n        Args:\n            swarm_id: Local swarm ID\n            request: Task orchestration request\n            project: Associated project (optional)\n            \n        Returns:\n            OrchestrationResult: Orchestration result\n        \"\"\"\n        logger.info(f\"Orchestrating task: {request.description[:50]}...\")\n        \n        start_time = datetime.now()\n        \n        try:\n            # Get remote swarm ID\n            remote_swarm_id = self._active_swarms.get(swarm_id)\n            if not remote_swarm_id:\n                raise ValueError(f\"Swarm {swarm_id} not found\")\n            \n            payload = {\n                \"task\": request.description,\n                \"priority\": request.priority,\n                \"maxAgents\": request.max_agents,\n                \"strategy\": request.strategy,\n                \"swarmId\": remote_swarm_id,\n                \"requirements\": request.requirements\n            }\n            \n            response = await self._make_request(\n                'POST', '/api/task/orchestrate',\n                json=payload\n            )\n            \n            execution_time = (datetime.now() - start_time).total_seconds()\n            \n            result = OrchestrationResult(\n                task_id=response.get('task_id'),\n                success=response.get('success', False),\n                assigned_agents=response.get('assigned_agents', []),\n                execution_time=execution_time,\n                result_data=response.get('result', {})\n            )\n            \n            if not result.success:\n                result.error_message = response.get('error', 'Task orchestration failed')\n            \n            logger.info(f\"Task orchestration completed in {execution_time:.2f}s\")\n            return result\n            \n        except Exception as e:\n            execution_time = (datetime.now() - start_time).total_seconds()\n            logger.error(f\"Task orchestration failed: {e}\")\n            \n            return OrchestrationResult(\n                task_id=None,\n                success=False,\n                assigned_agents=[],\n                execution_time=execution_time,\n                error_message=str(e)\n            )\n    \n    async def get_swarm_status(self, swarm_id: str) -> Dict[str, Any]:\n        \"\"\"\n        Get current status of a swarm.\n        \n        Args:\n            swarm_id: Local swarm ID\n            \n        Returns:\n            Dict containing swarm status information\n        \"\"\"\n        try:\n            remote_swarm_id = self._active_swarms.get(swarm_id)\n            if not remote_swarm_id:\n                raise ValueError(f\"Swarm {swarm_id} not found\")\n            \n            response = await self._make_request(\n                'GET', f'/api/swarm/{remote_swarm_id}/status'\n            )\n            \n            return response\n            \n        except Exception as e:\n            logger.error(f\"Failed to get swarm status: {e}\")\n            return {'error': str(e)}\n    \n    async def shutdown_swarm(self, swarm_id: str) -> bool:\n        \"\"\"\n        Shutdown and cleanup a swarm.\n        \n        Args:\n            swarm_id: Local swarm ID\n            \n        Returns:\n            bool: True if successful\n        \"\"\"\n        try:\n            remote_swarm_id = self._active_swarms.get(swarm_id)\n            if not remote_swarm_id:\n                return False\n            \n            await self._make_request(\n                'DELETE', f'/api/swarm/{remote_swarm_id}'\n            )\n            \n            # Remove from local tracking\n            del self._active_swarms[swarm_id]\n            \n            # Remove associated agents\n            agents_to_remove = []\n            for local_agent_id, remote_agent_id in self._active_agents.items():\n                # This would need to be tracked better in practice\n                pass\n            \n            logger.info(f\"Swarm {swarm_id} shutdown successfully\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Failed to shutdown swarm: {e}\")\n            return False\n    \n    async def cleanup(self) -> None:\n        \"\"\"\n        Cleanup Claude Flow client resources.\n        \"\"\"\n        logger.info(\"Cleaning up Claude Flow client\")\n        \n        # Shutdown all active swarms\n        for swarm_id in list(self._active_swarms.keys()):\n            await self.shutdown_swarm(swarm_id)\n        \n        # Close HTTP session\n        if self.session:\n            await self.session.close()\n            self.session = None\n        \n        logger.info(\"Claude Flow client cleanup completed\")\n    \n    # Private helper methods\n    \n    async def _make_request(\n        self,\n        method: str,\n        endpoint: str,\n        json: Optional[Dict] = None,\n        params: Optional[Dict] = None,\n        retries: int = 3\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Make HTTP request to Claude Flow API.\n        \"\"\"\n        if not self.session:\n            await self.initialize()\n        \n        url = f\"{self.base_url}{endpoint}\"\n        self._request_count += 1\n        \n        for attempt in range(retries + 1):\n            try:\n                async with self.session.request(\n                    method=method,\n                    url=url,\n                    json=json,\n                    params=params\n                ) as response:\n                    response.raise_for_status()\n                    \n                    if response.content_type == 'application/json':\n                        return await response.json()\n                    else:\n                        text = await response.text()\n                        return {'response': text}\n                    \n            except aiohttp.ClientError as e:\n                if attempt == retries:\n                    logger.error(f\"HTTP request failed after {retries} retries: {e}\")\n                    raise\n                \n                wait_time = 2 ** attempt  # Exponential backoff\n                logger.warning(f\"Request failed, retrying in {wait_time}s: {e}\")\n                await asyncio.sleep(wait_time)\n            \n            except Exception as e:\n                logger.error(f\"Unexpected error in HTTP request: {e}\")\n                raise\n    \n    async def _test_connection(self) -> None:\n        \"\"\"\n        Test connection to Claude Flow API.\n        \"\"\"\n        try:\n            response = await self._make_request('GET', '/api/health')\n            \n            if response.get('status') != 'healthy':\n                logger.warning(\"Claude Flow API health check returned non-healthy status\")\n            else:\n                logger.info(\"Claude Flow API connection test successful\")\n                \n        except Exception as e:\n            logger.warning(f\"Claude Flow API connection test failed: {e}\")\n            # Don't raise - allow client to work in degraded mode