"""
Claude Flow Client - REST API client for Claude Flow orchestration.

Provides async HTTP operations, swarm management, and workflow orchestration
for complex multi-agent AI tasks.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import aiohttp

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.models.project import Project
from src.claude_tui.models.ai_models import (
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
    
    def __init__(self, config_manager: ConfigManager = None):
        """
        Initialize the Claude Flow client.
        
        Args:
            config_manager: Configuration management instance (optional)
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
        self._connection_healthy = True
        self._last_health_check = None
        
        # Performance monitoring
        self._performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'connection_errors': 0,
            'timeouts': 0
        }
        
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
                    'User-Agent': 'Claude-TUI/0.1.0',
                    'Content-Type': 'application/json'
                }
            )
            
            # Test connection
            await self._test_connection()
            
            self._connection_healthy = True
            self._last_health_check = datetime.now()
            
            logger.info(f"Claude Flow client initialized with endpoint: {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude Flow client: {e}")
            self._connection_healthy = False
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
        
        try:
            payload = {
                "workflow": workflow_request.workflow_name,
                "parameters": workflow_request.parameters,
                "variables": workflow_request.variables,
                "project_context": {
                    "name": project.name if project else None,
                    "path": str(project.path) if project else None
                }
            }
            
            response = await self._make_request(
                'POST', '/api/workflow/execute',
                json=payload
            )
            
            # Parse workflow result
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = WorkflowResult(
                workflow_id=response.get('workflow_id', workflow_request.workflow_name),
                success=response.get('success', False),
                output=response.get('output', ''),
                execution_time=execution_time,
                agent_count=response.get('agent_count', 0),
                task_count=response.get('task_count', 0)
            )
            
            if not result.success:
                result.error_message = response.get('error', 'Workflow execution failed')
            
            logger.info(
                f"Workflow '{workflow_request.workflow_name}' completed in {execution_time:.2f}s "
                f"(success: {result.success})"
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Workflow execution failed: {e}")
            
            return WorkflowResult(
                workflow_id=workflow_request.workflow_name,
                success=False,
                output="",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def orchestrate_task(
        self,
        swarm_id: str,
        request: TaskOrchestrationRequest,
        project: Optional[Project] = None
    ) -> OrchestrationResult:
        """
        Orchestrate a task across the swarm.
        
        Args:
            swarm_id: Local swarm ID
            request: Task orchestration request
            project: Associated project (optional)
            
        Returns:
            OrchestrationResult: Orchestration result
        """
        logger.info(f"Orchestrating task: {request.description[:50]}...")
        
        start_time = datetime.now()
        
        try:
            # Get remote swarm ID
            remote_swarm_id = self._active_swarms.get(swarm_id)
            if not remote_swarm_id:
                raise ValueError(f"Swarm {swarm_id} not found")
            
            payload = {
                "task": request.description,
                "priority": request.priority,
                "maxAgents": request.max_agents,
                "strategy": request.strategy,
                "swarmId": remote_swarm_id,
                "requirements": request.requirements
            }
            
            response = await self._make_request(
                'POST', '/api/task/orchestrate',
                json=payload
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = OrchestrationResult(
                task_id=response.get('task_id'),
                success=response.get('success', False),
                assigned_agents=response.get('assigned_agents', []),
                execution_time=execution_time,
                result_data=response.get('result', {})
            )
            
            if not result.success:
                result.error_message = response.get('error', 'Task orchestration failed')
            
            logger.info(f"Task orchestration completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Task orchestration failed: {e}")
            
            return OrchestrationResult(
                task_id=None,
                success=False,
                assigned_agents=[],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def get_swarm_status(self, swarm_id: str) -> Dict[str, Any]:
        """
        Get current status of a swarm.
        
        Args:
            swarm_id: Local swarm ID
            
        Returns:
            Dict containing swarm status information
        """
        try:
            remote_swarm_id = self._active_swarms.get(swarm_id)
            if not remote_swarm_id:
                raise ValueError(f"Swarm {swarm_id} not found")
            
            response = await self._make_request(
                'GET', f'/api/swarm/{remote_swarm_id}/status'
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get swarm status: {e}")
            return {'error': str(e)}
    
    async def shutdown_swarm(self, swarm_id: str) -> bool:
        """
        Shutdown and cleanup a swarm.
        
        Args:
            swarm_id: Local swarm ID
            
        Returns:
            bool: True if successful
        """
        try:
            remote_swarm_id = self._active_swarms.get(swarm_id)
            if not remote_swarm_id:
                return False
            
            await self._make_request(
                'DELETE', f'/api/swarm/{remote_swarm_id}'
            )
            
            # Remove from local tracking
            del self._active_swarms[swarm_id]
            
            # Remove associated agents
            agents_to_remove = []
            for local_agent_id, remote_agent_id in self._active_agents.items():
                # This would need to be tracked better in practice
                pass
            
            logger.info(f"Swarm {swarm_id} shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown swarm: {e}")
            return False
    
    async def cleanup(self) -> None:
        """
        Cleanup Claude Flow client resources.
        """
        logger.info("Cleaning up Claude Flow client")
        
        # Shutdown all active swarms
        for swarm_id in list(self._active_swarms.keys()):
            await self.shutdown_swarm(swarm_id)
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        # Reset state
        self._connection_healthy = False
        self._last_health_check = None
        
        logger.info("Claude Flow client cleanup completed")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics."""
        return {
            'connection_healthy': self._connection_healthy,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'active_swarms': len(self._active_swarms),
            'active_agents': len(self._active_agents),
            'request_count': self._request_count,
            'performance_metrics': self._performance_metrics.copy()
        }
    
    async def health_check(self) -> bool:
        """Perform comprehensive health check."""
        try:
            is_healthy = await self._test_connection()
            self._last_health_check = datetime.now()
            self._connection_healthy = is_healthy
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._connection_healthy = False
            return False
    
    def is_healthy(self) -> bool:
        """Check if client is in healthy state."""
        if not self._connection_healthy:
            return False
        
        # Check if last health check was recent (within 5 minutes)
        if self._last_health_check:
            time_since_check = datetime.now() - self._last_health_check
            if time_since_check.total_seconds() > 300:  # 5 minutes
                return False
        
        return True
    
    # Private helper methods
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
        retries: int = 3,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Claude Flow API with enhanced error handling and monitoring.
        """
        if not self.session:
            await self.initialize()
        
        url = f"{self.base_url}{endpoint}"
        self._request_count += 1
        request_start = time.time()
        
        # Apply request-specific timeout
        request_timeout = aiohttp.ClientTimeout(
            total=timeout if timeout else self.timeout.total
        )
        
        for attempt in range(retries + 1):
            try:
                logger.debug(f"Making {method} request to {url} (attempt {attempt + 1}/{retries + 1})")
                
                async with self.session.request(
                    method=method,
                    url=url,
                    json=json,
                    params=params,
                    timeout=request_timeout
                ) as response:
                    # Enhanced status code handling
                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    if response.status == 503:  # Service unavailable
                        if attempt < retries:
                            wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                            logger.warning(f"Service unavailable, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    response.raise_for_status()
                    
                    # Parse response based on content type
                    if response.content_type == 'application/json':
                        response_data = await response.json()
                    else:
                        text = await response.text()
                        response_data = {'response': text}
                    
                    # Add performance metadata
                    request_time = time.time() - request_start
                    response_data['_metadata'] = {
                        'request_time': request_time,
                        'status_code': response.status,
                        'attempt': attempt + 1
                    }
                    
                    logger.debug(f"Request completed in {request_time:.3f}s")
                    return response_data
                    
            except asyncio.TimeoutError as e:
                if attempt == retries:
                    logger.error(f"Request timeout after {retries} attempts: {e}")
                    raise aiohttp.ClientError(f"Timeout after {retries} attempts")
                
                wait_time = min(2 ** attempt, 30)
                logger.warning(f"Request timeout, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            
            except aiohttp.ClientError as e:
                if attempt == retries:
                    logger.error(f"HTTP request failed after {retries} retries: {e}")
                    raise
                
                wait_time = min(2 ** attempt, 30)  # Exponential backoff with cap
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                logger.error(f"Unexpected error in HTTP request: {e}")
                raise
    
    async def _test_connection(self) -> bool:
        """
        Test connection to Claude Flow API with comprehensive health checking.
        """
        try:
            start_time = time.time()
            response = await self._make_request('GET', '/api/health', timeout=10)
            response_time = time.time() - start_time
            
            status = response.get('status', 'unknown')
            
            if status == 'healthy':
                logger.info(f"Claude Flow API health check passed ({response_time:.3f}s)")
                return True
            else:
                logger.warning(f"Claude Flow API health check returned status: {status}")
                # Still return True for 'degraded' status to allow operation
                return status in ['healthy', 'degraded']
                
        except Exception as e:
            logger.warning(f"Claude Flow API connection test failed: {e}")
            return False  # Return False to indicate unhealthy state