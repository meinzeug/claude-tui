"""
Claude Flow Integration Module

Provides comprehensive integration with Claude Flow workflow engine including:
- Swarm initialization and management
- Agent coordination and spawning
- Workflow orchestration and execution
- Memory management and persistence
- Real-time monitoring and metrics
- Neural network integration
"""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Callable
import tempfile
import aiohttp
import yaml

logger = logging.getLogger(__name__)


class ClaudeFlowError(Exception):
    """Base exception for Claude Flow integration errors"""
    pass


class SwarmInitializationError(ClaudeFlowError):
    """Raised when swarm initialization fails"""
    pass


class WorkflowExecutionError(ClaudeFlowError):
    """Raised when workflow execution fails"""
    pass


class AgentSpawnError(ClaudeFlowError):
    """Raised when agent spawning fails"""
    pass


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class SwarmTopology(Enum):
    """Swarm topology types"""
    MESH = "mesh"
    HIERARCHICAL = "hierarchical"
    RING = "ring"
    STAR = "star"


class AgentType(Enum):
    """Available agent types"""
    RESEARCHER = "researcher"
    CODER = "coder" 
    ANALYST = "analyst"
    OPTIMIZER = "optimizer"
    COORDINATOR = "coordinator"
    BACKEND_DEV = "backend-dev"
    FRONTEND_DEV = "frontend-dev"
    DATABASE_ARCHITECT = "database-architect"
    TEST_ENGINEER = "tester"
    DEVOPS_ENGINEER = "cicd-engineer"
    SECURITY_AUDITOR = "reviewer"
    SYSTEM_ARCHITECT = "system-architect"


@dataclass
class Agent:
    """Agent configuration and state"""
    id: str
    type: AgentType
    name: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    status: str = "idle"
    memory_context: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None


@dataclass
class WorkflowStep:
    """Individual workflow step configuration"""
    id: str
    name: str
    agent_type: AgentType
    instructions: str
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 3
    output_validation: Dict[str, Any] = field(default_factory=dict)
    memory_keys: List[str] = field(default_factory=list)


@dataclass 
class WorkflowResult:
    """Result from workflow execution"""
    workflow_id: str
    status: WorkflowStatus
    execution_time: float = 0.0
    steps_completed: int = 0
    steps_total: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    agent_results: Dict[str, Any] = field(default_factory=dict)
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == WorkflowStatus.COMPLETED

    @property
    def progress_percentage(self) -> float:
        if self.steps_total == 0:
            return 0.0
        return (self.steps_completed / self.steps_total) * 100.0


@dataclass
class SwarmConfig:
    """Swarm configuration"""
    topology: SwarmTopology
    max_agents: int = 8
    strategy: str = "adaptive"
    enable_coordination: bool = True
    enable_learning: bool = True
    persistence_mode: str = "auto"
    memory_ttl: int = 3600
    auto_scaling: bool = True


class SwarmManager:
    """
    Advanced swarm management for Claude Flow integration
    
    Features:
    - Dynamic topology selection and optimization
    - Agent lifecycle management
    - Load balancing and scaling
    - Performance monitoring
    - Memory management and persistence
    """
    
    def __init__(
        self,
        claude_flow_binary: str = "npx claude-flow@alpha",
        max_swarms: int = 5,
        enable_monitoring: bool = True
    ):
        self.claude_flow_binary = claude_flow_binary
        self.max_swarms = max_swarms
        self.enable_monitoring = enable_monitoring
        
        # Swarm tracking
        self.active_swarms: Dict[str, SwarmConfig] = {}
        self.swarm_agents: Dict[str, List[Agent]] = {}
        self.swarm_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Session management
        self.session_memory: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for Claude Flow integration"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    async def initialize_swarm(
        self,
        config: SwarmConfig,
        swarm_id: Optional[str] = None
    ) -> str:
        """
        Initialize a new swarm with specified configuration
        
        Args:
            config: Swarm configuration
            swarm_id: Optional swarm identifier
            
        Returns:
            Swarm ID for future operations
        """
        if swarm_id is None:
            swarm_id = f"swarm-{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Initializing swarm {swarm_id} with topology {config.topology.value}")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Initializing swarm {swarm_id}")
            
            # Initialize swarm via Claude Flow
            cmd = [
                *self.claude_flow_binary.split(),
                "swarm", "init",
                "--topology", config.topology.value,
                "--max-agents", str(config.max_agents),
                "--strategy", config.strategy
            ]
            
            result = await self._execute_command(cmd)
            
            if result.returncode != 0:
                raise SwarmInitializationError(f"Failed to initialize swarm: {result.stderr}")
            
            # Store swarm configuration
            self.active_swarms[swarm_id] = config
            self.swarm_agents[swarm_id] = []
            self.swarm_metrics[swarm_id] = {
                'created_at': datetime.utcnow(),
                'total_tasks': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'total_execution_time': 0.0
            }
            
            # Initialize DAA if enabled
            if config.enable_learning:
                await self._initialize_daa(swarm_id, config)
            
            # Execute hooks post-task
            await self._execute_hook("post-task", f"Swarm {swarm_id} initialized successfully")
            
            logger.info(f"Swarm {swarm_id} initialized successfully")
            return swarm_id
            
        except Exception as e:
            logger.error(f"Failed to initialize swarm {swarm_id}: {e}")
            raise SwarmInitializationError(f"Swarm initialization failed: {e}") from e

    async def spawn_agent(
        self,
        swarm_id: str,
        agent_type: AgentType,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None
    ) -> Agent:
        """
        Spawn a specialized agent in the swarm
        
        Args:
            swarm_id: Target swarm identifier
            agent_type: Type of agent to spawn
            name: Optional custom agent name
            capabilities: Optional agent capabilities
            
        Returns:
            Agent instance with configuration
        """
        if swarm_id not in self.active_swarms:
            raise AgentSpawnError(f"Swarm {swarm_id} not found")
        
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        if name is None:
            name = f"{agent_type.value}-{agent_id[-4:]}"
        
        if capabilities is None:
            capabilities = self._get_default_capabilities(agent_type)
        
        logger.info(f"Spawning agent {agent_id} ({agent_type.value}) in swarm {swarm_id}")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Spawning agent {agent_id}")
            
            # Spawn agent via Claude Flow
            cmd = [
                *self.claude_flow_binary.split(),
                "agent", "spawn",
                "--type", agent_type.value,
                "--name", name,
                "--capabilities", ",".join(capabilities)
            ]
            
            result = await self._execute_command(cmd)
            
            if result.returncode != 0:
                raise AgentSpawnError(f"Failed to spawn agent: {result.stderr}")
            
            # Create agent instance
            agent = Agent(
                id=agent_id,
                type=agent_type,
                name=name,
                capabilities=capabilities,
                status="active"
            )
            
            # Add to swarm
            self.swarm_agents[swarm_id].append(agent)
            
            # Execute hooks post-edit
            await self._execute_hook(
                "post-edit",
                f"Agent {agent_id} spawned",
                memory_key=f"swarm/{swarm_id}/agents/{agent_id}"
            )
            
            logger.info(f"Agent {agent_id} spawned successfully in swarm {swarm_id}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to spawn agent {agent_id}: {e}")
            raise AgentSpawnError(f"Agent spawn failed: {e}") from e

    async def get_swarm_status(self, swarm_id: str) -> Dict[str, Any]:
        """Get comprehensive swarm status and metrics"""
        
        if swarm_id not in self.active_swarms:
            raise ClaudeFlowError(f"Swarm {swarm_id} not found")
        
        try:
            # Get status via Claude Flow
            cmd = [
                *self.claude_flow_binary.split(),
                "swarm", "status",
                "--verbose"
            ]
            
            result = await self._execute_command(cmd)
            
            if result.returncode == 0:
                try:
                    flow_status = json.loads(result.stdout)
                except json.JSONDecodeError:
                    flow_status = {"status": "unknown", "raw_output": result.stdout}
            else:
                flow_status = {"status": "error", "error": result.stderr}
            
            # Combine with internal tracking
            agents = self.swarm_agents.get(swarm_id, [])
            metrics = self.swarm_metrics.get(swarm_id, {})
            config = self.active_swarms.get(swarm_id)
            
            return {
                'swarm_id': swarm_id,
                'config': {
                    'topology': config.topology.value if config else 'unknown',
                    'max_agents': config.max_agents if config else 0,
                    'strategy': config.strategy if config else 'unknown'
                },
                'agents': [
                    {
                        'id': agent.id,
                        'type': agent.type.value,
                        'name': agent.name,
                        'status': agent.status,
                        'capabilities': agent.capabilities,
                        'performance': agent.performance_metrics
                    }
                    for agent in agents
                ],
                'metrics': metrics,
                'claude_flow_status': flow_status,
                'active_agents': len([a for a in agents if a.status == 'active']),
                'total_agents': len(agents)
            }
            
        except Exception as e:
            logger.error(f"Failed to get swarm status for {swarm_id}: {e}")
            raise ClaudeFlowError(f"Status check failed: {e}") from e

    def _get_default_capabilities(self, agent_type: AgentType) -> List[str]:
        """Get default capabilities for agent type"""
        
        capability_map = {
            AgentType.RESEARCHER: ["research", "analysis", "documentation"],
            AgentType.CODER: ["coding", "debugging", "testing"],
            AgentType.ANALYST: ["analysis", "metrics", "reporting"],
            AgentType.OPTIMIZER: ["optimization", "performance", "refactoring"],
            AgentType.COORDINATOR: ["coordination", "planning", "monitoring"],
            AgentType.BACKEND_DEV: ["backend", "api", "database", "authentication"],
            AgentType.FRONTEND_DEV: ["frontend", "ui", "javascript", "react"],
            AgentType.DATABASE_ARCHITECT: ["database", "schema", "optimization", "migration"],
            AgentType.TEST_ENGINEER: ["testing", "qa", "automation", "validation"],
            AgentType.DEVOPS_ENGINEER: ["devops", "ci/cd", "docker", "deployment"],
            AgentType.SECURITY_AUDITOR: ["security", "audit", "vulnerability", "compliance"],
            AgentType.SYSTEM_ARCHITECT: ["architecture", "design", "scalability", "patterns"]
        }
        
        return capability_map.get(agent_type, ["general"])

    async def _initialize_daa(self, swarm_id: str, config: SwarmConfig):
        """Initialize Decentralized Autonomous Agents"""
        
        try:
            cmd = [
                *self.claude_flow_binary.split(),
                "daa", "init",
                "--enable-coordination", str(config.enable_coordination).lower(),
                "--enable-learning", str(config.enable_learning).lower(),
                "--persistence-mode", config.persistence_mode
            ]
            
            result = await self._execute_command(cmd)
            
            if result.returncode != 0:
                logger.warning(f"DAA initialization failed for swarm {swarm_id}: {result.stderr}")
            else:
                logger.info(f"DAA initialized for swarm {swarm_id}")
                
        except Exception as e:
            logger.warning(f"Could not initialize DAA for swarm {swarm_id}: {e}")

    async def _execute_hook(
        self,
        hook_type: str,
        message: str,
        memory_key: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        """Execute Claude Flow hooks for coordination"""
        
        try:
            if hook_type == "pre-task":
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "pre-task",
                    "--description", message
                ]
            elif hook_type == "post-task":
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "post-task",
                    "--task-id", "integration-task"
                ]
            elif hook_type == "post-edit" and memory_key:
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "post-edit",
                    "--memory-key", memory_key
                ]
                if file_path:
                    cmd.extend(["--file", file_path])
            elif hook_type == "notify":
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "notify",
                    "--message", message
                ]
            else:
                return  # Unsupported hook type
            
            await self._execute_command(cmd, timeout=10)
            
        except Exception as e:
            logger.debug(f"Hook execution failed ({hook_type}): {e}")

    async def _execute_command(
        self,
        cmd: List[str],
        timeout: int = 60,
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Execute shell command with timeout and error handling"""
        
        logger.debug(f"Executing command: {' '.join(cmd)}")
        
        try:
            if capture_output:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=process.returncode,
                    stdout=stdout.decode() if stdout else "",
                    stderr=stderr.decode() if stderr else ""
                )
            else:
                process = await asyncio.create_subprocess_exec(*cmd)
                returncode = await asyncio.wait_for(process.wait(), timeout=timeout)
                
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=returncode,
                    stdout="",
                    stderr=""
                )
                
        except asyncio.TimeoutError:
            logger.error(f"Command timeout after {timeout}s: {' '.join(cmd)}")
            raise ClaudeFlowError(f"Command timeout: {' '.join(cmd)}")
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise ClaudeFlowError(f"Command failed: {e}") from e


class ClaudeFlowOrchestrator:
    """
    Advanced Claude Flow workflow orchestration
    
    Features:
    - Complex workflow management and execution
    - Multi-agent coordination
    - Memory management and persistence
    - Real-time monitoring and progress tracking
    - Error handling and recovery
    - Performance optimization
    """
    
    def __init__(
        self,
        swarm_manager: Optional[SwarmManager] = None,
        memory_manager: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True
    ):
        self.swarm_manager = swarm_manager or SwarmManager()
        self.memory_manager = memory_manager or {}
        self.enable_caching = enable_caching
        
        # Workflow tracking
        self.active_workflows: Dict[str, WorkflowResult] = {}
        self.workflow_definitions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.execution_metrics: Dict[str, Any] = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_execution_time': 0.0
        }
        
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for workflow orchestration"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    async def orchestrate_development_workflow(
        self,
        project_spec: Dict[str, Any],
        workflow_config: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Orchestrate a comprehensive development workflow
        
        Args:
            project_spec: Project specification and requirements
            workflow_config: Optional workflow configuration
            
        Returns:
            WorkflowResult with execution details
        """
        workflow_id = f"workflow-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        logger.info(f"Starting development workflow {workflow_id}")
        
        try:
            # Execute hooks pre-task
            await self.swarm_manager._execute_hook(
                "pre-task", 
                f"Orchestrating development workflow {workflow_id}"
            )
            
            # Initialize swarm if needed
            swarm_id = await self._ensure_swarm_initialized(project_spec)
            
            # Generate workflow steps
            workflow_steps = await self._generate_workflow_steps(project_spec)
            
            # Create workflow result
            result = WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.RUNNING,
                steps_total=len(workflow_steps)
            )
            
            self.active_workflows[workflow_id] = result
            
            # Execute workflow steps
            for i, step in enumerate(workflow_steps):
                logger.info(f"Executing step {i+1}/{len(workflow_steps)}: {step.name}")
                
                step_result = await self._execute_workflow_step(
                    swarm_id, step, workflow_id
                )
                
                result.agent_results[step.id] = step_result
                result.steps_completed += 1
                
                if not step_result.get('success', False):
                    result.status = WorkflowStatus.FAILED
                    result.error_message = step_result.get('error', 'Step execution failed')
                    break
                    
                # Update memory with step results
                if step.memory_keys:
                    for key in step.memory_keys:
                        self.memory_manager[key] = step_result
            
            # Finalize result
            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.COMPLETED
            
            result.execution_time = time.time() - start_time
            
            # Execute hooks post-task
            await self.swarm_manager._execute_hook(
                "post-task",
                f"Workflow {workflow_id} completed with status {result.status.value}"
            )
            
            # Update metrics
            self._update_metrics(result)
            
            logger.info(
                f"Workflow {workflow_id} completed in {result.execution_time:.2f}s "
                f"with status {result.status.value}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
            
            self.active_workflows[workflow_id] = result
            self._update_metrics(result)
            
            raise WorkflowExecutionError(f"Workflow execution failed: {e}") from e

    async def _ensure_swarm_initialized(self, project_spec: Dict[str, Any]) -> str:
        """Ensure appropriate swarm is initialized for project"""
        
        # Determine optimal topology based on project complexity
        complexity_score = self._calculate_project_complexity(project_spec)
        
        if complexity_score < 30:
            topology = SwarmTopology.STAR  # Simple coordination
        elif complexity_score < 70:
            topology = SwarmTopology.HIERARCHICAL  # Structured coordination  
        else:
            topology = SwarmTopology.MESH  # Full coordination
        
        # Create swarm config
        config = SwarmConfig(
            topology=topology,
            max_agents=min(8, max(3, complexity_score // 20)),
            strategy="adaptive",
            enable_coordination=True,
            enable_learning=True
        )
        
        # Initialize swarm
        swarm_id = await self.swarm_manager.initialize_swarm(config)
        
        return swarm_id

    def _calculate_project_complexity(self, project_spec: Dict[str, Any]) -> int:
        """Calculate project complexity score for optimal topology selection"""
        
        complexity = 0
        
        # Feature complexity
        features = project_spec.get('features', [])
        complexity += len(features) * 10
        
        # Technology stack complexity
        tech_stack = project_spec.get('technology_stack', {})
        complexity += len(tech_stack) * 15
        
        # Integration requirements
        if project_spec.get('requires_database', False):
            complexity += 20
        if project_spec.get('requires_authentication', False):
            complexity += 15
        if project_spec.get('requires_api', False):
            complexity += 10
        if project_spec.get('requires_testing', False):
            complexity += 10
        
        # File count estimate
        estimated_files = project_spec.get('estimated_files', 10)
        complexity += min(estimated_files * 2, 40)
        
        return complexity

    async def _generate_workflow_steps(
        self,
        project_spec: Dict[str, Any]
    ) -> List[WorkflowStep]:
        """Generate workflow steps based on project specification"""
        
        steps = []
        
        # 1. Project Analysis and Planning
        steps.append(WorkflowStep(
            id="analyze-requirements",
            name="Analyze Requirements and Plan Architecture",
            agent_type=AgentType.SYSTEM_ARCHITECT,
            instructions=f"Analyze project requirements and create architecture plan: {project_spec.get('description', '')}",
            memory_keys=["workflow/architecture/plan"]
        ))
        
        # 2. Database Design (if needed)
        if project_spec.get('requires_database', False):
            steps.append(WorkflowStep(
                id="design-database",
                name="Design Database Schema",
                agent_type=AgentType.DATABASE_ARCHITECT,
                instructions="Design database schema based on requirements",
                dependencies=["analyze-requirements"],
                memory_keys=["workflow/database/schema"]
            ))
        
        # 3. Backend Development (if needed)
        if project_spec.get('requires_backend', True):
            steps.append(WorkflowStep(
                id="develop-backend",
                name="Develop Backend APIs",
                agent_type=AgentType.BACKEND_DEV,
                instructions="Implement backend APIs and business logic",
                dependencies=["analyze-requirements"] + (["design-database"] if project_spec.get('requires_database') else []),
                memory_keys=["workflow/backend/apis"]
            ))
        
        # 4. Frontend Development (if needed)  
        if project_spec.get('requires_frontend', True):
            steps.append(WorkflowStep(
                id="develop-frontend",
                name="Develop Frontend Interface",
                agent_type=AgentType.FRONTEND_DEV,
                instructions="Create user interface and frontend components",
                dependencies=["analyze-requirements"],
                memory_keys=["workflow/frontend/components"]
            ))
        
        # 5. Testing
        if project_spec.get('requires_testing', True):
            steps.append(WorkflowStep(
                id="create-tests",
                name="Create Comprehensive Tests",
                agent_type=AgentType.TEST_ENGINEER,
                instructions="Create unit, integration, and end-to-end tests",
                dependencies=[step.id for step in steps if step.agent_type in [AgentType.BACKEND_DEV, AgentType.FRONTEND_DEV]],
                memory_keys=["workflow/testing/suite"]
            ))
        
        # 6. Deployment Configuration
        steps.append(WorkflowStep(
            id="setup-deployment",
            name="Setup Deployment Configuration",
            agent_type=AgentType.DEVOPS_ENGINEER,
            instructions="Create deployment configuration and CI/CD pipeline",
            dependencies=[step.id for step in steps],
            memory_keys=["workflow/deployment/config"]
        ))
        
        # 7. Security Review
        steps.append(WorkflowStep(
            id="security-review",
            name="Perform Security Review",
            agent_type=AgentType.SECURITY_AUDITOR,
            instructions="Review code for security vulnerabilities and best practices",
            dependencies=[step.id for step in steps if step.agent_type != AgentType.DEVOPS_ENGINEER],
            memory_keys=["workflow/security/report"]
        ))
        
        return steps

    async def _execute_workflow_step(
        self,
        swarm_id: str,
        step: WorkflowStep,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Execute individual workflow step with agent"""
        
        try:
            # Spawn agent if needed
            agents = self.swarm_manager.swarm_agents.get(swarm_id, [])
            suitable_agent = None
            
            for agent in agents:
                if agent.type == step.agent_type and agent.status == "active":
                    suitable_agent = agent
                    break
            
            if not suitable_agent:
                suitable_agent = await self.swarm_manager.spawn_agent(
                    swarm_id, step.agent_type
                )
            
            # Execute step via Claude Flow task orchestration
            cmd = [
                *self.swarm_manager.claude_flow_binary.split(),
                "task", "orchestrate",
                "--task", step.instructions,
                "--priority", "high",
                "--strategy", "adaptive"
            ]
            
            result = await self.swarm_manager._execute_command(cmd, timeout=step.timeout)
            
            if result.returncode == 0:
                # Parse result
                try:
                    task_result = json.loads(result.stdout)
                except json.JSONDecodeError:
                    task_result = {"output": result.stdout, "success": True}
                
                # Update agent metrics
                suitable_agent.last_active = datetime.utcnow()
                suitable_agent.performance_metrics['tasks_completed'] = (
                    suitable_agent.performance_metrics.get('tasks_completed', 0) + 1
                )
                
                # Execute memory hooks
                for memory_key in step.memory_keys:
                    await self.swarm_manager._execute_hook(
                        "post-edit",
                        f"Step {step.id} completed",
                        memory_key=memory_key
                    )
                
                return {
                    'success': True,
                    'agent_id': suitable_agent.id,
                    'output': task_result,
                    'execution_time': time.time()
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr,
                    'agent_id': suitable_agent.id if suitable_agent else None
                }
                
        except Exception as e:
            logger.error(f"Workflow step {step.id} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': None
            }

    def _update_metrics(self, result: WorkflowResult):
        """Update execution metrics"""
        
        self.execution_metrics['total_workflows'] += 1
        
        if result.is_success:
            self.execution_metrics['successful_workflows'] += 1
        else:
            self.execution_metrics['failed_workflows'] += 1
        
        # Update average execution time
        total_time = (
            self.execution_metrics['average_execution_time'] * 
            (self.execution_metrics['total_workflows'] - 1) +
            result.execution_time
        )
        self.execution_metrics['average_execution_time'] = (
            total_time / self.execution_metrics['total_workflows']
        )

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get current workflow status"""
        return self.active_workflows.get(workflow_id)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel running workflow"""
        
        if workflow_id in self.active_workflows:
            result = self.active_workflows[workflow_id]
            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.CANCELLED
                return True
        
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestration performance metrics"""
        
        success_rate = 0.0
        if self.execution_metrics['total_workflows'] > 0:
            success_rate = (
                self.execution_metrics['successful_workflows'] / 
                self.execution_metrics['total_workflows']
            )
        
        return {
            **self.execution_metrics,
            'success_rate': success_rate,
            'active_workflows': len([
                w for w in self.active_workflows.values() 
                if w.status == WorkflowStatus.RUNNING
            ])
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check health of Claude Flow orchestration"""
        
        try:
            # Check swarm manager health
            swarm_status = await self.swarm_manager._execute_command(
                [*self.swarm_manager.claude_flow_binary.split(), "swarm", "status"],
                timeout=10
            )
            
            return {
                'status': 'healthy' if swarm_status.returncode == 0 else 'degraded',
                'active_workflows': len(self.active_workflows),
                'active_swarms': len(self.swarm_manager.active_swarms),
                'total_agents': sum(
                    len(agents) for agents in self.swarm_manager.swarm_agents.values()
                ),
                'metrics': self.get_metrics(),
                'claude_flow_status': swarm_status.stdout if swarm_status.returncode == 0 else swarm_status.stderr
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }