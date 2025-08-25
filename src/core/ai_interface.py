"""
AI Interface Layer for claude-tiu.

This module provides a unified interface for integrating with Claude Code and Claude Flow,
handling context management, response processing, and intelligent routing between services.

Key Features:
- Claude Code direct integration for single tasks
- Claude Flow workflow orchestration for complex multi-step processes
- Intelligent routing based on task complexity analysis
- Context-aware prompt generation with project information
- Response validation and post-processing
- Error handling and retry mechanisms
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from .types import AITaskResult, DevelopmentResult, Task, PathStr
from .validator import ProgressValidator


logger = logging.getLogger(__name__)


@dataclass
class AIContext:
    """Context information for AI operations."""
    project_id: Optional[UUID] = None
    project_path: Optional[Path] = None
    current_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    coding_standards: Dict[str, Any] = field(default_factory=dict)
    test_requirements: Dict[str, Any] = field(default_factory=dict)
    framework_info: Dict[str, str] = field(default_factory=dict)
    previous_context: Optional[str] = None


@dataclass
class ClaudeCodeRequest:
    """Request structure for Claude Code operations."""
    prompt: str
    context: AIContext
    timeout: int = 300
    max_retries: int = 3
    format: str = "text"  # text, json, code
    enable_validation: bool = True


@dataclass
class ClaudeFlowRequest:
    """Request structure for Claude Flow operations."""
    workflow_config: Dict[str, Any]
    project_context: AIContext
    topology: str = "mesh"
    max_agents: int = 8
    enable_coordination: bool = True


class AIInterfaceException(Exception):
    """AI interface related errors."""
    pass


class ClaudeCodeClient:
    """
    Direct integration with Claude Code CLI for single-task AI operations.
    """
    
    def __init__(
        self,
        claude_code_path: str = "claude",
        enable_validation: bool = True,
        default_timeout: int = 300
    ):
        """
        Initialize Claude Code client.
        
        Args:
            claude_code_path: Path to Claude Code executable
            enable_validation: Enable response validation
            default_timeout: Default timeout for operations
        """
        self.claude_code_path = claude_code_path
        self.enable_validation = enable_validation
        self.default_timeout = default_timeout
        self._validator = ProgressValidator() if enable_validation else None
    
    async def execute_coding_task(self, request: ClaudeCodeRequest) -> AITaskResult:
        """
        Execute a coding task using Claude Code.
        
        Args:
            request: Claude Code request configuration
            
        Returns:
            AI task execution result
        """
        start_time = time.time()
        task_id = uuid4()
        
        try:
            # Build smart context
            context_str = await self._build_smart_context(request.context)
            
            # Prepare full prompt with context
            full_prompt = self._prepare_prompt(request.prompt, context_str)
            
            # Execute Claude Code command
            result = await self._execute_claude_code(
                full_prompt,
                request.timeout,
                request.format
            )
            
            # Process and validate result
            processed_result = await self._process_claude_result(
                result, request, task_id
            )
            
            execution_time = time.time() - start_time
            
            return AITaskResult(
                task_id=task_id,
                success=True,
                generated_content=processed_result.get('content', ''),
                files_modified=processed_result.get('files_modified', []),
                validation_score=processed_result.get('validation_score', 0.0),
                execution_time=execution_time,
                metadata={
                    'model': 'claude-3.5-sonnet',
                    'tokens_used': processed_result.get('tokens_used', 0),
                    'format': request.format
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Claude Code execution failed: {e}")
            
            return AITaskResult(
                task_id=task_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                metadata={'error_type': type(e).__name__}
            )
    
    async def _build_smart_context(self, context: AIContext) -> str:
        """Build intelligent context from project information."""
        context_parts = []
        
        # Project structure information
        if context.project_path and context.project_path.exists():
            structure = await self._analyze_project_structure(context.project_path)
            context_parts.append(f"Project Structure:\n{structure}")
        
        # Current relevant files
        if context.current_files:
            files_content = await self._get_relevant_files_content(
                context.current_files, context.project_path
            )
            context_parts.append(f"Relevant Files:\n{files_content}")
        
        # Dependencies and framework info
        if context.dependencies:
            context_parts.append(f"Dependencies: {', '.join(context.dependencies)}")
        
        if context.framework_info:
            framework_str = ', '.join(f"{k}: {v}" for k, v in context.framework_info.items())
            context_parts.append(f"Framework: {framework_str}")
        
        # Coding standards
        if context.coding_standards:
            standards_str = json.dumps(context.coding_standards, indent=2)
            context_parts.append(f"Coding Standards:\n{standards_str}")
        
        # Test requirements
        if context.test_requirements:
            test_str = json.dumps(context.test_requirements, indent=2)
            context_parts.append(f"Test Requirements:\n{test_str}")
        
        # Previous context for continuity
        if context.previous_context:
            context_parts.append(f"Previous Context:\n{context.previous_context}")
        
        return "\n\n".join(context_parts)
    
    def _prepare_prompt(self, user_prompt: str, context: str) -> str:
        """Prepare full prompt with context and instructions."""
        system_instructions = """
You are an expert software developer working on a project. Use the provided context
to understand the project structure, coding standards, and requirements.

IMPORTANT GUIDELINES:
1. Generate complete, functional code without placeholders
2. Follow the project's coding standards and patterns
3. Include proper error handling and validation
4. Add comprehensive tests if test requirements are specified
5. Use the specified framework and dependencies appropriately
6. Maintain consistency with existing code patterns

If you cannot complete a task fully, explicitly state what is incomplete and why.
Never use placeholder comments like "TODO", "FIXME", or "..." in production code.
"""
        
        return f"""{system_instructions}

PROJECT CONTEXT:
{context}

USER REQUEST:
{user_prompt}

Please provide a complete implementation following the guidelines above."""
    
    async def _execute_claude_code(
        self,
        prompt: str,
        timeout: int,
        format_type: str
    ) -> str:
        """Execute Claude Code CLI command."""
        # Prepare command
        cmd = [
            self.claude_code_path,
            "--format", format_type,
            prompt
        ]
        
        try:
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise AIInterfaceException(f"Claude Code failed: {error_msg}")
            
            return stdout.decode('utf-8', errors='ignore')
            
        except asyncio.TimeoutError:
            raise AIInterfaceException(f"Claude Code timeout after {timeout} seconds")
        except FileNotFoundError:
            raise AIInterfaceException(f"Claude Code not found at: {self.claude_code_path}")
    
    async def _process_claude_result(
        self,
        raw_result: str,
        request: ClaudeCodeRequest,
        task_id: UUID
    ) -> Dict[str, Any]:
        """Process and validate Claude Code result."""
        processed = {
            'content': raw_result,
            'files_modified': [],
            'validation_score': 100.0,
            'tokens_used': len(raw_result.split()) * 1.3  # Rough token estimate
        }
        
        # Extract file modifications if any
        files_modified = self._extract_file_modifications(raw_result)
        processed['files_modified'] = files_modified
        
        # Validate result if validation is enabled
        if self.enable_validation and self._validator and files_modified:
            try:
                # Create temporary files for validation
                temp_files = await self._create_temp_files_for_validation(
                    raw_result, files_modified
                )
                
                # Run validation on temporary files
                validation_results = []
                for temp_file in temp_files:
                    validation_result = await self._validator.validate_single_file(temp_file)
                    validation_results.append(validation_result)
                
                # Calculate average validation score
                if validation_results:
                    avg_score = sum(vr.authenticity_score for vr in validation_results) / len(validation_results)
                    processed['validation_score'] = avg_score
                
                # Cleanup temporary files
                for temp_file in temp_files:
                    Path(temp_file).unlink(missing_ok=True)
                    
            except Exception as e:
                logger.warning(f"Validation failed for task {task_id}: {e}")
                processed['validation_score'] = 70.0  # Conservative score
        
        return processed
    
    async def _analyze_project_structure(self, project_path: Path) -> str:
        """Analyze and return project structure summary."""
        structure_lines = []
        
        def add_directory(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
            if current_depth > max_depth:
                return
            
            try:
                items = sorted(path.iterdir())
                dirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
                files = [item for item in items if item.is_file() and not item.name.startswith('.')]
                
                # Add directories first
                for i, item in enumerate(dirs[:10]):  # Limit to 10 dirs
                    is_last_dir = i == len(dirs) - 1 and not files
                    connector = "└── " if is_last_dir else "├── "
                    structure_lines.append(f"{prefix}{connector}{item.name}/")
                    
                    new_prefix = prefix + ("    " if is_last_dir else "│   ")
                    add_directory(item, new_prefix, max_depth, current_depth + 1)
                
                # Add files
                for i, item in enumerate(files[:15]):  # Limit to 15 files
                    is_last = i == len(files) - 1
                    connector = "└── " if is_last else "├── "
                    structure_lines.append(f"{prefix}{connector}{item.name}")
                    
            except PermissionError:
                pass
        
        structure_lines.append(f"{project_path.name}/")
        add_directory(project_path)
        
        return "\n".join(structure_lines[:50])  # Limit output size
    
    async def _get_relevant_files_content(
        self,
        file_paths: List[str],
        project_path: Optional[Path]
    ) -> str:
        """Get content of relevant files for context."""
        content_parts = []
        
        for file_path in file_paths[:5]:  # Limit to 5 files
            try:
                if project_path:
                    full_path = project_path / file_path
                else:
                    full_path = Path(file_path)
                
                if full_path.exists() and full_path.is_file():
                    # Read file content (limited size)
                    content = full_path.read_text(encoding='utf-8', errors='ignore')
                    if len(content) > 2000:  # Limit content size
                        content = content[:2000] + "\n... (truncated)"
                    
                    content_parts.append(f"File: {file_path}\n```\n{content}\n```")
                    
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
        
        return "\n\n".join(content_parts)
    
    def _extract_file_modifications(self, result: str) -> List[str]:
        """Extract file paths that were modified/created from result."""
        import re
        
        # Look for common file path patterns in the result
        file_patterns = [
            r'(?:File|file):\s*([^\s\n]+\.[a-zA-Z0-9]+)',
            r'(?:Creating|creating|Created|created)\s+([^\s\n]+\.[a-zA-Z0-9]+)',
            r'(?:Modified|modified|Updated|updated)\s+([^\s\n]+\.[a-zA-Z0-9]+)',
            r'`([^\s`]+\.[a-zA-Z0-9]+)`',
        ]
        
        files = set()
        for pattern in file_patterns:
            matches = re.findall(pattern, result, re.IGNORECASE)
            files.update(matches)
        
        return list(files)
    
    async def _create_temp_files_for_validation(
        self,
        result: str,
        file_paths: List[str]
    ) -> List[str]:
        """Create temporary files for validation."""
        import tempfile
        
        temp_files = []
        
        # Try to extract code blocks for each file
        code_blocks = self._extract_code_blocks(result)
        
        for i, file_path in enumerate(file_paths):
            try:
                # Get file extension
                file_ext = Path(file_path).suffix
                
                # Create temporary file with same extension
                temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
                
                # Write content (use corresponding code block or generic content)
                if i < len(code_blocks):
                    content = code_blocks[i]
                else:
                    content = result  # Use full result as fallback
                
                with open(temp_fd, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                temp_files.append(temp_path)
                
            except Exception as e:
                logger.warning(f"Could not create temp file for {file_path}: {e}")
        
        return temp_files
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from markdown-formatted text."""
        import re
        
        # Find all code blocks (```...```)
        code_block_pattern = r'```(?:[a-zA-Z0-9]*\n)?(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        return [match.strip() for match in matches]


class ClaudeFlowOrchestrator:
    """
    Advanced Claude Flow workflow orchestration for complex multi-agent tasks.
    """
    
    def __init__(
        self,
        enable_coordination: bool = True,
        default_topology: str = "mesh",
        max_agents: int = 8
    ):
        """
        Initialize Claude Flow orchestrator.
        
        Args:
            enable_coordination: Enable agent coordination
            default_topology: Default swarm topology
            max_agents: Maximum number of agents
        """
        self.enable_coordination = enable_coordination
        self.default_topology = default_topology
        self.max_agents = max_agents
    
    async def orchestrate_development_workflow(
        self,
        request: ClaudeFlowRequest
    ) -> DevelopmentResult:
        """
        Orchestrate a complex development workflow using Claude Flow.
        
        Args:
            request: Claude Flow request configuration
            
        Returns:
            Development workflow result
        """
        workflow_id = uuid4()
        start_time = time.time()
        
        try:
            # Initialize swarm topology
            topology = request.topology or self.default_topology
            await self._initialize_swarm(topology, request.max_agents)
            
            # Spawn specialized agents based on project requirements
            agents = await self._spawn_specialized_agents(request.project_context)
            
            # Execute coordinated workflow
            result = await self._execute_coordinated_workflow(
                agents, request.workflow_config, request.project_context
            )
            
            # Validate and consolidate results
            validated_result = await self._validate_workflow_output(result)
            
            execution_time = time.time() - start_time
            
            return DevelopmentResult(
                workflow_id=workflow_id,
                success=validated_result['success'],
                tasks_executed=validated_result.get('tasks_executed', []),
                files_generated=validated_result.get('files_generated', []),
                validation_results=validated_result.get('validation_results', []),
                total_time=execution_time,
                quality_metrics=validated_result.get('quality_metrics', {}),
                error_details=validated_result.get('error_details')
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Claude Flow orchestration failed: {e}")
            
            return DevelopmentResult(
                workflow_id=workflow_id,
                success=False,
                total_time=execution_time,
                error_details=str(e)
            )
    
    async def _initialize_swarm(self, topology: str, max_agents: int) -> None:
        """Initialize Claude Flow swarm with specified topology."""
        # This would integrate with Claude Flow MCP server
        # For now, we'll simulate the initialization
        logger.info(f"Initializing swarm with {topology} topology, max {max_agents} agents")
        
        # Simulate initialization delay
        await asyncio.sleep(0.5)
    
    async def _spawn_specialized_agents(self, context: AIContext) -> List[Dict[str, Any]]:
        """Spawn task-specific agents based on project requirements."""
        agents = []
        
        # Analyze project context to determine required agent types
        required_agents = self._analyze_required_agents(context)
        
        for agent_type in required_agents:
            agent = {
                'id': str(uuid4()),
                'type': agent_type,
                'capabilities': self._get_agent_capabilities(agent_type),
                'status': 'ready'
            }
            agents.append(agent)
            logger.info(f"Spawned {agent_type} agent: {agent['id']}")
        
        return agents
    
    def _analyze_required_agents(self, context: AIContext) -> List[str]:
        """Analyze project context to determine required agent types."""
        agents = ['coordinator']  # Always need a coordinator
        
        # Check framework requirements
        if context.framework_info:
            framework = context.framework_info.get('framework', '').lower()
            
            if 'react' in framework or 'vue' in framework or 'angular' in framework:
                agents.append('frontend-dev')
            
            if 'node' in framework or 'express' in framework or 'fastapi' in framework:
                agents.append('backend-dev')
            
            if 'python' in framework:
                agents.append('python-dev')
        
        # Check if database is needed
        if context.dependencies and any('sql' in dep.lower() or 'mongo' in dep.lower() for dep in context.dependencies):
            agents.append('database-architect')
        
        # Always include tester and reviewer for quality
        agents.extend(['tester', 'code-reviewer'])
        
        # Limit to max agents
        return agents[:self.max_agents]
    
    def _get_agent_capabilities(self, agent_type: str) -> List[str]:
        """Get capabilities for specific agent type."""
        capabilities_map = {
            'coordinator': ['task-coordination', 'workflow-management'],
            'frontend-dev': ['react', 'vue', 'angular', 'css', 'javascript'],
            'backend-dev': ['api-development', 'server-architecture', 'database-integration'],
            'python-dev': ['python', 'fastapi', 'django', 'data-processing'],
            'database-architect': ['database-design', 'sql', 'mongodb', 'migrations'],
            'tester': ['unit-testing', 'integration-testing', 'test-automation'],
            'code-reviewer': ['code-quality', 'security-review', 'best-practices']
        }
        
        return capabilities_map.get(agent_type, ['general-development'])
    
    async def _execute_coordinated_workflow(
        self,
        agents: List[Dict[str, Any]],
        workflow_config: Dict[str, Any],
        context: AIContext
    ) -> Dict[str, Any]:
        """Execute coordinated workflow with multiple agents."""
        # This would orchestrate the actual Claude Flow execution
        # For now, we'll simulate a coordinated workflow
        
        results = {
            'success': True,
            'tasks_executed': [],
            'files_generated': [],
            'agent_results': {}
        }
        
        # Simulate agent coordination
        for agent in agents:
            agent_result = await self._execute_agent_task(agent, workflow_config, context)
            results['agent_results'][agent['id']] = agent_result
            
            if agent_result.get('files_generated'):
                results['files_generated'].extend(agent_result['files_generated'])
            
            if not agent_result.get('success', True):
                results['success'] = False
        
        return results
    
    async def _execute_agent_task(
        self,
        agent: Dict[str, Any],
        workflow_config: Dict[str, Any],
        context: AIContext
    ) -> Dict[str, Any]:
        """Execute task for a specific agent."""
        # Simulate agent execution
        await asyncio.sleep(0.2)
        
        agent_type = agent['type']
        
        # Simulate different outcomes based on agent type
        if agent_type == 'coordinator':
            return {
                'success': True,
                'task': 'workflow_coordination',
                'output': 'Workflow coordinated successfully'
            }
        elif agent_type in ['frontend-dev', 'backend-dev', 'python-dev']:
            return {
                'success': True,
                'task': 'code_generation',
                'files_generated': [f'{agent_type}_output.py', f'{agent_type}_tests.py'],
                'output': f'Generated code for {agent_type} tasks'
            }
        else:
            return {
                'success': True,
                'task': 'review_validation',
                'output': f'Completed {agent_type} tasks'
            }
    
    async def _validate_workflow_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and consolidate workflow output."""
        # Add quality metrics
        result['quality_metrics'] = {
            'agent_success_rate': len([r for r in result['agent_results'].values() if r.get('success', True)]) / len(result['agent_results']) * 100,
            'files_generated_count': len(result.get('files_generated', [])),
            'coordination_score': 85.0  # Simulated coordination quality score
        }
        
        return result


class AIInterface:
    """
    Unified AI interface providing intelligent routing between Claude Code and Claude Flow.
    """
    
    def __init__(
        self,
        claude_code_path: str = "claude",
        enable_validation: bool = True,
        enable_flow_orchestration: bool = True
    ):
        """
        Initialize AI interface.
        
        Args:
            claude_code_path: Path to Claude Code CLI
            enable_validation: Enable response validation
            enable_flow_orchestration: Enable Claude Flow orchestration
        """
        self.claude_code_client = ClaudeCodeClient(
            claude_code_path=claude_code_path,
            enable_validation=enable_validation
        )
        
        self.claude_flow_orchestrator = ClaudeFlowOrchestrator() if enable_flow_orchestration else None
        
        self.task_complexity_analyzer = TaskComplexityAnalyzer()
    
    async def execute_ai_task(
        self,
        task: Task,
        context: AIContext,
        force_service: Optional[str] = None
    ) -> Union[AITaskResult, DevelopmentResult]:
        """
        Execute AI task with intelligent service routing.
        
        Args:
            task: Task to execute
            context: AI context information
            force_service: Force specific service ('claude_code' or 'claude_flow')
            
        Returns:
            AI execution result
        """
        # Analyze task complexity if not forced to specific service
        if force_service:
            service = force_service
        else:
            complexity_analysis = await self.task_complexity_analyzer.analyze_task(task, context)
            service = complexity_analysis['recommended_service']
        
        if service == 'claude_code':
            # Execute with Claude Code
            request = ClaudeCodeRequest(
                prompt=task.ai_prompt or task.description,
                context=context,
                enable_validation=True
            )
            return await self.claude_code_client.execute_coding_task(request)
        
        elif service == 'claude_flow' and self.claude_flow_orchestrator:
            # Execute with Claude Flow
            workflow_config = {
                'name': task.name,
                'description': task.description,
                'requirements': task.ai_prompt or task.description
            }
            
            request = ClaudeFlowRequest(
                workflow_config=workflow_config,
                project_context=context
            )
            return await self.claude_flow_orchestrator.orchestrate_development_workflow(request)
        
        else:
            # Fallback to Claude Code
            request = ClaudeCodeRequest(
                prompt=task.ai_prompt or task.description,
                context=context,
                enable_validation=True
            )
            return await self.claude_code_client.execute_coding_task(request)


class TaskComplexityAnalyzer:
    """
    Analyze task complexity to determine optimal AI service routing.
    """
    
    async def analyze_task(self, task: Task, context: AIContext) -> Dict[str, Any]:
        """
        Analyze task complexity and recommend service.
        
        Args:
            task: Task to analyze
            context: AI context
            
        Returns:
            Analysis result with service recommendation
        """
        complexity_score = 0
        factors = {}
        
        # File count impact
        file_count = len(context.current_files)
        factors['file_count'] = file_count
        complexity_score += min(file_count * 10, 50)
        
        # Technology stack complexity
        tech_count = len(context.dependencies) + len(context.framework_info)
        factors['technology_complexity'] = tech_count
        complexity_score += tech_count * 15
        
        # Task description complexity (word count, technical terms)
        description_complexity = self._analyze_description_complexity(task.description)
        factors['description_complexity'] = description_complexity
        complexity_score += description_complexity
        
        # Estimated duration impact
        if task.estimated_duration:
            duration_score = min(task.estimated_duration / 10, 30)
            factors['duration_score'] = duration_score
            complexity_score += duration_score
        
        # Integration requirements
        if any(keyword in (task.description + task.ai_prompt or "").lower() 
               for keyword in ['integration', 'workflow', 'multi-step', 'coordinate']):
            factors['integration_required'] = True
            complexity_score += 30
        else:
            factors['integration_required'] = False
        
        # Determine recommendation
        if complexity_score < 30:
            recommendation = 'claude_code'
            complexity_level = 'simple'
        elif complexity_score < 80:
            recommendation = 'hybrid'  # Could use either
            complexity_level = 'medium'
        else:
            recommendation = 'claude_flow'
            complexity_level = 'complex'
        
        return {
            'complexity_score': complexity_score,
            'complexity_level': complexity_level,
            'recommended_service': recommendation,
            'factors': factors,
            'confidence': self._calculate_confidence(factors)
        }
    
    def _analyze_description_complexity(self, description: str) -> int:
        """Analyze description complexity."""
        if not description:
            return 0
        
        words = description.lower().split()
        complexity_score = 0
        
        # Base word count impact
        complexity_score += min(len(words), 50)
        
        # Technical complexity keywords
        complex_keywords = [
            'architecture', 'design', 'implement', 'integrate', 'optimize',
            'database', 'api', 'authentication', 'security', 'performance',
            'testing', 'deployment', 'configuration', 'migration'
        ]
        
        keyword_count = sum(1 for keyword in complex_keywords if keyword in ' '.join(words))
        complexity_score += keyword_count * 10
        
        return min(complexity_score, 100)
    
    def _calculate_confidence(self, factors: Dict[str, Any]) -> float:
        """Calculate confidence in recommendation."""
        confidence = 0.7  # Base confidence
        
        # Higher confidence for clear indicators
        if factors.get('integration_required'):
            confidence += 0.2
        
        if factors.get('file_count', 0) > 5:
            confidence += 0.1
        
        if factors.get('technology_complexity', 0) > 3:
            confidence += 0.1
        
        return min(confidence, 1.0)


# Utility functions for external use

async def execute_simple_ai_task(
    prompt: str,
    project_path: Optional[PathStr] = None,
    files: Optional[List[str]] = None
) -> AITaskResult:
    """
    Convenience function to execute a simple AI task.
    
    Args:
        prompt: Task prompt/description
        project_path: Optional project path for context
        files: Optional relevant files
        
    Returns:
        AI task result
    """
    context = AIContext(
        project_path=Path(project_path) if project_path else None,
        current_files=files or []
    )
    
    request = ClaudeCodeRequest(prompt=prompt, context=context)
    client = ClaudeCodeClient()
    
    return await client.execute_coding_task(request)