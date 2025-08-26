"""
Automatic Programming Pipeline - Core system that transforms natural language requirements 
into production-ready code using Claude Code direct CLI and Claude Flow orchestration.

Implements:
1. Requirements analysis and task decomposition
2. Multi-agent coordination via Claude Flow
3. Code generation via Claude Code CLI
4. Validation and quality assurance
5. File-based communication and dependency management
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.models.project import Project
from src.claude_tui.models.ai_models import (
    CodeResult, WorkflowRequest, WorkflowResult, 
    SwarmConfig, AgentConfig, TaskOrchestrationRequest
)
from src.claude_tui.integrations.claude_code_client import ClaudeCodeClient
from src.claude_tui.integrations.claude_flow_client import ClaudeFlowClient
from src.claude_tui.utils.file_system import FileSystemManager
from src.claude_tui.core.logger import get_logger
from src.claude_tui.validation.real_time_validator import RealTimeValidator

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    INITIALIZING = "initializing"
    ANALYZING_REQUIREMENTS = "analyzing_requirements"
    DECOMPOSING_TASKS = "decomposing_tasks"
    COORDINATING_AGENTS = "coordinating_agents"
    GENERATING_CODE = "generating_code"
    VALIDATING_RESULTS = "validating_results"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskComponent:
    """Represents a decomposed task component."""
    id: str
    description: str
    file_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    agent_type: str = "coder"
    priority: int = 5
    estimated_complexity: int = 3  # 1-10 scale
    requirements: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequirementsAnalysis:
    """Results from requirements analysis."""
    original_requirements: str
    parsed_requirements: Dict[str, Any]
    project_type: str
    estimated_complexity: int
    recommended_architecture: str
    suggested_technologies: List[str]
    quality_attributes: List[str]
    constraints: List[str]
    assumptions: List[str]
    risks: List[str]


@dataclass
class PipelineResult:
    """Final result from automatic programming pipeline."""
    success: bool
    generated_files: List[str]
    execution_time: float
    requirements_analysis: Optional[RequirementsAnalysis]
    task_components: List[TaskComponent]
    validation_results: Dict[str, Any]
    agent_reports: Dict[str, Dict[str, Any]]
    error_message: Optional[str] = None
    quality_score: float = 0.0
    coverage_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class RequirementsAnalyzer:
    """Analyzes natural language requirements and extracts actionable information."""
    
    def __init__(self, claude_code_client: ClaudeCodeClient):
        self.claude_code_client = claude_code_client
        self.logger = get_logger(__name__)
    
    async def analyze(self, requirements: str, project_path: Optional[str] = None) -> RequirementsAnalysis:
        """
        Analyze natural language requirements.
        
        Args:
            requirements: Natural language requirements
            project_path: Optional project context path
            
        Returns:
            RequirementsAnalysis: Structured analysis results
        """
        self.logger.info("Analyzing requirements with AI assistance")
        
        context = {
            'task_type': 'requirements_analysis',
            'requirements': requirements,
            'project_path': project_path,
            'analysis_framework': 'SPARC',  # Specification, Pseudocode, Architecture, Refinement, Completion
            'output_format': 'structured_analysis'
        }
        
        analysis_prompt = f"""
        Analyze these software requirements and provide a structured analysis:
        
        Requirements: {requirements}
        
        Please analyze and extract:
        1. Project Type (web_app, api, desktop_app, mobile_app, library, etc.)
        2. Estimated Complexity (1-10 scale)
        3. Recommended Architecture Pattern
        4. Suggested Technologies and Frameworks
        5. Quality Attributes (performance, security, scalability, etc.)
        6. Constraints and Limitations
        7. Assumptions Made
        8. Potential Risks
        9. Core Features List
        10. Non-functional Requirements
        
        Format as JSON with clear structure for automated processing.
        """
        
        try:
            response = await self.claude_code_client.execute_task(
                task_description=analysis_prompt,
                context=context
            )
            
            if response.get('success', False):
                # Parse AI response into structured format
                analysis_data = response.get('analysis', {})
                
                return RequirementsAnalysis(
                    original_requirements=requirements,
                    parsed_requirements=analysis_data.get('core_features', {}),
                    project_type=analysis_data.get('project_type', 'unknown'),
                    estimated_complexity=int(analysis_data.get('complexity_score', 5)),
                    recommended_architecture=analysis_data.get('architecture_pattern', 'layered'),
                    suggested_technologies=analysis_data.get('technologies', []),
                    quality_attributes=analysis_data.get('quality_attributes', []),
                    constraints=analysis_data.get('constraints', []),
                    assumptions=analysis_data.get('assumptions', []),
                    risks=analysis_data.get('risks', [])
                )
            else:
                # Fallback analysis
                return self._create_fallback_analysis(requirements)
                
        except Exception as e:
            self.logger.error(f"Requirements analysis failed: {e}")
            return self._create_fallback_analysis(requirements)
    
    def _create_fallback_analysis(self, requirements: str) -> RequirementsAnalysis:
        """Create basic fallback analysis when AI analysis fails."""
        return RequirementsAnalysis(
            original_requirements=requirements,
            parsed_requirements={'description': requirements},
            project_type='general',
            estimated_complexity=5,
            recommended_architecture='modular',
            suggested_technologies=['python'],
            quality_attributes=['maintainability'],
            constraints=['time', 'resources'],
            assumptions=['standard_environment'],
            risks=['complexity_underestimation']
        )


class TaskDecomposer:
    """Decomposes complex programming tasks into manageable components."""
    
    def __init__(self, claude_code_client: ClaudeCodeClient):
        self.claude_code_client = claude_code_client
        self.logger = get_logger(__name__)
    
    async def decompose(
        self, 
        requirements_analysis: RequirementsAnalysis,
        project_path: Optional[str] = None
    ) -> List[TaskComponent]:
        """
        Decompose requirements into actionable task components.
        
        Args:
            requirements_analysis: Analyzed requirements
            project_path: Target project path
            
        Returns:
            List[TaskComponent]: Decomposed task components
        """
        self.logger.info("Decomposing requirements into task components")
        
        context = {
            'task_type': 'task_decomposition',
            'requirements_analysis': {
                'project_type': requirements_analysis.project_type,
                'complexity': requirements_analysis.estimated_complexity,
                'architecture': requirements_analysis.recommended_architecture,
                'technologies': requirements_analysis.suggested_technologies
            },
            'project_path': project_path,
            'decomposition_strategy': 'dependency_aware'
        }
        
        decomposition_prompt = f"""
        Decompose this software project into specific, actionable task components:
        
        Project Type: {requirements_analysis.project_type}
        Architecture: {requirements_analysis.recommended_architecture}
        Technologies: {', '.join(requirements_analysis.suggested_technologies)}
        Requirements: {requirements_analysis.original_requirements}
        
        Create task components that:
        1. Are specific and actionable
        2. Have clear file paths and locations
        3. Identify dependencies between tasks
        4. Specify appropriate agent types (coder, tester, reviewer, etc.)
        5. Include validation criteria
        6. Are ordered by priority and dependencies
        
        Format as JSON array of task components.
        """
        
        try:
            response = await self.claude_code_client.execute_task(
                task_description=decomposition_prompt,
                context=context
            )
            
            if response.get('success', False):
                task_data = response.get('tasks', [])
                return self._parse_task_components(task_data, requirements_analysis)
            else:
                return self._create_fallback_tasks(requirements_analysis)
                
        except Exception as e:
            self.logger.error(f"Task decomposition failed: {e}")
            return self._create_fallback_tasks(requirements_analysis)
    
    def _parse_task_components(
        self, 
        task_data: List[Dict], 
        requirements_analysis: RequirementsAnalysis
    ) -> List[TaskComponent]:
        """Parse AI-generated task data into TaskComponent objects."""
        components = []
        
        for i, task in enumerate(task_data):
            component = TaskComponent(
                id=task.get('id', f"task_{i}"),
                description=task.get('description', f"Task {i}"),
                file_path=task.get('file_path'),
                dependencies=task.get('dependencies', []),
                agent_type=task.get('agent_type', 'coder'),
                priority=task.get('priority', 5),
                estimated_complexity=task.get('complexity', 3),
                requirements=task.get('requirements', []),
                validation_criteria=task.get('validation_criteria', []),
                context=task.get('context', {})
            )
            components.append(component)
        
        return components
    
    def _create_fallback_tasks(self, requirements_analysis: RequirementsAnalysis) -> List[TaskComponent]:
        """Create basic fallback tasks when decomposition fails."""
        base_tasks = [
            TaskComponent(
                id="setup_project",
                description="Set up project structure and configuration",
                file_path="main.py",
                agent_type="coder",
                priority=1,
                estimated_complexity=2
            ),
            TaskComponent(
                id="implement_core",
                description="Implement core functionality",
                file_path="core/main.py",
                dependencies=["setup_project"],
                agent_type="coder",
                priority=5,
                estimated_complexity=requirements_analysis.estimated_complexity
            ),
            TaskComponent(
                id="add_tests",
                description="Add comprehensive tests",
                file_path="tests/test_main.py",
                dependencies=["implement_core"],
                agent_type="tester",
                priority=7,
                estimated_complexity=3
            )
        ]
        
        return base_tasks


class CodeGenerator:
    """Coordinates code generation across multiple agents."""
    
    def __init__(
        self, 
        claude_code_client: ClaudeCodeClient, 
        claude_flow_client: ClaudeFlowClient
    ):
        self.claude_code_client = claude_code_client
        self.claude_flow_client = claude_flow_client
        self.logger = get_logger(__name__)
        self._active_swarm: Optional[str] = None
    
    async def generate_code(
        self, 
        task_components: List[TaskComponent],
        project_path: str,
        requirements_analysis: RequirementsAnalysis
    ) -> Dict[str, Any]:
        """
        Generate code for all task components using coordinated agents.
        
        Args:
            task_components: List of task components to implement
            project_path: Target project path
            requirements_analysis: Original requirements analysis
            
        Returns:
            Dict containing generation results and agent reports
        """
        self.logger.info(f"Generating code for {len(task_components)} components")
        
        try:
            # Initialize swarm for coordination
            swarm_config = SwarmConfig(
                topology="mesh",
                max_agents=min(len(task_components), 8),  # Cap at 8 agents
                strategy="collaborative"
            )
            
            self._active_swarm = await self.claude_flow_client.initialize_swarm(swarm_config)
            
            # Group tasks by dependencies and priority
            execution_groups = self._group_tasks_by_dependencies(task_components)
            
            generation_results = {
                'generated_files': [],
                'agent_reports': {},
                'execution_time': 0.0,
                'success': True
            }
            
            start_time = time.time()
            
            # Execute task groups sequentially (dependencies) but tasks within groups in parallel
            for group_index, task_group in enumerate(execution_groups):
                self.logger.info(f"Executing task group {group_index + 1}/{len(execution_groups)}")
                
                # Execute tasks in this group in parallel
                group_tasks = []
                for task_component in task_group:
                    group_tasks.append(
                        self._execute_task_component(task_component, project_path, requirements_analysis)
                    )
                
                # Wait for all tasks in group to complete
                group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(group_results):
                    task_component = task_group[i]
                    
                    if isinstance(result, Exception):
                        self.logger.error(f"Task {task_component.id} failed: {result}")
                        generation_results['success'] = False
                        generation_results['agent_reports'][task_component.id] = {
                            'success': False,
                            'error': str(result)
                        }
                    else:
                        generation_results['agent_reports'][task_component.id] = result
                        if result.get('generated_files'):
                            generation_results['generated_files'].extend(result['generated_files'])
            
            generation_results['execution_time'] = time.time() - start_time
            
            return generation_results
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'generated_files': [],
                'agent_reports': {},
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
            }
        finally:
            # Cleanup swarm
            if self._active_swarm:
                await self.claude_flow_client.shutdown_swarm(self._active_swarm)
                self._active_swarm = None
    
    def _group_tasks_by_dependencies(self, task_components: List[TaskComponent]) -> List[List[TaskComponent]]:
        """Group tasks into execution groups based on dependencies."""
        # Simple topological sort for dependency ordering
        groups = []
        remaining_tasks = task_components.copy()
        completed_task_ids = set()
        
        while remaining_tasks:
            # Find tasks with no unfulfilled dependencies
            ready_tasks = []
            for task in remaining_tasks:
                if all(dep_id in completed_task_ids for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Break circular dependencies by taking highest priority task
                ready_tasks = [max(remaining_tasks, key=lambda t: t.priority)]
            
            # Remove ready tasks from remaining
            for task in ready_tasks:
                remaining_tasks.remove(task)
                completed_task_ids.add(task.id)
            
            groups.append(ready_tasks)
        
        return groups
    
    async def _execute_task_component(
        self, 
        task_component: TaskComponent,
        project_path: str,
        requirements_analysis: RequirementsAnalysis
    ) -> Dict[str, Any]:
        """Execute a single task component."""
        self.logger.info(f"Executing task: {task_component.id}")
        
        # Build context for the task
        context = {
            'task_id': task_component.id,
            'task_type': 'code_generation',
            'project_path': project_path,
            'file_path': task_component.file_path,
            'agent_type': task_component.agent_type,
            'requirements': task_component.requirements,
            'validation_criteria': task_component.validation_criteria,
            'project_context': {
                'project_type': requirements_analysis.project_type,
                'architecture': requirements_analysis.recommended_architecture,
                'technologies': requirements_analysis.suggested_technologies
            },
            **task_component.context
        }
        
        # Generate detailed prompt based on task type and agent type
        prompt = self._build_task_prompt(task_component, requirements_analysis)
        
        try:
            # Execute via Claude Code
            response = await self.claude_code_client.execute_task(
                task_description=prompt,
                context=context
            )
            
            result = {
                'task_id': task_component.id,
                'success': response.get('success', False),
                'generated_files': response.get('generated_files', []),
                'execution_time': response.get('execution_time', 0),
                'agent_type': task_component.agent_type,
                'validation_passed': response.get('validation_passed', False)
            }
            
            if not result['success']:
                result['error'] = response.get('error', 'Task execution failed')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task_component.id} execution failed: {e}")
            return {
                'task_id': task_component.id,
                'success': False,
                'error': str(e),
                'generated_files': [],
                'execution_time': 0,
                'agent_type': task_component.agent_type
            }
    
    def _build_task_prompt(
        self, 
        task_component: TaskComponent,
        requirements_analysis: RequirementsAnalysis
    ) -> str:
        """Build detailed prompt for task execution."""
        base_prompt = f"""
        Task: {task_component.description}
        
        Context:
        - Project Type: {requirements_analysis.project_type}
        - Architecture: {requirements_analysis.recommended_architecture}
        - Technologies: {', '.join(requirements_analysis.suggested_technologies)}
        - Target File: {task_component.file_path or 'TBD'}
        - Agent Role: {task_component.agent_type}
        
        Requirements:
        {chr(10).join(f"- {req}" for req in task_component.requirements)}
        
        Validation Criteria:
        {chr(10).join(f"- {criteria}" for criteria in task_component.validation_criteria)}
        
        Please implement this task following best practices for {requirements_analysis.project_type} development.
        """
        
        # Add agent-specific instructions
        if task_component.agent_type == 'tester':
            base_prompt += """
            
            Focus on:
            - Comprehensive test coverage
            - Edge case testing
            - Performance testing if applicable
            - Integration testing
            """
        elif task_component.agent_type == 'reviewer':
            base_prompt += """
            
            Focus on:
            - Code quality and style
            - Security vulnerabilities
            - Performance implications
            - Maintainability
            """
        
        return base_prompt


class ValidationEngine:
    """Validates generated code and results."""
    
    def __init__(self, claude_code_client: ClaudeCodeClient, config_manager: Optional[ConfigManager] = None):
        self.claude_code_client = claude_code_client
        self.config_manager = config_manager
        # Initialize real_time_validator only when needed to avoid dependency issues
        self.real_time_validator = None
        self.logger = get_logger(__name__)
    
    async def validate_results(
        self, 
        generation_results: Dict[str, Any],
        project_path: str,
        requirements_analysis: RequirementsAnalysis
    ) -> Dict[str, Any]:
        """
        Validate generated code and results.
        
        Args:
            generation_results: Results from code generation
            project_path: Project path
            requirements_analysis: Original requirements
            
        Returns:
            Dict containing validation results
        """
        self.logger.info("Validating generated code and results")
        
        validation_results = {
            'overall_success': True,
            'quality_score': 0.0,
            'coverage_metrics': {},
            'performance_metrics': {},
            'issues': [],
            'suggestions': [],
            'validation_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Validate each generated file
            file_validations = []
            for file_path in generation_results.get('generated_files', []):
                file_validations.append(
                    self._validate_file(file_path, project_path, requirements_analysis)
                )
            
            if file_validations:
                file_results = await asyncio.gather(*file_validations, return_exceptions=True)
                
                total_quality = 0.0
                valid_files = 0
                
                for i, result in enumerate(file_results):
                    file_path = generation_results['generated_files'][i]
                    
                    if isinstance(result, Exception):
                        validation_results['issues'].append({
                            'file': file_path,
                            'type': 'validation_error',
                            'message': str(result)
                        })
                        validation_results['overall_success'] = False
                    else:
                        total_quality += result.get('quality_score', 0.0)
                        valid_files += 1
                        
                        # Collect issues and suggestions
                        validation_results['issues'].extend(
                            result.get('issues', [])
                        )
                        validation_results['suggestions'].extend(
                            result.get('suggestions', [])
                        )
                
                # Calculate overall quality score
                if valid_files > 0:
                    validation_results['quality_score'] = total_quality / valid_files
            
            # Validate against original requirements
            requirements_validation = await self._validate_requirements_fulfillment(
                generation_results, requirements_analysis
            )
            validation_results.update(requirements_validation)
            
            validation_results['validation_time'] = time.time() - start_time
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            validation_results.update({
                'overall_success': False,
                'error': str(e),
                'validation_time': time.time() - start_time
            })
            return validation_results
    
    async def _validate_file(
        self, 
        file_path: str, 
        project_path: str,
        requirements_analysis: RequirementsAnalysis
    ) -> Dict[str, Any]:
        """Validate a single generated file."""
        try:
            full_path = Path(project_path) / file_path
            
            # Use real-time validator for basic checks if available
            validation_result = {}
            if self.real_time_validator:
                validation_result = await self.real_time_validator.validate_output(
                    str(full_path), {'file_path': str(full_path)}
                )
            
            # Enhanced validation via Claude Code API
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                context = {
                    'file_path': file_path,
                    'project_type': requirements_analysis.project_type,
                    'technologies': requirements_analysis.suggested_technologies,
                    'validation_type': 'comprehensive'
                }
                
                api_validation = await self.claude_code_client.validate_output(
                    code_content, context
                )
                
                # Merge validation results
                return {
                    'file_path': file_path,
                    'quality_score': api_validation.get('quality_score', 0.5),
                    'issues': api_validation.get('issues', []),
                    'suggestions': api_validation.get('suggestions', []),
                    'syntax_valid': validation_result.get('syntax_valid', True),
                    'style_score': validation_result.get('style_score', 0.5)
                }
            else:
                return {
                    'file_path': file_path,
                    'quality_score': 0.0,
                    'issues': [{'type': 'missing_file', 'message': f'File {file_path} was not created'}],
                    'suggestions': [],
                    'syntax_valid': False
                }
                
        except Exception as e:
            return {
                'file_path': file_path,
                'quality_score': 0.0,
                'issues': [{'type': 'validation_error', 'message': str(e)}],
                'suggestions': [],
                'error': str(e)
            }
    
    async def _validate_requirements_fulfillment(
        self, 
        generation_results: Dict[str, Any],
        requirements_analysis: RequirementsAnalysis
    ) -> Dict[str, Any]:
        """Validate that generated code fulfills original requirements."""
        context = {
            'task_type': 'requirements_validation',
            'original_requirements': requirements_analysis.original_requirements,
            'generated_files': generation_results.get('generated_files', []),
            'project_type': requirements_analysis.project_type
        }
        
        validation_prompt = f"""
        Validate if the generated code fulfills the original requirements:
        
        Original Requirements: {requirements_analysis.original_requirements}
        Generated Files: {', '.join(generation_results.get('generated_files', []))}
        
        Please assess:
        1. Requirements coverage (0-100%)
        2. Functional completeness
        3. Quality attributes fulfillment
        4. Missing functionality
        5. Overall alignment with requirements
        
        Provide structured assessment with scores and recommendations.
        """
        
        try:
            response = await self.claude_code_client.execute_task(
                task_description=validation_prompt,
                context=context
            )
            
            if response.get('success', False):
                assessment = response.get('assessment', {})
                return {
                    'requirements_coverage': assessment.get('coverage_percentage', 50.0),
                    'functional_completeness': assessment.get('completeness_score', 0.5),
                    'missing_features': assessment.get('missing_features', []),
                    'alignment_score': assessment.get('alignment_score', 0.5)
                }
            else:
                return {'requirements_coverage': 0.0, 'functional_completeness': 0.0}
                
        except Exception as e:
            self.logger.error(f"Requirements validation failed: {e}")
            return {'requirements_coverage': 0.0, 'functional_completeness': 0.0, 'error': str(e)}


class AutomaticProgrammingCoordinator:
    """
    Main orchestration class for the automatic programming pipeline.
    
    Transforms natural language requirements into production-ready code using:
    - Claude Code CLI for actual code generation
    - Claude Flow for multi-agent coordination
    - File-based communication for reliability
    - Hive Mind memory for context sharing
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the automatic programming coordinator.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        self.logger = get_logger(__name__)
        
        # Initialize clients
        self.claude_code_client = ClaudeCodeClient(config_manager)
        self.claude_flow_client = ClaudeFlowClient(config_manager)
        self.file_system = FileSystemManager()
        
        # Initialize pipeline components
        self.requirements_analyzer = RequirementsAnalyzer(self.claude_code_client)
        self.task_decomposer = TaskDecomposer(self.claude_code_client)
        self.code_generator = CodeGenerator(self.claude_code_client, self.claude_flow_client)
        self.validation_engine = ValidationEngine(self.claude_code_client, config_manager)
        
        # Pipeline state
        self.current_status = PipelineStatus.INITIALIZING
        self.pipeline_id = str(uuid.uuid4())
        self.start_time: Optional[datetime] = None
        
        # Memory integration for context sharing
        self.memory_store: Dict[str, Any] = {}
        
        self.logger.info(f"Automatic Programming Coordinator initialized - Pipeline ID: {self.pipeline_id}")
    
    async def initialize(self) -> None:
        """Initialize the automatic programming coordinator."""
        try:
            self.logger.info("Initializing automatic programming pipeline")
            
            # Initialize Claude Flow client
            await self.claude_flow_client.initialize()
            
            # Test Claude Code client
            if not await self.claude_code_client.health_check():
                self.logger.warning("Claude Code client health check failed")
            
            self.logger.info("Automatic programming pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize automatic programming pipeline: {e}")
            raise
    
    async def generate_code(
        self,
        requirements: str,
        project_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Generate code from natural language requirements.
        
        Args:
            requirements: Natural language requirements description
            project_path: Target project directory path
            options: Optional generation options
            
        Returns:
            PipelineResult: Complete pipeline execution result
        """
        self.logger.info(f"Starting automatic programming pipeline for: {requirements[:100]}...")
        
        self.start_time = datetime.now()
        options = options or {}
        
        # Initialize result structure
        result = PipelineResult(
            success=False,
            generated_files=[],
            execution_time=0.0,
            requirements_analysis=None,
            task_components=[],
            validation_results={},
            agent_reports={}
        )
        
        try:
            # Ensure project directory exists
            project_path_obj = Path(project_path)
            project_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Store context in memory for agent coordination
            self.memory_store['pipeline_context'] = {
                'pipeline_id': self.pipeline_id,
                'requirements': requirements,
                'project_path': project_path,
                'options': options,
                'start_time': self.start_time.isoformat()
            }
            
            # Step 1: Analyze Requirements
            self.current_status = PipelineStatus.ANALYZING_REQUIREMENTS
            self.logger.info("Step 1: Analyzing requirements")
            
            requirements_analysis = await self.requirements_analyzer.analyze(
                requirements, project_path
            )
            result.requirements_analysis = requirements_analysis
            
            self.memory_store['requirements_analysis'] = {
                'project_type': requirements_analysis.project_type,
                'complexity': requirements_analysis.estimated_complexity,
                'architecture': requirements_analysis.recommended_architecture,
                'technologies': requirements_analysis.suggested_technologies
            }
            
            # Step 2: Decompose Tasks
            self.current_status = PipelineStatus.DECOMPOSING_TASKS
            self.logger.info("Step 2: Decomposing tasks")
            
            task_components = await self.task_decomposer.decompose(
                requirements_analysis, project_path
            )
            result.task_components = task_components
            
            self.memory_store['task_components'] = [
                {
                    'id': task.id,
                    'description': task.description,
                    'file_path': task.file_path,
                    'dependencies': task.dependencies,
                    'agent_type': task.agent_type
                }
                for task in task_components
            ]
            
            # Step 3: Generate Code
            self.current_status = PipelineStatus.GENERATING_CODE
            self.logger.info("Step 3: Generating code with coordinated agents")
            
            generation_results = await self.code_generator.generate_code(
                task_components, project_path, requirements_analysis
            )
            
            result.generated_files = generation_results.get('generated_files', [])
            result.agent_reports = generation_results.get('agent_reports', {})
            
            if not generation_results.get('success', False):
                raise Exception(f"Code generation failed: {generation_results.get('error', 'Unknown error')}")
            
            # Step 4: Validate Results
            self.current_status = PipelineStatus.VALIDATING_RESULTS
            self.logger.info("Step 4: Validating generated code")
            
            validation_results = await self.validation_engine.validate_results(
                generation_results, project_path, requirements_analysis
            )
            result.validation_results = validation_results
            result.quality_score = validation_results.get('quality_score', 0.0)
            
            # Step 5: Complete Pipeline
            self.current_status = PipelineStatus.COMPLETING
            self.logger.info("Step 5: Completing pipeline")
            
            # Calculate final metrics
            execution_time = (datetime.now() - self.start_time).total_seconds()
            result.execution_time = execution_time
            result.success = validation_results.get('overall_success', False)
            
            # Store final results in memory for future reference
            self.memory_store['pipeline_result'] = {
                'success': result.success,
                'execution_time': result.execution_time,
                'generated_files': result.generated_files,
                'quality_score': result.quality_score,
                'completed_at': datetime.now().isoformat()
            }
            
            self.current_status = PipelineStatus.COMPLETED
            self.logger.info(f"Automatic programming pipeline completed successfully in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.current_status = PipelineStatus.FAILED
            execution_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            self.logger.error(f"Automatic programming pipeline failed: {e}")
            
            result.success = False
            result.error_message = str(e)
            result.execution_time = execution_time
            
            return result
        
        finally:
            # Cleanup
            await self._cleanup()
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and progress."""
        return {
            'pipeline_id': self.pipeline_id,
            'status': self.current_status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'execution_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'memory_context': self.memory_store
        }
    
    async def get_memory_context(self, key: str) -> Optional[Any]:
        """Retrieve context from memory store."""
        return self.memory_store.get(key)
    
    async def set_memory_context(self, key: str, value: Any) -> None:
        """Store context in memory for agent coordination."""
        self.memory_store[key] = value
    
    async def _cleanup(self) -> None:
        """Cleanup pipeline resources."""
        try:
            # Cleanup clients
            await self.claude_code_client.cleanup()
            await self.claude_flow_client.cleanup()
            
            self.logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Pipeline cleanup failed: {e}")
    
    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()


# Utility functions for easy pipeline usage

async def create_programming_pipeline(
    config_path: Optional[str] = None
) -> AutomaticProgrammingCoordinator:
    """
    Create and initialize an automatic programming pipeline.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        AutomaticProgrammingCoordinator: Initialized pipeline
    """
    config_manager = ConfigManager(config_path) if config_path else ConfigManager()
    
    pipeline = AutomaticProgrammingCoordinator(config_manager)
    await pipeline.initialize()
    
    return pipeline


async def generate_project_from_requirements(
    requirements: str,
    project_path: str,
    config_path: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """
    High-level function to generate a complete project from requirements.
    
    Args:
        requirements: Natural language project requirements
        project_path: Target project directory
        config_path: Optional configuration file path
        options: Generation options
        
    Returns:
        PipelineResult: Complete generation result
    """
    async with await create_programming_pipeline(config_path) as pipeline:
        return await pipeline.generate_code(requirements, project_path, options)


# Example usage
if __name__ == "__main__":
    async def main():
        # Example: Generate a REST API project
        requirements = """
        Create a REST API for user management with the following features:
        - User registration and authentication
        - CRUD operations for user profiles
        - JWT-based authentication
        - Input validation and error handling
        - Basic logging and monitoring
        - Docker configuration for deployment
        - Comprehensive tests with good coverage
        """
        
        project_path = "/tmp/user_management_api"
        
        result = await generate_project_from_requirements(
            requirements=requirements,
            project_path=project_path,
            options={
                'include_docker': True,
                'include_tests': True,
                'preferred_framework': 'FastAPI'
            }
        )
        
        print(f"Generation Success: {result.success}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Generated Files: {len(result.generated_files)}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
    
    # Run example
    # asyncio.run(main())