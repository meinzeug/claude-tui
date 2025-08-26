#!/usr/bin/env python3
"""
Automatic Programming Workflow Manager
=====================================

Unified workflow manager that orchestrates:
- Claude Code direct CLI for code generation
- Claude Flow API for swarm coordination  
- Hive Mind memory for context sharing
- File system operations for project management

This is the core orchestration engine for end-to-end AI workflow integration.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import uuid

from .claude_code_client import ClaudeCodeClient
from .claude_flow_client import ClaudeFlowClient
from .integration_manager import IntegrationManager

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Individual workflow step configuration"""
    step_id: str
    name: str
    description: str
    step_type: str  # 'claude_code', 'claude_flow', 'file_operation', 'validation'
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    timeout: int = 300  # 5 minutes default
    retry_count: int = 3
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    workflow_id: str
    status: WorkflowStatus
    steps_completed: int
    steps_total: int
    results: Dict[str, Any]
    errors: List[str]
    duration: float
    created_files: List[str]
    modified_files: List[str]


@dataclass
class ProgressUpdate:
    """Progress update for real-time monitoring"""
    workflow_id: str
    step_id: str
    step_name: str
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: float
    step_status: str


class AutomaticProgrammingWorkflow:
    """
    Main workflow orchestrator for automatic programming features
    """
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, asyncio.Task] = {}
        self.progress_callbacks: List[Callable[[ProgressUpdate], None]] = []
        
        # Initialize component clients
        self.claude_code_client = ClaudeCodeClient(config_manager) if config_manager else None
        self.claude_flow_client = ClaudeFlowClient(config_manager) if config_manager else None
        self.integration_manager = IntegrationManager() if config_manager else None
        
        # Workflow templates
        self._load_workflow_templates()
        
        logger.info("AutomaticProgrammingWorkflow initialized")
    
    def _load_workflow_templates(self):
        """Load predefined workflow templates"""
        self.workflow_templates = {
            "fastapi_app": {
                "name": "FastAPI Application Generator",
                "description": "Generate a complete FastAPI application with authentication",
                "steps": [
                    WorkflowStep(
                        step_id="init_project",
                        name="Initialize Project Structure",
                        description="Create project directories and basic structure",
                        step_type="file_operation",
                        parameters={"operation": "create_structure"}
                    ),
                    WorkflowStep(
                        step_id="generate_requirements",
                        name="Generate Requirements",
                        description="Create requirements.txt with FastAPI dependencies",
                        step_type="claude_code",
                        parameters={"prompt": "Generate requirements.txt for FastAPI app with authentication"},
                        dependencies=["init_project"]
                    ),
                    WorkflowStep(
                        step_id="generate_main_app",
                        name="Generate Main Application",
                        description="Create main FastAPI application with basic structure",
                        step_type="claude_code",
                        parameters={"prompt": "Create FastAPI main application with router setup"},
                        dependencies=["generate_requirements"]
                    ),
                    WorkflowStep(
                        step_id="generate_auth",
                        name="Generate Authentication",
                        description="Create authentication system with JWT",
                        step_type="claude_code",
                        parameters={"prompt": "Create JWT authentication system for FastAPI"},
                        dependencies=["generate_main_app"]
                    ),
                    WorkflowStep(
                        step_id="generate_models",
                        name="Generate Data Models",
                        description="Create Pydantic models and database schemas",
                        step_type="claude_code",
                        parameters={"prompt": "Create Pydantic models and SQLAlchemy schemas"},
                        dependencies=["generate_auth"]
                    ),
                    WorkflowStep(
                        step_id="generate_tests",
                        name="Generate Tests",
                        description="Create comprehensive test suite",
                        step_type="claude_code",
                        parameters={"prompt": "Create pytest test suite for FastAPI app"},
                        dependencies=["generate_models"]
                    ),
                    WorkflowStep(
                        step_id="validate_project",
                        name="Validate Generated Project",
                        description="Validate the generated code and run basic checks",
                        step_type="validation",
                        parameters={"validation_type": "comprehensive"},
                        dependencies=["generate_tests"]
                    )
                ]
            },
            
            "react_dashboard": {
                "name": "React Dashboard Generator",
                "description": "Generate a complete React dashboard with components",
                "steps": [
                    WorkflowStep(
                        step_id="init_react_project",
                        name="Initialize React Project",
                        description="Create React project structure with TypeScript",
                        step_type="file_operation",
                        parameters={"operation": "create_react_structure"}
                    ),
                    WorkflowStep(
                        step_id="generate_package_json",
                        name="Generate Package Configuration",
                        description="Create package.json with React dependencies",
                        step_type="claude_code",
                        parameters={"prompt": "Generate package.json for React dashboard with TypeScript"},
                        dependencies=["init_react_project"]
                    ),
                    WorkflowStep(
                        step_id="generate_main_components",
                        name="Generate Main Components",
                        description="Create main dashboard components",
                        step_type="claude_code",
                        parameters={"prompt": "Create React dashboard components with routing"},
                        dependencies=["generate_package_json"]
                    ),
                    WorkflowStep(
                        step_id="generate_styles",
                        name="Generate Styling",
                        description="Create CSS/SCSS styles for the dashboard",
                        step_type="claude_code",
                        parameters={"prompt": "Create modern CSS styling for React dashboard"},
                        dependencies=["generate_main_components"]
                    ),
                    WorkflowStep(
                        step_id="validate_react_project",
                        name="Validate React Project",
                        description="Validate React code and check for issues",
                        step_type="validation",
                        parameters={"validation_type": "react_specific"},
                        dependencies=["generate_styles"]
                    )
                ]
            }
        }
    
    def add_progress_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Add a callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Remove a progress callback"""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    def _notify_progress(self, update: ProgressUpdate):
        """Notify all progress callbacks of an update"""
        for callback in self.progress_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    async def create_workflow_from_template(
        self,
        template_name: str,
        project_name: str,
        project_path: Path,
        custom_parameters: Dict[str, Any] = None
    ) -> str:
        """
        Create a workflow from a predefined template
        
        Args:
            template_name: Name of the template to use
            project_name: Name of the project being created
            project_path: Path where the project should be created
            custom_parameters: Additional parameters to customize the workflow
            
        Returns:
            Workflow ID for tracking
        """
        if template_name not in self.workflow_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.workflow_templates[template_name]
        workflow_id = str(uuid.uuid4())
        
        # Customize workflow with project-specific parameters
        workflow = {
            "id": workflow_id,
            "name": f"{template['name']} - {project_name}",
            "description": template["description"],
            "project_name": project_name,
            "project_path": str(project_path),
            "status": WorkflowStatus.PENDING,
            "steps": template["steps"],
            "created_at": time.time(),
            "custom_parameters": custom_parameters or {}
        }
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow {workflow_id} from template {template_name}")
        
        return workflow_id
    
    async def create_custom_workflow(
        self,
        name: str,
        description: str,
        prompt: str,
        project_path: Path,
        workflow_type: str = "general"
    ) -> str:
        """
        Create a custom workflow from a natural language prompt
        
        Args:
            name: Workflow name
            description: Workflow description
            prompt: Natural language description of what to build
            project_path: Path where the project should be created
            workflow_type: Type of workflow (fastapi, react, python, etc.)
            
        Returns:
            Workflow ID for tracking
        """
        workflow_id = str(uuid.uuid4())
        
        # Use Claude Flow to decompose the prompt into workflow steps
        try:
            workflow_steps = await self._generate_workflow_steps_from_prompt(
                prompt, workflow_type
            )
        except Exception as e:
            logger.error(f"Failed to generate workflow steps: {e}")
            # Fallback to basic workflow
            workflow_steps = await self._create_basic_workflow_steps(prompt)
        
        workflow = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "project_path": str(project_path),
            "status": WorkflowStatus.PENDING,
            "steps": workflow_steps,
            "created_at": time.time(),
            "original_prompt": prompt,
            "workflow_type": workflow_type
        }
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created custom workflow {workflow_id}: {name}")
        
        return workflow_id
    
    async def _generate_workflow_steps_from_prompt(
        self,
        prompt: str,
        workflow_type: str
    ) -> List[WorkflowStep]:
        """Generate workflow steps using Claude Flow API"""
        try:
            # Use Claude Flow to decompose the task
            response = await self.claude_flow_client.decompose_task(
                task_description=prompt,
                task_type=workflow_type
            )
            
            steps = []
            for i, step_data in enumerate(response.get("steps", [])):
                step = WorkflowStep(
                    step_id=f"step_{i+1}",
                    name=step_data.get("name", f"Step {i+1}"),
                    description=step_data.get("description", ""),
                    step_type=step_data.get("type", "claude_code"),
                    parameters=step_data.get("parameters", {}),
                    dependencies=step_data.get("dependencies", [])
                )
                steps.append(step)
            
            return steps
            
        except Exception as e:
            logger.error(f"Error generating workflow steps from Claude Flow: {e}")
            raise
    
    async def _create_basic_workflow_steps(self, prompt: str) -> List[WorkflowStep]:
        """Create basic workflow steps as fallback"""
        steps = [
            WorkflowStep(
                step_id="analyze_requirements",
                name="Analyze Requirements",
                description="Analyze the project requirements",
                step_type="claude_code",
                parameters={"prompt": f"Analyze requirements for: {prompt}"}
            ),
            WorkflowStep(
                step_id="generate_code",
                name="Generate Code",
                description="Generate the main code implementation",
                step_type="claude_code",
                parameters={"prompt": f"Implement: {prompt}"},
                dependencies=["analyze_requirements"]
            ),
            WorkflowStep(
                step_id="create_tests",
                name="Create Tests",
                description="Generate test cases",
                step_type="claude_code",
                parameters={"prompt": f"Create tests for: {prompt}"},
                dependencies=["generate_code"]
            ),
            WorkflowStep(
                step_id="validate_result",
                name="Validate Result",
                description="Validate the generated code",
                step_type="validation",
                parameters={"validation_type": "basic"},
                dependencies=["create_tests"]
            )
        ]
        
        return steps
    
    async def execute_workflow(self, workflow_id: str) -> WorkflowResult:
        """
        Execute a workflow with real-time progress tracking
        
        Args:
            workflow_id: ID of the workflow to execute
            
        Returns:
            WorkflowResult with execution details
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        workflow["status"] = WorkflowStatus.INITIALIZING
        
        logger.info(f"Starting execution of workflow {workflow_id}: {workflow['name']}")
        
        # Create execution task
        execution_task = asyncio.create_task(
            self._execute_workflow_steps(workflow_id)
        )
        self.active_workflows[workflow_id] = execution_task
        
        try:
            result = await execution_task
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow["status"] = WorkflowStatus.FAILED
            raise
        finally:
            # Clean up
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
    
    async def _execute_workflow_steps(self, workflow_id: str) -> WorkflowResult:
        """Execute individual workflow steps"""
        workflow = self.workflows[workflow_id]
        steps = workflow["steps"]
        project_path = Path(workflow["project_path"])
        
        workflow["status"] = WorkflowStatus.IN_PROGRESS
        start_time = time.time()
        
        results = {}
        errors = []
        created_files = []
        modified_files = []
        completed_steps = 0
        
        # Create execution context
        execution_context = {
            "project_name": workflow.get("project_name", "generated_project"),
            "project_path": project_path,
            "workflow_id": workflow_id,
            "custom_parameters": workflow.get("custom_parameters", {}),
            "results": results
        }
        
        try:
            for i, step in enumerate(steps):
                # Check dependencies
                if not self._check_step_dependencies(step, results):
                    error_msg = f"Step {step.step_id} dependencies not met"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                # Update progress
                progress = (i / len(steps))
                self._notify_progress(ProgressUpdate(
                    workflow_id=workflow_id,
                    step_id=step.step_id,
                    step_name=step.name,
                    progress=progress,
                    message=f"Starting {step.name}",
                    timestamp=time.time(),
                    step_status="starting"
                ))
                
                # Execute step
                try:
                    step_result = await self._execute_single_step(
                        step, execution_context
                    )
                    results[step.step_id] = step_result
                    completed_steps += 1
                    
                    # Track file changes
                    if "created_files" in step_result:
                        created_files.extend(step_result["created_files"])
                    if "modified_files" in step_result:
                        modified_files.extend(step_result["modified_files"])
                    
                    # Update progress
                    progress = ((i + 1) / len(steps))
                    self._notify_progress(ProgressUpdate(
                        workflow_id=workflow_id,
                        step_id=step.step_id,
                        step_name=step.name,
                        progress=progress,
                        message=f"Completed {step.name}",
                        timestamp=time.time(),
                        step_status="completed"
                    ))
                    
                except Exception as e:
                    error_msg = f"Step {step.step_id} failed: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    
                    # Update progress with error
                    self._notify_progress(ProgressUpdate(
                        workflow_id=workflow_id,
                        step_id=step.step_id,
                        step_name=step.name,
                        progress=(i / len(steps)),
                        message=f"Failed: {step.name} - {str(e)}",
                        timestamp=time.time(),
                        step_status="failed"
                    ))
        
        except Exception as e:
            logger.error(f"Critical error in workflow execution: {e}")
            errors.append(f"Critical error: {str(e)}")
            workflow["status"] = WorkflowStatus.FAILED
        
        # Determine final status
        duration = time.time() - start_time
        if errors and completed_steps == 0:
            final_status = WorkflowStatus.FAILED
        elif errors:
            final_status = WorkflowStatus.COMPLETED  # Partial success
        else:
            final_status = WorkflowStatus.COMPLETED
        
        workflow["status"] = final_status
        
        # Create final result
        result = WorkflowResult(
            workflow_id=workflow_id,
            status=final_status,
            steps_completed=completed_steps,
            steps_total=len(steps),
            results=results,
            errors=errors,
            duration=duration,
            created_files=created_files,
            modified_files=modified_files
        )
        
        logger.info(f"Workflow {workflow_id} completed with status {final_status.value}")
        logger.info(f"Completed {completed_steps}/{len(steps)} steps in {duration:.2f}s")
        
        return result
    
    def _check_step_dependencies(self, step: WorkflowStep, results: Dict[str, Any]) -> bool:
        """Check if step dependencies are satisfied"""
        for dep_step_id in step.dependencies:
            if dep_step_id not in results:
                return False
            # Check if dependency step was successful
            dep_result = results[dep_step_id]
            if isinstance(dep_result, dict) and dep_result.get("status") == "failed":
                return False
        return True
    
    async def _execute_single_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        logger.info(f"Executing step {step.step_id}: {step.name}")
        
        if step.step_type == "claude_code":
            return await self._execute_claude_code_step(step, context)
        elif step.step_type == "claude_flow":
            return await self._execute_claude_flow_step(step, context)
        elif step.step_type == "file_operation":
            return await self._execute_file_operation_step(step, context)
        elif step.step_type == "validation":
            return await self._execute_validation_step(step, context)
        else:
            raise ValueError(f"Unknown step type: {step.step_type}")
    
    async def _execute_claude_code_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a Claude Code step"""
        try:
            prompt = step.parameters.get("prompt", "")
            # Enhance prompt with context
            enhanced_prompt = self._enhance_prompt_with_context(prompt, context)
            
            # Execute via Claude Code client
            response = await self.claude_code_client.execute_code_generation(
                prompt=enhanced_prompt,
                project_path=context["project_path"]
            )
            
            return {
                "status": "success",
                "response": response,
                "created_files": response.get("created_files", []),
                "modified_files": response.get("modified_files", []),
                "step_type": "claude_code"
            }
            
        except Exception as e:
            logger.error(f"Claude Code step failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "step_type": "claude_code"
            }
    
    async def _execute_claude_flow_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a Claude Flow step"""
        try:
            # Execute coordination task via Claude Flow
            task_type = step.parameters.get("task_type", "general")
            task_description = step.parameters.get("description", step.description)
            
            response = await self.claude_flow_client.execute_coordinated_task(
                task_description=task_description,
                task_type=task_type,
                context=context
            )
            
            return {
                "status": "success",
                "response": response,
                "step_type": "claude_flow"
            }
            
        except Exception as e:
            logger.error(f"Claude Flow step failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "step_type": "claude_flow"
            }
    
    async def _execute_file_operation_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a file operation step"""
        try:
            operation = step.parameters.get("operation")
            project_path = context["project_path"]
            
            created_files = []
            modified_files = []
            
            if operation == "create_structure":
                # Create basic project structure
                directories = [
                    "src", "tests", "docs", "config"
                ]
                for dir_name in directories:
                    dir_path = project_path / dir_name
                    dir_path.mkdir(parents=True, exist_ok=True)
                
                # Create basic files
                basic_files = {
                    "README.md": f"# {context['project_name']}\n\nGenerated project.",
                    ".gitignore": "*.pyc\n__pycache__/\n.env\n",
                    "setup.py": f"from setuptools import setup\n\nsetup(name='{context['project_name']}')\n"
                }
                
                for file_name, content in basic_files.items():
                    file_path = project_path / file_name
                    file_path.write_text(content)
                    created_files.append(str(file_path))
            
            elif operation == "create_react_structure":
                # Create React project structure
                directories = [
                    "src", "src/components", "src/styles", "src/utils",
                    "public", "tests"
                ]
                for dir_name in directories:
                    dir_path = project_path / dir_name
                    dir_path.mkdir(parents=True, exist_ok=True)
                
                # Create React basic files
                react_files = {
                    "public/index.html": "<!DOCTYPE html><html><head><title>React App</title></head><body><div id='root'></div></body></html>",
                    "src/index.tsx": "import React from 'react';\nimport ReactDOM from 'react-dom';\nimport App from './App';\n\nReactDOM.render(<App />, document.getElementById('root'));",
                    "src/App.tsx": "import React from 'react';\n\nconst App: React.FC = () => {\n  return <div>React Dashboard</div>;\n};\n\nexport default App;",
                    ".gitignore": "node_modules/\nbuild/\n.env\n"
                }
                
                for file_name, content in react_files.items():
                    file_path = project_path / file_name
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)
                    created_files.append(str(file_path))
            
            return {
                "status": "success",
                "created_files": created_files,
                "modified_files": modified_files,
                "step_type": "file_operation"
            }
            
        except Exception as e:
            logger.error(f"File operation step failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "step_type": "file_operation"
            }
    
    async def _execute_validation_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a validation step"""
        try:
            validation_type = step.parameters.get("validation_type", "basic")
            project_path = context["project_path"]
            
            # Perform validation based on type
            validation_results = {
                "files_checked": 0,
                "issues_found": 0,
                "recommendations": []
            }
            
            if validation_type == "comprehensive":
                # Check all Python files for basic syntax
                for py_file in project_path.rglob("*.py"):
                    validation_results["files_checked"] += 1
                    try:
                        compile(py_file.read_text(), str(py_file), 'exec')
                    except SyntaxError as e:
                        validation_results["issues_found"] += 1
                        validation_results["recommendations"].append(
                            f"Syntax error in {py_file}: {str(e)}"
                        )
            
            elif validation_type == "react_specific":
                # Check TypeScript/JavaScript files
                for ts_file in project_path.rglob("*.tsx"):
                    validation_results["files_checked"] += 1
                    # Basic checks for React components
                    content = ts_file.read_text()
                    if "export default" not in content:
                        validation_results["recommendations"].append(
                            f"Missing default export in {ts_file}"
                        )
            
            return {
                "status": "success",
                "validation_results": validation_results,
                "step_type": "validation"
            }
            
        except Exception as e:
            logger.error(f"Validation step failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "step_type": "validation"
            }
    
    def _enhance_prompt_with_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """Enhance prompt with workflow context"""
        enhanced_prompt = f"""
Project Context:
- Project Name: {context.get('project_name', 'Unknown')}
- Project Path: {context.get('project_path', 'Unknown')}
- Workflow ID: {context.get('workflow_id', 'Unknown')}

Previous Results:
{json.dumps(context.get('results', {}), indent=2)}

Task:
{prompt}

Please provide a complete, production-ready implementation.
"""
        return enhanced_prompt
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        is_active = workflow_id in self.active_workflows
        
        return {
            "id": workflow_id,
            "name": workflow["name"],
            "status": workflow["status"].value if isinstance(workflow["status"], WorkflowStatus) else workflow["status"],
            "is_active": is_active,
            "created_at": workflow["created_at"],
            "steps_total": len(workflow["steps"]),
            "project_path": workflow["project_path"]
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        return [self.get_workflow_status(wf_id) for wf_id in self.workflows.keys()]
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id in self.active_workflows:
            task = self.active_workflows[workflow_id]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            workflow = self.workflows[workflow_id]
            workflow["status"] = WorkflowStatus.CANCELLED
            
            del self.active_workflows[workflow_id]
            logger.info(f"Cancelled workflow {workflow_id}")
            return True
        
        return False
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all available workflow templates"""
        return {
            name: {
                "name": template["name"],
                "description": template["description"],
                "steps_count": len(template["steps"])
            }
            for name, template in self.workflow_templates.items()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        # Cancel any active workflows
        for workflow_id in list(self.active_workflows.keys()):
            await self.cancel_workflow(workflow_id)
        
        # Cleanup clients
        if hasattr(self.claude_code_client, 'cleanup'):
            await self.claude_code_client.cleanup()
        if hasattr(self.claude_flow_client, 'cleanup'):
            await self.claude_flow_client.cleanup()
        if hasattr(self.integration_manager, 'cleanup'):
            await self.integration_manager.cleanup()
        
        logger.info("AutomaticProgrammingWorkflow cleanup completed")


# Convenience function for creating workflow manager
def create_workflow_manager(config_manager=None) -> AutomaticProgrammingWorkflow:
    """Create and initialize workflow manager"""
    return AutomaticProgrammingWorkflow(config_manager)