"""
Task Data Models.

Defines task-related data structures for development workflows,
task execution, and workflow orchestration.
"""

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, validator


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(Enum):
    """Types of development tasks."""
    CODE_GENERATION = "code_generation"
    FILE_CREATION = "file_creation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    CUSTOM = "custom"


@dataclass
class TaskDependency:
    """Represents a task dependency."""
    task_id: str
    dependency_type: str = "requires"  # requires, suggests, blocks
    condition: Optional[str] = None


class DevelopmentTask(BaseModel):
    """Represents a development task within a workflow."""
    
    # Basic identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    task_type: TaskType = TaskType.CODE_GENERATION
    
    # Task execution details
    ai_prompt: Optional[str] = None
    file_path: Optional[Path] = None
    target_directory: Optional[Path] = None
    expected_output: Optional[str] = None
    
    # Dependencies and ordering
    dependencies: List[TaskDependency] = Field(default_factory=list)
    blocks: List[str] = Field(default_factory=list)  # Tasks this blocks
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Execution parameters
    timeout: Optional[int] = None  # Seconds
    max_retries: int = 3
    retry_delay: int = 5  # Seconds
    parallel_safe: bool = True
    
    # Context and configuration
    context: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    
    # Validation settings
    validation_enabled: bool = True
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    allow_placeholders: bool = False
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0
    
    # Results and metrics
    execution_time: float = 0.0
    quality_score: Optional[float] = None
    generated_content: Optional[str] = None
    error_message: Optional[str] = None
    
    # Custom metadata
    tags: List[str] = Field(default_factory=list)
    custom_properties: Dict[str, Any] = Field(default_factory=dict)
    
    # Associated project (will be set by project manager)
    project: Optional[Any] = Field(None, exclude=True)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: str
        }
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Task name cannot be empty')
        return v.strip()
    
    def add_dependency(self, task_id: str, dependency_type: str = "requires") -> None:
        """Add a dependency to this task."""
        dependency = TaskDependency(task_id=task_id, dependency_type=dependency_type)
        if dependency not in self.dependencies:
            self.dependencies.append(dependency)
    
    def remove_dependency(self, task_id: str) -> None:
        """Remove a dependency from this task."""
        self.dependencies = [d for d in self.dependencies if d.task_id != task_id]
    
    def has_dependency(self, task_id: str) -> bool:
        """Check if this task depends on another task."""
        return any(d.task_id == task_id for d in self.dependencies)
    
    def get_required_dependencies(self) -> List[str]:
        """Get list of required dependency task IDs."""
        return [d.task_id for d in self.dependencies if d.dependency_type == "requires"]
    
    def start_execution(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self.attempts += 1
    
    def complete_execution(self, success: bool = True, error_message: Optional[str] = None) -> None:
        """Mark task as completed or failed."""
        self.completed_at = datetime.now()
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
        
        if success:
            self.status = TaskStatus.COMPLETED
        else:
            self.status = TaskStatus.FAILED
            self.error_message = error_message
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.attempts < self.max_retries and self.status == TaskStatus.FAILED
    
    def reset_for_retry(self) -> None:
        """Reset task state for retry."""
        self.status = TaskStatus.PENDING
        self.error_message = None
        self.generated_content = None
    
    def estimate_duration(self) -> timedelta:
        """Estimate task duration based on type and complexity."""
        # Base estimates in seconds
        base_estimates = {
            TaskType.CODE_GENERATION: 120,
            TaskType.FILE_CREATION: 30,
            TaskType.REFACTORING: 180,
            TaskType.TESTING: 90,
            TaskType.DOCUMENTATION: 60,
            TaskType.VALIDATION: 45,
            TaskType.DEPLOYMENT: 300,
            TaskType.CUSTOM: 120
        }
        
        base_seconds = base_estimates.get(self.task_type, 120)
        
        # Adjust based on priority and complexity indicators
        multiplier = 1.0
        if self.priority == TaskPriority.CRITICAL:
            multiplier *= 0.8  # Rush critical tasks
        elif self.priority == TaskPriority.LOW:
            multiplier *= 1.3  # Allow more time for low priority
        
        # Adjust based on context complexity
        if len(self.context) > 10:
            multiplier *= 1.2
        if len(self.dependencies) > 3:
            multiplier *= 1.1
        
        return timedelta(seconds=int(base_seconds * multiplier))


class TaskResult(BaseModel):
    """Result of a task execution."""
    
    task_id: str
    success: bool
    
    # Execution metadata
    execution_time: float = 0.0
    attempts: int = 1
    completed_at: datetime = Field(default_factory=datetime.now)
    
    # Result content
    generated_content: Optional[str] = None
    modified_files: List[Path] = Field(default_factory=list)
    created_files: List[Path] = Field(default_factory=list)
    deleted_files: List[Path] = Field(default_factory=list)
    
    # Quality and validation
    quality_score: Optional[float] = None
    validation_passed: bool = True
    validation_issues: List[str] = Field(default_factory=list)
    placeholder_count: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # AI-specific results
    ai_model_used: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Custom results
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: str
        }
    
    def add_validation_issue(self, issue: str) -> None:
        """Add a validation issue to the result."""
        if issue not in self.validation_issues:
            self.validation_issues.append(issue)
            self.validation_passed = False
    
    def get_file_changes_summary(self) -> Dict[str, int]:
        """Get summary of file changes."""
        return {
            'created': len(self.created_files),
            'modified': len(self.modified_files),
            'deleted': len(self.deleted_files),
            'total': len(self.created_files) + len(self.modified_files) + len(self.deleted_files)
        }


@dataclass
class WorkflowSuccessCriteria:
    """Defines success criteria for a workflow."""
    min_success_rate: float = 1.0  # 100% success rate required
    min_quality_score: float = 0.8
    max_placeholder_count: int = 0
    allow_failed_optional_tasks: bool = True
    custom_validation: Optional[Callable] = None
    
    def is_met(self, task_results: List[TaskResult]) -> bool:
        """Check if success criteria are met."""
        if not task_results:
            return False
        
        # Check success rate
        success_count = sum(1 for r in task_results if r.success)
        success_rate = success_count / len(task_results)
        
        if success_rate < self.min_success_rate:
            return False
        
        # Check quality scores
        quality_scores = [r.quality_score for r in task_results if r.quality_score is not None]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality < self.min_quality_score:
                return False
        
        # Check placeholder count
        total_placeholders = sum(r.placeholder_count for r in task_results)
        if total_placeholders > self.max_placeholder_count:
            return False
        
        # Custom validation
        if self.custom_validation:
            return self.custom_validation(task_results)
        
        return True


class Workflow(BaseModel):
    """Represents a complete development workflow."""
    
    # Basic identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    version: str = "1.0.0"
    
    # Workflow configuration
    tasks: List[DevelopmentTask] = Field(default_factory=list)
    execution_strategy: str = "adaptive"  # sequential, parallel, adaptive, balanced
    
    # Success and failure handling
    success_criteria: WorkflowSuccessCriteria = Field(default_factory=WorkflowSuccessCriteria)
    fail_fast: bool = False
    continue_on_failure: bool = True
    
    # Execution parameters
    timeout: Optional[int] = None  # Total workflow timeout in seconds
    max_concurrent_tasks: int = 5
    validate_results: bool = True
    
    # Context and variables
    variables: Dict[str, Any] = Field(default_factory=dict)
    global_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Associated project (will be set by project manager)
    project: Optional[Any] = Field(None, exclude=True)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_task(self, task: DevelopmentTask) -> None:
        """Add a task to the workflow."""
        if task not in self.tasks:
            self.tasks.append(task)
            self.updated_at = datetime.now()
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the workflow."""
        original_count = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.id != task_id]
        
        if len(self.tasks) < original_count:
            self.updated_at = datetime.now()
            return True
        return False
    
    def get_task(self, task_id: str) -> Optional[DevelopmentTask]:
        """Get a task by ID."""
        return next((t for t in self.tasks if t.id == task_id), None)
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[DevelopmentTask]:
        """Get all tasks with a specific status."""
        return [t for t in self.tasks if t.status == status]
    
    def get_ready_tasks(self) -> List[DevelopmentTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        completed_task_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                # Check if all required dependencies are completed
                required_deps = task.get_required_dependencies()
                if all(dep_id in completed_task_ids for dep_id in required_deps):
                    ready_tasks.append(task)
        
        return ready_tasks
    
    def validate_workflow(self) -> List[str]:
        """Validate the workflow for issues."""
        issues = []
        
        if not self.tasks:
            issues.append("Workflow has no tasks")
            return issues
        
        task_ids = {task.id for task in self.tasks}
        
        # Check for dependency cycles
        for task in self.tasks:
            visited = set()
            if self._has_cycle(task, task_ids, visited):
                issues.append(f"Circular dependency detected involving task '{task.name}'")
        
        # Check for missing dependencies
        for task in self.tasks:
            for dep in task.dependencies:
                if dep.task_id not in task_ids:
                    issues.append(f"Task '{task.name}' depends on missing task '{dep.task_id}'")
        
        # Check for orphaned tasks (no path from any root task)
        root_tasks = [t for t in self.tasks if not t.dependencies]
        if not root_tasks:
            issues.append("Workflow has no root tasks (all tasks have dependencies)")
        
        return issues
    
    def _has_cycle(self, task: DevelopmentTask, all_task_ids: set, visited: set) -> bool:
        """Check for circular dependencies starting from a task."""
        if task.id in visited:
            return True
        
        visited.add(task.id)
        
        for dep in task.dependencies:
            if dep.task_id in all_task_ids:
                dep_task = next(t for t in self.tasks if t.id == dep.task_id)
                if self._has_cycle(dep_task, all_task_ids, visited.copy()):
                    return True
        
        return False
    
    def estimate_total_duration(self) -> timedelta:
        """Estimate total workflow duration."""
        if not self.tasks:
            return timedelta(0)
        
        # For sequential execution, sum all durations
        if self.execution_strategy == "sequential":
            return sum((task.estimate_duration() for task in self.tasks), timedelta(0))
        
        # For parallel execution, use critical path
        # This is a simplified estimation - a full implementation would
        # use proper critical path analysis
        max_duration = max(task.estimate_duration() for task in self.tasks)
        return max_duration


class WorkflowResult(BaseModel):
    """Result of a workflow execution."""
    
    workflow_id: str
    success: bool
    
    # Execution statistics
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int = 0
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration: float = 0.0  # Total execution time in seconds
    
    # Results
    task_results: List[TaskResult] = Field(default_factory=list)
    overall_quality_score: Optional[float] = None
    total_placeholder_count: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    failed_task_ids: List[str] = Field(default_factory=list)
    
    # Resource usage
    peak_concurrent_tasks: int = 0
    total_ai_tokens: Optional[int] = None
    execution_cost: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def calculate_success_rate(self) -> float:
        """Calculate the success rate of the workflow."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100.0
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow execution."""
        return {
            'workflow_id': self.workflow_id,
            'success': self.success,
            'success_rate': self.calculate_success_rate(),
            'duration_seconds': self.duration,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'skipped_tasks': self.skipped_tasks,
            'overall_quality_score': self.overall_quality_score,
            'total_placeholder_count': self.total_placeholder_count,
            'peak_concurrent_tasks': self.peak_concurrent_tasks,
            'error_message': self.error_message
        }