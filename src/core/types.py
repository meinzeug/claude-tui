"""
Core data types and models for claude-tiu.

This module defines the fundamental data structures used throughout the system
for type safety, validation, and clear API contracts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

import pydantic
from pydantic import BaseModel, Field, field_validator


class ProjectState(str, Enum):
    """Project lifecycle states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class TaskStatus(str, Enum):
    """Task execution states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Task/issue priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationType(str, Enum):
    """Validation pipeline types."""
    QUICK = "quick"
    DEEP = "deep"
    COMPREHENSIVE = "comprehensive"


class IssueType(str, Enum):
    """Types of validation issues."""
    PLACEHOLDER = "placeholder"
    EMPTY_FUNCTION = "empty_function"
    MOCK_DATA = "mock_data"
    BROKEN_LOGIC = "broken_logic"
    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"
    SECURITY_VULNERABILITY = "security_vulnerability"


class Severity(str, Enum):
    """Issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Context information for logging."""
    request_id: str = "unknown"
    user_id: str = "anonymous"
    project_id: str = "none"
    session_id: str = "none"
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    cpu_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_percent: float = 0.0
    response_time_ms: float = 0.0
    active_connections: int = 0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FileInfo:
    """Information about a project file."""
    path: Path
    size: int = 0
    last_modified: datetime = field(default_factory=datetime.utcnow)
    validation_score: float = 0.0
    status: str = "unknown"
    checksum: Optional[str] = None


@dataclass
class ProgressMetrics:
    """Progress tracking metrics."""
    real_progress: float = 0.0
    fake_progress: float = 0.0
    authenticity_rate: float = 100.0
    quality_score: float = 0.0
    tasks_completed: int = 0
    tasks_total: int = 0
    estimated_completion: Optional[datetime] = None

    @property
    def total_progress(self) -> float:
        """Calculate total progress percentage."""
        return self.real_progress + self.fake_progress

    def update_authenticity_rate(self) -> None:
        """Recalculate authenticity rate based on real vs fake progress."""
        total = self.total_progress
        if total > 0:
            self.authenticity_rate = (self.real_progress / total) * 100
        else:
            self.authenticity_rate = 100.0


class Issue(BaseModel):
    """Validation issue model."""
    id: UUID = Field(default_factory=uuid4)
    type: IssueType
    severity: Severity
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    auto_fix_available: bool = False
    suggested_fix: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ValidationResult(BaseModel):
    """Validation pipeline result."""
    is_authentic: bool
    authenticity_score: float = Field(ge=0.0, le=100.0)
    real_progress: float = Field(ge=0.0, le=100.0)
    fake_progress: float = Field(ge=0.0, le=100.0)
    issues: List[Issue] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('authenticity_score')
    def validate_authenticity_score(cls, v: float) -> float:
        """Ensure authenticity score is within valid range."""
        return max(0.0, min(100.0, v))

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ProgressReport(BaseModel):
    """Comprehensive progress reporting."""
    project_id: UUID
    metrics: ProgressMetrics
    validation: ValidationResult
    recent_activities: List[str] = Field(default_factory=list)
    performance_data: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


@dataclass
class AITaskResult:
    """Result from AI task execution."""
    task_id: UUID
    success: bool
    generated_content: Optional[str] = None
    files_modified: List[str] = field(default_factory=list)
    validation_score: float = 0.0
    tokens_used: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class DevelopmentResult:
    """Result from development workflow execution."""
    workflow_id: UUID
    success: bool
    tasks_executed: List[UUID] = field(default_factory=list)
    files_generated: List[str] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    total_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    error_details: Optional[str] = None


class ProjectConfig(BaseModel):
    """Project configuration model."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    template: str = "basic"
    framework: str = "python"
    database: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    
    # AI configuration
    ai_settings: Dict[str, Any] = Field(default_factory=lambda: {
        "auto_completion": True,
        "validation_level": "strict",
        "anti_hallucination": True,
        "max_tokens_per_request": 4000,
        "timeout_seconds": 300
    })
    
    # Validation configuration  
    validation_config: Dict[str, Any] = Field(default_factory=lambda: {
        "placeholder_tolerance": 5,
        "min_test_coverage": 80,
        "max_complexity": 10,
        "auto_fix_enabled": True,
        "validation_interval": 30
    })
    
    # Workflow configuration
    workflow_config: Dict[str, Any] = Field(default_factory=lambda: {
        "max_concurrent_tasks": 5,
        "retry_attempts": 3,
        "parallel_execution": True,
        "dependency_timeout": 600
    })

    @field_validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate project name."""
        if not v.strip():
            raise ValueError("Project name cannot be empty")
        return v.strip()

    @field_validator('features')  
    def validate_features(cls, v: List[str]) -> List[str]:
        """Validate project features."""
        valid_features = {
            'auth', 'api', 'database', 'testing', 'docker', 
            'ci_cd', 'monitoring', 'documentation'
        }
        invalid = set(v) - valid_features
        if invalid:
            raise ValueError(f"Invalid features: {invalid}")
        return v

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    project_id: UUID
    overall_authenticity: float
    quality_metrics: Dict[str, float]
    issues: List[Issue]
    metrics: Dict[str, int]
    last_validated: datetime
    next_validation: Optional[datetime] = None

    def __post_init__(self):
        """Post-initialization validation."""
        if not 0.0 <= self.overall_authenticity <= 100.0:
            raise ValueError("Authenticity must be between 0 and 100")


class Task(BaseModel):
    """Task model for workflow management."""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    dependencies: Set[UUID] = Field(default_factory=set)
    ai_prompt: Optional[str] = None
    expected_outputs: List[str] = Field(default_factory=list)
    validation_criteria: Dict[str, Any] = Field(default_factory=dict)
    progress: ProgressMetrics = Field(default_factory=ProgressMetrics)
    estimated_duration: Optional[int] = None  # minutes
    actual_duration: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @field_validator('dependencies', mode='before')
    def convert_dependencies(cls, v):
        """Convert dependencies to set if needed."""
        if isinstance(v, list):
            return set(v)
        return v

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.actual_duration = int(delta.total_seconds() / 60)

    def fail(self, error: str) -> None:
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.error_message = error
        if self.started_at:
            delta = datetime.utcnow() - self.started_at
            self.actual_duration = int(delta.total_seconds() / 60)

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            set: list
        }


class Workflow(BaseModel):
    """Workflow model for orchestrating multiple tasks."""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    description: str = ""
    tasks: List[Task] = Field(default_factory=list)
    global_settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING

    def add_task(self, task: Task) -> None:
        """Add task to workflow."""
        self.tasks.append(task)

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies met)."""
        completed_task_ids = {
            task.id for task in self.tasks 
            if task.status == TaskStatus.COMPLETED
        }
        
        ready_tasks = []
        for task in self.tasks:
            if (task.status == TaskStatus.PENDING and 
                task.dependencies.issubset(completed_task_ids)):
                ready_tasks.append(task)
        
        return ready_tasks

    def is_complete(self) -> bool:
        """Check if all tasks in workflow are completed."""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)

    def has_failed_tasks(self) -> bool:
        """Check if any tasks have failed."""
        return any(task.status == TaskStatus.FAILED for task in self.tasks)

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


# Context managers and utilities

class AsyncContextManager:
    """Base async context manager for resource management."""
    
    async def __aenter__(self):
        """Enter async context."""
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        await self.cleanup()
    
    async def setup(self) -> None:
        """Setup resources."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources.""" 
        pass


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used: int = 0
    disk_percent: float = 0.0
    active_tasks: int = 0
    cache_hit_rate: float = 0.0
    ai_response_time: float = 0.0


# Additional enums needed by other modules

class ExecutionStrategy(str, Enum):
    """Task execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel" 
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"


class TaskType(str, Enum):
    """Types of development tasks."""
    CODING = "coding"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    REVIEW = "review"
    SETUP = "setup"
    OPTIMIZATION = "optimization"


# Factory functions for creating common objects

def create_task(
    name: str,
    description: str,
    task_type: TaskType = TaskType.CODING,
    priority: Priority = Priority.MEDIUM,
    estimated_duration: Optional[int] = None,
    dependencies: Optional[List[UUID]] = None,
    **kwargs
) -> Task:
    """
    Factory function to create a task with common defaults.
    
    Args:
        name: Task name
        description: Task description
        task_type: Type of task
        priority: Task priority
        estimated_duration: Estimated duration in minutes
        dependencies: Task dependencies (UUIDs)
        **kwargs: Additional task fields
    
    Returns:
        Created task instance
    """
    task_data = {
        'name': name,
        'description': description,
        'priority': priority,
        'estimated_duration': estimated_duration,
        'dependencies': set(dependencies) if dependencies else set(),
        **kwargs
    }
    
    return Task(**task_data)


def create_workflow(
    name: str,
    description: str = "",
    tasks: Optional[List[Task]] = None,
    **kwargs
) -> Workflow:
    """
    Factory function to create a workflow with tasks.
    
    Args:
        name: Workflow name
        description: Workflow description
        tasks: Initial tasks to add
        **kwargs: Additional workflow fields
    
    Returns:
        Created workflow instance
    """
    workflow = Workflow(
        name=name,
        description=description,
        tasks=tasks or [],
        **kwargs
    )
    
    return workflow


def create_validation_issue(
    issue_type: IssueType,
    severity: Severity,
    description: str,
    file_path: Optional[str] = None,
    line_number: Optional[int] = None,
    suggestion: Optional[str] = None,
    auto_fixable: bool = False
) -> Issue:
    """
    Factory function to create a validation issue.
    
    Args:
        issue_type: Type of issue
        severity: Issue severity
        description: Issue description
        file_path: File where issue was found
        line_number: Line number of issue
        suggestion: Suggested fix
        auto_fixable: Whether issue can be auto-fixed
    
    Returns:
        Created validation issue
    """
    return Issue(
        type=issue_type,
        severity=severity,
        description=description,
        file_path=file_path,
        line_number=line_number,
        suggested_fix=suggestion,
        auto_fix_available=auto_fixable
    )


# Type aliases for better readability
ProjectDict = Dict[str, Any]
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, Union[int, float, str]]
PathStr = Union[str, Path]


# Additional dataclasses needed for compatibility
# Note: Project class is implemented in project_manager.py to avoid circular imports


# Exception classes for better error handling

class ClaudeTIUException(Exception):
    """Base exception for claude-tiu."""
    pass


class TaskException(ClaudeTIUException):
    """Task-related exceptions."""
    pass


class WorkflowException(ClaudeTIUException):
    """Workflow-related exceptions."""
    pass


class ValidationException(ClaudeTIUException):
    """Validation-related exceptions."""
    pass


class AIInterfaceException(ClaudeTIUException):
    """AI interface exceptions."""
    pass