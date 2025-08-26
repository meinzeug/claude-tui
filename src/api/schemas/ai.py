"""
Pydantic schemas for AI API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class OrchestrationStrategy(str, Enum):
    """Orchestration strategy options."""
    ADAPTIVE = "adaptive"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    BALANCED = "balanced"


class ValidationTypeEnum(str, Enum):
    """AI validation type options."""
    CODE = "code"
    TEXT = "text"
    GENERAL = "general"
    JSON = "json"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# Request Schemas
class CodeGenerationRequest(BaseModel):
    """Request schema for AI code generation."""
    prompt: str = Field(..., min_length=1, max_length=5000, description="Code generation prompt")
    language: str = Field(default="python", description="Target programming language")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    validate_response: bool = Field(default=True, description="Enable response validation")
    use_cache: bool = Field(default=True, description="Use response caching")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="AI creativity level")
    
    @validator('language')
    def validate_language(cls, v):
        allowed = [
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go', 
            'rust', 'php', 'html', 'css', 'sql', 'bash', 'yaml', 'json'
        ]
        if v.lower() not in allowed:
            raise ValueError(f'Language must be one of: {", ".join(allowed)}')
        return v.lower()


class TaskOrchestrationRequest(BaseModel):
    """Request schema for AI task orchestration."""
    task_description: str = Field(..., min_length=1, max_length=2000, description="Task description")
    requirements: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task requirements")
    agents: Optional[List[str]] = Field(default_factory=list, description="Specific agents to use")
    strategy: OrchestrationStrategy = Field(default=OrchestrationStrategy.ADAPTIVE, description="Orchestration strategy")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    max_agents: Optional[int] = Field(default=5, ge=1, le=20, description="Maximum number of agents")
    timeout: Optional[int] = Field(default=300, ge=30, le=3600, description="Timeout in seconds")


class ValidationRequest(BaseModel):
    """Request schema for AI response validation."""
    response: Union[str, Dict[str, Any]] = Field(..., description="Content to validate")
    validation_type: ValidationTypeEnum = Field(default=ValidationTypeEnum.GENERAL, description="Validation type")
    criteria: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validation criteria")
    strict_mode: bool = Field(default=False, description="Enable strict validation mode")
    language: Optional[str] = Field(None, description="Language for code validation")


class CodeReviewRequest(BaseModel):
    """Request schema for AI code review."""
    code: str = Field(..., min_length=1, max_length=10000, description="Code to review")
    language: str = Field(..., description="Programming language")
    review_criteria: Optional[List[str]] = Field(
        default_factory=lambda: ["quality", "security", "performance", "maintainability"],
        description="Review criteria"
    )
    severity_level: str = Field(default="medium", regex="^(low|medium|high|critical)$")


class RefactorRequest(BaseModel):
    """Request schema for AI code refactoring."""
    code: str = Field(..., min_length=1, max_length=10000, description="Code to refactor")
    language: str = Field(..., description="Programming language")
    instructions: str = Field(..., min_length=1, max_length=1000, description="Refactoring instructions")
    preserve_functionality: bool = Field(default=True, description="Preserve original functionality")
    optimization_targets: Optional[List[str]] = Field(
        default_factory=list,
        description="Optimization targets (performance, readability, maintainability)"
    )


# Response Schemas
class CodeGenerationResponse(BaseModel):
    """Response schema for code generation."""
    code: str = Field(..., description="Generated code")
    language: str = Field(..., description="Programming language")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    validation: Optional[Dict[str, Any]] = Field(None, description="Validation results")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    cached: bool = Field(default=False, description="Whether result was cached")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TaskOrchestrationResponse(BaseModel):
    """Response schema for task orchestration."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Orchestration status")
    agents_assigned: List[str] = Field(default_factory=list, description="Assigned agents")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Task progress (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Orchestration metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class ValidationResponse(BaseModel):
    """Response schema for AI response validation."""
    is_valid: bool = Field(..., description="Whether response is valid")
    score: float = Field(..., ge=0.0, le=1.0, description="Validation score (0-1)")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Validation metadata")
    validated_at: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class AIHealthResponse(BaseModel):
    """Response schema for AI service health."""
    status: str = Field(..., description="Overall health status")
    claude_code_available: bool = Field(..., description="Claude Code availability")
    claude_flow_available: bool = Field(..., description="Claude Flow availability")
    services: Dict[str, Any] = Field(default_factory=dict, description="Detailed service status")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class PerformanceMetricsResponse(BaseModel):
    """Response schema for AI performance metrics."""
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")
    average_response_time: float = Field(..., description="Average response time in seconds")
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Cache hit rate (0-1)")
    providers_available: int = Field(..., description="Number of available providers")
    timeframe: str = Field(..., description="Metrics timeframe")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics")
    collected_at: datetime = Field(default_factory=datetime.utcnow, description="Collection timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TaskStatusResponse(BaseModel):
    """Response schema for task status."""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Current status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress (0-1)")
    current_step: Optional[str] = Field(None, description="Current step description")
    completed_steps: List[str] = Field(default_factory=list, description="Completed steps")
    remaining_steps: List[str] = Field(default_factory=list, description="Remaining steps")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TaskResultsResponse(BaseModel):
    """Response schema for task results."""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Final status")
    results: Dict[str, Any] = Field(default_factory=dict, description="Task results")
    artifacts: List[str] = Field(default_factory=list, description="Generated artifacts")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    validation_report: Dict[str, Any] = Field(default_factory=dict, description="Validation report")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class CodeReviewResponse(BaseModel):
    """Response schema for code review."""
    review: str = Field(..., description="Detailed code review")
    language: str = Field(..., description="Programming language")
    issues_found: List[Dict[str, Any]] = Field(default_factory=list, description="Issues found")
    suggestions: List[Dict[str, Any]] = Field(default_factory=list, description="Improvement suggestions")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Code quality score")
    security_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Security score")
    maintainability_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Maintainability score")
    reviewed_at: datetime = Field(default_factory=datetime.utcnow, description="Review timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class RefactorResponse(BaseModel):
    """Response schema for code refactoring."""
    refactored_code: str = Field(..., description="Refactored code")
    original_code: str = Field(..., description="Original code")
    language: str = Field(..., description="Programming language")
    changes_summary: str = Field(..., description="Summary of changes made")
    improvements: List[str] = Field(default_factory=list, description="Improvements made")
    warnings: List[str] = Field(default_factory=list, description="Refactoring warnings")
    quality_improvement: float = Field(default=0.0, description="Quality improvement score")
    refactored_at: datetime = Field(default_factory=datetime.utcnow, description="Refactoring timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# Streaming Response Schemas
class StreamingChunk(BaseModel):
    """Schema for streaming response chunks."""
    chunk_type: str = Field(..., description="Type of chunk (data, status, error)")
    content: str = Field(..., description="Chunk content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Chunk metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Chunk timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# Batch Operation Schemas
class BatchCodeGenerationRequest(BaseModel):
    """Request schema for batch code generation."""
    requests: List[CodeGenerationRequest] = Field(..., min_items=1, max_items=10, description="Batch requests")
    parallel_execution: bool = Field(default=True, description="Execute requests in parallel")
    timeout: Optional[int] = Field(default=600, ge=30, le=3600, description="Timeout for batch operation")


class BatchCodeGenerationResponse(BaseModel):
    """Response schema for batch code generation."""
    results: List[CodeGenerationResponse] = Field(..., description="Batch results")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Batch summary")
    execution_time: float = Field(..., description="Total execution time")
    success_count: int = Field(..., description="Number of successful requests")
    error_count: int = Field(..., description="Number of failed requests")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# Error Response Schema
class AIErrorResponse(BaseModel):
    """Schema for AI API error responses."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }