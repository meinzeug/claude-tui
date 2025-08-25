"""
AI Models - Data models for AI service integration.

Defines Pydantic models for AI requests, responses, and related data structures
used throughout the AI interface and validation systems.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class AIRequestType(str, Enum):
    """Types of AI requests."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_REFACTORING = "code_refactoring"
    WORKFLOW_EXECUTION = "workflow_execution"
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    TASK_ORCHESTRATION = "task_orchestration"


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Base AI Models
class AIRequest(BaseModel):
    """Base AI request model."""
    id: str = Field(..., description="Request identifier")
    request_type: AIRequestType = Field(..., description="Type of AI request")
    prompt: str = Field(..., description="AI prompt")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request context")
    created_at: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    timeout: Optional[int] = Field(None, description="Request timeout in seconds")


class AIResponse(BaseModel):
    """Base AI response model."""
    request_id: str = Field(..., description="Original request ID")
    success: bool = Field(..., description="Request success status")
    content: str = Field(default="", description="Response content")
    model_used: Optional[str] = Field(None, description="AI model used")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# Code-specific Models
class CodeContext(BaseModel):
    """Context for code generation requests."""
    language: str = Field(..., description="Programming language")
    framework: Optional[str] = Field(None, description="Framework to use")
    file_path: Optional[str] = Field(None, description="Target file path")
    existing_code: Optional[str] = Field(None, description="Existing code context")
    project_structure: Dict[str, Any] = Field(default_factory=dict, description="Project structure")
    dependencies: List[str] = Field(default_factory=list, description="Project dependencies")
    coding_standards: Dict[str, str] = Field(default_factory=dict, description="Coding standards")


class CodeResult(BaseModel):
    """Result of code generation."""
    success: bool = Field(..., description="Generation success status")
    content: str = Field(default="", description="Generated code")
    language: Optional[str] = Field(None, description="Code language")
    generated_files: List[str] = Field(default_factory=list, description="Generated file paths")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Code quality score")
    validation_passed: bool = Field(default=False, description="Validation status")
    execution_time: float = Field(default=0.0, description="Generation time")
    model_used: Optional[str] = Field(None, description="AI model used")
    error_message: Optional[str] = Field(None, description="Error message")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


# Review Models
class ReviewCriteria(BaseModel):
    """Criteria for code review."""
    focus_areas: List[str] = Field(default_factory=list, description="Areas to focus on")
    min_severity: Optional[ValidationSeverity] = Field(None, description="Minimum issue severity")
    include_suggestions: bool = Field(True, description="Include improvement suggestions")
    include_compliments: bool = Field(True, description="Include positive feedback")
    style_guide: Optional[str] = Field(None, description="Style guide to follow")
    security_check: bool = Field(True, description="Include security checks")
    performance_check: bool = Field(True, description="Include performance checks")


class ReviewIssue(BaseModel):
    """Individual review issue."""
    description: str = Field(..., description="Issue description")
    severity: ValidationSeverity = Field(..., description="Issue severity")
    line_number: Optional[int] = Field(None, description="Line number")
    column_number: Optional[int] = Field(None, description="Column number")
    category: str = Field(default="general", description="Issue category")
    suggested_fix: Optional[str] = Field(None, description="Suggested fix")


class CodeReview(BaseModel):
    """Code review result."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall code score")
    issues: List[ReviewIssue] = Field(default_factory=list, description="Found issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    compliments: List[str] = Field(default_factory=list, description="Positive aspects")
    summary: str = Field(default="", description="Review summary")
    categories_analyzed: List[str] = Field(default_factory=list, description="Analyzed categories")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Code metrics")


# Workflow Models
class WorkflowRequest(BaseModel):
    """Workflow execution request."""
    workflow_name: str = Field(..., description="Workflow name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Workflow parameters")
    variables: Dict[str, str] = Field(default_factory=dict, description="Template variables")
    priority: str = Field(default="medium", description="Execution priority")
    timeout: Optional[int] = Field(None, description="Execution timeout")
    requirements: List[str] = Field(default_factory=list, description="Workflow requirements")


class WorkflowResult(BaseModel):
    """Workflow execution result."""
    workflow_id: str = Field(..., description="Workflow identifier")
    success: bool = Field(..., description="Execution success")
    output: str = Field(default="", description="Workflow output")
    execution_time: float = Field(default=0.0, description="Execution time")
    agent_count: int = Field(default=0, description="Number of agents used")
    task_count: int = Field(default=0, description="Number of tasks executed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Swarm Models
class SwarmTopology(str, Enum):
    """Swarm topology types."""
    MESH = "mesh"
    HIERARCHICAL = "hierarchical"
    RING = "ring"
    STAR = "star"


class SwarmConfig(BaseModel):
    """Swarm configuration."""
    topology: SwarmTopology = Field(..., description="Swarm topology")
    max_agents: int = Field(default=5, ge=1, le=100, description="Maximum number of agents")
    strategy: str = Field(default="adaptive", description="Distribution strategy")
    coordination_enabled: bool = Field(default=True, description="Enable agent coordination")
    persistence_enabled: bool = Field(default=False, description="Enable state persistence")


class AgentConfig(BaseModel):
    """Agent configuration."""
    type: str = Field(..., description="Agent type")
    name: Optional[str] = Field(None, description="Agent name")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    resources: Dict[str, Any] = Field(default_factory=dict, description="Agent resources")
    learning_enabled: bool = Field(default=False, description="Enable learning")
    coordination_enabled: bool = Field(default=True, description="Enable coordination")


# Task Models
class TaskOrchestrationRequest(BaseModel):
    """Task orchestration request."""
    description: str = Field(..., description="Task description")
    priority: str = Field(default="medium", description="Task priority")
    max_agents: Optional[int] = Field(None, description="Maximum agents to use")
    strategy: str = Field(default="adaptive", description="Orchestration strategy")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Task requirements")
    timeout: Optional[int] = Field(None, description="Task timeout")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")


class OrchestrationResult(BaseModel):
    """Task orchestration result."""
    task_id: Optional[str] = Field(None, description="Task identifier")
    success: bool = Field(..., description="Orchestration success")
    assigned_agents: List[str] = Field(default_factory=list, description="Assigned agent IDs")
    execution_time: float = Field(default=0.0, description="Execution time")
    result_data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")


# Validation Models
class ValidationContext(BaseModel):
    """Context for validation operations."""
    file_path: Optional[str] = Field(None, description="File being validated")
    language: Optional[str] = Field(None, description="Programming language")
    project_context: Dict[str, Any] = Field(default_factory=dict, description="Project context")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rules to apply")
    strictness_level: str = Field(default="medium", description="Validation strictness")


class ValidationResult(BaseModel):
    """Validation result."""
    is_valid: bool = Field(..., description="Overall validation status")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Validation score")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Validation issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    execution_time: float = Field(default=0.0, description="Validation time")
    validator_used: str = Field(default="default", description="Validator used")


# Analysis Models
class RequirementsAnalysis(BaseModel):
    """Requirements analysis result."""
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Requirements complexity")
    estimated_effort: int = Field(default=0, description="Estimated effort in hours")
    risk_level: str = Field(default="medium", description="Project risk level")
    recommended_approach: str = Field(default="", description="Recommended implementation approach")
    key_challenges: List[str] = Field(default_factory=list, description="Identified challenges")
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    dependencies: List[str] = Field(default_factory=list, description="External dependencies")
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Project milestones")


class ContextAnalysis(BaseModel):
    """Context analysis result."""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Context relevance")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts identified")
    relationships: Dict[str, List[str]] = Field(default_factory=dict, description="Concept relationships")
    missing_context: List[str] = Field(default_factory=list, description="Missing context elements")
    recommendations: List[str] = Field(default_factory=list, description="Context recommendations")


# Performance Models
class PerformanceMetrics(BaseModel):
    """Performance metrics."""
    response_time: float = Field(default=0.0, description="Response time in seconds")
    throughput: float = Field(default=0.0, description="Requests per second")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Error rate")
    resource_usage: Dict[str, float] = Field(default_factory=dict, description="Resource usage metrics")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality metrics")


class BenchmarkResult(BaseModel):
    """Benchmark execution result."""
    benchmark_name: str = Field(..., description="Benchmark name")
    execution_time: float = Field(..., description="Total execution time")
    iterations: int = Field(..., description="Number of iterations")
    metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    baseline_comparison: Optional[Dict[str, float]] = Field(None, description="Comparison to baseline")
    environment_info: Dict[str, str] = Field(default_factory=dict, description="Environment information")


# Configuration Models
class AIServiceConfig(BaseModel):
    """AI service configuration."""
    service_name: str = Field(..., description="Service name")
    endpoint_url: str = Field(..., description="Service endpoint URL")
    api_key_ref: Optional[str] = Field(None, description="API key reference")
    timeout: int = Field(default=300, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")
    model_preferences: Dict[str, str] = Field(default_factory=dict, description="Model preferences")
    enabled: bool = Field(default=True, description="Service enabled status")


class IntegrationConfig(BaseModel):
    """Integration configuration."""
    claude_code_enabled: bool = Field(default=True, description="Claude Code integration enabled")
    claude_flow_enabled: bool = Field(default=True, description="Claude Flow integration enabled")
    validation_enabled: bool = Field(default=True, description="Validation enabled")
    auto_fix_enabled: bool = Field(default=True, description="Auto-fix enabled")
    execution_testing_enabled: bool = Field(default=True, description="Execution testing enabled")
    performance_monitoring: bool = Field(default=True, description="Performance monitoring enabled")
    
    # Service-specific settings
    claude_code_path: Optional[str] = Field(None, description="Claude Code executable path")
    claude_flow_endpoint: str = Field(default="http://localhost:3000", description="Claude Flow endpoint")
    
    # Safety settings
    sandbox_enabled: bool = Field(default=True, description="Sandbox execution enabled")
    max_execution_time: int = Field(default=30, description="Maximum execution time")
    max_memory_usage: int = Field(default=256, description="Maximum memory usage in MB")


# Error Models
class AIError(BaseModel):
    """AI service error."""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    service_name: str = Field(..., description="Service that generated the error")
    request_id: Optional[str] = Field(None, description="Associated request ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    retryable: bool = Field(default=False, description="Whether error is retryable")


# Response aggregation models
class AggregatedResponse(BaseModel):
    """Aggregated response from multiple AI services."""
    primary_response: AIResponse = Field(..., description="Primary response")
    secondary_responses: List[AIResponse] = Field(default_factory=list, description="Secondary responses")
    consensus_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Response consensus score")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence score")
    aggregation_strategy: str = Field(default="weighted", description="Aggregation strategy used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Aggregation metadata")