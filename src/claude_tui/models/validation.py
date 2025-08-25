"""
Validation models and data structures for Claude-TUI.

Contains Pydantic models for validation results, authenticity scoring,
and anti-hallucination engine output.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationType(str, Enum):
    """Types of validation performed."""
    PLACEHOLDER_DETECTION = "placeholder_detection"
    SEMANTIC_ANALYSIS = "semantic_analysis" 
    EXECUTION_TEST = "execution_test"
    AUTHENTICITY_CHECK = "authenticity_check"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_CHECK = "performance_check"


class SeverityLevel(str, Enum):
    """Issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Alias for backwards compatibility
ValidationSeverity = SeverityLevel


class ValidationIssue(BaseModel):
    """Individual validation issue."""
    description: str = Field(..., description="Description of the issue")
    severity: SeverityLevel = Field(..., description="Issue severity level")
    line_number: Optional[int] = Field(None, description="Line number where issue occurs")
    column_number: Optional[int] = Field(None, description="Column number where issue occurs")
    file_path: Optional[str] = Field(None, description="File path containing the issue")
    issue_type: str = Field(..., description="Type of issue")
    suggested_fix: Optional[str] = Field(None, description="Suggested fix for the issue")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in issue detection")


class PlaceholderDetectionResult(BaseModel):
    """Results from placeholder detection."""
    placeholders_found: int = Field(0, ge=0, description="Number of placeholders found")
    placeholder_patterns: List[str] = Field(default_factory=list, description="Detected placeholder patterns")
    completion_suggestions: List[str] = Field(default_factory=list, description="Suggested completions")
    fake_progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Percentage of fake progress")


class SemanticAnalysisResult(BaseModel):
    """Results from semantic code analysis."""
    syntax_valid: bool = Field(True, description="Whether syntax is valid")
    semantic_coherence_score: float = Field(1.0, ge=0.0, le=1.0, description="Semantic coherence score")
    complexity_score: float = Field(0.0, ge=0.0, le=10.0, description="Code complexity score")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality metrics")


class ExecutionTestResult(BaseModel):
    """Results from code execution testing."""
    execution_successful: bool = Field(False, description="Whether execution was successful")
    execution_time: float = Field(0.0, ge=0.0, description="Execution time in seconds")
    output: str = Field("", description="Execution output")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    test_coverage: float = Field(0.0, ge=0.0, le=1.0, description="Test coverage percentage")


class AuthenticityScore(BaseModel):
    """Authenticity scoring results."""
    overall_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall authenticity score")
    human_like_probability: float = Field(0.0, ge=0.0, le=1.0, description="Probability code was written by human")
    ai_generated_probability: float = Field(0.0, ge=0.0, le=1.0, description="Probability code was AI generated")
    hallucination_probability: float = Field(0.0, ge=0.0, le=1.0, description="Probability of hallucination")
    confidence_level: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in scoring")


class ValidationResult(BaseModel):
    """Comprehensive validation result."""
    validation_id: str = Field(..., description="Unique validation identifier")
    status: ValidationStatus = Field(..., description="Validation status")
    validation_types: List[ValidationType] = Field(..., description="Types of validation performed")
    
    # Core results
    is_authentic: bool = Field(True, description="Whether code is authentic")
    authenticity_score: AuthenticityScore = Field(..., description="Authenticity scoring")
    
    # Specific validation results
    placeholder_detection: PlaceholderDetectionResult = Field(..., description="Placeholder detection results")
    semantic_analysis: SemanticAnalysisResult = Field(..., description="Semantic analysis results") 
    execution_test: Optional[ExecutionTestResult] = Field(None, description="Execution test results")
    
    # Issues and suggestions
    issues: List[ValidationIssue] = Field(default_factory=list, description="Issues found")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    auto_fix_suggestions: List[str] = Field(default_factory=list, description="Auto-fix suggestions")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    execution_time: float = Field(0.0, ge=0.0, description="Total validation execution time")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    # Performance metrics
    performance_score: float = Field(1.0, ge=0.0, le=1.0, description="Performance score")
    quality_score: float = Field(1.0, ge=0.0, le=1.0, description="Overall quality score")
    
    @validator('authenticity_score', always=True)
    def validate_authenticity_score(cls, v):
        """Validate authenticity score consistency."""
        if v.human_like_probability + v.ai_generated_probability > 1.1:
            raise ValueError("Human-like + AI probabilities cannot exceed 1.0")
        return v
    
    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get only critical severity issues."""
        return [issue for issue in self.issues if issue.severity == SeverityLevel.CRITICAL]
    
    def get_overall_score(self) -> float:
        """Calculate overall validation score."""
        scores = [
            self.authenticity_score.overall_score,
            self.semantic_analysis.semantic_coherence_score,
            self.performance_score,
            self.quality_score
        ]
        return sum(scores) / len(scores)


class CompletionRequest(BaseModel):
    """Request for code completion."""
    code: str = Field(..., description="Code to complete")
    context: Dict[str, Any] = Field(default_factory=dict, description="Completion context")
    completion_type: str = Field("auto", description="Type of completion requested")
    max_suggestions: int = Field(5, ge=1, le=20, description="Maximum number of suggestions")


class CompletionResult(BaseModel):
    """Result from code completion."""
    original_code: str = Field(..., description="Original code")
    completed_code: str = Field(..., description="Completed code")
    suggestions: List[str] = Field(default_factory=list, description="Alternative suggestions")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in completion")
    completion_time: float = Field(0.0, ge=0.0, description="Time taken for completion")


class CrossValidationResult(BaseModel):
    """Results from cross-validation with multiple models."""
    model_results: Dict[str, ValidationResult] = Field(..., description="Results from each model")
    consensus_score: float = Field(0.0, ge=0.0, le=1.0, description="Consensus score across models")
    agreement_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Agreement percentage")
    final_verdict: bool = Field(True, description="Final consensus verdict")
    disagreement_areas: List[str] = Field(default_factory=list, description="Areas of disagreement")


@dataclass
class ProgressReport:
    """Progress validation report for UI display."""
    real_progress: float  # 0.0 to 1.0
    claimed_progress: float  # 0.0 to 1.0
    fake_progress: float  # 0.0 to 1.0  
    quality_score: float  # 0.0 to 10.0
    authenticity_score: float  # 0.0 to 1.0
    placeholders_found: int
    todos_found: int
    validation_timestamp: datetime = None
    
    def __post_init__(self):
        if self.validation_timestamp is None:
            self.validation_timestamp = datetime.now()
    
    @property
    def is_authentic(self) -> bool:
        """Check if progress appears authentic."""
        return self.authenticity_score > 0.8 and self.fake_progress < 0.2
    
    @property
    def needs_attention(self) -> bool:
        """Check if progress needs immediate attention."""
        return (self.fake_progress > 0.3 or 
                self.authenticity_score < 0.6 or
                self.placeholders_found > 10)


# Export main classes
__all__ = [
    'ValidationStatus',
    'ValidationType',
    'SeverityLevel',
    'ValidationSeverity',  # Alias for backwards compatibility
    'ValidationIssue',
    'PlaceholderDetectionResult',
    'SemanticAnalysisResult',
    'ExecutionTestResult',
    'AuthenticityScore',
    'ValidationResult',
    'CompletionRequest',
    'CompletionResult',
    'CrossValidationResult',
    'ProgressReport'
]