"""
Template Validation Models - Models for template validation and quality scoring.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"
    SKIPPED = "skipped"


class ValidationSeverity(str, Enum):
    """Validation issue severity."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityMetric(str, Enum):
    """Quality metric types."""
    COMPLETENESS = "completeness"
    DOCUMENTATION = "documentation"
    CODE_QUALITY = "code_quality"
    STRUCTURE = "structure"
    USABILITY = "usability"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"


class TemplateValidation(Base):
    """Template validation results."""
    
    __tablename__ = "template_validations"
    
    # Primary identifiers
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    template_version = Column(String(20))
    
    # Validation metadata
    validation_type = Column(String(50), nullable=False)  # automated, manual, security
    validator_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    
    # Overall status
    status = Column(String(20), nullable=False, default=ValidationStatus.PENDING.value)
    overall_score = Column(Float, default=0.0)
    
    # Individual metrics
    completeness_score = Column(Float, default=0.0)
    documentation_score = Column(Float, default=0.0)
    code_quality_score = Column(Float, default=0.0)
    structure_score = Column(Float, default=0.0)
    usability_score = Column(Float, default=0.0)
    security_score = Column(Float, default=0.0)
    performance_score = Column(Float, default=0.0)
    maintainability_score = Column(Float, default=0.0)
    
    # Validation details
    issues = Column(JSONB, default=list)
    warnings = Column(JSONB, default=list)
    suggestions = Column(JSONB, default=list)
    
    # Check results
    has_readme = Column(Boolean, default=False)
    has_examples = Column(Boolean, default=False)
    has_tests = Column(Boolean, default=False)
    has_documentation = Column(Boolean, default=False)
    has_license = Column(Boolean, default=False)
    
    # File analysis
    total_files = Column(Integer, default=0)
    code_files = Column(Integer, default=0)
    config_files = Column(Integer, default=0)
    doc_files = Column(Integer, default=0)
    test_files = Column(Integer, default=0)
    
    # Security checks
    security_issues_count = Column(Integer, default=0)
    has_secrets = Column(Boolean, default=False)
    has_vulnerabilities = Column(Boolean, default=False)
    
    # Validation metadata
    validation_duration = Column(Float)  # seconds
    validation_tools_used = Column(JSONB, default=list)
    
    # Comments and notes
    reviewer_notes = Column(Text)
    auto_generated_feedback = Column(Text)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    template = relationship("Template")
    validator = relationship("UserProfile")
    
    def calculate_weighted_score(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'completeness_score': 0.20,
            'documentation_score': 0.15,
            'code_quality_score': 0.20,
            'structure_score': 0.15,
            'usability_score': 0.10,
            'security_score': 0.10,
            'performance_score': 0.05,
            'maintainability_score': 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            score = getattr(self, metric, 0.0)
            if score > 0:
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation to dictionary."""
        return {
            'id': str(self.id),
            'template_id': str(self.template_id),
            'template_version': self.template_version,
            'validation_type': self.validation_type,
            'validator_id': str(self.validator_id) if self.validator_id else None,
            'status': self.status,
            'overall_score': self.overall_score,
            'completeness_score': self.completeness_score,
            'documentation_score': self.documentation_score,
            'code_quality_score': self.code_quality_score,
            'structure_score': self.structure_score,
            'usability_score': self.usability_score,
            'security_score': self.security_score,
            'performance_score': self.performance_score,
            'maintainability_score': self.maintainability_score,
            'issues': self.issues,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'has_readme': self.has_readme,
            'has_examples': self.has_examples,
            'has_tests': self.has_tests,
            'has_documentation': self.has_documentation,
            'has_license': self.has_license,
            'security_issues_count': self.security_issues_count,
            'validation_duration': self.validation_duration,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class QualityScore(Base):
    """Quality scoring history and trends."""
    
    __tablename__ = "quality_scores"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    validation_id = Column(PG_UUID(as_uuid=True), ForeignKey("template_validations.id"))
    
    # Score details
    metric_type = Column(String(50), nullable=False)  # QualityMetric enum value
    score = Column(Float, nullable=False)
    max_score = Column(Float, default=100.0)
    
    # Score breakdown
    score_details = Column(JSONB)  # Detailed scoring breakdown
    improvement_suggestions = Column(JSONB, default=list)
    
    # Context
    scoring_algorithm = Column(String(50))
    algorithm_version = Column(String(20))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    template = relationship("Template")
    validation = relationship("TemplateValidation")


class ValidationRule(Base):
    """Template validation rules configuration."""
    
    __tablename__ = "validation_rules"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Rule identification
    rule_name = Column(String(100), unique=True, nullable=False)
    rule_type = Column(String(50), nullable=False)  # structure, content, security, quality
    description = Column(Text)
    
    # Rule configuration
    rule_config = Column(JSONB)
    severity = Column(String(20), nullable=False, default=ValidationSeverity.WARNING.value)
    
    # Rule metadata
    is_active = Column(Boolean, default=True)
    is_required = Column(Boolean, default=False)
    applies_to_types = Column(JSONB, default=list)  # Template types this rule applies to
    
    # Scoring
    weight = Column(Float, default=1.0)
    penalty_score = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ValidationIssue(Base):
    """Individual validation issues."""
    
    __tablename__ = "validation_issues"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    validation_id = Column(PG_UUID(as_uuid=True), ForeignKey("template_validations.id"), nullable=False)
    rule_id = Column(PG_UUID(as_uuid=True), ForeignKey("validation_rules.id"))
    
    # Issue details
    issue_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    description = Column(Text)
    
    # Location information
    file_path = Column(String(500))
    line_number = Column(Integer)
    column_number = Column(Integer)
    
    # Fix suggestions
    suggested_fix = Column(Text)
    auto_fixable = Column(Boolean, default=False)
    
    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text)
    resolved_by_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    resolved_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    validation = relationship("TemplateValidation")
    rule = relationship("ValidationRule")
    resolved_by = relationship("UserProfile")


# Pydantic models for API

class ValidationIssueBase(BaseModel):
    """Base validation issue schema."""
    issue_type: str
    severity: ValidationSeverity
    message: str
    description: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False


class ValidationIssueResponse(ValidationIssueBase):
    """Schema for validation issue responses."""
    id: UUID
    validation_id: UUID
    rule_id: Optional[UUID] = None
    is_resolved: bool
    resolution_notes: Optional[str] = None
    resolved_by_id: Optional[UUID] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class TemplateValidationBase(BaseModel):
    """Base template validation schema."""
    template_version: Optional[str] = None
    validation_type: str
    reviewer_notes: Optional[str] = None


class TemplateValidationCreate(TemplateValidationBase):
    """Schema for creating template validations."""
    template_id: UUID


class TemplateValidationResponse(TemplateValidationBase):
    """Schema for template validation responses."""
    id: UUID
    template_id: UUID
    validator_id: Optional[UUID] = None
    status: ValidationStatus
    overall_score: float
    completeness_score: float
    documentation_score: float
    code_quality_score: float
    structure_score: float
    usability_score: float
    security_score: float
    performance_score: float
    maintainability_score: float
    issues: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    has_readme: bool
    has_examples: bool
    has_tests: bool
    has_documentation: bool
    has_license: bool
    security_issues_count: int
    total_files: int
    validation_duration: Optional[float] = None
    validation_tools_used: List[str]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class QualityScoreResponse(BaseModel):
    """Schema for quality score responses."""
    id: UUID
    template_id: UUID
    validation_id: Optional[UUID] = None
    metric_type: QualityMetric
    score: float
    max_score: float
    score_details: Optional[Dict[str, Any]] = None
    improvement_suggestions: List[str]
    scoring_algorithm: Optional[str] = None
    algorithm_version: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class ValidationRuleCreate(BaseModel):
    """Schema for creating validation rules."""
    rule_name: str = Field(..., min_length=3, max_length=100)
    rule_type: str
    description: Optional[str] = None
    rule_config: Dict[str, Any] = Field(default_factory=dict)
    severity: ValidationSeverity = ValidationSeverity.WARNING
    is_required: bool = False
    applies_to_types: List[str] = Field(default_factory=list)
    weight: float = Field(1.0, ge=0, le=10)
    penalty_score: float = Field(0.0, ge=0, le=100)


class ValidationRuleResponse(BaseModel):
    """Schema for validation rule responses."""
    id: UUID
    rule_name: str
    rule_type: str
    description: Optional[str] = None
    rule_config: Dict[str, Any]
    severity: ValidationSeverity
    is_active: bool
    is_required: bool
    applies_to_types: List[str]
    weight: float
    penalty_score: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ValidationSummary(BaseModel):
    """Validation summary for templates."""
    template_id: UUID
    latest_validation_id: Optional[UUID] = None
    overall_score: float
    status: ValidationStatus
    issue_count: int
    warning_count: int
    critical_issues: List[ValidationIssueResponse]
    last_validated: Optional[datetime] = None
    requires_revalidation: bool = False


class ValidationReport(BaseModel):
    """Comprehensive validation report."""
    template_id: UUID
    template_name: str
    template_version: str
    validation_summary: ValidationSummary
    quality_scores: Dict[str, float]
    issues_by_severity: Dict[str, List[ValidationIssueResponse]]
    improvements_needed: List[str]
    compliance_status: Dict[str, bool]
    validation_history: List[Dict[str, Any]]
    generated_at: datetime


class BulkValidationRequest(BaseModel):
    """Request for bulk template validation."""
    template_ids: List[UUID]
    validation_type: str = "automated"
    force_revalidation: bool = False
    include_security_scan: bool = True


class BulkValidationResponse(BaseModel):
    """Response for bulk template validation."""
    request_id: UUID
    total_templates: int
    validations_started: int
    validations_failed: int
    estimated_completion: datetime
    status: str