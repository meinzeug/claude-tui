"""
Content Moderation Models - Advanced moderation system with AI assistance.
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


class ModerationAction(str, Enum):
    """Moderation action enumeration."""
    APPROVE = "approve"
    REJECT = "reject"
    FLAG = "flag"
    REMOVE = "remove"
    BAN = "ban"
    WARN = "warn"
    NO_ACTION = "no_action"


class ContentType(str, Enum):
    """Content type enumeration."""
    TEMPLATE = "template"
    PLUGIN = "plugin"
    REVIEW = "review"
    COMMENT = "comment"
    USER_PROFILE = "user_profile"
    COLLECTION = "collection"


class ViolationType(str, Enum):
    """Content violation type enumeration."""
    SPAM = "spam"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    OFFENSIVE_LANGUAGE = "offensive_language"
    COPYRIGHT_VIOLATION = "copyright_violation"
    MALICIOUS_CODE = "malicious_code"
    FAKE_CONTENT = "fake_content"
    HARASSMENT = "harassment"
    ADULT_CONTENT = "adult_content"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"


class ContentModerationEntry(Base):
    """Central moderation tracking for all content types."""
    
    __tablename__ = "content_moderation"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Content identification
    content_type = Column(String(20), nullable=False)
    content_id = Column(PG_UUID(as_uuid=True), nullable=False)
    content_author_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    
    # Moderation details
    status = Column(String(20), default="pending")  # pending, approved, rejected, escalated
    priority = Column(String(10), default="medium")  # low, medium, high, critical
    
    # Detection source
    detection_method = Column(String(20), nullable=False)  # ai_auto, user_report, manual_review
    reporter_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    
    # AI Analysis Results
    ai_confidence_score = Column(Float, default=0.0)  # 0-1
    ai_spam_probability = Column(Float, default=0.0)
    ai_toxicity_score = Column(Float, default=0.0)
    ai_adult_content_score = Column(Float, default=0.0)
    ai_violence_score = Column(Float, default=0.0)
    ai_hate_speech_score = Column(Float, default=0.0)
    ai_recommendation = Column(String(20))  # approve, reject, escalate
    
    # Detected violations
    violations = Column(JSONB, default=list)  # List of ViolationType values
    risk_factors = Column(JSONB, default=list)
    
    # Human moderation
    assigned_moderator_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    moderator_action = Column(String(20))  # ModerationAction
    moderator_notes = Column(Text)
    
    # Resolution
    resolution_reason = Column(String(100))
    action_taken = Column(String(50))
    user_notified = Column(Boolean, default=False)
    notification_sent_at = Column(DateTime)
    
    # Appeals
    appeal_count = Column(Integer, default=0)
    last_appeal_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Relationships
    content_author = relationship("UserProfile", foreign_keys=[content_author_id])
    reporter = relationship("UserProfile", foreign_keys=[reporter_id])
    assigned_moderator = relationship("UserProfile", foreign_keys=[assigned_moderator_id])
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert moderation entry to dictionary."""
        data = {
            'id': str(self.id),
            'content_type': self.content_type,
            'content_id': str(self.content_id),
            'status': self.status,
            'priority': self.priority,
            'detection_method': self.detection_method,
            'violations': self.violations,
            'moderator_action': self.moderator_action,
            'action_taken': self.action_taken,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }
        
        if include_sensitive:
            data.update({
                'ai_confidence_score': float(self.ai_confidence_score) if self.ai_confidence_score else 0.0,
                'ai_spam_probability': float(self.ai_spam_probability) if self.ai_spam_probability else 0.0,
                'ai_toxicity_score': float(self.ai_toxicity_score) if self.ai_toxicity_score else 0.0,
                'ai_recommendation': self.ai_recommendation,
                'risk_factors': self.risk_factors,
                'moderator_notes': self.moderator_notes,
                'resolution_reason': self.resolution_reason
            })
        
        return data


class SpamDetectionResult(Base):
    """Spam detection analysis results."""
    
    __tablename__ = "spam_detection_results"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    content_type = Column(String(20), nullable=False)
    content_id = Column(PG_UUID(as_uuid=True), nullable=False)
    
    # Detection results
    is_spam = Column(Boolean, default=False)
    confidence_score = Column(Float, default=0.0)  # 0-1
    
    # Spam indicators
    duplicate_content_score = Column(Float, default=0.0)
    suspicious_links_count = Column(Integer, default=0)
    promotional_keywords_count = Column(Integer, default=0)
    repetitive_patterns_score = Column(Float, default=0.0)
    
    # Content analysis
    text_quality_score = Column(Float, default=0.0)
    language_coherence_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    
    # Behavioral indicators
    author_reputation_score = Column(Float, default=0.0)
    posting_frequency_score = Column(Float, default=0.0)
    account_age_score = Column(Float, default=0.0)
    
    # Detection metadata
    detection_model_version = Column(String(20), default="1.0")
    processing_time_ms = Column(Integer, default=0)
    
    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ToxicityAnalysis(Base):
    """Content toxicity analysis results."""
    
    __tablename__ = "toxicity_analysis"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    content_type = Column(String(20), nullable=False)
    content_id = Column(PG_UUID(as_uuid=True), nullable=False)
    
    # Overall toxicity
    toxicity_score = Column(Float, default=0.0)  # 0-1
    is_toxic = Column(Boolean, default=False)
    
    # Specific toxicity categories
    severe_toxicity_score = Column(Float, default=0.0)
    identity_attack_score = Column(Float, default=0.0)
    insult_score = Column(Float, default=0.0)
    profanity_score = Column(Float, default=0.0)
    threat_score = Column(Float, default=0.0)
    
    # Language analysis
    detected_language = Column(String(10))
    confidence_language = Column(Float, default=0.0)
    
    # Flagged phrases
    flagged_phrases = Column(JSONB, default=list)
    context_analysis = Column(JSONB)
    
    # Detection metadata
    analysis_model_version = Column(String(20), default="1.0")
    processing_time_ms = Column(Integer, default=0)
    
    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ModerationRule(Base):
    """Configurable moderation rules."""
    
    __tablename__ = "moderation_rules"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Rule identification
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(50), nullable=False)  # spam, toxicity, content_policy
    
    # Rule configuration
    content_types = Column(JSONB, nullable=False)  # Which content types this applies to
    conditions = Column(JSONB, nullable=False)  # Rule conditions
    thresholds = Column(JSONB)  # Score thresholds
    
    # Actions
    auto_action = Column(String(20))  # Automatic action if triggered
    escalate_to_human = Column(Boolean, default=False)
    
    # Rule status
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=1)
    
    # Performance tracking
    times_triggered = Column(Integer, default=0)
    false_positive_count = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_triggered_at = Column(DateTime)


class ModerationAppeal(Base):
    """User appeals for moderation decisions."""
    
    __tablename__ = "moderation_appeals"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    moderation_id = Column(PG_UUID(as_uuid=True), ForeignKey("content_moderation.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Appeal details
    reason = Column(Text, nullable=False)
    additional_context = Column(Text)
    evidence_urls = Column(JSONB, default=list)
    
    # Status
    status = Column(String(20), default="pending")  # pending, under_review, approved, rejected
    
    # Review
    reviewed_by = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    review_notes = Column(Text)
    final_decision = Column(String(20))  # upheld, overturned, modified
    
    # Timestamps
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime)
    
    # Relationships
    moderation_entry = relationship("ContentModerationEntry")
    user = relationship("UserProfile", foreign_keys=[user_id])
    reviewer = relationship("UserProfile", foreign_keys=[reviewed_by])


class AutoModerationConfig(Base):
    """Configuration for automatic moderation system."""
    
    __tablename__ = "auto_moderation_config"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # System settings
    is_enabled = Column(Boolean, default=True)
    strictness_level = Column(String(10), default="medium")  # low, medium, high
    
    # Thresholds
    spam_threshold = Column(Float, default=0.7)
    toxicity_threshold = Column(Float, default=0.8)
    auto_approve_threshold = Column(Float, default=0.2)
    escalation_threshold = Column(Float, default=0.9)
    
    # Feature flags
    enable_spam_detection = Column(Boolean, default=True)
    enable_toxicity_detection = Column(Boolean, default=True)
    enable_malware_scan = Column(Boolean, default=True)
    enable_copyright_check = Column(Boolean, default=True)
    
    # Rate limiting
    max_reports_per_user_per_day = Column(Integer, default=10)
    max_content_per_user_per_hour = Column(Integer, default=5)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic models for API

class ModerationEntryResponse(BaseModel):
    """Schema for moderation entry responses."""
    id: UUID
    content_type: ContentType
    content_id: UUID
    status: str
    priority: str
    detection_method: str
    violations: List[str]
    moderator_action: Optional[str] = None
    action_taken: Optional[str] = None
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ModerationStatsResponse(BaseModel):
    """Schema for moderation statistics."""
    total_items_moderated: int
    pending_items: int
    approved_items: int
    rejected_items: int
    auto_moderated_percentage: float
    average_resolution_time_hours: float
    top_violation_types: List[Dict[str, Any]]
    moderator_workload: List[Dict[str, Any]]
    appeal_rate: float
    accuracy_metrics: Dict[str, float]


class ContentReportCreate(BaseModel):
    """Schema for reporting content."""
    content_type: ContentType
    content_id: UUID
    violation_type: ViolationType
    description: str = Field(..., max_length=500)
    additional_context: Optional[str] = Field(None, max_length=1000)


class ModerationAppealCreate(BaseModel):
    """Schema for creating moderation appeals."""
    moderation_id: UUID
    reason: str = Field(..., max_length=1000)
    additional_context: Optional[str] = Field(None, max_length=2000)
    evidence_urls: List[str] = Field(default_factory=list)


class ModerationRuleCreate(BaseModel):
    """Schema for creating moderation rules."""
    name: str = Field(..., max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    category: str = Field(..., max_length=50)
    content_types: List[ContentType]
    conditions: Dict[str, Any]
    thresholds: Optional[Dict[str, float]] = None
    auto_action: Optional[ModerationAction] = None
    escalate_to_human: bool = False
    priority: int = Field(1, ge=1, le=10)


class AutoModerationConfigUpdate(BaseModel):
    """Schema for updating auto-moderation configuration."""
    is_enabled: Optional[bool] = None
    strictness_level: Optional[str] = Field(None, regex=r'^(low|medium|high)$')
    spam_threshold: Optional[float] = Field(None, ge=0, le=1)
    toxicity_threshold: Optional[float] = Field(None, ge=0, le=1)
    auto_approve_threshold: Optional[float] = Field(None, ge=0, le=1)
    escalation_threshold: Optional[float] = Field(None, ge=0, le=1)
    enable_spam_detection: Optional[bool] = None
    enable_toxicity_detection: Optional[bool] = None
    enable_malware_scan: Optional[bool] = None
    enable_copyright_check: Optional[bool] = None
