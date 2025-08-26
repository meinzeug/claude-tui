"""
Rating and Review Models - Enhanced rating system with moderation.
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


class ReviewStatus(str, Enum):
    """Review status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    HIDDEN = "hidden"
    FLAGGED = "flagged"


class ReportReason(str, Enum):
    """Report reason enumeration."""
    SPAM = "spam"
    INAPPROPRIATE = "inappropriate"
    OFFENSIVE = "offensive"
    FAKE = "fake"
    COPYRIGHT = "copyright"
    OTHER = "other"


class TemplateRating(Base):
    """Enhanced template rating system."""
    
    __tablename__ = "template_ratings"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Rating details
    overall_rating = Column(Integer, nullable=False)  # 1-5 stars
    
    # Detailed ratings
    quality_rating = Column(Integer)  # Code quality
    usability_rating = Column(Integer)  # Ease of use
    documentation_rating = Column(Integer)  # Documentation quality
    support_rating = Column(Integer)  # Author support
    
    # Review content
    title = Column(String(200))
    content = Column(Text)
    
    # Context
    template_version_reviewed = Column(String(20))
    use_case = Column(String(200))  # What user used template for
    experience_level = Column(String(20))  # beginner, intermediate, advanced
    
    # Verification
    is_verified_usage = Column(Boolean, default=False)  # User actually used the template
    usage_duration_days = Column(Integer)  # How long they used it
    
    # Moderation
    status = Column(String(20), default=ReviewStatus.PENDING.value)
    moderation_notes = Column(Text)
    moderated_by = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    moderated_at = Column(DateTime)
    
    # Helpfulness tracking
    helpful_votes = Column(Integer, default=0)
    not_helpful_votes = Column(Integer, default=0)
    total_votes = Column(Integer, default=0)
    helpfulness_score = Column(Float, default=0.0)
    
    # Analytics
    view_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    template = relationship("Template")
    user = relationship("UserProfile", foreign_keys=[user_id])
    moderator = relationship("UserProfile", foreign_keys=[moderated_by])
    reports = relationship("ReviewReport", back_populates="review")
    votes = relationship("ReviewHelpfulness", back_populates="review")
    
    def to_dict(self, include_private: bool = False) -> Dict[str, Any]:
        """Convert rating to dictionary."""
        data = {
            'id': str(self.id),
            'template_id': str(self.template_id),
            'user_id': str(self.user_id),
            'overall_rating': self.overall_rating,
            'quality_rating': self.quality_rating,
            'usability_rating': self.usability_rating,
            'documentation_rating': self.documentation_rating,
            'support_rating': self.support_rating,
            'title': self.title,
            'content': self.content,
            'template_version_reviewed': self.template_version_reviewed,
            'use_case': self.use_case,
            'experience_level': self.experience_level,
            'is_verified_usage': self.is_verified_usage,
            'helpful_votes': self.helpful_votes,
            'not_helpful_votes': self.not_helpful_votes,
            'total_votes': self.total_votes,
            'helpfulness_score': float(self.helpfulness_score) if self.helpfulness_score else 0.0,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_private:
            data.update({
                'status': self.status,
                'moderation_notes': self.moderation_notes,
                'moderated_by': str(self.moderated_by) if self.moderated_by else None,
                'moderated_at': self.moderated_at.isoformat() if self.moderated_at else None,
                'view_count': self.view_count
            })
        
        return data


class ReviewHelpfulness(Base):
    """Review helpfulness voting system."""
    
    __tablename__ = "review_helpfulness"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    review_id = Column(PG_UUID(as_uuid=True), ForeignKey("template_ratings.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Vote details
    is_helpful = Column(Boolean, nullable=False)  # True = helpful, False = not helpful
    
    # Timestamps
    voted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    review = relationship("TemplateRating", back_populates="votes")
    user = relationship("UserProfile")


class ReviewReport(Base):
    """Review reporting system for moderation."""
    
    __tablename__ = "review_reports"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    review_id = Column(PG_UUID(as_uuid=True), ForeignKey("template_ratings.id"), nullable=False)
    reporter_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Report details
    reason = Column(String(20), nullable=False)
    description = Column(Text)
    
    # Status
    status = Column(String(20), default="pending")  # pending, reviewed, resolved, dismissed
    resolution = Column(Text)
    resolved_by = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    resolved_at = Column(DateTime)
    
    # Timestamps
    reported_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    review = relationship("TemplateRating", back_populates="reports")
    reporter = relationship("UserProfile", foreign_keys=[reporter_id])
    resolver = relationship("UserProfile", foreign_keys=[resolved_by])


class UserReputation(Base):
    """User reputation system based on reviews and contributions."""
    
    __tablename__ = "user_reputation"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, unique=True)
    
    # Reputation scores
    total_reputation = Column(Integer, default=0)
    reviewer_reputation = Column(Integer, default=0)  # Based on review quality
    creator_reputation = Column(Integer, default=0)  # Based on template quality
    community_reputation = Column(Integer, default=0)  # Based on community participation
    
    # Review metrics
    total_reviews_written = Column(Integer, default=0)
    helpful_reviews_count = Column(Integer, default=0)
    average_review_helpfulness = Column(Float, default=0.0)
    
    # Template metrics
    templates_created = Column(Integer, default=0)
    average_template_rating = Column(Float, default=0.0)
    total_template_downloads = Column(Integer, default=0)
    
    # Badges and achievements
    badges = Column(JSONB, default=list)
    achievements = Column(JSONB, default=list)
    
    # Status
    is_trusted_reviewer = Column(Boolean, default=False)
    is_verified_creator = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("UserProfile")


class ModerationQueue(Base):
    """Content moderation queue."""
    
    __tablename__ = "moderation_queue"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Content details
    content_type = Column(String(20), nullable=False)  # review, template, comment
    content_id = Column(PG_UUID(as_uuid=True), nullable=False)
    
    # Moderation details
    priority = Column(String(10), default="medium")  # low, medium, high, urgent
    reason = Column(String(50), nullable=False)  # auto_flagged, user_reported, manual_review
    flags = Column(JSONB, default=list)  # List of automated flags
    
    # AI moderation results
    ai_spam_score = Column(Float, default=0.0)
    ai_toxicity_score = Column(Float, default=0.0)
    ai_recommendation = Column(String(20))  # approve, reject, needs_human_review
    
    # Status
    status = Column(String(20), default="pending")  # pending, in_review, completed
    assigned_to = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    
    # Resolution
    resolution = Column(String(20))  # approved, rejected, no_action
    resolution_notes = Column(Text)
    resolved_by = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    resolved_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    assigned_moderator = relationship("UserProfile", foreign_keys=[assigned_to])
    resolver = relationship("UserProfile", foreign_keys=[resolved_by])


# Pydantic models for API

class TemplateRatingCreate(BaseModel):
    """Schema for creating template ratings."""
    template_id: UUID
    overall_rating: int = Field(..., ge=1, le=5)
    quality_rating: Optional[int] = Field(None, ge=1, le=5)
    usability_rating: Optional[int] = Field(None, ge=1, le=5)
    documentation_rating: Optional[int] = Field(None, ge=1, le=5)
    support_rating: Optional[int] = Field(None, ge=1, le=5)
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, max_length=2000)
    template_version_reviewed: Optional[str] = None
    use_case: Optional[str] = Field(None, max_length=200)
    experience_level: Optional[str] = Field(None, regex=r'^(beginner|intermediate|advanced)$')


class TemplateRatingUpdate(BaseModel):
    """Schema for updating template ratings."""
    overall_rating: Optional[int] = Field(None, ge=1, le=5)
    quality_rating: Optional[int] = Field(None, ge=1, le=5)
    usability_rating: Optional[int] = Field(None, ge=1, le=5)
    documentation_rating: Optional[int] = Field(None, ge=1, le=5)
    support_rating: Optional[int] = Field(None, ge=1, le=5)
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, max_length=2000)
    use_case: Optional[str] = Field(None, max_length=200)


class TemplateRatingResponse(BaseModel):
    """Schema for template rating responses."""
    id: UUID
    template_id: UUID
    user_id: UUID
    overall_rating: int
    quality_rating: Optional[int] = None
    usability_rating: Optional[int] = None
    documentation_rating: Optional[int] = None
    support_rating: Optional[int] = None
    title: Optional[str] = None
    content: Optional[str] = None
    template_version_reviewed: Optional[str] = None
    use_case: Optional[str] = None
    experience_level: Optional[str] = None
    is_verified_usage: bool
    helpful_votes: int
    not_helpful_votes: int
    total_votes: int
    helpfulness_score: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ReviewHelpfulnessVote(BaseModel):
    """Schema for review helpfulness voting."""
    review_id: UUID
    is_helpful: bool


class ReviewReportCreate(BaseModel):
    """Schema for reporting reviews."""
    review_id: UUID
    reason: ReportReason
    description: Optional[str] = Field(None, max_length=500)


class UserReputationResponse(BaseModel):
    """Schema for user reputation responses."""
    user_id: UUID
    total_reputation: int
    reviewer_reputation: int
    creator_reputation: int
    community_reputation: int
    total_reviews_written: int
    helpful_reviews_count: int
    average_review_helpfulness: float
    templates_created: int
    average_template_rating: float
    total_template_downloads: int
    badges: List[str]
    achievements: List[str]
    is_trusted_reviewer: bool
    is_verified_creator: bool
    
    class Config:
        from_attributes = True


class ModerationStats(BaseModel):
    """Moderation statistics."""
    pending_items: int
    items_reviewed_today: int
    total_reports: int
    auto_approved_rate: float
    average_resolution_time_hours: float
    top_flagged_reasons: List[Dict[str, Any]]
