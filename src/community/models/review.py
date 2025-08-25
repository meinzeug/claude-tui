"""
Review and Rating Models - Models for template reviews and ratings system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
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
    FLAGGED = "flagged"
    HIDDEN = "hidden"


class RatingCriteria(str, Enum):
    """Rating criteria enumeration."""
    OVERALL = "overall"
    EASE_OF_USE = "ease_of_use"
    DOCUMENTATION = "documentation"
    CODE_QUALITY = "code_quality"
    CUSTOMIZABILITY = "customizability"
    PERFORMANCE = "performance"
    COMPLETENESS = "completeness"


class Review(Base):
    """Template review model."""
    
    __tablename__ = "reviews"
    
    # Primary identifiers
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    author_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Review content
    title = Column(String(200))
    content = Column(Text, nullable=False)
    summary = Column(String(500))
    
    # Ratings
    overall_rating = Column(Float, nullable=False)
    ease_of_use_rating = Column(Float)
    documentation_rating = Column(Float)
    code_quality_rating = Column(Float)
    customizability_rating = Column(Float)
    performance_rating = Column(Float)
    completeness_rating = Column(Float)
    
    # Review metadata
    status = Column(String(20), nullable=False, default=ReviewStatus.PENDING.value)
    is_verified_purchase = Column(Boolean, default=False)
    is_recommended = Column(Boolean)
    
    # Usage context
    use_case = Column(String(200))
    project_size = Column(String(20))  # small, medium, large, enterprise
    experience_level = Column(String(20))  # beginner, intermediate, advanced
    
    # Interaction metrics
    helpful_count = Column(Integer, default=0)
    not_helpful_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    
    # Moderation
    flagged_count = Column(Integer, default=0)
    moderation_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reviewed_at = Column(DateTime)
    
    # Relationships
    template = relationship("Template", back_populates="reviews")
    author = relationship("UserProfile", back_populates="reviews")
    replies = relationship("ReviewReply", back_populates="review")
    votes = relationship("ReviewVote", back_populates="review")
    
    def calculate_weighted_rating(self) -> float:
        """Calculate weighted average rating."""
        ratings = [
            (self.overall_rating, 0.3),
            (self.ease_of_use_rating or 0, 0.2),
            (self.code_quality_rating or 0, 0.2),
            (self.documentation_rating or 0, 0.15),
            (self.customizability_rating or 0, 0.1),
            (self.performance_rating or 0, 0.05)
        ]
        
        total_weight = sum(weight for rating, weight in ratings if rating > 0)
        if total_weight == 0:
            return self.overall_rating
        
        weighted_sum = sum(rating * weight for rating, weight in ratings if rating > 0)
        return weighted_sum / total_weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert review to dictionary."""
        return {
            'id': str(self.id),
            'template_id': str(self.template_id),
            'author_id': str(self.author_id),
            'title': self.title,
            'content': self.content,
            'summary': self.summary,
            'overall_rating': self.overall_rating,
            'ease_of_use_rating': self.ease_of_use_rating,
            'documentation_rating': self.documentation_rating,
            'code_quality_rating': self.code_quality_rating,
            'customizability_rating': self.customizability_rating,
            'performance_rating': self.performance_rating,
            'completeness_rating': self.completeness_rating,
            'status': self.status,
            'is_verified_purchase': self.is_verified_purchase,
            'is_recommended': self.is_recommended,
            'use_case': self.use_case,
            'project_size': self.project_size,
            'experience_level': self.experience_level,
            'helpful_count': self.helpful_count,
            'not_helpful_count': self.not_helpful_count,
            'reply_count': self.reply_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ReviewReply(Base):
    """Review reply model."""
    
    __tablename__ = "review_replies"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    review_id = Column(PG_UUID(as_uuid=True), ForeignKey("reviews.id"), nullable=False)
    author_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    parent_reply_id = Column(PG_UUID(as_uuid=True), ForeignKey("review_replies.id"))
    
    # Reply content
    content = Column(Text, nullable=False)
    
    # Metadata
    is_template_author = Column(Boolean, default=False)
    helpful_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    review = relationship("Review", back_populates="replies")
    author = relationship("UserProfile")
    parent_reply = relationship("ReviewReply", remote_side=[id])


class ReviewVote(Base):
    """Review helpfulness voting."""
    
    __tablename__ = "review_votes"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    review_id = Column(PG_UUID(as_uuid=True), ForeignKey("reviews.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Vote
    is_helpful = Column(Boolean, nullable=False)  # True = helpful, False = not helpful
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    review = relationship("Review", back_populates="votes")
    user = relationship("UserProfile")


class Rating(Base):
    """Individual rating model (for detailed analytics)."""
    
    __tablename__ = "ratings"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    review_id = Column(PG_UUID(as_uuid=True), ForeignKey("reviews.id"))
    
    # Rating details
    criteria = Column(String(50), nullable=False)  # RatingCriteria enum value
    value = Column(Float, nullable=False)
    
    # Context
    user_experience_level = Column(String(20))
    project_context = Column(String(200))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    template = relationship("Template")
    user = relationship("UserProfile")
    review = relationship("Review")


class TemplateRating(Base):
    """Aggregated template rating statistics."""
    
    __tablename__ = "template_ratings"
    
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), primary_key=True)
    
    # Overall metrics
    average_rating = Column(Float, default=0.0)
    total_ratings = Column(Integer, default=0)
    
    # Detailed ratings
    overall_avg = Column(Float, default=0.0)
    ease_of_use_avg = Column(Float, default=0.0)
    documentation_avg = Column(Float, default=0.0)
    code_quality_avg = Column(Float, default=0.0)
    customizability_avg = Column(Float, default=0.0)
    performance_avg = Column(Float, default=0.0)
    completeness_avg = Column(Float, default=0.0)
    
    # Rating distribution
    five_star_count = Column(Integer, default=0)
    four_star_count = Column(Integer, default=0)
    three_star_count = Column(Integer, default=0)
    two_star_count = Column(Integer, default=0)
    one_star_count = Column(Integer, default=0)
    
    # Quality indicators
    verified_ratings_count = Column(Integer, default=0)
    recommendation_percentage = Column(Float, default=0.0)
    
    # Timestamps
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    template = relationship("Template")
    
    def calculate_rating_score(self) -> float:
        """Calculate weighted rating score for ranking."""
        if self.total_ratings == 0:
            return 0.0
        
        # Wilson score interval for better ranking
        n = self.total_ratings
        p = self.average_rating / 5.0  # Convert to 0-1 scale
        
        if n == 0:
            return 0.0
        
        # Wilson score with z=1.96 (95% confidence)
        z = 1.96
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        spread = z * (p * (1 - p) + z * z / (4 * n))**0.5 / (n**0.5 * denominator)
        
        return max(0, centre - spread) * 5  # Convert back to 1-5 scale


# Pydantic models for API

class ReviewBase(BaseModel):
    """Base review schema."""
    title: Optional[str] = Field(None, max_length=200)
    content: str = Field(..., min_length=10, max_length=5000)
    summary: Optional[str] = Field(None, max_length=500)
    overall_rating: float = Field(..., ge=1, le=5)
    ease_of_use_rating: Optional[float] = Field(None, ge=1, le=5)
    documentation_rating: Optional[float] = Field(None, ge=1, le=5)
    code_quality_rating: Optional[float] = Field(None, ge=1, le=5)
    customizability_rating: Optional[float] = Field(None, ge=1, le=5)
    performance_rating: Optional[float] = Field(None, ge=1, le=5)
    completeness_rating: Optional[float] = Field(None, ge=1, le=5)
    is_recommended: Optional[bool] = None
    use_case: Optional[str] = Field(None, max_length=200)
    project_size: Optional[str] = Field(None, regex="^(small|medium|large|enterprise)$")
    experience_level: Optional[str] = Field(None, regex="^(beginner|intermediate|advanced)$")


class ReviewCreate(ReviewBase):
    """Schema for creating reviews."""
    template_id: UUID


class ReviewUpdate(BaseModel):
    """Schema for updating reviews."""
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, min_length=10, max_length=5000)
    summary: Optional[str] = Field(None, max_length=500)
    overall_rating: Optional[float] = Field(None, ge=1, le=5)
    ease_of_use_rating: Optional[float] = Field(None, ge=1, le=5)
    documentation_rating: Optional[float] = Field(None, ge=1, le=5)
    code_quality_rating: Optional[float] = Field(None, ge=1, le=5)
    customizability_rating: Optional[float] = Field(None, ge=1, le=5)
    performance_rating: Optional[float] = Field(None, ge=1, le=5)
    completeness_rating: Optional[float] = Field(None, ge=1, le=5)
    is_recommended: Optional[bool] = None
    use_case: Optional[str] = Field(None, max_length=200)
    project_size: Optional[str] = Field(None, regex="^(small|medium|large|enterprise)$")
    experience_level: Optional[str] = Field(None, regex="^(beginner|intermediate|advanced)$")


class ReviewResponse(ReviewBase):
    """Schema for review responses."""
    id: UUID
    template_id: UUID
    author_id: UUID
    status: ReviewStatus
    is_verified_purchase: bool
    helpful_count: int
    not_helpful_count: int
    reply_count: int
    created_at: datetime
    updated_at: datetime
    author: Optional[Dict[str, Any]] = None  # UserProfilePublic
    
    class Config:
        from_attributes = True


class ReviewReplyCreate(BaseModel):
    """Schema for creating review replies."""
    content: str = Field(..., min_length=5, max_length=1000)
    parent_reply_id: Optional[UUID] = None


class ReviewReplyResponse(BaseModel):
    """Schema for review reply responses."""
    id: UUID
    review_id: UUID
    author_id: UUID
    parent_reply_id: Optional[UUID] = None
    content: str
    is_template_author: bool
    helpful_count: int
    created_at: datetime
    updated_at: datetime
    author: Optional[Dict[str, Any]] = None  # UserProfilePublic
    
    class Config:
        from_attributes = True


class ReviewVoteCreate(BaseModel):
    """Schema for creating review votes."""
    is_helpful: bool


class RatingCreate(BaseModel):
    """Schema for creating individual ratings."""
    template_id: UUID
    criteria: RatingCriteria
    value: float = Field(..., ge=1, le=5)
    user_experience_level: Optional[str] = Field(None, regex="^(beginner|intermediate|advanced)$")
    project_context: Optional[str] = Field(None, max_length=200)


class TemplateRatingResponse(BaseModel):
    """Schema for template rating responses."""
    template_id: UUID
    average_rating: float
    total_ratings: int
    overall_avg: float
    ease_of_use_avg: float
    documentation_avg: float
    code_quality_avg: float
    customizability_avg: float
    performance_avg: float
    completeness_avg: float
    five_star_count: int
    four_star_count: int
    three_star_count: int
    two_star_count: int
    one_star_count: int
    verified_ratings_count: int
    recommendation_percentage: float
    last_updated: datetime
    
    class Config:
        from_attributes = True


class ReviewSearchFilters(BaseModel):
    """Review search filters."""
    template_id: Optional[UUID] = None
    author_id: Optional[UUID] = None
    min_rating: Optional[float] = Field(None, ge=1, le=5)
    max_rating: Optional[float] = Field(None, ge=1, le=5)
    is_recommended: Optional[bool] = None
    project_size: Optional[str] = None
    experience_level: Optional[str] = None
    verified_only: bool = False
    created_after: Optional[datetime] = None
    sort_by: str = "created_at"
    sort_order: str = "desc"


class ReviewStats(BaseModel):
    """Review statistics."""
    total_reviews: int
    average_rating: float
    rating_distribution: Dict[str, int]
    verified_reviews_count: int
    recommended_percentage: float
    recent_reviews: List[ReviewResponse]
    top_rated_reviews: List[ReviewResponse]