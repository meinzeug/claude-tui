"""
User Profile Models - User-related models for community features.
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


class UserRole(str, Enum):
    """User role enumeration."""
    USER = "user"
    CONTRIBUTOR = "contributor"
    MAINTAINER = "maintainer"
    MODERATOR = "moderator"
    ADMIN = "admin"


class ContributionType(str, Enum):
    """Contribution type enumeration."""
    TEMPLATE_CREATED = "template_created"
    TEMPLATE_UPDATED = "template_updated"
    REVIEW_POSTED = "review_posted"
    BUG_REPORTED = "bug_reported"
    IMPROVEMENT_SUGGESTED = "improvement_suggested"
    DOCUMENTATION_IMPROVED = "documentation_improved"


class UserProfile(Base):
    """Extended user profile for community features."""
    
    __tablename__ = "user_profiles"
    
    # Primary identifiers
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False, index=True)
    
    # Profile information
    display_name = Column(String(100))
    bio = Column(Text)
    avatar_url = Column(String(500))
    location = Column(String(100))
    website_url = Column(String(500))
    github_username = Column(String(100))
    twitter_username = Column(String(100))
    
    # Community status
    role = Column(String(20), nullable=False, default=UserRole.USER.value)
    reputation_score = Column(Integer, default=0)
    contribution_points = Column(Integer, default=0)
    
    # Profile statistics
    templates_created = Column(Integer, default=0)
    templates_forked = Column(Integer, default=0)
    reviews_written = Column(Integer, default=0)
    followers_count = Column(Integer, default=0)
    following_count = Column(Integer, default=0)
    
    # Skills and expertise
    skills = Column(JSONB, default=list)
    specializations = Column(JSONB, default=list)
    preferred_frameworks = Column(JSONB, default=list)
    programming_languages = Column(JSONB, default=list)
    
    # Activity metrics
    last_active_at = Column(DateTime)
    total_downloads = Column(Integer, default=0)
    total_stars_received = Column(Integer, default=0)
    average_template_rating = Column(Float, default=0.0)
    
    # Profile settings
    is_public = Column(Boolean, default=True)
    show_email = Column(Boolean, default=False)
    show_activity = Column(Boolean, default=True)
    email_notifications = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="profile")
    contributions = relationship("UserContribution", back_populates="user")
    templates = relationship("Template", foreign_keys="[Template.author_id]")
    reviews = relationship("Review", back_populates="author")
    
    def to_dict(self, include_private: bool = False) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        data = {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'username': self.username,
            'display_name': self.display_name,
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'location': self.location,
            'website_url': self.website_url,
            'github_username': self.github_username,
            'twitter_username': self.twitter_username,
            'role': self.role,
            'reputation_score': self.reputation_score,
            'contribution_points': self.contribution_points,
            'templates_created': self.templates_created,
            'templates_forked': self.templates_forked,
            'reviews_written': self.reviews_written,
            'followers_count': self.followers_count,
            'following_count': self.following_count,
            'skills': self.skills,
            'specializations': self.specializations,
            'preferred_frameworks': self.preferred_frameworks,
            'programming_languages': self.programming_languages,
            'total_downloads': self.total_downloads,
            'total_stars_received': self.total_stars_received,
            'average_template_rating': self.average_template_rating,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_private:
            data.update({
                'show_email': self.show_email,
                'show_activity': self.show_activity,
                'email_notifications': self.email_notifications,
                'last_active_at': self.last_active_at.isoformat() if self.last_active_at else None
            })
        
        return data


class UserContribution(Base):
    """User contribution tracking."""
    
    __tablename__ = "user_contributions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Contribution details
    contribution_type = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Related entities
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"))
    review_id = Column(PG_UUID(as_uuid=True), ForeignKey("reviews.id"))
    
    # Contribution metadata
    points_awarded = Column(Integer, default=0)
    impact_score = Column(Float, default=0.0)
    visibility = Column(String(20), default="public")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("UserProfile", back_populates="contributions")
    template = relationship("Template")
    review = relationship("Review")


class UserFollowing(Base):
    """User following relationships."""
    
    __tablename__ = "user_following"
    
    follower_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), primary_key=True)
    following_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    follower = relationship("UserProfile", foreign_keys=[follower_id])
    following = relationship("UserProfile", foreign_keys=[following_id])


class UserActivity(Base):
    """User activity tracking."""
    
    __tablename__ = "user_activities"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Activity details
    activity_type = Column(String(50), nullable=False)
    activity_data = Column(JSONB)
    
    # Related entities
    template_id = Column(PG_UUID(as_uuid=True))
    target_user_id = Column(PG_UUID(as_uuid=True))
    
    # Activity metadata
    is_public = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("UserProfile")


# Pydantic models for API

class UserProfileBase(BaseModel):
    """Base user profile schema."""
    display_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=1000)
    location: Optional[str] = Field(None, max_length=100)
    website_url: Optional[str] = Field(None, max_length=500)
    github_username: Optional[str] = Field(None, max_length=100)
    twitter_username: Optional[str] = Field(None, max_length=100)
    skills: List[str] = Field(default_factory=list)
    specializations: List[str] = Field(default_factory=list)
    preferred_frameworks: List[str] = Field(default_factory=list)
    programming_languages: List[str] = Field(default_factory=list)
    is_public: bool = True
    show_activity: bool = True
    email_notifications: bool = True


class UserProfileCreate(UserProfileBase):
    """Schema for creating user profiles."""
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')


class UserProfileUpdate(UserProfileBase):
    """Schema for updating user profiles."""
    username: Optional[str] = Field(None, min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')


class UserProfileResponse(UserProfileBase):
    """Schema for user profile responses."""
    id: UUID
    user_id: UUID
    username: str
    role: UserRole
    reputation_score: int
    contribution_points: int
    templates_created: int
    templates_forked: int
    reviews_written: int
    followers_count: int
    following_count: int
    total_downloads: int
    total_stars_received: int
    average_template_rating: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserProfilePublic(BaseModel):
    """Public user profile schema."""
    id: UUID
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    location: Optional[str] = None
    website_url: Optional[str] = None
    github_username: Optional[str] = None
    twitter_username: Optional[str] = None
    reputation_score: int
    templates_created: int
    reviews_written: int
    followers_count: int
    following_count: int
    skills: List[str] = Field(default_factory=list)
    specializations: List[str] = Field(default_factory=list)
    created_at: datetime


class UserContributionResponse(BaseModel):
    """Schema for user contribution responses."""
    id: UUID
    contribution_type: ContributionType
    title: str
    description: Optional[str] = None
    template_id: Optional[UUID] = None
    points_awarded: int
    impact_score: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserActivityResponse(BaseModel):
    """Schema for user activity responses."""
    id: UUID
    activity_type: str
    activity_data: Dict[str, Any]
    template_id: Optional[UUID] = None
    target_user_id: Optional[UUID] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserStatsResponse(BaseModel):
    """User statistics response."""
    total_templates: int
    public_templates: int
    total_downloads: int
    total_stars: int
    total_forks: int
    average_rating: float
    contribution_streak: int
    rank_in_community: int
    badges_earned: List[Dict[str, Any]]
    recent_activity: List[UserActivityResponse]
    top_templates: List[Dict[str, Any]]


class LeaderboardEntry(BaseModel):
    """Leaderboard entry schema."""
    rank: int
    user: UserProfilePublic
    metric_value: float
    metric_type: str
    change_from_previous: Optional[int] = None


class CommunityLeaderboard(BaseModel):
    """Community leaderboard response."""
    leaderboard_type: str
    period: str
    entries: List[LeaderboardEntry]
    total_participants: int
    updated_at: datetime