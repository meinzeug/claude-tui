"""
Marketplace Models - Models for template marketplace organization and features.
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


class CollectionType(str, Enum):
    """Collection type enumeration."""
    FEATURED = "featured"
    CURATED = "curated"
    TRENDING = "trending"
    NEW_RELEASES = "new_releases"
    STAFF_PICKS = "staff_picks"
    COMMUNITY_FAVORITES = "community_favorites"
    SEASONAL = "seasonal"


class FeaturedCollection(Base):
    """Featured template collections."""
    
    __tablename__ = "featured_collections"
    
    # Primary identifiers
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    
    # Collection details
    description = Column(Text)
    short_description = Column(String(300))
    collection_type = Column(String(30), nullable=False)
    
    # Visual presentation
    cover_image_url = Column(String(500))
    banner_image_url = Column(String(500))
    icon = Column(String(50))
    color_theme = Column(String(10))
    
    # Collection metadata
    curator_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    template_ids = Column(JSONB, default=list)
    tags = Column(JSONB, default=list)
    
    # Display settings
    sort_order = Column(Integer, default=0)
    max_templates = Column(Integer, default=20)
    is_dynamic = Column(Boolean, default=False)  # Auto-populated based on criteria
    
    # Dynamic collection criteria (if is_dynamic=True)
    auto_criteria = Column(JSONB)  # Filters for auto-populating collections
    
    # Status and visibility
    is_active = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)
    is_public = Column(Boolean, default=True)
    
    # Analytics
    view_count = Column(Integer, default=0)
    template_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    featured_until = Column(DateTime)
    
    # Relationships
    curator = relationship("UserProfile")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary."""
        return {
            'id': str(self.id),
            'slug': self.slug,
            'name': self.name,
            'description': self.description,
            'short_description': self.short_description,
            'collection_type': self.collection_type,
            'cover_image_url': self.cover_image_url,
            'banner_image_url': self.banner_image_url,
            'icon': self.icon,
            'color_theme': self.color_theme,
            'curator_id': str(self.curator_id) if self.curator_id else None,
            'template_ids': self.template_ids,
            'tags': self.tags,
            'template_count': self.template_count,
            'view_count': self.view_count,
            'is_featured': self.is_featured,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class TemplateMarketplace(Base):
    """Main marketplace configuration and settings."""
    
    __tablename__ = "marketplace_config"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Marketplace settings
    name = Column(String(200), default="Claude-TIU Template Marketplace")
    description = Column(Text)
    welcome_message = Column(Text)
    
    # Featured content
    featured_templates = Column(JSONB, default=list)
    featured_collections = Column(JSONB, default=list)
    trending_templates = Column(JSONB, default=list)
    
    # Category organization
    categories = Column(JSONB, default=list)
    featured_categories = Column(JSONB, default=list)
    
    # Display settings
    templates_per_page = Column(Integer, default=20)
    max_search_results = Column(Integer, default=100)
    
    # Quality standards
    min_quality_score = Column(Float, default=3.0)
    require_documentation = Column(Boolean, default=True)
    require_examples = Column(Boolean, default=False)
    
    # Marketplace analytics
    total_templates = Column(Integer, default=0)
    total_users = Column(Integer, default=0)
    total_downloads = Column(Integer, default=0)
    monthly_active_users = Column(Integer, default=0)
    
    # Configuration
    auto_approve_templates = Column(Boolean, default=False)
    enable_user_uploads = Column(Boolean, default=True)
    enable_reviews = Column(Boolean, default=True)
    enable_ratings = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TemplateTag(Base):
    """Template tagging system."""
    
    __tablename__ = "template_tags"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(50), unique=True, nullable=False, index=True)
    slug = Column(String(50), unique=True, nullable=False, index=True)
    
    # Tag metadata
    description = Column(Text)
    category = Column(String(50))  # framework, language, feature, etc.
    color = Column(String(10))
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    template_count = Column(Integer, default=0)
    
    # Status
    is_featured = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TemplateDownload(Base):
    """Template download tracking."""
    
    __tablename__ = "template_downloads"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    
    # Download details
    download_type = Column(String(20), default="direct")  # direct, api, cli
    client_info = Column(JSONB)  # User agent, IP (hashed), etc.
    
    # Context
    project_name = Column(String(200))
    intended_use = Column(String(200))
    
    # Timestamps
    downloaded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    template = relationship("Template")
    user = relationship("UserProfile")


class TemplateView(Base):
    """Template view tracking for analytics."""
    
    __tablename__ = "template_views"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    
    # View details
    view_duration = Column(Integer)  # seconds
    referrer = Column(String(200))
    
    # Session info (hashed/anonymized)
    session_id = Column(String(100))
    
    # Timestamps
    viewed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    template = relationship("Template")
    user = relationship("UserProfile")


class TemplateStar(Base):
    """Template starring/favoriting system."""
    
    __tablename__ = "template_stars"
    
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), primary_key=True)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), primary_key=True)
    
    # Timestamps
    starred_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("UserProfile")
    template = relationship("Template")


class TemplateFork(Base):
    """Template forking system."""
    
    __tablename__ = "template_forks"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    original_template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    forked_template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Fork details
    fork_reason = Column(String(200))
    changes_description = Column(Text)
    
    # Timestamps
    forked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    original_template = relationship("Template", foreign_keys=[original_template_id])
    forked_template = relationship("Template", foreign_keys=[forked_template_id])
    user = relationship("UserProfile")


# Pydantic models for API

class FeaturedCollectionBase(BaseModel):
    """Base featured collection schema."""
    name: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    short_description: Optional[str] = Field(None, max_length=300)
    collection_type: CollectionType
    cover_image_url: Optional[str] = Field(None, max_length=500)
    banner_image_url: Optional[str] = Field(None, max_length=500)
    icon: Optional[str] = Field(None, max_length=50)
    color_theme: Optional[str] = Field(None, max_length=10)
    tags: List[str] = Field(default_factory=list)
    max_templates: int = Field(20, ge=1, le=100)
    is_dynamic: bool = False


class FeaturedCollectionCreate(FeaturedCollectionBase):
    """Schema for creating featured collections."""
    slug: str = Field(..., min_length=3, max_length=100, regex=r'^[a-z0-9-]+$')
    template_ids: List[UUID] = Field(default_factory=list)
    auto_criteria: Optional[Dict[str, Any]] = None


class FeaturedCollectionUpdate(BaseModel):
    """Schema for updating featured collections."""
    name: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    short_description: Optional[str] = Field(None, max_length=300)
    template_ids: Optional[List[UUID]] = None
    tags: Optional[List[str]] = None
    cover_image_url: Optional[str] = Field(None, max_length=500)
    banner_image_url: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None
    is_featured: Optional[bool] = None


class FeaturedCollectionResponse(FeaturedCollectionBase):
    """Schema for featured collection responses."""
    id: UUID
    slug: str
    curator_id: Optional[UUID] = None
    template_count: int
    view_count: int
    is_active: bool
    is_featured: bool
    created_at: datetime
    updated_at: datetime
    featured_until: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TemplateTagCreate(BaseModel):
    """Schema for creating template tags."""
    name: str = Field(..., min_length=2, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=50)
    color: Optional[str] = Field(None, max_length=10)


class TemplateTagResponse(BaseModel):
    """Schema for template tag responses."""
    id: UUID
    name: str
    slug: str
    description: Optional[str] = None
    category: Optional[str] = None
    color: Optional[str] = None
    usage_count: int
    template_count: int
    is_featured: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class TemplateDownloadCreate(BaseModel):
    """Schema for recording template downloads."""
    template_id: UUID
    download_type: str = "direct"
    project_name: Optional[str] = Field(None, max_length=200)
    intended_use: Optional[str] = Field(None, max_length=200)


class MarketplaceStats(BaseModel):
    """Marketplace statistics."""
    total_templates: int
    public_templates: int
    featured_templates: int
    total_users: int
    total_downloads: int
    monthly_active_users: int
    categories: Dict[str, int]
    languages: Dict[str, int]
    frameworks: Dict[str, int]
    recent_templates: List[Dict[str, Any]]
    trending_templates: List[Dict[str, Any]]
    top_rated_templates: List[Dict[str, Any]]


class MarketplaceTrends(BaseModel):
    """Marketplace trend analysis."""
    trending_templates: List[Dict[str, Any]]
    trending_categories: List[Dict[str, Any]]
    trending_tags: List[Dict[str, Any]]
    rising_creators: List[Dict[str, Any]]
    popular_searches: List[str]
    download_trends: Dict[str, List[int]]
    rating_trends: Dict[str, List[float]]
    period: str
    generated_at: datetime


class SearchSuggestion(BaseModel):
    """Search suggestion model."""
    query: str
    type: str  # template, category, tag, user
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarketplaceSearch(BaseModel):
    """Marketplace search configuration."""
    enable_fuzzy_search: bool = True
    enable_autocomplete: bool = True
    max_suggestions: int = 10
    search_weights: Dict[str, float] = Field(default_factory=lambda: {
        "name": 3.0,
        "description": 1.0,
        "tags": 2.0,
        "author": 1.5
    })
    boost_featured: bool = True
    boost_high_rated: bool = True