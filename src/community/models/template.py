"""
Template Models - Core template marketplace data structures.
"""

import json
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class TemplateStatus(str, Enum):
    """Template status enumeration."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class TemplateType(str, Enum):
    """Template type enumeration."""
    PROJECT = "project"
    COMPONENT = "component"
    SNIPPET = "snippet"
    BOILERPLATE = "boilerplate"
    FRAMEWORK = "framework"


class ComplexityLevel(str, Enum):
    """Template complexity level."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Template(Base):
    """Template model for marketplace."""
    
    __tablename__ = "templates"
    
    # Primary identifiers
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False, index=True)
    
    # Basic information
    description = Column(Text)
    short_description = Column(String(300))
    author_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Template content and structure
    template_data = Column(JSONB)
    template_files = Column(JSONB)  # File structure and content
    template_config = Column(JSONB)
    
    # Metadata
    template_type = Column(String(20), nullable=False, default=TemplateType.PROJECT.value)
    status = Column(String(20), nullable=False, default=TemplateStatus.DRAFT.value)
    complexity_level = Column(String(20), nullable=False, default=ComplexityLevel.BEGINNER.value)
    
    # Tags and categories
    tags = Column(JSONB, default=list)
    categories = Column(JSONB, default=list)
    frameworks = Column(JSONB, default=list)
    languages = Column(JSONB, default=list)
    
    # Marketplace metrics
    download_count = Column(Integer, default=0)
    usage_count = Column(Integer, default=0)
    star_count = Column(Integer, default=0)
    fork_count = Column(Integer, default=0)
    
    # Rating and reviews
    average_rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    
    # Quality metrics
    quality_score = Column(Float, default=0.0)
    completeness_score = Column(Float, default=0.0)
    documentation_score = Column(Float, default=0.0)
    
    # Versioning
    version = Column(String(20), nullable=False, default="1.0.0")
    is_latest_version = Column(Boolean, default=True)
    parent_template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"))
    
    # Licensing and usage
    license_type = Column(String(50), default="MIT")
    is_public = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    
    # Requirements and compatibility
    min_python_version = Column(String(10))
    max_python_version = Column(String(10))
    dependencies = Column(JSONB, default=list)
    requirements = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime)
    last_used_at = Column(DateTime)
    
    # Relationships
    versions = relationship("TemplateVersion", back_populates="template")
    reviews = relationship("Review", back_populates="template")
    author = relationship("User", back_populates="templates")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            'id': str(self.id),
            'slug': self.slug,
            'name': self.name,
            'description': self.description,
            'short_description': self.short_description,
            'author_id': str(self.author_id),
            'template_type': self.template_type,
            'status': self.status,
            'complexity_level': self.complexity_level,
            'tags': self.tags,
            'categories': self.categories,
            'frameworks': self.frameworks,
            'languages': self.languages,
            'download_count': self.download_count,
            'usage_count': self.usage_count,
            'star_count': self.star_count,
            'fork_count': self.fork_count,
            'average_rating': self.average_rating,
            'rating_count': self.rating_count,
            'quality_score': self.quality_score,
            'version': self.version,
            'license_type': self.license_type,
            'is_public': self.is_public,
            'is_featured': self.is_featured,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'published_at': self.published_at.isoformat() if self.published_at else None
        }


class TemplateVersion(Base):
    """Template version tracking."""
    
    __tablename__ = "template_versions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False)
    version = Column(String(20), nullable=False)
    
    # Version metadata
    changelog = Column(Text)
    breaking_changes = Column(Text)
    migration_notes = Column(Text)
    
    # Template snapshot
    template_data = Column(JSONB)
    template_files = Column(JSONB)
    template_config = Column(JSONB)
    
    # Version status
    is_stable = Column(Boolean, default=False)
    is_deprecated = Column(Boolean, default=False)
    deprecation_reason = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deprecated_at = Column(DateTime)
    
    # Relationships
    template = relationship("Template", back_populates="versions")


class TemplateCategory(Base):
    """Template categories for organization."""
    
    __tablename__ = "template_categories"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    
    # Category metadata
    parent_id = Column(PG_UUID(as_uuid=True), ForeignKey("template_categories.id"))
    icon = Column(String(50))
    color = Column(String(10))
    sort_order = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    template_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parent = relationship("TemplateCategory", remote_side=[id])


# Pydantic models for API

class TemplateBase(BaseModel):
    """Base template schema."""
    name: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    short_description: Optional[str] = Field(None, max_length=300)
    template_type: TemplateType = TemplateType.PROJECT
    complexity_level: ComplexityLevel = ComplexityLevel.BEGINNER
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    license_type: str = "MIT"
    is_public: bool = True


class TemplateCreate(TemplateBase):
    """Schema for creating templates."""
    template_data: Dict[str, Any]
    template_files: Dict[str, Any] = Field(default_factory=dict)
    template_config: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    requirements: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('template_data')
    def validate_template_data(cls, v):
        """Validate template data structure."""
        required_fields = ['name', 'description']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'template_data must contain {field}')
        return v


class TemplateUpdate(BaseModel):
    """Schema for updating templates."""
    name: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    short_description: Optional[str] = Field(None, max_length=300)
    template_data: Optional[Dict[str, Any]] = None
    template_files: Optional[Dict[str, Any]] = None
    template_config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    frameworks: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    complexity_level: Optional[ComplexityLevel] = None
    license_type: Optional[str] = None
    is_public: Optional[bool] = None


class TemplateResponse(TemplateBase):
    """Schema for template responses."""
    id: UUID
    slug: str
    author_id: UUID
    status: TemplateStatus
    version: str
    download_count: int
    usage_count: int
    star_count: int
    fork_count: int
    average_rating: float
    rating_count: int
    quality_score: float
    is_featured: bool
    is_premium: bool
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TemplateSearchFilters(BaseModel):
    """Template search filters."""
    query: Optional[str] = None
    template_type: Optional[TemplateType] = None
    complexity_level: Optional[ComplexityLevel] = None
    categories: Optional[List[str]] = None
    frameworks: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    min_rating: Optional[float] = Field(None, ge=0, le=5)
    max_rating: Optional[float] = Field(None, ge=0, le=5)
    is_featured: Optional[bool] = None
    is_free: Optional[bool] = None
    author_id: Optional[UUID] = None
    created_after: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    sort_by: str = "updated_at"
    sort_order: str = "desc"


class TemplateStats(BaseModel):
    """Template statistics."""
    total_templates: int
    public_templates: int
    featured_templates: int
    categories: Dict[str, int]
    languages: Dict[str, int]
    frameworks: Dict[str, int]
    complexity_distribution: Dict[str, int]
    rating_distribution: Dict[str, int]
    recent_activity: List[Dict[str, Any]]
    
    
class TemplateInheritanceConfig(BaseModel):
    """Configuration for template inheritance."""
    inherit_from: UUID
    override_files: Dict[str, Any] = Field(default_factory=dict)
    merge_config: bool = True
    preserve_structure: bool = True
    custom_variables: Dict[str, Any] = Field(default_factory=dict)


class TemplateBuildResult(BaseModel):
    """Result of template build process."""
    success: bool
    template_id: UUID
    build_time: float
    files_generated: List[str]
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)