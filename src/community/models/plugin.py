"""
Plugin Models - Models for plugin management and marketplace.
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


class PluginStatus(str, Enum):
    """Plugin status enumeration."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    BANNED = "banned"


class PluginType(str, Enum):
    """Plugin type enumeration."""
    INTEGRATION = "integration"
    THEME = "theme"
    EXTENSION = "extension"
    TOOL = "tool"
    WORKFLOW = "workflow"
    AUTOMATION = "automation"


class Plugin(Base):
    """Plugin model for the marketplace."""
    
    __tablename__ = "plugins"
    
    # Primary identifiers
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    
    # Plugin details
    description = Column(Text)
    short_description = Column(String(300))
    plugin_type = Column(String(20), nullable=False)
    
    # Author and ownership
    author_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    organization_id = Column(PG_UUID(as_uuid=True), ForeignKey("organizations.id"))
    
    # Plugin metadata
    version = Column(String(20), nullable=False, default="1.0.0")
    homepage_url = Column(String(500))
    repository_url = Column(String(500))
    documentation_url = Column(String(500))
    
    # Categories and tags
    categories = Column(JSONB, default=list)
    tags = Column(JSONB, default=list)
    
    # Technical requirements
    compatibility = Column(JSONB)  # Supported platforms, versions
    dependencies = Column(JSONB, default=list)  # Required dependencies
    permissions = Column(JSONB, default=list)  # Required permissions
    
    # Installation and distribution
    download_url = Column(String(500))
    install_command = Column(Text)
    package_size = Column(Integer, default=0)  # Size in bytes
    
    # Status and visibility
    status = Column(String(20), default=PluginStatus.DRAFT.value)
    is_public = Column(Boolean, default=False)
    is_featured = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)  # Verified by maintainers
    is_premium = Column(Boolean, default=False)
    
    # Pricing (if premium)
    price = Column(Float, default=0.0)
    currency = Column(String(3), default="USD")
    license_type = Column(String(50), default="MIT")
    
    # Analytics
    download_count = Column(Integer, default=0)
    install_count = Column(Integer, default=0)
    active_installs = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    
    # Ratings and reviews
    rating_count = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    star_count = Column(Integer, default=0)
    
    # Security
    security_scan_status = Column(String(20), default="pending")
    security_scan_date = Column(DateTime)
    security_issues_count = Column(Integer, default=0)
    is_security_approved = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime)
    last_used_at = Column(DateTime)
    
    # Relationships
    author = relationship("UserProfile")
    organization = relationship("Organization")
    reviews = relationship("PluginReview", back_populates="plugin")
    installs = relationship("PluginInstall", back_populates="plugin")
    
    def to_dict(self, include_private: bool = False) -> Dict[str, Any]:
        """Convert plugin to dictionary."""
        data = {
            'id': str(self.id),
            'slug': self.slug,
            'name': self.name,
            'description': self.description,
            'short_description': self.short_description,
            'plugin_type': self.plugin_type,
            'version': self.version,
            'homepage_url': self.homepage_url,
            'repository_url': self.repository_url,
            'documentation_url': self.documentation_url,
            'categories': self.categories,
            'tags': self.tags,
            'compatibility': self.compatibility,
            'dependencies': self.dependencies,
            'download_url': self.download_url,
            'package_size': self.package_size,
            'is_featured': self.is_featured,
            'is_verified': self.is_verified,
            'is_premium': self.is_premium,
            'price': float(self.price) if self.price else 0.0,
            'currency': self.currency,
            'license_type': self.license_type,
            'download_count': self.download_count,
            'install_count': self.install_count,
            'active_installs': self.active_installs,
            'rating_count': self.rating_count,
            'average_rating': float(self.average_rating) if self.average_rating else 0.0,
            'star_count': self.star_count,
            'is_security_approved': self.is_security_approved,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'published_at': self.published_at.isoformat() if self.published_at else None
        }
        
        if include_private:
            data.update({
                'author_id': str(self.author_id),
                'status': self.status,
                'security_scan_status': self.security_scan_status,
                'security_issues_count': self.security_issues_count,
                'permissions': self.permissions
            })
        
        return data


class PluginDependency(Base):
    """Plugin dependency relationships."""
    
    __tablename__ = "plugin_dependencies"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    plugin_id = Column(PG_UUID(as_uuid=True), ForeignKey("plugins.id"), nullable=False)
    dependency_plugin_id = Column(PG_UUID(as_uuid=True), ForeignKey("plugins.id"))
    
    # Dependency details
    dependency_name = Column(String(100), nullable=False)
    dependency_version = Column(String(50))
    dependency_type = Column(String(20), default="required")  # required, optional, dev
    
    # External dependencies (not plugins)
    is_external = Column(Boolean, default=False)
    external_package_url = Column(String(500))
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    plugin = relationship("Plugin")
    dependency_plugin = relationship("Plugin", foreign_keys=[dependency_plugin_id])


class PluginInstall(Base):
    """Plugin installation tracking."""
    
    __tablename__ = "plugin_installs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    plugin_id = Column(PG_UUID(as_uuid=True), ForeignKey("plugins.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Installation details
    version_installed = Column(String(20))
    installation_method = Column(String(20), default="manual")  # manual, cli, api
    client_info = Column(JSONB)  # User agent, platform, etc.
    
    # Status
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime)
    
    # Timestamps
    installed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    uninstalled_at = Column(DateTime)
    
    # Relationships
    plugin = relationship("Plugin", back_populates="installs")
    user = relationship("UserProfile")


class PluginReview(Base):
    """Plugin reviews and ratings."""
    
    __tablename__ = "plugin_reviews"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    plugin_id = Column(PG_UUID(as_uuid=True), ForeignKey("plugins.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Review content
    rating = Column(Integer, nullable=False)  # 1-5 stars
    title = Column(String(200))
    content = Column(Text)
    
    # Review metadata
    plugin_version_reviewed = Column(String(20))
    is_verified_purchase = Column(Boolean, default=False)
    
    # Moderation
    is_approved = Column(Boolean, default=True)
    is_hidden = Column(Boolean, default=False)
    moderation_reason = Column(Text)
    
    # Helpfulness tracking
    helpful_count = Column(Integer, default=0)
    not_helpful_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    plugin = relationship("Plugin", back_populates="reviews")
    user = relationship("UserProfile")


class PluginSecurityScan(Base):
    """Plugin security scan results."""
    
    __tablename__ = "plugin_security_scans"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    plugin_id = Column(PG_UUID(as_uuid=True), ForeignKey("plugins.id"), nullable=False)
    
    # Scan details
    scan_version = Column(String(20))
    scanner_name = Column(String(50), default="internal")
    scan_type = Column(String(20), default="automated")  # automated, manual
    
    # Results
    status = Column(String(20), default="pending")  # pending, completed, failed
    overall_score = Column(Float, default=0.0)  # 0-100
    risk_level = Column(String(10), default="unknown")  # low, medium, high, critical
    
    # Issues found
    vulnerabilities = Column(JSONB, default=list)
    security_issues = Column(JSONB, default=list)
    code_quality_issues = Column(JSONB, default=list)
    
    # Scan metadata
    scan_duration = Column(Integer, default=0)  # seconds
    files_scanned = Column(Integer, default=0)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    
    # Relationships
    plugin = relationship("Plugin")


# Pydantic models for API

class PluginBase(BaseModel):
    """Base plugin schema."""
    name: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    short_description: Optional[str] = Field(None, max_length=300)
    plugin_type: PluginType
    version: str = Field("1.0.0", regex=r'^\d+\.\d+\.\d+$')
    homepage_url: Optional[str] = Field(None, max_length=500)
    repository_url: Optional[str] = Field(None, max_length=500)
    documentation_url: Optional[str] = Field(None, max_length=500)
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    license_type: str = Field("MIT", max_length=50)


class PluginCreate(PluginBase):
    """Schema for creating plugins."""
    slug: str = Field(..., min_length=3, max_length=100, regex=r'^[a-z0-9-]+$')
    compatibility: Optional[Dict[str, Any]] = None
    dependencies: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    install_command: Optional[str] = None
    is_premium: bool = False
    price: Optional[float] = Field(None, ge=0)
    
    @validator('price')
    def validate_price(cls, v, values):
        if values.get('is_premium') and (v is None or v <= 0):
            raise ValueError('Premium plugins must have a price greater than 0')
        return v


class PluginUpdate(BaseModel):
    """Schema for updating plugins."""
    name: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    short_description: Optional[str] = Field(None, max_length=300)
    version: Optional[str] = Field(None, regex=r'^\d+\.\d+\.\d+$')
    homepage_url: Optional[str] = Field(None, max_length=500)
    repository_url: Optional[str] = Field(None, max_length=500)
    documentation_url: Optional[str] = Field(None, max_length=500)
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    install_command: Optional[str] = None
    is_public: Optional[bool] = None


class PluginResponse(PluginBase):
    """Schema for plugin responses."""
    id: UUID
    slug: str
    author_id: UUID
    status: PluginStatus
    is_public: bool
    is_featured: bool
    is_verified: bool
    is_premium: bool
    price: float
    currency: str
    download_count: int
    install_count: int
    active_installs: int
    rating_count: int
    average_rating: float
    star_count: int
    is_security_approved: bool
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class PluginSearchFilters(BaseModel):
    """Plugin search filters."""
    query: Optional[str] = None
    plugin_type: Optional[PluginType] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    is_featured: Optional[bool] = None
    is_free: Optional[bool] = None
    is_verified: Optional[bool] = None
    min_rating: Optional[float] = Field(None, ge=1, le=5)
    max_rating: Optional[float] = Field(None, ge=1, le=5)
    author_id: Optional[UUID] = None
    sort_by: str = Field("updated_at", regex=r'^(created_at|updated_at|download_count|rating|install_count)$')
    sort_order: str = Field("desc", regex=r'^(asc|desc)$')


class PluginReviewCreate(BaseModel):
    """Schema for creating plugin reviews."""
    plugin_id: UUID
    rating: int = Field(..., ge=1, le=5)
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, max_length=2000)
    plugin_version_reviewed: Optional[str] = None


class PluginReviewResponse(BaseModel):
    """Schema for plugin review responses."""
    id: UUID
    plugin_id: UUID
    user_id: UUID
    rating: int
    title: Optional[str] = None
    content: Optional[str] = None
    plugin_version_reviewed: Optional[str] = None
    helpful_count: int
    not_helpful_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
