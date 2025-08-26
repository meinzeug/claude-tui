"""
Community Database Schema - PostgreSQL schema definitions for community features.
"""

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, Boolean, ForeignKey, 
    Index, UniqueConstraint, CheckConstraint, func
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, TSVECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime

Base = declarative_base()


# ============================================================================
# TEMPLATES
# ============================================================================

class Template(Base):
    """Enhanced template model with full-text search."""
    
    __tablename__ = "templates"
    
    # Primary identifiers
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    
    # Template details
    description = Column(Text)
    short_description = Column(String(300))
    template_type = Column(String(30), nullable=False, index=True)
    complexity_level = Column(String(20), index=True)
    
    # Author and ownership
    author_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, index=True)
    organization_id = Column(PG_UUID(as_uuid=True), ForeignKey("organizations.id"))
    
    # Template metadata
    version = Column(String(20), nullable=False, default="1.0.0")
    homepage_url = Column(String(500))
    repository_url = Column(String(500))
    documentation_url = Column(String(500))
    demo_url = Column(String(500))
    
    # Categories and tags (JSON arrays for flexibility)
    categories = Column(JSONB, default=list, index=True)
    tags = Column(JSONB, default=list, index=True)
    frameworks = Column(JSONB, default=list, index=True)
    languages = Column(JSONB, default=list, index=True)
    
    # Template content
    template_files = Column(JSONB)  # File structure and content
    installation_guide = Column(Text)
    usage_examples = Column(JSONB)
    
    # Requirements
    requirements = Column(JSONB)  # Dependencies, system requirements
    compatibility = Column(JSONB)  # Supported platforms, versions
    
    # Status and visibility
    is_public = Column(Boolean, default=False, index=True)
    is_featured = Column(Boolean, default=False, index=True)
    is_premium = Column(Boolean, default=False, index=True)
    is_verified = Column(Boolean, default=False)  # Staff verified
    
    # Pricing (if premium)
    price = Column(Float, default=0.0)
    currency = Column(String(3), default="USD")
    license_type = Column(String(50), default="MIT")
    
    # Analytics and engagement
    download_count = Column(Integer, default=0, index=True)
    view_count = Column(Integer, default=0)
    star_count = Column(Integer, default=0, index=True)
    fork_count = Column(Integer, default=0)
    
    # Ratings
    rating_count = Column(Integer, default=0, index=True)
    average_rating = Column(Float, default=0.0, index=True)
    
    # Full-text search
    search_vector = Column(TSVECTOR)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    published_at = Column(DateTime, index=True)
    last_used_at = Column(DateTime)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('price >= 0', name='positive_price'),
        CheckConstraint('average_rating >= 0 AND average_rating <= 5', name='valid_rating'),
        Index('ix_templates_search', search_vector, postgresql_using='gin'),
        Index('ix_templates_categories', categories, postgresql_using='gin'),
        Index('ix_templates_tags', tags, postgresql_using='gin'),
        Index('ix_templates_frameworks', frameworks, postgresql_using='gin'),
        Index('ix_templates_languages', languages, postgresql_using='gin'),
        Index('ix_templates_composite_search', 'is_public', 'average_rating', 'download_count'),
    )


# ============================================================================
# PLUGINS
# ============================================================================

class Plugin(Base):
    """Plugin model with security scanning and dependency management."""
    
    __tablename__ = "plugins"
    
    # Primary identifiers
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    
    # Plugin details
    description = Column(Text)
    short_description = Column(String(300))
    plugin_type = Column(String(20), nullable=False, index=True)
    
    # Author and ownership
    author_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, index=True)
    organization_id = Column(PG_UUID(as_uuid=True), ForeignKey("organizations.id"))
    
    # Plugin metadata
    version = Column(String(20), nullable=False, default="1.0.0")
    homepage_url = Column(String(500))
    repository_url = Column(String(500))
    documentation_url = Column(String(500))
    
    # Categories and tags
    categories = Column(JSONB, default=list, index=True)
    tags = Column(JSONB, default=list, index=True)
    
    # Technical requirements
    compatibility = Column(JSONB)  # Supported platforms, versions
    dependencies = Column(JSONB, default=list)  # Required dependencies
    permissions = Column(JSONB, default=list)  # Required permissions
    
    # Installation and distribution
    download_url = Column(String(500))
    install_command = Column(Text)
    package_size = Column(Integer, default=0)  # Size in bytes
    checksum = Column(String(128))  # SHA-256 checksum
    
    # Status and visibility
    status = Column(String(20), default="draft", index=True)
    is_public = Column(Boolean, default=False, index=True)
    is_featured = Column(Boolean, default=False, index=True)
    is_verified = Column(Boolean, default=False, index=True)
    is_premium = Column(Boolean, default=False, index=True)
    
    # Pricing (if premium)
    price = Column(Float, default=0.0)
    currency = Column(String(3), default="USD")
    license_type = Column(String(50), default="MIT")
    
    # Analytics
    download_count = Column(Integer, default=0, index=True)
    install_count = Column(Integer, default=0, index=True)
    active_installs = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    
    # Ratings and reviews
    rating_count = Column(Integer, default=0, index=True)
    average_rating = Column(Float, default=0.0, index=True)
    star_count = Column(Integer, default=0, index=True)
    
    # Security
    security_scan_status = Column(String(20), default="pending", index=True)
    security_scan_date = Column(DateTime)
    security_issues_count = Column(Integer, default=0)
    is_security_approved = Column(Boolean, default=False, index=True)
    
    # Full-text search
    search_vector = Column(TSVECTOR)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    published_at = Column(DateTime, index=True)
    last_used_at = Column(DateTime)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('price >= 0', name='positive_plugin_price'),
        CheckConstraint('average_rating >= 0 AND average_rating <= 5', name='valid_plugin_rating'),
        Index('ix_plugins_search', search_vector, postgresql_using='gin'),
        Index('ix_plugins_categories', categories, postgresql_using='gin'),
        Index('ix_plugins_tags', tags, postgresql_using='gin'),
        Index('ix_plugins_composite_search', 'is_public', 'status', 'is_security_approved'),
    )


# ============================================================================
# RATINGS & REVIEWS
# ============================================================================

class TemplateRating(Base):
    """Enhanced template rating system with detailed metrics."""
    
    __tablename__ = "template_ratings"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, index=True)
    
    # Rating details
    overall_rating = Column(Integer, nullable=False)  # 1-5 stars
    quality_rating = Column(Integer)  # Code quality
    usability_rating = Column(Integer)  # Ease of use
    documentation_rating = Column(Integer)  # Documentation quality
    support_rating = Column(Integer)  # Author support
    
    # Review content
    title = Column(String(200))
    content = Column(Text)
    
    # Context
    template_version_reviewed = Column(String(20))
    use_case = Column(String(200))
    experience_level = Column(String(20))
    
    # Verification
    is_verified_usage = Column(Boolean, default=False, index=True)
    usage_duration_days = Column(Integer)
    
    # Moderation
    status = Column(String(20), default="pending", index=True)
    moderation_notes = Column(Text)
    moderated_by = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    moderated_at = Column(DateTime)
    
    # Helpfulness tracking
    helpful_votes = Column(Integer, default=0, index=True)
    not_helpful_votes = Column(Integer, default=0)
    total_votes = Column(Integer, default=0)
    helpfulness_score = Column(Float, default=0.0)
    
    # Analytics
    view_count = Column(Integer, default=0)
    
    # Full-text search for review content
    search_vector = Column(TSVECTOR)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('template_id', 'user_id', name='unique_user_template_rating'),
        CheckConstraint('overall_rating >= 1 AND overall_rating <= 5', name='valid_overall_rating'),
        CheckConstraint('quality_rating IS NULL OR (quality_rating >= 1 AND quality_rating <= 5)', name='valid_quality_rating'),
        CheckConstraint('usability_rating IS NULL OR (usability_rating >= 1 AND usability_rating <= 5)', name='valid_usability_rating'),
        CheckConstraint('documentation_rating IS NULL OR (documentation_rating >= 1 AND documentation_rating <= 5)', name='valid_documentation_rating'),
        CheckConstraint('support_rating IS NULL OR (support_rating >= 1 AND support_rating <= 5)', name='valid_support_rating'),
        Index('ix_template_ratings_search', search_vector, postgresql_using='gin'),
        Index('ix_template_ratings_composite', 'template_id', 'status', 'created_at'),
    )


class PluginReview(Base):
    """Plugin reviews and ratings."""
    
    __tablename__ = "plugin_reviews"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    plugin_id = Column(PG_UUID(as_uuid=True), ForeignKey("plugins.id"), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, index=True)
    
    # Review content
    rating = Column(Integer, nullable=False)  # 1-5 stars
    title = Column(String(200))
    content = Column(Text)
    
    # Review metadata
    plugin_version_reviewed = Column(String(20))
    is_verified_purchase = Column(Boolean, default=False, index=True)
    
    # Moderation
    is_approved = Column(Boolean, default=True, index=True)
    is_hidden = Column(Boolean, default=False)
    moderation_reason = Column(Text)
    
    # Helpfulness tracking
    helpful_count = Column(Integer, default=0)
    not_helpful_count = Column(Integer, default=0)
    
    # Full-text search
    search_vector = Column(TSVECTOR)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('plugin_id', 'user_id', name='unique_user_plugin_review'),
        CheckConstraint('rating >= 1 AND rating <= 5', name='valid_plugin_rating'),
        Index('ix_plugin_reviews_search', search_vector, postgresql_using='gin'),
        Index('ix_plugin_reviews_composite', 'plugin_id', 'is_approved', 'created_at'),
    )


# ============================================================================
# USER SYSTEM
# ============================================================================

class UserProfile(Base):
    """Enhanced user profile with community features."""
    
    __tablename__ = "user_profiles"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    
    # Profile information
    display_name = Column(String(100))
    bio = Column(Text)
    avatar_url = Column(String(500))
    website_url = Column(String(500))
    location = Column(String(100))
    
    # Social links
    github_url = Column(String(500))
    twitter_url = Column(String(500))
    linkedin_url = Column(String(500))
    
    # Community metrics
    templates_count = Column(Integer, default=0)
    plugins_count = Column(Integer, default=0)
    reviews_count = Column(Integer, default=0)
    downloads_received = Column(Integer, default=0)
    
    # Account settings
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    
    # Privacy settings
    profile_visibility = Column(String(20), default="public")  # public, private, limited
    show_email = Column(Boolean, default=False)
    show_real_name = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)
    
    # Full-text search
    search_vector = Column(TSVECTOR)
    
    # Constraints
    __table_args__ = (
        Index('ix_user_profiles_search', search_vector, postgresql_using='gin'),
        Index('ix_user_profiles_composite', 'is_active', 'created_at'),
    )


class UserReputation(Base):
    """User reputation system based on community contributions."""
    
    __tablename__ = "user_reputation"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, unique=True)
    
    # Reputation scores
    total_reputation = Column(Integer, default=0, index=True)
    reviewer_reputation = Column(Integer, default=0)
    creator_reputation = Column(Integer, default=0)
    community_reputation = Column(Integer, default=0)
    
    # Review metrics
    total_reviews_written = Column(Integer, default=0)
    helpful_reviews_count = Column(Integer, default=0)
    average_review_helpfulness = Column(Float, default=0.0)
    
    # Template metrics
    templates_created = Column(Integer, default=0)
    average_template_rating = Column(Float, default=0.0)
    total_template_downloads = Column(Integer, default=0)
    
    # Plugin metrics
    plugins_created = Column(Integer, default=0)
    average_plugin_rating = Column(Float, default=0.0)
    total_plugin_installs = Column(Integer, default=0)
    
    # Badges and achievements
    badges = Column(JSONB, default=list)
    achievements = Column(JSONB, default=list)
    
    # Status
    is_trusted_reviewer = Column(Boolean, default=False, index=True)
    is_verified_creator = Column(Boolean, default=False, index=True)
    is_community_moderator = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        Index('ix_user_reputation_composite', 'total_reputation', 'is_trusted_reviewer'),
    )


# ============================================================================
# MODERATION SYSTEM
# ============================================================================

class ContentModerationEntry(Base):
    """Central moderation tracking for all content types."""
    
    __tablename__ = "content_moderation"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Content identification
    content_type = Column(String(20), nullable=False, index=True)
    content_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    content_author_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), index=True)
    
    # Moderation details
    status = Column(String(20), default="pending", index=True)
    priority = Column(String(10), default="medium", index=True)
    
    # Detection source
    detection_method = Column(String(20), nullable=False, index=True)
    reporter_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    
    # AI Analysis Results
    ai_confidence_score = Column(Float, default=0.0)
    ai_spam_probability = Column(Float, default=0.0)
    ai_toxicity_score = Column(Float, default=0.0)
    ai_adult_content_score = Column(Float, default=0.0)
    ai_violence_score = Column(Float, default=0.0)
    ai_hate_speech_score = Column(Float, default=0.0)
    ai_recommendation = Column(String(20))
    
    # Detected violations
    violations = Column(JSONB, default=list)
    risk_factors = Column(JSONB, default=list)
    
    # Human moderation
    assigned_moderator_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"))
    moderator_action = Column(String(20))
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
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    reviewed_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Constraints
    __table_args__ = (
        Index('ix_content_moderation_composite', 'content_type', 'status', 'priority', 'created_at'),
        Index('ix_content_moderation_ai_scores', 'ai_spam_probability', 'ai_toxicity_score'),
        UniqueConstraint('content_type', 'content_id', name='unique_content_moderation'),
    )


# ============================================================================
# ANALYTICS & TRACKING
# ============================================================================

class TemplateDownload(Base):
    """Template download tracking with detailed analytics."""
    
    __tablename__ = "template_downloads"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), index=True)
    
    # Download details
    download_type = Column(String(20), default="direct", index=True)
    client_info = Column(JSONB)
    ip_address_hash = Column(String(64))  # Hashed for privacy
    
    # Context
    project_name = Column(String(200))
    intended_use = Column(String(200))
    referrer_url = Column(String(500))
    
    # Geographic data (anonymized)
    country_code = Column(String(2))
    city = Column(String(100))
    
    # Timestamps
    downloaded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Constraints
    __table_args__ = (
        Index('ix_template_downloads_analytics', 'template_id', 'downloaded_at', 'country_code'),
        Index('ix_template_downloads_user_timeline', 'user_id', 'downloaded_at'),
    )


class PluginInstall(Base):
    """Plugin installation tracking."""
    
    __tablename__ = "plugin_installs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    plugin_id = Column(PG_UUID(as_uuid=True), ForeignKey("plugins.id"), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, index=True)
    
    # Installation details
    version_installed = Column(String(20))
    installation_method = Column(String(20), default="manual", index=True)
    client_info = Column(JSONB)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    last_used_at = Column(DateTime)
    
    # Timestamps
    installed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    uninstalled_at = Column(DateTime)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('plugin_id', 'user_id', name='unique_user_plugin_install'),
        Index('ix_plugin_installs_analytics', 'plugin_id', 'is_active', 'installed_at'),
    )


# ============================================================================
# MARKETPLACE FEATURES
# ============================================================================

class FeaturedCollection(Base):
    """Featured template collections."""
    
    __tablename__ = "featured_collections"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    
    # Collection details
    description = Column(Text)
    short_description = Column(String(300))
    collection_type = Column(String(30), nullable=False, index=True)
    
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
    sort_order = Column(Integer, default=0, index=True)
    max_templates = Column(Integer, default=20)
    is_dynamic = Column(Boolean, default=False)
    auto_criteria = Column(JSONB)
    
    # Status and visibility
    is_active = Column(Boolean, default=True, index=True)
    is_featured = Column(Boolean, default=False, index=True)
    is_public = Column(Boolean, default=True)
    
    # Analytics
    view_count = Column(Integer, default=0)
    template_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    featured_until = Column(DateTime)
    
    # Constraints
    __table_args__ = (
        Index('ix_featured_collections_composite', 'is_active', 'is_featured', 'sort_order'),
    )


# ============================================================================
# CACHING OPTIMIZATION VIEWS
# ============================================================================

# Create materialized views for frequently accessed data
# These would be created via SQL migrations

"""
-- Materialized view for template statistics
CREATE MATERIALIZED VIEW template_stats AS
SELECT 
    t.id,
    t.name,
    t.download_count,
    t.average_rating,
    t.rating_count,
    COUNT(DISTINCT td.id) as recent_downloads,
    COUNT(DISTINCT tr.id) as recent_reviews
FROM templates t
LEFT JOIN template_downloads td ON t.id = td.template_id 
    AND td.downloaded_at >= NOW() - INTERVAL '30 days'
LEFT JOIN template_ratings tr ON t.id = tr.template_id 
    AND tr.created_at >= NOW() - INTERVAL '30 days'
    AND tr.status = 'approved'
WHERE t.is_public = true
GROUP BY t.id, t.name, t.download_count, t.average_rating, t.rating_count;

CREATE UNIQUE INDEX ON template_stats (id);

-- Materialized view for trending content
CREATE MATERIALIZED VIEW trending_content AS
SELECT 
    'template' as content_type,
    t.id,
    t.name,
    t.average_rating,
    COUNT(DISTINCT td.id) as recent_activity,
    (COUNT(DISTINCT td.id) * 2 + COUNT(DISTINCT tr.id) * 3 + t.average_rating * t.rating_count) as trending_score
FROM templates t
LEFT JOIN template_downloads td ON t.id = td.template_id 
    AND td.downloaded_at >= NOW() - INTERVAL '7 days'
LEFT JOIN template_ratings tr ON t.id = tr.template_id 
    AND tr.created_at >= NOW() - INTERVAL '7 days'
    AND tr.status = 'approved'
WHERE t.is_public = true
GROUP BY t.id, t.name, t.average_rating, t.rating_count
UNION ALL
SELECT 
    'plugin' as content_type,
    p.id,
    p.name,
    p.average_rating,
    COUNT(DISTINCT pi.id) as recent_activity,
    (COUNT(DISTINCT pi.id) * 2 + COUNT(DISTINCT pr.id) * 3 + p.average_rating * p.rating_count) as trending_score
FROM plugins p
LEFT JOIN plugin_installs pi ON p.id = pi.plugin_id 
    AND pi.installed_at >= NOW() - INTERVAL '7 days'
LEFT JOIN plugin_reviews pr ON p.id = pr.plugin_id 
    AND pr.created_at >= NOW() - INTERVAL '7 days'
    AND pr.is_approved = true
WHERE p.is_public = true AND p.status = 'published'
GROUP BY p.id, p.name, p.average_rating, p.rating_count
ORDER BY trending_score DESC;

-- Refresh these views periodically (via cron job or background task)
-- REFRESH MATERIALIZED VIEW CONCURRENTLY template_stats;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY trending_content;
"""


# Function to create full-text search triggers
def create_search_triggers():
    """SQL commands to create full-text search triggers."""
    return """
    -- Template search vector trigger
    CREATE OR REPLACE FUNCTION update_template_search_vector() RETURNS trigger AS $$
    BEGIN
        NEW.search_vector := 
            setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(NEW.short_description, '')), 'B') ||
            setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'C') ||
            setweight(to_tsvector('english', COALESCE(array_to_string(NEW.tags, ' '), '')), 'D');
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    CREATE TRIGGER template_search_vector_update 
        BEFORE INSERT OR UPDATE ON templates
        FOR EACH ROW EXECUTE FUNCTION update_template_search_vector();

    -- Plugin search vector trigger
    CREATE OR REPLACE FUNCTION update_plugin_search_vector() RETURNS trigger AS $$
    BEGIN
        NEW.search_vector := 
            setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(NEW.short_description, '')), 'B') ||
            setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'C') ||
            setweight(to_tsvector('english', COALESCE(array_to_string(NEW.tags, ' '), '')), 'D');
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    CREATE TRIGGER plugin_search_vector_update 
        BEFORE INSERT OR UPDATE ON plugins
        FOR EACH ROW EXECUTE FUNCTION update_plugin_search_vector();

    -- Rating search vector trigger
    CREATE OR REPLACE FUNCTION update_rating_search_vector() RETURNS trigger AS $$
    BEGIN
        NEW.search_vector := 
            setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B');
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    CREATE TRIGGER rating_search_vector_update 
        BEFORE INSERT OR UPDATE ON template_ratings
        FOR EACH ROW EXECUTE FUNCTION update_rating_search_vector();

    -- User profile search vector trigger
    CREATE OR REPLACE FUNCTION update_user_search_vector() RETURNS trigger AS $$
    BEGIN
        NEW.search_vector := 
            setweight(to_tsvector('english', COALESCE(NEW.username, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(NEW.display_name, '')), 'B') ||
            setweight(to_tsvector('english', COALESCE(NEW.bio, '')), 'C');
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    CREATE TRIGGER user_search_vector_update 
        BEFORE INSERT OR UPDATE ON user_profiles
        FOR EACH ROW EXECUTE FUNCTION update_user_search_vector();
    """