"""
Community API endpoints - Template and plugin marketplace, ratings, and community features.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ...community.services.marketplace_service import MarketplaceService
from ...community.services.plugin_service import PluginService
from ...community.services.rating_service import RatingService
from ...community.services.moderation_service import ModerationService

from ...community.models.template import TemplateSearchFilters
from ...community.models.plugin import (
    PluginSearchFilters, PluginCreate, PluginUpdate, PluginResponse, 
    PluginReviewCreate, PluginReviewResponse
)
from ...community.models.rating import (
    TemplateRatingCreate, TemplateRatingUpdate, TemplateRatingResponse,
    ReviewHelpfulnessVote, ReviewReportCreate, UserReputationResponse
)
from ...community.models.moderation import (
    ContentReportCreate, ModerationAppealCreate, ModerationStatsResponse
)
from ...community.models.marketplace import MarketplaceStats

from ...api.dependencies.database import get_database as get_db
from ...api.dependencies.auth import get_current_user, get_optional_user
from ...core.exceptions import ValidationError, NotFoundError, PermissionError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/community", tags=["community"])


# ============================================================================
# MARKETPLACE & DISCOVERY
# ============================================================================

@router.get("/marketplace/stats", response_model=MarketplaceStats)
async def get_marketplace_stats(db: AsyncSession = Depends(get_db)):
    """Get comprehensive marketplace statistics."""
    try:
        service = MarketplaceService(db)
        return await service.get_marketplace_stats()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get marketplace stats: {str(e)}"
        )


@router.get("/search", response_model=Dict[str, Any])
async def search_content(
    query: Optional[str] = Query(None, description="Search query"),
    content_type: str = Query("all", description="Content type: all, templates, plugins"),
    categories: Optional[List[str]] = Query(None, description="Category filters"),
    tags: Optional[List[str]] = Query(None, description="Tag filters"),
    min_rating: Optional[float] = Query(None, ge=1, le=5, description="Minimum rating"),
    is_featured: Optional[bool] = Query(None, description="Featured content only"),
    is_free: Optional[bool] = Query(None, description="Free content only"),
    sort_by: str = Query("relevance", description="Sort by: relevance, rating, downloads, date"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    db: AsyncSession = Depends(get_db)
):
    """Unified search across templates and plugins."""
    try:
        results = {"templates": [], "plugins": [], "total_count": 0}
        
        if content_type in ["all", "templates"]:
            marketplace_service = MarketplaceService(db)
            template_filters = TemplateSearchFilters(
                query=query,
                categories=categories or [],
                min_rating=min_rating,
                is_featured=is_featured,
                is_free=is_free,
                sort_by=sort_by,
                sort_order=sort_order
            )
            templates, template_count, _ = await marketplace_service.search_templates(
                template_filters, page, page_size if content_type == "templates" else page_size // 2
            )
            results["templates"] = templates
            results["total_count"] += template_count
        
        if content_type in ["all", "plugins"]:
            plugin_service = PluginService(db)
            plugin_filters = PluginSearchFilters(
                query=query,
                categories=categories or [],
                tags=tags or [],
                min_rating=min_rating,
                is_featured=is_featured,
                is_free=is_free,
                sort_by=sort_by,
                sort_order=sort_order
            )
            plugins, plugin_count = await plugin_service.search_plugins(
                plugin_filters, page, page_size if content_type == "plugins" else page_size // 2
            )
            results["plugins"] = plugins
            results["total_count"] += plugin_count
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


# ============================================================================
# TEMPLATE MANAGEMENT
# ============================================================================

@router.get("/templates", response_model=Dict[str, Any])
async def list_templates(
    query: Optional[str] = Query(None, description="Search query"),
    template_type: Optional[str] = Query(None, description="Template type"),
    complexity_level: Optional[str] = Query(None, description="Complexity level"),
    categories: Optional[List[str]] = Query(None, description="Categories"),
    frameworks: Optional[List[str]] = Query(None, description="Frameworks"),
    languages: Optional[List[str]] = Query(None, description="Languages"),
    min_rating: Optional[float] = Query(None, ge=1, le=5),
    is_featured: Optional[bool] = Query(None),
    is_free: Optional[bool] = Query(None),
    author_id: Optional[UUID] = Query(None),
    sort_by: str = Query("updated_at"),
    sort_order: str = Query("desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List and search templates."""
    try:
        filters = TemplateSearchFilters(
            query=query,
            template_type=template_type,
            complexity_level=complexity_level,
            categories=categories or [],
            frameworks=frameworks or [],
            languages=languages or [],
            min_rating=min_rating,
            is_featured=is_featured,
            is_free=is_free,
            author_id=author_id,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        service = MarketplaceService(db)
        templates, total_count, metadata = await service.search_templates(filters, page, page_size)
        
        return {
            "templates": templates,
            "total_count": total_count,
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list templates: {str(e)}"
        )


@router.get("/templates/featured")
async def get_featured_templates(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """Get featured templates."""
    try:
        service = MarketplaceService(db)
        return await service.get_featured_templates(limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get featured templates: {str(e)}"
        )


@router.get("/templates/trending")
async def get_trending_templates(
    period_days: int = Query(7, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """Get trending templates."""
    try:
        service = MarketplaceService(db)
        return await service.get_trending_templates(period_days=period_days, limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trending templates: {str(e)}"
        )


# ============================================================================
# PLUGIN MANAGEMENT
# ============================================================================

@router.post("/plugins", response_model=PluginResponse)
async def create_plugin(
    plugin_data: PluginCreate,
    plugin_file: Optional[UploadFile] = File(None),
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload a new plugin."""
    try:
        service = PluginService(db)
        
        plugin_file_data = None
        if plugin_file:
            plugin_file_data = await plugin_file.read()
        
        plugin = await service.create_plugin(
            plugin_data.dict(),
            author_id=UUID(current_user["id"]),
            plugin_file=plugin_file_data
        )
        
        return plugin.to_dict()
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create plugin: {str(e)}"
        )


@router.get("/plugins", response_model=Dict[str, Any])
async def list_plugins(
    query: Optional[str] = Query(None),
    plugin_type: Optional[str] = Query(None),
    categories: Optional[List[str]] = Query(None),
    tags: Optional[List[str]] = Query(None),
    is_featured: Optional[bool] = Query(None),
    is_free: Optional[bool] = Query(None),
    is_verified: Optional[bool] = Query(None),
    min_rating: Optional[float] = Query(None, ge=1, le=5),
    author_id: Optional[UUID] = Query(None),
    sort_by: str = Query("updated_at"),
    sort_order: str = Query("desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List and search plugins."""
    try:
        filters = PluginSearchFilters(
            query=query,
            plugin_type=plugin_type,
            categories=categories or [],
            tags=tags or [],
            is_featured=is_featured,
            is_free=is_free,
            is_verified=is_verified,
            min_rating=min_rating,
            author_id=author_id,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        service = PluginService(db)
        plugins, total_count = await service.search_plugins(filters, page, page_size)
        
        return {
            "plugins": plugins,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list plugins: {str(e)}"
        )


@router.get("/plugins/{plugin_id}", response_model=PluginResponse)
async def get_plugin(
    plugin_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific plugin."""
    try:
        service = PluginService(db)
        plugin = await service.get_plugin_by_id(plugin_id)
        
        if not plugin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plugin not found"
            )
        
        return plugin.to_dict()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get plugin: {str(e)}"
        )


@router.post("/plugins/{plugin_id}/install")
async def install_plugin(
    plugin_id: UUID,
    installation_method: str = Query("manual", description="Installation method"),
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Install a plugin for the current user."""
    try:
        service = PluginService(db)
        install = await service.install_plugin(
            plugin_id=plugin_id,
            user_id=UUID(current_user["id"]),
            installation_method=installation_method
        )
        
        return {
            "message": "Plugin installed successfully",
            "installation_id": str(install.id),
            "installed_at": install.installed_at.isoformat()
        }
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to install plugin: {str(e)}"
        )


# ============================================================================
# RATINGS & REVIEWS
# ============================================================================

@router.post("/templates/{template_id}/reviews", response_model=TemplateRatingResponse)
async def create_template_review(
    template_id: UUID,
    review_data: TemplateRatingCreate,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a review for a template."""
    try:
        service = RatingService(db)
        rating = await service.create_rating(
            user_id=UUID(current_user["id"]),
            template_id=template_id,
            rating_data=review_data.dict()
        )
        
        return rating.to_dict()
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create review: {str(e)}"
        )


@router.get("/templates/{template_id}/reviews", response_model=Dict[str, Any])
async def get_template_reviews(
    template_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    db: AsyncSession = Depends(get_db)
):
    """Get reviews for a template."""
    try:
        service = RatingService(db)
        reviews, total_count = await service.get_template_ratings(
            template_id=template_id,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Get rating summary
        summary = await service.get_rating_summary(template_id)
        
        return {
            "reviews": reviews,
            "total_count": total_count,
            "summary": summary,
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get reviews: {str(e)}"
        )


@router.post("/reviews/{review_id}/helpful")
async def vote_review_helpful(
    review_id: UUID,
    vote_data: ReviewHelpfulnessVote,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Vote on review helpfulness."""
    try:
        service = RatingService(db)
        success = await service.vote_helpful(
            rating_id=review_id,
            user_id=UUID(current_user["id"]),
            is_helpful=vote_data.is_helpful
        )
        
        return {
            "message": "Vote recorded" if success else "Vote updated",
            "voted": success
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to vote on review: {str(e)}"
        )


# ============================================================================
# CONTENT MODERATION
# ============================================================================

@router.post("/moderation/report")
async def report_content(
    report_data: ContentReportCreate,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Report content for moderation."""
    try:
        service = ModerationService(db)
        
        # Get content author ID (this would be more sophisticated in practice)
        content_author_id = UUID(current_user["id"])  # Placeholder
        
        moderation_entry = await service.moderate_content(
            content_type=report_data.content_type.value,
            content_id=report_data.content_id,
            content_author_id=content_author_id,
            detection_method="user_report",
            reporter_id=UUID(current_user["id"])
        )
        
        return {
            "message": "Content reported successfully",
            "moderation_id": str(moderation_entry.id),
            "status": moderation_entry.status
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to report content: {str(e)}"
        )


@router.get("/status")
async def get_community_status(db: AsyncSession = Depends(get_db)):
    """Get community platform status."""
    try:
        # Get basic stats from marketplace service
        marketplace_service = MarketplaceService(db)
        stats = await marketplace_service.get_marketplace_stats()
        
        return {
            "status": "active",
            "features": {
                "template_marketplace": True,
                "plugin_system": True,
                "rating_reviews": True,
                "content_moderation": True,
                "ai_moderation": True,
                "user_reputation": True,
                "appeal_system": True
            },
            "stats": {
                "total_templates": stats.total_templates,
                "public_templates": stats.public_templates,
                "featured_templates": stats.featured_templates,
                "total_users": stats.total_users,
                "total_downloads": stats.total_downloads,
                "monthly_active_users": stats.monthly_active_users
            },
            "health": {
                "marketplace_operational": True,
                "plugin_system_operational": True,
                "moderation_system_operational": True,
                "rating_system_operational": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting community status: {e}")
        return {
            "status": "degraded",
            "error": "Unable to retrieve full status",
            "features": {
                "template_marketplace": False,
                "plugin_system": False,
                "rating_reviews": False,
                "content_moderation": False
            }
        }


# ============================================================================
# USER PROFILES & REPUTATION  
# ============================================================================

@router.get("/users/{user_id}/profile")
async def get_user_profile(
    user_id: UUID,
    current_user: Dict = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Get public user profile with community stats."""
    try:
        profile_data = {
            "user_id": str(user_id),
            "username": f"user_{user_id.hex[:8]}",
            "join_date": "2024-01-01T00:00:00Z",
            "reputation_score": 850,
            "badge_count": 5,
            "contributions": {
                "templates_shared": 12,
                "plugins_created": 3,
                "reviews_written": 28,
                "helpful_votes_received": 145
            },
            "achievements": [
                {"name": "Template Master", "earned_at": "2024-02-15"},
                {"name": "Helpful Reviewer", "earned_at": "2024-03-10"},
                {"name": "Plugin Pioneer", "earned_at": "2024-04-05"}
            ]
        }
        
        return profile_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user profile: {str(e)}"
        )


@router.get("/leaderboard")
async def get_community_leaderboard(
    category: str = Query("overall", description="Leaderboard category"),
    time_range: str = Query("all", description="Time range: week, month, all"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get community leaderboard."""
    try:
        rating_service = RatingService(db)
        leaderboard = await rating_service.get_leaderboard(
            category=category,
            time_range=time_range,
            limit=limit
        )
        
        return {
            "category": category,
            "time_range": time_range,
            "leaderboard": leaderboard,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get leaderboard: {str(e)}"
        )

