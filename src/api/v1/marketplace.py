"""
Community Marketplace API Endpoints - Comprehensive REST API for Claude-TUI marketplace.

Features:
- Template sharing and distribution
- Plugin discovery and installation 
- Rating and review system
- Monetization capabilities
- Advanced search and filtering
- Secure plugin sandboxing
- Real-time analytics and metrics
"""

import asyncio
import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks, status
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

# Community marketplace imports
from ...community.models.marketplace import (
    MarketplaceStats, MarketplaceTrends, FeaturedCollectionResponse,
    TemplateTagResponse, TemplateDownloadCreate, SearchSuggestion
)
from ...community.models.plugin import (
    Plugin, PluginInstallation, PluginManifest, PluginSecurity,
    PluginStatus, PluginType, SecurityLevel
)
from ...community.models.rating import TemplateRating, TemplateReview
from ...community.models.template import Template, TemplateResponse, TemplateSearchFilters
from ...community.services.marketplace_service_enhanced import EnhancedMarketplaceService
from ...community.services.plugin_service_enhanced import EnhancedPluginService
from ...community.services.rating_service import RatingService
from ...community.services.moderation_service import ModerationService
from ...community.services.recommendation_engine import RecommendationEngine
from ...community.services.search_service import SearchService, SearchQuery, SearchResults
from ...community.security.rate_limiter import RateLimiter
from ...security.input_validator import InputValidator
from ...security.code_sandbox import CodeSandbox
from ...auth.decorators import require_auth, require_permissions
from ...api.dependencies.database import get_database as get_db
from ...api.dependencies.auth import get_current_user, get_optional_user
from ...core.exceptions import ValidationError, NotFoundError, SecurityError

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/marketplace", tags=["Community Marketplace"])

# Security and validation
security = HTTPBearer(auto_error=False)
input_validator = InputValidator()
rate_limiter = RateLimiter()


# Request/Response Models

class MarketplaceSearchRequest(BaseModel):
    """Enhanced marketplace search request."""
    query: Optional[str] = Field(None, max_length=200, description="Search query")
    template_type: Optional[str] = Field(None, description="Template type filter")
    plugin_type: Optional[str] = Field(None, description="Plugin type filter")
    categories: List[str] = Field(default_factory=list, description="Category filters")
    frameworks: List[str] = Field(default_factory=list, description="Framework filters") 
    languages: List[str] = Field(default_factory=list, description="Language filters")
    min_rating: Optional[float] = Field(None, ge=1, le=5, description="Minimum rating")
    max_rating: Optional[float] = Field(None, ge=1, le=5, description="Maximum rating")
    is_featured: Optional[bool] = Field(None, description="Featured items only")
    is_free: Optional[bool] = Field(None, description="Free items only")
    is_verified: Optional[bool] = Field(None, description="Verified items only")
    security_level: Optional[str] = Field(None, description="Minimum security level")
    compatibility_version: Optional[str] = Field(None, description="Compatible version")
    author_id: Optional[UUID] = Field(None, description="Author ID filter")
    organization_id: Optional[UUID] = Field(None, description="Organization filter")
    created_after: Optional[datetime] = Field(None, description="Created after date")
    updated_after: Optional[datetime] = Field(None, description="Updated after date")
    sort_by: str = Field("relevance", description="Sort field")
    sort_order: str = Field("desc", description="Sort order")
    include_beta: bool = Field(False, description="Include beta versions")
    include_experimental: bool = Field(False, description="Include experimental")
    
    # Advanced search options
    search_type: str = Field("fuzzy", description="Search type: exact, fuzzy, semantic")
    include_facets: bool = Field(True, description="Include faceted search results")
    min_score: float = Field(0.1, ge=0, le=1, description="Minimum relevance score")
    boost_fields: Dict[str, float] = Field(default_factory=dict, description="Field boost weights")
    price_range: Optional[str] = Field(None, description="Price range filter")
    downloads_range: Optional[str] = Field(None, description="Downloads range filter")
    updated_period: Optional[str] = Field(None, description="Updated within period")


class PluginInstallRequest(BaseModel):
    """Plugin installation request."""
    plugin_id: UUID = Field(..., description="Plugin ID to install")
    version: Optional[str] = Field(None, description="Specific version")
    environment: str = Field("production", description="Installation environment")
    auto_enable: bool = Field(True, description="Auto-enable after install")
    sandbox_mode: bool = Field(True, description="Install in sandbox mode")
    allowed_permissions: List[str] = Field(default_factory=list, description="Allowed permissions")
    installation_path: Optional[str] = Field(None, description="Custom installation path")


class PluginPublishRequest(BaseModel):
    """Plugin publishing request."""
    name: str = Field(..., min_length=3, max_length=100, description="Plugin name")
    description: str = Field(..., min_length=10, max_length=5000, description="Plugin description")
    short_description: str = Field(..., min_length=10, max_length=200, description="Short description")
    version: str = Field(..., regex=r'^\d+\.\d+\.\d+(-[a-z0-9]+)?$', description="Semantic version")
    plugin_type: str = Field(..., description="Plugin type")
    categories: List[str] = Field(..., min_items=1, description="Plugin categories")
    frameworks: List[str] = Field(default_factory=list, description="Supported frameworks")
    languages: List[str] = Field(..., min_items=1, description="Programming languages")
    required_permissions: List[str] = Field(default_factory=list, description="Required permissions")
    optional_permissions: List[str] = Field(default_factory=list, description="Optional permissions")
    compatibility: Dict[str, str] = Field(..., description="Compatibility requirements")
    dependencies: List[str] = Field(default_factory=list, description="Plugin dependencies")
    license: str = Field("MIT", description="License type")
    homepage_url: Optional[str] = Field(None, description="Homepage URL")
    repository_url: Optional[str] = Field(None, description="Repository URL")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    icon_url: Optional[str] = Field(None, description="Icon URL")
    screenshots: List[str] = Field(default_factory=list, description="Screenshot URLs")
    tags: List[str] = Field(default_factory=list, description="Plugin tags")
    is_premium: bool = Field(False, description="Is premium plugin")
    price: Optional[float] = Field(None, ge=0, description="Price in USD")
    pricing_model: Optional[str] = Field(None, description="Pricing model")


class RatingSubmissionRequest(BaseModel):
    """Rating submission request."""
    item_id: UUID = Field(..., description="Template/Plugin ID")
    item_type: str = Field(..., description="Item type (template/plugin)")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    review_title: Optional[str] = Field(None, max_length=100, description="Review title")
    review_text: Optional[str] = Field(None, max_length=5000, description="Review text")
    pros: List[str] = Field(default_factory=list, description="Positive points")
    cons: List[str] = Field(default_factory=list, description="Negative points")
    recommended: bool = Field(True, description="Recommend to others")
    experience_level: str = Field("intermediate", description="User experience level")


class MonetizationRequest(BaseModel):
    """Monetization configuration request."""
    item_id: UUID = Field(..., description="Item ID")
    item_type: str = Field(..., description="Item type")
    pricing_model: str = Field(..., description="Pricing model")
    base_price: Optional[float] = Field(None, ge=0, description="Base price")
    subscription_price: Optional[float] = Field(None, ge=0, description="Monthly subscription")
    free_tier_limits: Optional[Dict[str, Any]] = Field(None, description="Free tier limitations")
    premium_features: List[str] = Field(default_factory=list, description="Premium features")
    trial_period_days: int = Field(0, ge=0, le=90, description="Trial period days")


class MarketplaceAnalyticsResponse(BaseModel):
    """Marketplace analytics response."""
    overview: Dict[str, Any]
    trending_items: List[Dict[str, Any]]
    category_stats: Dict[str, int]
    user_engagement: Dict[str, Any]
    revenue_metrics: Dict[str, float]
    growth_trends: Dict[str, List[float]]
    popular_searches: List[str]
    conversion_rates: Dict[str, float]
    generated_at: datetime


# Dependency Functions

async def get_marketplace_service(db: AsyncSession = Depends(get_db)) -> EnhancedMarketplaceService:
    """Get marketplace service instance."""
    return EnhancedMarketplaceService(db)


async def get_plugin_service(db: AsyncSession = Depends(get_db)) -> EnhancedPluginService:
    """Get plugin service instance."""
    return EnhancedPluginService(db)


async def get_rating_service(db: AsyncSession = Depends(get_db)) -> RatingService:
    """Get rating service instance."""
    return RatingService(db)


async def get_recommendation_engine(db: AsyncSession = Depends(get_db)) -> RecommendationEngine:
    """Get recommendation engine instance."""
    return RecommendationEngine(db)


async def get_moderation_service(db: AsyncSession = Depends(get_db)) -> ModerationService:
    """Get moderation service instance."""
    return ModerationService(db)


async def get_search_service(db: AsyncSession = Depends(get_db)) -> SearchService:
    """Get advanced search service instance."""
    return SearchService(db)


# Template and Plugin Discovery Endpoints

@router.get("/search/autocomplete")
async def search_autocomplete(
    q: str = Query(..., min_length=1, max_length=100, description="Search query"),
    limit: int = Query(10, ge=1, le=20, description="Max suggestions"),
    search_service: SearchService = Depends(get_search_service),
    current_user: Optional[Dict] = Depends(get_optional_user)
):
    """Get autocomplete suggestions for search queries."""
    try:
        # Rate limiting for autocomplete
        await rate_limiter.check_rate_limit("autocomplete", current_user.get("id") if current_user else "anonymous", max_requests=50, window_minutes=1)
        
        suggestions = await search_service.get_autocomplete(q, limit)
        
        return {
            "query": q,
            "suggestions": suggestions,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Autocomplete failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Autocomplete failed: {str(e)}"
        )


@router.get("/search/trending")
async def get_trending_searches(
    limit: int = Query(10, ge=1, le=20, description="Max trending terms"),
    search_service: SearchService = Depends(get_search_service)
):
    """Get trending search terms."""
    try:
        trending = await search_service.get_trending_searches(limit)
        
        return {
            "trending_searches": trending,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get trending searches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trending searches: {str(e)}"
        )


@router.get("/search", response_model=Dict[str, Any])
async def search_marketplace(
    search_request: MarketplaceSearchRequest = Depends(),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    marketplace_service: EnhancedMarketplaceService = Depends(get_marketplace_service),
    search_service: SearchService = Depends(get_search_service),
    current_user: Optional[Dict] = Depends(get_optional_user)
):
    """Advanced marketplace search with AI-powered recommendations and facets."""
    try:
        # Rate limiting
        await rate_limiter.check_rate_limit("search", current_user.get("id") if current_user else "anonymous")
        
        # Input validation
        search_request = input_validator.sanitize_search_request(search_request)
        
        # Build advanced search query
        search_query = SearchQuery(
            query=search_request.query or "",
            filters={
                "type": [t for t in [search_request.template_type, search_request.plugin_type] if t],
                "category": search_request.categories,
                "tags": search_request.frameworks + search_request.languages,
                "price_range": [search_request.price_range] if search_request.price_range else [],
                "rating": [f"{search_request.min_rating}+"] if search_request.min_rating else [],
                "updated": [search_request.updated_period] if search_request.updated_period else [],
            },
            sort_by=search_request.sort_by,
            sort_order=search_request.sort_order,
            page=page,
            page_size=page_size,
            search_type=search_request.search_type,
            include_facets=search_request.include_facets,
            min_score=search_request.min_score,
            boost_fields=search_request.boost_fields or {}
        )
        
        # Execute advanced search
        results = await search_service.search(search_query)
        
        # Log search analytics
        await search_service.analytics.log_search(
            query=search_query.query,
            results_count=results.total_count,
            search_time_ms=results.search_time_ms,
            filters=search_query.filters,
            user_id=UUID(current_user["id"]) if current_user else None
        )
        
        return {
            "items": [item.__dict__ for item in results.items],
            "total_count": results.total_count,
            "total_pages": results.total_pages,
            "current_page": results.page,
            "page_size": results.page_size,
            "facets": {name: {
                "display_name": facet.display_name,
                "type": facet.facet_type,
                "values": facet.values,
                "selected": list(facet.selected_values)
            } for name, facet in results.facets.items()},
            "suggestions": results.suggestions,
            "did_you_mean": results.did_you_mean,
            "search_time_ms": results.search_time_ms,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Marketplace search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/featured", response_model=Dict[str, Any])
async def get_featured_content(
    content_type: Optional[str] = Query(None, description="Content type filter"),
    limit: int = Query(20, ge=1, le=100, description="Number of items"),
    marketplace_service: EnhancedMarketplaceService = Depends(get_marketplace_service)
):
    """Get featured templates and plugins."""
    try:
        featured_content = await marketplace_service.get_featured_content(
            content_type=content_type,
            limit=limit
        )
        
        return {
            "featured_items": featured_content.items,
            "collections": featured_content.collections,
            "spotlights": featured_content.spotlights,
            "editorial_picks": featured_content.editorial_picks,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get featured content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get featured content: {str(e)}"
        )


@router.get("/trending", response_model=Dict[str, Any])
async def get_trending_content(
    period: str = Query("week", description="Trending period"),
    content_type: Optional[str] = Query(None, description="Content type filter"),
    limit: int = Query(20, ge=1, le=100, description="Number of items"),
    marketplace_service: EnhancedMarketplaceService = Depends(get_marketplace_service)
):
    """Get trending templates and plugins."""
    try:
        trending_content = await marketplace_service.get_trending_content(
            period=period,
            content_type=content_type,
            limit=limit
        )
        
        return {
            "trending_items": trending_content.items,
            "trending_metrics": trending_content.metrics,
            "period": period,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get trending content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trending content: {str(e)}"
        )


@router.get("/recommendations", response_model=List[Dict[str, Any]])
async def get_personalized_recommendations(
    content_type: Optional[str] = Query(None, description="Content type filter"),
    based_on: Optional[UUID] = Query(None, description="Base recommendations on item"),
    limit: int = Query(20, ge=1, le=50, description="Number of recommendations"),
    current_user: Dict = Depends(get_current_user),
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get AI-powered personalized recommendations."""
    try:
        user_id = UUID(current_user["id"])
        
        recommendations = await recommendation_engine.get_personalized_recommendations(
            user_id=user_id,
            content_type=content_type,
            based_on=based_on,
            limit=limit
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )


# Plugin Installation and Management

@router.post("/plugins/install")
@require_auth
async def install_plugin(
    install_request: PluginInstallRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
    plugin_service: EnhancedPluginService = Depends(get_plugin_service)
):
    """Install a plugin with security validation and sandboxing."""
    try:
        user_id = UUID(current_user["id"])
        
        # Check user permissions and quotas
        await plugin_service.validate_installation_permissions(user_id, install_request.plugin_id)
        
        # Start installation in background
        installation_id = await plugin_service.initiate_installation(
            plugin_id=install_request.plugin_id,
            user_id=user_id,
            installation_config=install_request.dict(exclude={'plugin_id'})
        )
        
        # Schedule installation process
        background_tasks.add_task(
            execute_plugin_installation,
            installation_id,
            plugin_service
        )
        
        return {
            "installation_id": installation_id,
            "status": "initiated",
            "message": "Plugin installation started",
            "estimated_duration": "2-5 minutes",
            "sandbox_mode": install_request.sandbox_mode,
            "initiated_at": datetime.utcnow()
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Security validation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Plugin installation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Installation failed: {str(e)}"
        )


@router.get("/plugins/installations/{installation_id}/status")
@require_auth
async def get_installation_status(
    installation_id: UUID,
    current_user: Dict = Depends(get_current_user),
    plugin_service: EnhancedPluginService = Depends(get_plugin_service)
):
    """Get plugin installation status and progress."""
    try:
        user_id = UUID(current_user["id"])
        
        status_info = await plugin_service.get_installation_status(
            installation_id, user_id
        )
        
        return status_info
        
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Installation not found"
        )
    except Exception as e:
        logger.error(f"Failed to get installation status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {str(e)}"
        )


@router.post("/plugins/uninstall/{plugin_id}")
@require_auth
async def uninstall_plugin(
    plugin_id: UUID,
    force: bool = Query(False, description="Force uninstall"),
    current_user: Dict = Depends(get_current_user),
    plugin_service: EnhancedPluginService = Depends(get_plugin_service)
):
    """Uninstall a plugin safely."""
    try:
        user_id = UUID(current_user["id"])
        
        result = await plugin_service.uninstall_plugin(
            plugin_id=plugin_id,
            user_id=user_id,
            force=force
        )
        
        return {
            "plugin_id": plugin_id,
            "status": "uninstalled",
            "cleanup_performed": result.cleanup_performed,
            "data_preserved": result.data_preserved,
            "uninstalled_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Plugin uninstallation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Uninstallation failed: {str(e)}"
        )


@router.get("/plugins/installed")
@require_auth
async def get_installed_plugins(
    current_user: Dict = Depends(get_current_user),
    plugin_service: EnhancedPluginService = Depends(get_plugin_service)
):
    """Get user's installed plugins."""
    try:
        user_id = UUID(current_user["id"])
        
        installed_plugins = await plugin_service.get_user_installed_plugins(user_id)
        
        return {
            "plugins": installed_plugins,
            "total_count": len(installed_plugins),
            "active_count": len([p for p in installed_plugins if p.get("status") == "active"]),
            "retrieved_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get installed plugins: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get installed plugins: {str(e)}"
        )


# Plugin Publishing and Management

@router.post("/plugins/publish")
@require_auth 
async def publish_plugin(
    publish_request: PluginPublishRequest,
    plugin_file: UploadFile = File(..., description="Plugin package file"),
    current_user: Dict = Depends(get_current_user),
    plugin_service: EnhancedPluginService = Depends(get_plugin_service),
    moderation_service: ModerationService = Depends(get_moderation_service)
):
    """Publish a new plugin to the marketplace."""
    try:
        user_id = UUID(current_user["id"])
        
        # Validate file format and size
        if not plugin_file.filename.endswith(('.zip', '.tar.gz')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plugin must be packaged as ZIP or TAR.GZ"
            )
        
        # Check user publishing permissions
        await plugin_service.validate_publishing_permissions(user_id)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            content = await plugin_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Security scan and validation
            security_scan = await plugin_service.security_scan_plugin(temp_file_path)
            
            if security_scan.risk_level == "HIGH":
                raise SecurityError("Plugin contains high-risk code patterns")
            
            # Extract and validate manifest
            manifest = await plugin_service.extract_and_validate_manifest(temp_file_path)
            
            # Create plugin entry
            plugin_id = await plugin_service.create_plugin(
                author_id=user_id,
                plugin_data=publish_request.dict(),
                manifest=manifest,
                security_scan=security_scan,
                package_path=temp_file_path
            )
            
            # Submit for moderation if required
            if security_scan.requires_review or publish_request.is_premium:
                moderation_id = await moderation_service.submit_for_review(
                    item_id=plugin_id,
                    item_type="plugin",
                    submitter_id=user_id
                )
                
                return {
                    "plugin_id": plugin_id,
                    "status": "pending_review",
                    "moderation_id": moderation_id,
                    "message": "Plugin submitted for review",
                    "estimated_review_time": "24-72 hours",
                    "submitted_at": datetime.utcnow()
                }
            else:
                # Auto-approve for low-risk plugins
                await plugin_service.approve_plugin(plugin_id)
                
                return {
                    "plugin_id": plugin_id,
                    "status": "published",
                    "message": "Plugin published successfully",
                    "published_at": datetime.utcnow()
                }
            
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Plugin publishing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Publishing failed: {str(e)}"
        )


@router.put("/plugins/{plugin_id}")
@require_auth
async def update_plugin(
    plugin_id: UUID,
    update_data: Dict[str, Any],
    plugin_file: Optional[UploadFile] = File(None, description="Updated plugin package"),
    current_user: Dict = Depends(get_current_user),
    plugin_service: EnhancedPluginService = Depends(get_plugin_service)
):
    """Update an existing plugin."""
    try:
        user_id = UUID(current_user["id"])
        
        # Validate ownership
        await plugin_service.validate_plugin_ownership(plugin_id, user_id)
        
        # Process update
        updated_plugin = await plugin_service.update_plugin(
            plugin_id=plugin_id,
            update_data=update_data,
            new_package=plugin_file,
            author_id=user_id
        )
        
        return {
            "plugin_id": plugin_id,
            "status": "updated",
            "version": updated_plugin.version,
            "message": "Plugin updated successfully",
            "updated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Plugin update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update failed: {str(e)}"
        )


# Rating and Review System

@router.post("/ratings")
@require_auth
async def submit_rating(
    rating_request: RatingSubmissionRequest,
    current_user: Dict = Depends(get_current_user),
    rating_service: RatingService = Depends(get_rating_service)
):
    """Submit a rating and review for a marketplace item."""
    try:
        user_id = UUID(current_user["id"])
        
        # Validate user can rate this item
        await rating_service.validate_rating_eligibility(
            user_id, rating_request.item_id, rating_request.item_type
        )
        
        # Submit rating
        rating_id = await rating_service.submit_rating(
            user_id=user_id,
            item_id=rating_request.item_id,
            item_type=rating_request.item_type,
            rating=rating_request.rating,
            review_data={
                "title": rating_request.review_title,
                "text": rating_request.review_text,
                "pros": rating_request.pros,
                "cons": rating_request.cons,
                "recommended": rating_request.recommended,
                "experience_level": rating_request.experience_level
            }
        )
        
        return {
            "rating_id": rating_id,
            "status": "submitted",
            "message": "Rating submitted successfully",
            "submitted_at": datetime.utcnow()
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Rating submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rating submission failed: {str(e)}"
        )


@router.get("/ratings/{item_id}")
async def get_item_ratings(
    item_id: UUID,
    item_type: str = Query(..., description="Item type (template/plugin)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    sort_by: str = Query("newest", description="Sort order"),
    rating_service: RatingService = Depends(get_rating_service)
):
    """Get ratings and reviews for a marketplace item."""
    try:
        ratings_data = await rating_service.get_item_ratings(
            item_id=item_id,
            item_type=item_type,
            page=page,
            page_size=page_size,
            sort_by=sort_by
        )
        
        return {
            "ratings": ratings_data.ratings,
            "summary": ratings_data.summary,
            "distribution": ratings_data.distribution,
            "total_count": ratings_data.total_count,
            "average_rating": ratings_data.average_rating,
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        logger.error(f"Failed to get ratings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ratings: {str(e)}"
        )


# Monetization and Premium Features

@router.post("/monetization/configure")
@require_auth
@require_permissions(["marketplace.monetize"])
async def configure_monetization(
    monetization_request: MonetizationRequest,
    current_user: Dict = Depends(get_current_user),
    marketplace_service: EnhancedMarketplaceService = Depends(get_marketplace_service)
):
    """Configure monetization for a marketplace item."""
    try:
        user_id = UUID(current_user["id"])
        
        # Validate ownership
        await marketplace_service.validate_item_ownership(
            monetization_request.item_id, 
            monetization_request.item_type, 
            user_id
        )
        
        # Configure monetization
        config_id = await marketplace_service.configure_monetization(
            item_id=monetization_request.item_id,
            item_type=monetization_request.item_type,
            monetization_config=monetization_request.dict(exclude={'item_id', 'item_type'}),
            owner_id=user_id
        )
        
        return {
            "config_id": config_id,
            "status": "configured",
            "message": "Monetization configured successfully",
            "configured_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Monetization configuration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration failed: {str(e)}"
        )


@router.post("/purchases/{item_id}")
@require_auth
async def purchase_premium_item(
    item_id: UUID,
    item_type: str = Query(..., description="Item type"),
    license_type: str = Query("standard", description="License type"),
    current_user: Dict = Depends(get_current_user),
    marketplace_service: EnhancedMarketplaceService = Depends(get_marketplace_service)
):
    """Purchase a premium marketplace item."""
    try:
        user_id = UUID(current_user["id"])
        
        # Process purchase
        purchase_id = await marketplace_service.process_purchase(
            item_id=item_id,
            item_type=item_type,
            buyer_id=user_id,
            license_type=license_type
        )
        
        return {
            "purchase_id": purchase_id,
            "status": "completed",
            "message": "Purchase completed successfully",
            "access_granted": True,
            "purchased_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Purchase failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Purchase failed: {str(e)}"
        )


# Analytics and Metrics

@router.get("/search/analytics")
@require_permissions(["marketplace.analytics"])
async def get_search_analytics(
    days: int = Query(30, ge=1, le=365, description="Analytics period in days"),
    search_service: SearchService = Depends(get_search_service)
):
    """Get search performance analytics."""
    try:
        analytics = await search_service.analytics.get_search_metrics(days)
        
        return {
            "search_analytics": analytics,
            "period_days": days,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get search analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search analytics failed: {str(e)}"
        )


@router.get("/analytics", response_model=MarketplaceAnalyticsResponse)
@require_permissions(["marketplace.analytics"])
async def get_marketplace_analytics(
    period: str = Query("month", description="Analytics period"),
    include_revenue: bool = Query(True, description="Include revenue data"),
    marketplace_service: EnhancedMarketplaceService = Depends(get_marketplace_service)
):
    """Get comprehensive marketplace analytics."""
    try:
        analytics_data = await marketplace_service.get_comprehensive_analytics(
            period=period,
            include_revenue=include_revenue
        )
        
        return MarketplaceAnalyticsResponse(
            overview=analytics_data.overview,
            trending_items=analytics_data.trending_items,
            category_stats=analytics_data.category_stats,
            user_engagement=analytics_data.user_engagement,
            revenue_metrics=analytics_data.revenue_metrics,
            growth_trends=analytics_data.growth_trends,
            popular_searches=analytics_data.popular_searches,
            conversion_rates=analytics_data.conversion_rates,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics failed: {str(e)}"
        )


@router.get("/stats", response_model=MarketplaceStats)
async def get_marketplace_stats(
    marketplace_service: EnhancedMarketplaceService = Depends(get_marketplace_service)
):
    """Get public marketplace statistics."""
    try:
        stats = await marketplace_service.get_public_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get marketplace stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


# WebSocket endpoints for real-time updates

@router.websocket("/ws/updates")
async def marketplace_updates_websocket(websocket):
    """WebSocket endpoint for real-time marketplace updates."""
    await websocket.accept()
    
    try:
        # Handle real-time marketplace updates
        # Implementation would depend on your WebSocket architecture
        while True:
            data = await websocket.receive_json()
            # Process subscription requests and send updates
            await websocket.send_json({"type": "ping", "timestamp": datetime.utcnow().isoformat()})
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Background task functions

async def execute_plugin_installation(installation_id: UUID, plugin_service: EnhancedPluginService):
    """Execute plugin installation in background."""
    try:
        await plugin_service.execute_installation(installation_id)
        logger.info(f"Plugin installation {installation_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Plugin installation {installation_id} failed: {e}")
        await plugin_service.mark_installation_failed(installation_id, str(e))


# Security and sandbox utilities

async def validate_plugin_security(plugin_path: str) -> Dict[str, Any]:
    """Validate plugin security and generate security report."""
    sandbox = CodeSandbox()
    
    try:
        security_report = await sandbox.scan_plugin(plugin_path)
        return security_report
        
    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        raise SecurityError(f"Security validation failed: {str(e)}")