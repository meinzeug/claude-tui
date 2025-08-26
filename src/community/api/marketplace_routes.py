"""
Marketplace API Routes - REST endpoints for template marketplace operations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.template import TemplateSearchFilters, TemplateResponse
from ..models.marketplace import (
    FeaturedCollectionCreate, FeaturedCollectionResponse, FeaturedCollectionUpdate,
    TemplateTagCreate, TemplateTagResponse, MarketplaceStats, TemplateDownloadCreate
)
from ..services.marketplace_service import MarketplaceService
from ...api.dependencies.database import get_database as get_db
from ...api.dependencies.auth import get_current_user, get_optional_user

marketplace_router = APIRouter(prefix="/marketplace", tags=["marketplace"])


@marketplace_router.get("/search", response_model=Dict[str, Any])
async def search_templates(
    query: Optional[str] = Query(None, description="Search query"),
    template_type: Optional[str] = Query(None, description="Template type filter"),
    complexity_level: Optional[str] = Query(None, description="Complexity level filter"),
    categories: Optional[List[str]] = Query(None, description="Category filters"),
    frameworks: Optional[List[str]] = Query(None, description="Framework filters"),
    languages: Optional[List[str]] = Query(None, description="Language filters"),
    min_rating: Optional[float] = Query(None, ge=1, le=5, description="Minimum rating"),
    max_rating: Optional[float] = Query(None, ge=1, le=5, description="Maximum rating"),
    is_featured: Optional[bool] = Query(None, description="Featured templates only"),
    is_free: Optional[bool] = Query(None, description="Free templates only"),
    author_id: Optional[UUID] = Query(None, description="Author ID filter"),
    created_after: Optional[datetime] = Query(None, description="Created after date"),
    updated_after: Optional[datetime] = Query(None, description="Updated after date"),
    sort_by: str = Query("updated_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    db: AsyncSession = Depends(get_db)
):
    """Search templates with advanced filtering."""
    try:
        # Build search filters
        filters = TemplateSearchFilters(
            query=query,
            template_type=template_type,
            complexity_level=complexity_level,
            categories=categories or [],
            frameworks=frameworks or [],
            languages=languages or [],
            min_rating=min_rating,
            max_rating=max_rating,
            is_featured=is_featured,
            is_free=is_free,
            author_id=author_id,
            created_after=created_after,
            updated_after=updated_after,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        service = MarketplaceService(db)
        templates, total_count, metadata = await service.search_templates(
            filters, page, page_size
        )
        
        return {
            "templates": templates,
            "total_count": total_count,
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@marketplace_router.get("/featured", response_model=List[TemplateResponse])
async def get_featured_templates(
    limit: int = Query(10, ge=1, le=50, description="Number of templates to return"),
    db: AsyncSession = Depends(get_db)
):
    """Get featured templates."""
    try:
        service = MarketplaceService(db)
        templates = await service.get_featured_templates(limit=limit)
        return templates
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get featured templates: {str(e)}"
        )


@marketplace_router.get("/trending", response_model=List[TemplateResponse])
async def get_trending_templates(
    period_days: int = Query(7, ge=1, le=365, description="Trending period in days"),
    limit: int = Query(10, ge=1, le=50, description="Number of templates to return"),
    db: AsyncSession = Depends(get_db)
):
    """Get trending templates."""
    try:
        service = MarketplaceService(db)
        templates = await service.get_trending_templates(
            period_days=period_days, 
            limit=limit
        )
        return templates
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trending templates: {str(e)}"
        )


@marketplace_router.get("/recommendations", response_model=List[TemplateResponse])
async def get_recommendations(
    template_id: Optional[UUID] = Query(None, description="Template ID for similar recommendations"),
    limit: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Get personalized template recommendations."""
    try:
        service = MarketplaceService(db)
        user_id = UUID(current_user["id"]) if current_user else None
        
        recommendations = await service.get_template_recommendations(
            user_id=user_id,
            template_id=template_id,
            limit=limit
        )
        return recommendations
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )


@marketplace_router.get("/suggestions", response_model=List[Dict[str, Any]])
async def get_search_suggestions(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=20, description="Number of suggestions"),
    db: AsyncSession = Depends(get_db)
):
    """Get search suggestions for autocomplete."""
    try:
        service = MarketplaceService(db)
        suggestions = await service.get_template_suggestions(query=query, limit=limit)
        return suggestions
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get suggestions: {str(e)}"
        )


@marketplace_router.get("/stats", response_model=MarketplaceStats)
async def get_marketplace_stats(db: AsyncSession = Depends(get_db)):
    """Get comprehensive marketplace statistics."""
    try:
        service = MarketplaceService(db)
        stats = await service.get_marketplace_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get marketplace stats: {str(e)}"
        )


@marketplace_router.post("/templates/{template_id}/download")
async def record_template_download(
    template_id: UUID,
    download_data: TemplateDownloadCreate,
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Record a template download for analytics."""
    try:
        service = MarketplaceService(db)
        user_id = UUID(current_user["id"]) if current_user else None
        
        await service.record_template_download(
            template_id=template_id,
            user_id=user_id,
            download_type=download_data.download_type,
            project_name=download_data.project_name,
            intended_use=download_data.intended_use
        )
        
        return {"message": "Download recorded successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record download: {str(e)}"
        )


@marketplace_router.post("/templates/{template_id}/view")
async def record_template_view(
    template_id: UUID,
    view_duration: Optional[int] = Query(None, description="View duration in seconds"),
    referrer: Optional[str] = Query(None, description="Referrer URL"),
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Record a template view for analytics."""
    try:
        service = MarketplaceService(db)
        user_id = UUID(current_user["id"]) if current_user else None
        
        await service.record_template_view(
            template_id=template_id,
            user_id=user_id,
            view_duration=view_duration,
            referrer=referrer
        )
        
        return {"message": "View recorded successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record view: {str(e)}"
        )


@marketplace_router.post("/templates/{template_id}/star")
async def star_template(
    template_id: UUID,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Star/favorite a template."""
    try:
        service = MarketplaceService(db)
        user_id = UUID(current_user["id"])
        
        result = await service.star_template(template_id=template_id, user_id=user_id)
        
        if result:
            return {"message": "Template starred successfully"}
        else:
            return {"message": "Template already starred"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to star template: {str(e)}"
        )


@marketplace_router.delete("/templates/{template_id}/star")
async def unstar_template(
    template_id: UUID,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Unstar/unfavorite a template."""
    try:
        service = MarketplaceService(db)
        user_id = UUID(current_user["id"])
        
        result = await service.unstar_template(template_id=template_id, user_id=user_id)
        
        if result:
            return {"message": "Template unstarred successfully"}
        else:
            return {"message": "Template was not starred"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unstar template: {str(e)}"
        )


# Featured Collections endpoints

@marketplace_router.get("/collections", response_model=List[FeaturedCollectionResponse])
async def get_featured_collections(
    collection_type: Optional[str] = Query(None, description="Collection type filter"),
    is_featured: Optional[bool] = Query(None, description="Featured collections only"),
    limit: int = Query(20, ge=1, le=100, description="Number of collections"),
    db: AsyncSession = Depends(get_db)
):
    """Get featured collections."""
    # Implementation would use a collections service
    return []


@marketplace_router.post("/collections", response_model=FeaturedCollectionResponse)
async def create_featured_collection(
    collection_data: FeaturedCollectionCreate,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new featured collection (admin/curator only)."""
    # Implementation would check permissions and create collection
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Collection creation not yet implemented"
    )


@marketplace_router.get("/collections/{collection_id}", response_model=FeaturedCollectionResponse)
async def get_collection(
    collection_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific featured collection."""
    # Implementation would retrieve collection by ID
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Collection not found"
    )


@marketplace_router.put("/collections/{collection_id}", response_model=FeaturedCollectionResponse)
async def update_collection(
    collection_id: UUID,
    collection_data: FeaturedCollectionUpdate,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a featured collection (admin/curator only)."""
    # Implementation would check permissions and update collection
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Collection updates not yet implemented"
    )


# Template Tags endpoints

@marketplace_router.get("/tags", response_model=List[TemplateTagResponse])
async def get_template_tags(
    category: Optional[str] = Query(None, description="Tag category filter"),
    is_featured: Optional[bool] = Query(None, description="Featured tags only"),
    limit: int = Query(50, ge=1, le=200, description="Number of tags"),
    db: AsyncSession = Depends(get_db)
):
    """Get template tags."""
    # Implementation would retrieve tags with filtering
    return []


@marketplace_router.post("/tags", response_model=TemplateTagResponse)
async def create_template_tag(
    tag_data: TemplateTagCreate,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new template tag (admin only)."""
    # Implementation would check admin permissions and create tag
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Tag creation not yet implemented"
    )