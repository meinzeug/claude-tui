"""
Template API Routes - REST endpoints for template management operations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.template import (
    TemplateCreate, TemplateUpdate, TemplateResponse,
    TemplateInheritanceConfig, TemplateBuildResult
)
from ..services.template_service import TemplateService
from ...api.dependencies.database import get_db
from ...api.dependencies.auth import get_current_user, get_optional_user
from ...core.exceptions import NotFoundError, ValidationError, PermissionError

router = APIRouter(prefix="/templates", tags=["templates"])


@router.post("/", response_model=TemplateResponse)
async def create_template(
    template_data: TemplateCreate,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new template."""
    try:
        service = TemplateService(db)
        user_id = UUID(current_user["id"])
        
        template = await service.create_template(template_data, user_id)
        return TemplateResponse.from_orm(template)
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {str(e)}"
        )


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: UUID,
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific template."""
    try:
        service = TemplateService(db)
        template = await service.repository.get_template_by_id(template_id)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Check access permissions
        if not template.is_public:
            if not current_user or UUID(current_user["id"]) != template.author_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to private template"
                )
        
        return TemplateResponse.from_orm(template)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template: {str(e)}"
        )


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: UUID,
    template_data: TemplateUpdate,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update an existing template."""
    try:
        service = TemplateService(db)
        user_id = UUID(current_user["id"])
        
        template = await service.update_template(template_id, template_data, user_id)
        return TemplateResponse.from_orm(template)
        
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
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update template: {str(e)}"
        )


@router.delete("/{template_id}")
async def delete_template(
    template_id: UUID,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a template."""
    try:
        service = TemplateService(db)
        user_id = UUID(current_user["id"])
        
        # Get template to check ownership
        template = await service.repository.get_template_by_id(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Check permissions
        if template.author_id != user_id:
            # Add admin check here if needed
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this template"
            )
        
        # Delete template
        success = await service.repository.delete(template_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete template"
            )
        
        return {"message": "Template deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete template: {str(e)}"
        )


@router.post("/inherit", response_model=TemplateResponse)
async def create_inherited_template(
    inheritance_config: TemplateInheritanceConfig,
    template_name: str,
    description: str = "",
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a template by inheriting from another template."""
    try:
        service = TemplateService(db)
        user_id = UUID(current_user["id"])
        
        template = await service.create_template_from_inheritance(
            inheritance_config, user_id, template_name, description
        )
        
        return TemplateResponse.from_orm(template)
        
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
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create inherited template: {str(e)}"
        )


@router.post("/{template_id}/build", response_model=TemplateBuildResult)
async def build_template(
    template_id: UUID,
    build_config: Dict[str, Any],
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Build/render a template with configuration."""
    try:
        service = TemplateService(db)
        
        # Check if template exists and is accessible
        template = await service.repository.get_template_by_id(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        if not template.is_public:
            if not current_user or UUID(current_user["id"]) != template.author_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to private template"
                )
        
        # Validate build configuration
        is_valid, errors, warnings = await service.validate_template_build(
            template_id, build_config
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Invalid build configuration",
                    "errors": errors,
                    "warnings": warnings
                }
            )
        
        # Build template
        build_result = await service.build_template(template_id, build_config)
        return build_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build template: {str(e)}"
        )


@router.get("/{template_id}/variables")
async def get_template_variables(
    template_id: UUID,
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Get template variables for UI generation."""
    try:
        service = TemplateService(db)
        
        # Check access
        template = await service.repository.get_template_by_id(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        if not template.is_public:
            if not current_user or UUID(current_user["id"]) != template.author_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to private template"
                )
        
        variables = await service.get_template_variables(template_id)
        return {"variables": variables}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template variables: {str(e)}"
        )


@router.post("/{template_id}/validate-build")
async def validate_template_build(
    template_id: UUID,
    build_config: Dict[str, Any],
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Validate template build configuration."""
    try:
        service = TemplateService(db)
        
        # Check access
        template = await service.repository.get_template_by_id(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        if not template.is_public:
            if not current_user or UUID(current_user["id"]) != template.author_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to private template"
                )
        
        is_valid, errors, warnings = await service.validate_template_build(
            template_id, build_config
        )
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate build configuration: {str(e)}"
        )


@router.get("/{template_id}/stats")
async def get_template_stats(
    template_id: UUID,
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Get template statistics."""
    try:
        service = TemplateService(db)
        
        # Check access
        template = await service.repository.get_template_by_id(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        if not template.is_public:
            if not current_user or UUID(current_user["id"]) != template.author_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to private template"
                )
        
        stats = await service.repository.get_template_stats(template_id)
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template stats: {str(e)}"
        )


@router.get("/{template_id}/versions")
async def get_template_versions(
    template_id: UUID,
    limit: Optional[int] = 20,
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Get template version history."""
    try:
        service = TemplateService(db)
        
        # Check access
        template = await service.repository.get_template_by_id(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        if not template.is_public:
            if not current_user or UUID(current_user["id"]) != template.author_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to private template"
                )
        
        from ..repositories.template_repository import TemplateVersionRepository
        version_repo = TemplateVersionRepository(db)
        versions = await version_repo.get_versions_by_template(template_id, limit)
        
        return {
            "template_id": str(template_id),
            "versions": [
                {
                    "id": str(v.id),
                    "version": v.version,
                    "changelog": v.changelog,
                    "breaking_changes": v.breaking_changes,
                    "migration_notes": v.migration_notes,
                    "is_stable": v.is_stable,
                    "is_deprecated": v.is_deprecated,
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                }
                for v in versions
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template versions: {str(e)}"
        )


@router.get("/slug/{slug}", response_model=TemplateResponse)
async def get_template_by_slug(
    slug: str,
    current_user: Optional[Dict] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """Get template by slug."""
    try:
        service = TemplateService(db)
        template = await service.repository.get_template_by_slug(slug)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Check access permissions
        if not template.is_public:
            if not current_user or UUID(current_user["id"]) != template.author_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to private template"
                )
        
        return TemplateResponse.from_orm(template)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template by slug: {str(e)}"
        )