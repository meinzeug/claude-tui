"""
Project Management REST API Endpoints.

Provides comprehensive project management operations:
- Project creation and initialization
- Project listing and retrieval
- Project configuration management
- Project validation and health checks
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ..dependencies.auth import get_current_user
from ..dependencies.database import get_database
from ..middleware.rate_limiting import rate_limit
from ...services.project_service import ProjectService
from ...core.exceptions import (
    ProjectDirectoryError, ValidationError, ConfigurationError
)

# Initialize router
router = APIRouter()

# Pydantic Models
class ProjectCreateRequest(BaseModel):
    """Request model for project creation."""
    name: str = Field(..., min_length=1, max_length=100, description="Project name")
    path: str = Field(..., description="Project directory path")
    project_type: str = Field(default="python", description="Project type")
    template: Optional[str] = Field(None, description="Project template")
    description: Optional[str] = Field(None, max_length=500, description="Project description")
    initialize_git: bool = Field(default=True, description="Initialize git repository")
    create_venv: bool = Field(default=True, description="Create virtual environment (Python)")
    config: Optional[Dict] = Field(default_factory=dict, description="Additional configuration")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Project name cannot be empty')
        return v.strip()
    
    @validator('project_type')
    def validate_project_type(cls, v):
        allowed_types = ['python', 'nodejs', 'react', 'vue', 'angular', 'flask', 'django', 'fastapi']
        if v not in allowed_types:
            raise ValueError(f'Project type must be one of: {", ".join(allowed_types)}')
        return v

class ProjectResponse(BaseModel):
    """Response model for project operations."""
    project_id: str
    name: str
    path: str
    project_type: str = Field(alias="type")
    description: Optional[str]
    created_at: str
    status: str
    git_initialized: bool
    venv_created: bool
    config: Dict
    
    class Config:
        allow_population_by_field_name = True

class ProjectListResponse(BaseModel):
    """Response model for project listing."""
    projects: List[ProjectResponse]
    total: int
    page: int
    page_size: int

class ProjectValidationResponse(BaseModel):
    """Response model for project validation."""
    project_id: str
    is_valid: bool
    score: float
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict
    validated_at: str

class ProjectConfigUpdateRequest(BaseModel):
    """Request model for project configuration updates."""
    config: Dict = Field(..., description="Configuration updates")

# Dependency injection
async def get_project_service() -> ProjectService:
    """Get project service dependency."""
    service = ProjectService()
    await service.initialize()
    return service

# Routes
@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
@rate_limit(requests=5, window=60)  # 5 requests per minute
async def create_project(
    project_data: ProjectCreateRequest,
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Create a new project with AI-powered initialization.
    
    Creates a new project with specified configuration, including:
    - Directory structure setup
    - Template application (if specified)
    - Git repository initialization
    - Virtual environment creation (Python projects)
    - Configuration file generation
    """
    try:
        result = await project_service.create_project(
            name=project_data.name,
            path=project_data.path,
            project_type=project_data.project_type,
            template=project_data.template,
            config=project_data.config,
            initialize_git=project_data.initialize_git,
            create_venv=project_data.create_venv
        )
        
        return ProjectResponse(
            project_id=result['project_id'],
            name=result['name'],
            path=result['path'],
            type=result['type'],
            description=project_data.description,
            created_at=result['created_at'],
            status='created',
            git_initialized=result['config'].get('git_initialized', False),
            venv_created=result['config'].get('venv_created', False),
            config=result['config']
        )
        
    except ProjectDirectoryError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Project directory error: {str(e)}"
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create project: {str(e)}"
        )

@router.get("/", response_model=ProjectListResponse)
@rate_limit(requests=10, window=60)
async def list_projects(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    List all active projects with pagination.
    
    Returns a paginated list of all projects accessible to the current user.
    """
    try:
        projects = await project_service.list_active_projects()
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_projects = projects[start_idx:end_idx]
        
        project_responses = [
            ProjectResponse(
                project_id=project['project_id'],
                name=project['name'],
                path=project['path'],
                type=project['type'],
                description=project.get('config', {}).get('description'),
                created_at=project.get('created_at', ''),
                status=project.get('entry_status', 'active'),
                git_initialized=project.get('config', {}).get('git_initialized', False),
                venv_created=project.get('config', {}).get('venv_created', False),
                config=project.get('config', {})
            )
            for project in paginated_projects
        ]
        
        return ProjectListResponse(
            projects=project_responses,
            total=len(projects),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list projects: {str(e)}"
        )

@router.get("/{project_id}", response_model=ProjectResponse)
@rate_limit(requests=20, window=60)
async def get_project(
    project_id: str,
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Get detailed information about a specific project.
    
    Returns comprehensive project information including configuration,
    runtime status, and metadata.
    """
    try:
        project_info = await project_service.get_project_info(project_id)
        
        return ProjectResponse(
            project_id=project_info['project_id'],
            name=project_info['name'],
            path=project_info['path'],
            type=project_info['type'],
            description=project_info.get('config', {}).get('description'),
            created_at=project_info.get('created_at', ''),
            status=project_info.get('entry_status', 'active'),
            git_initialized=project_info.get('config', {}).get('git_initialized', False),
            venv_created=project_info.get('config', {}).get('venv_created', False),
            config=project_info.get('config', {})
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get project: {str(e)}"
        )

@router.post("/{project_id}/validate", response_model=ProjectValidationResponse)
@rate_limit(requests=5, window=60)
async def validate_project(
    project_id: str,
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Validate project structure and configuration.
    
    Performs comprehensive validation of project structure,
    configuration validity, and best practices compliance.
    """
    try:
        validation_result = await project_service.validate_project(project_id)
        
        return ProjectValidationResponse(
            project_id=project_id,
            is_valid=validation_result['is_valid'],
            score=validation_result['score'],
            issues=validation_result.get('issues', []),
            warnings=validation_result.get('warnings', []),
            recommendations=validation_result.get('recommendations', []),
            metadata=validation_result.get('metadata', {}),
            validated_at=datetime.utcnow().isoformat()
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate project: {str(e)}"
        )

@router.put("/{project_id}/config", response_model=ProjectResponse)
@rate_limit(requests=10, window=60)
async def update_project_config(
    project_id: str,
    config_update: ProjectConfigUpdateRequest,
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Update project configuration.
    
    Updates project configuration with new settings and
    saves changes to the project configuration file.
    """
    try:
        updated_project = await project_service.update_project_config(
            project_id=project_id,
            config_updates=config_update.config
        )
        
        return ProjectResponse(
            project_id=updated_project['project_id'],
            name=updated_project['name'],
            path=updated_project['path'],
            type=updated_project['type'],
            description=updated_project.get('config', {}).get('description'),
            created_at=updated_project.get('created_at', ''),
            status=updated_project.get('entry_status', 'active'),
            git_initialized=updated_project.get('config', {}).get('git_initialized', False),
            venv_created=updated_project.get('config', {}).get('venv_created', False),
            config=updated_project.get('config', {})
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update project config: {str(e)}"
        )

@router.post("/{project_id}/load")
@rate_limit(requests=5, window=60)
async def load_existing_project(
    project_id: str,
    path: str = Body(..., embed=True, description="Project directory path"),
    validate_structure: bool = Body(True, embed=True, description="Validate project structure"),
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Load an existing project from filesystem path.
    
    Loads an existing project directory and registers it
    in the active projects system.
    """
    try:
        result = await project_service.load_project(
            path=path,
            validate_structure=validate_structure
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Project loaded successfully",
                "project_id": result['project_id'],
                "name": result['name'],
                "path": result['path'],
                "validation": result.get('validation', {})
            }
        )
        
    except ProjectDirectoryError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Project directory error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load project: {str(e)}"
        )

@router.delete("/{project_id}")
@rate_limit(requests=5, window=60)
async def remove_project(
    project_id: str,
    delete_files: bool = Query(False, description="Delete project files from filesystem"),
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Remove project from active projects.
    
    Removes project from the active projects list and optionally
    deletes project files from the filesystem.
    """
    try:
        result = await project_service.remove_project(
            project_id=project_id,
            delete_files=delete_files
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Project removed successfully",
                "project_id": result['project_id'],
                "removed_at": result['removed_at'],
                "files_deleted": result['files_deleted']
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove project: {str(e)}"
        )

@router.get("/{project_id}/health")
@rate_limit(requests=20, window=60)
async def project_health_check(
    project_id: str,
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Get project health status and metrics.
    
    Returns project health information including file system status,
    git repository status, and basic metrics.
    """
    try:
        project_info = await project_service.get_project_info(project_id)
        runtime_info = project_info.get('runtime_info', {})
        
        health_status = {
            "project_id": project_id,
            "exists": runtime_info.get('exists', False),
            "is_git_repo": runtime_info.get('is_git_repo', False),
            "has_venv": runtime_info.get('has_venv', False),
            "file_count": runtime_info.get('file_count', 0),
            "size_mb": round(runtime_info.get('size_mb', 0), 2),
            "last_accessed": project_info.get('last_accessed'),
            "status": "healthy" if runtime_info.get('exists') else "missing",
            "checked_at": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=health_status
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check project health: {str(e)}"
        )

@router.post("/{project_id}/duplicate")
@rate_limit(requests=5, window=60)
async def duplicate_project(
    project_id: str,
    new_name: str = Body(..., embed=True, description="New project name"),
    new_path: str = Body(..., embed=True, description="New project path"),
    include_git_history: bool = Body(False, embed=True, description="Include git history"),
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Duplicate an existing project.
    
    Creates a copy of an existing project with a new name and path.
    Optionally includes git history in the duplication.
    """
    try:
        result = await project_service.duplicate_project(
            project_id=project_id,
            new_name=new_name,
            new_path=new_path,
            include_git_history=include_git_history
        )
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Project duplicated successfully",
                "original_project_id": project_id,
                "new_project_id": result['project_id'],
                "new_path": result['path'],
                "duplicated_at": result['created_at']
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to duplicate project: {str(e)}"
        )

@router.get("/{project_id}/dependencies")
@rate_limit(requests=10, window=60)
async def get_project_dependencies(
    project_id: str,
    dependency_type: Optional[str] = Query(None, description="Filter by dependency type"),
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Get project dependencies and their status.
    
    Returns information about project dependencies including
    installed packages, versions, and security status.
    """
    try:
        dependencies = await project_service.analyze_dependencies(
            project_id=project_id,
            dependency_type=dependency_type
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "project_id": project_id,
                "dependencies": dependencies,
                "analyzed_at": datetime.utcnow().isoformat()
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze dependencies: {str(e)}"
        )

@router.post("/{project_id}/backup")
@rate_limit(requests=3, window=300)  # 3 requests per 5 minutes
async def backup_project(
    project_id: str,
    backup_name: Optional[str] = Body(None, embed=True, description="Custom backup name"),
    include_git: bool = Body(True, embed=True, description="Include git repository"),
    include_dependencies: bool = Body(False, embed=True, description="Include node_modules/venv"),
    compression_level: int = Body(6, embed=True, ge=1, le=9, description="Compression level (1-9)"),
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Create a backup of the project.
    
    Creates a compressed backup of the project directory
    with configurable options for what to include.
    """
    try:
        result = await project_service.create_backup(
            project_id=project_id,
            backup_name=backup_name,
            include_git=include_git,
            include_dependencies=include_dependencies,
            compression_level=compression_level
        )
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Project backup created successfully",
                "backup_id": result['backup_id'],
                "backup_path": result['backup_path'],
                "size_mb": result['size_mb'],
                "created_at": result['created_at']
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create backup: {str(e)}"
        )

@router.get("/{project_id}/backups")
@rate_limit(requests=10, window=60)
async def list_project_backups(
    project_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Page size"),
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    List available backups for a project.
    
    Returns a paginated list of available backups with metadata.
    """
    try:
        backups = await project_service.list_backups(
            project_id=project_id,
            page=page,
            page_size=page_size
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "project_id": project_id,
                "backups": backups['items'],
                "total": backups['total'],
                "page": page,
                "page_size": page_size
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list backups: {str(e)}"
        )

@router.post("/{project_id}/restore/{backup_id}")
@rate_limit(requests=3, window=300)  # 3 requests per 5 minutes
async def restore_project_backup(
    project_id: str,
    backup_id: str,
    restore_path: Optional[str] = Body(None, embed=True, description="Custom restore path"),
    overwrite_existing: bool = Body(False, embed=True, description="Overwrite existing files"),
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Restore project from backup.
    
    Restores a project from a previously created backup.
    Can restore to original location or custom path.
    """
    try:
        result = await project_service.restore_from_backup(
            project_id=project_id,
            backup_id=backup_id,
            restore_path=restore_path,
            overwrite_existing=overwrite_existing
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Project restored successfully",
                "restored_project_id": result['project_id'],
                "restored_path": result['path'],
                "restored_at": result['restored_at']
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project or backup not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore backup: {str(e)}"
        )

@router.get("/{project_id}/analytics")
@rate_limit(requests=10, window=60)
async def get_project_analytics(
    project_id: str,
    time_range: str = Query("30d", description="Time range (e.g., 7d, 30d, 90d)"),
    include_code_metrics: bool = Query(True, description="Include code quality metrics"),
    current_user: Dict = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Get project analytics and metrics.
    
    Returns comprehensive analytics including activity patterns,
    code quality trends, and performance metrics.
    """
    try:
        analytics = await project_service.get_analytics(
            project_id=project_id,
            time_range=time_range,
            include_code_metrics=include_code_metrics
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "project_id": project_id,
                "time_range": time_range,
                "analytics": analytics,
                "generated_at": datetime.utcnow().isoformat()
            }
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )