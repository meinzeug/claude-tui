"""
Validation and Anti-Hallucination REST API Endpoints.

Provides comprehensive validation services:
- Code validation and quality checking
- AI response validation
- Placeholder detection
- Progress authenticity verification
- Security and semantic analysis
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ..dependencies.auth import get_current_user
from ..middleware.rate_limiting import rate_limit
from ...services.validation_service import ValidationService, ValidationLevel, ValidationCategory
from ...core.exceptions import (
    ValidationError, PlaceholderDetectionError, SemanticValidationError
)

# Initialize router
router = APIRouter()

# Enums for API
class ValidationLevelEnum(str, Enum):
    """Validation level enumeration for API."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"

class ValidationCategoryEnum(str, Enum):
    """Validation category enumeration for API."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    PLACEHOLDER = "placeholder"
    QUALITY = "quality"
    SECURITY = "security"
    PERFORMANCE = "performance"

class ResponseTypeEnum(str, Enum):
    """Response type enumeration for validation."""
    TEXT = "text"
    CODE = "code"
    JSON = "json"
    GENERAL = "general"

# Pydantic Models
class CodeValidationRequest(BaseModel):
    """Request model for code validation."""
    code: str = Field(..., min_length=1, description="Code content to validate")
    language: str = Field(default="python", description="Programming language")
    file_path: Optional[str] = Field(None, description="Optional file path for context")
    validation_level: ValidationLevelEnum = Field(default=ValidationLevelEnum.STANDARD, description="Validation level")
    check_placeholders: bool = Field(default=True, description="Check for placeholders")
    check_syntax: bool = Field(default=True, description="Check syntax")
    check_quality: bool = Field(default=True, description="Check code quality")
    
    @validator('language')
    def validate_language(cls, v):
        allowed_languages = ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go', 'rust', 'php']
        if v.lower() not in allowed_languages:
            raise ValueError(f'Language must be one of: {", ".join(allowed_languages)}')
        return v.lower()

class ResponseValidationRequest(BaseModel):
    """Request model for AI response validation."""
    response: Union[str, Dict[str, Any]] = Field(..., description="Response content to validate")
    response_type: ResponseTypeEnum = Field(default=ResponseTypeEnum.TEXT, description="Response type")
    validation_criteria: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validation criteria")
    
class ProgressValidationRequest(BaseModel):
    """Request model for progress authenticity validation."""
    file_path: str = Field(..., description="File path to validate")
    project_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Project context")

class ValidationResponse(BaseModel):
    """Response model for validation operations."""
    is_valid: bool
    score: float = Field(ge=0.0, le=1.0, description="Validation score (0-1)")
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]
    categories: Dict[str, Any]
    metadata: Dict[str, Any]
    
class ProgressValidationResponse(BaseModel):
    """Response model for progress validation."""
    is_authentic: bool
    authenticity_score: float = Field(ge=0.0, le=1.0, description="Authenticity score (0-1)")
    issues: List[str]
    suggestions: List[str]
    placeholder_count: int
    metadata: Dict[str, Any]

class ValidationReportResponse(BaseModel):
    """Response model for validation reports."""
    total_validations: int
    successful_validations: int
    success_rate: float = Field(ge=0.0, le=1.0, description="Success rate (0-1)")
    average_score: float = Field(ge=0.0, le=1.0, description="Average validation score (0-1)")
    type_breakdown: Dict[str, Any]
    recent_history: List[Dict[str, Any]]
    report_generated_at: str

class ValidationHistoryResponse(BaseModel):
    """Response model for validation history."""
    history: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int

# Dependency injection
async def get_validation_service() -> ValidationService:
    """Get validation service dependency."""
    service = ValidationService()
    await service.initialize()
    return service

# Routes
@router.post("/code", response_model=ValidationResponse, status_code=status.HTTP_200_OK)
@rate_limit(requests=20, window=60)  # 20 requests per minute
async def validate_code(
    validation_request: CodeValidationRequest,
    current_user: Dict = Depends(get_current_user),
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Comprehensive code validation with anti-hallucination checks.
    
    Validates code for:
    - Syntax correctness
    - Placeholder detection
    - Code quality metrics
    - Security issues (strict/comprehensive levels)
    - Semantic analysis (comprehensive level)
    """
    try:
        result = await validation_service.validate_code(
            code=validation_request.code,
            language=validation_request.language,
            file_path=validation_request.file_path,
            validation_level=ValidationLevel(validation_request.validation_level.value),
            check_placeholders=validation_request.check_placeholders,
            check_syntax=validation_request.check_syntax,
            check_quality=validation_request.check_quality
        )
        
        return ValidationResponse(
            is_valid=result['is_valid'],
            score=result['score'],
            issues=result.get('issues', []),
            warnings=result.get('warnings', []),
            suggestions=result.get('suggestions', []),
            categories=result.get('categories', {}),
            metadata=result.get('metadata', {})
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Code validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate code: {str(e)}"
        )

@router.post("/response", response_model=ValidationResponse)
@rate_limit(requests=30, window=60)
async def validate_response(
    validation_request: ResponseValidationRequest,
    current_user: Dict = Depends(get_current_user),
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Validate AI response for completeness and authenticity.
    
    Validates AI-generated responses for:
    - Content completeness
    - Format correctness
    - Authenticity indicators
    - Custom validation criteria
    """
    try:
        result = await validation_service.validate_response(
            response=validation_request.response,
            response_type=validation_request.response_type.value,
            validation_criteria=validation_request.validation_criteria
        )
        
        return ValidationResponse(
            is_valid=result['is_valid'],
            score=result['score'],
            issues=result.get('issues', []),
            warnings=result.get('warnings', []),
            suggestions=result.get('suggestions', []),
            categories=result.get('categories', {}),
            metadata=result.get('metadata', {})
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Response validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate response: {str(e)}"
        )

@router.post("/progress", response_model=ProgressValidationResponse)
@rate_limit(requests=10, window=60)
async def validate_progress(
    validation_request: ProgressValidationRequest,
    current_user: Dict = Depends(get_current_user),
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Validate file progress for authenticity and completeness.
    
    Uses advanced progress validator to check for:
    - Implementation completeness
    - Placeholder detection
    - Code authenticity
    - Progress consistency
    """
    try:
        result = await validation_service.check_progress_authenticity(
            file_path=validation_request.file_path,
            project_context=validation_request.project_context
        )
        
        return ProgressValidationResponse(
            is_authentic=result['is_authentic'],
            authenticity_score=result['authenticity_score'],
            issues=result.get('issues', []),
            suggestions=result.get('suggestions', []),
            placeholder_count=result.get('placeholder_count', 0),
            metadata=result.get('metadata', {})
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Progress validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate progress: {str(e)}"
        )

@router.post("/file", response_model=ValidationResponse)
@rate_limit(requests=10, window=60)
async def validate_file(
    file: UploadFile = File(..., description="File to validate"),
    language: Optional[str] = Query(None, description="Programming language (auto-detected if not provided)"),
    validation_level: ValidationLevelEnum = Query(ValidationLevelEnum.STANDARD, description="Validation level"),
    current_user: Dict = Depends(get_current_user),
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Validate uploaded file content for code quality and authenticity.
    
    Accepts file uploads and performs comprehensive validation
    including auto-detection of programming language.
    """
    try:
        # Read file content
        content = await file.read()
        code = content.decode('utf-8')
        
        # Auto-detect language if not provided
        if not language:
            file_extension = Path(file.filename).suffix.lower()
            language_mapping = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                '.go': 'go',
                '.rs': 'rust',
                '.php': 'php'
            }
            language = language_mapping.get(file_extension, 'python')
        
        result = await validation_service.validate_code(
            code=code,
            language=language,
            file_path=file.filename,
            validation_level=ValidationLevel(validation_level.value),
            check_placeholders=True,
            check_syntax=True,
            check_quality=True
        )
        
        return ValidationResponse(
            is_valid=result['is_valid'],
            score=result['score'],
            issues=result.get('issues', []),
            warnings=result.get('warnings', []),
            suggestions=result.get('suggestions', []),
            categories=result.get('categories', {}),
            metadata=result.get('metadata', {})
        )
        
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File contains invalid UTF-8 encoding"
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate file: {str(e)}"
        )

@router.get("/report", response_model=ValidationReportResponse)
@rate_limit(requests=10, window=60)
async def get_validation_report(
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Limit number of results"),
    validation_type_filter: Optional[str] = Query(None, description="Filter by validation type"),
    current_user: Dict = Depends(get_current_user),
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Get comprehensive validation report with metrics and analytics.
    
    Returns detailed validation statistics including:
    - Success rates and performance metrics
    - Validation type breakdowns
    - Recent validation history
    - Trend analysis
    """
    try:
        report = await validation_service.get_validation_report(
            limit=limit,
            validation_type_filter=validation_type_filter
        )
        
        return ValidationReportResponse(
            total_validations=report['total_validations'],
            successful_validations=report['successful_validations'],
            success_rate=report['success_rate'],
            average_score=report['average_score'],
            type_breakdown=report['type_breakdown'],
            recent_history=report['recent_history'],
            report_generated_at=report['report_generated_at']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get validation report: {str(e)}"
        )

@router.post("/batch/code", response_model=List[ValidationResponse])
@rate_limit(requests=5, window=60)
async def validate_code_batch(
    validation_requests: List[CodeValidationRequest] = Body(..., max_items=10, description="List of code validations"),
    current_user: Dict = Depends(get_current_user),
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Validate multiple code snippets in a single batch operation.
    
    Performs concurrent validation of up to 10 code snippets
    with individual error handling for each validation.
    """
    try:
        results = []
        
        for validation_request in validation_requests:
            try:
                result = await validation_service.validate_code(
                    code=validation_request.code,
                    language=validation_request.language,
                    file_path=validation_request.file_path,
                    validation_level=ValidationLevel(validation_request.validation_level.value),
                    check_placeholders=validation_request.check_placeholders,
                    check_syntax=validation_request.check_syntax,
                    check_quality=validation_request.check_quality
                )
                
                results.append(ValidationResponse(
                    is_valid=result['is_valid'],
                    score=result['score'],
                    issues=result.get('issues', []),
                    warnings=result.get('warnings', []),
                    suggestions=result.get('suggestions', []),
                    categories=result.get('categories', {}),
                    metadata=result.get('metadata', {})
                ))
                
            except Exception as e:
                # Add error result but continue with other validations
                results.append(ValidationResponse(
                    is_valid=False,
                    score=0.0,
                    issues=[f"Validation failed: {str(e)}"],
                    warnings=[],
                    suggestions=[],
                    categories={},
                    metadata={"error": str(e)}
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform batch validation: {str(e)}"
        )

@router.get("/patterns/placeholders")
@rate_limit(requests=20, window=60)
async def get_placeholder_patterns(
    language: Optional[str] = Query(None, description="Programming language filter"),
    current_user: Dict = Depends(get_current_user),
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Get placeholder detection patterns for different languages.
    
    Returns the patterns used for detecting placeholders
    and incomplete implementations in code.
    """
    try:
        # Access private attribute for patterns (in real implementation, make this a public method)
        patterns = validation_service._placeholder_patterns
        
        if language:
            language_patterns = patterns.get(language, [])
            general_patterns = patterns.get('general', [])
            filtered_patterns = {
                language: language_patterns,
                'general': general_patterns
            }
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "patterns": filtered_patterns,
                    "language_filter": language,
                    "total_patterns": len(language_patterns) + len(general_patterns)
                }
            )
        
        total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "patterns": patterns,
                "supported_languages": list(patterns.keys()),
                "total_patterns": total_patterns
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get placeholder patterns: {str(e)}"
        )

@router.get("/health/service")
@rate_limit(requests=20, window=60)
async def validation_service_health(
    current_user: Dict = Depends(get_current_user),
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Get validation service health status and diagnostics.
    
    Returns service health information including component
    availability and validation statistics.
    """
    try:
        health = await validation_service.health_check()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy" if health['status'] == 'healthy' else "degraded",
                "service": "validation_service",
                "components": {
                    "progress_validator": health.get('progress_validator_available', False),
                    "validation_rules": health.get('validation_rules_loaded', 0),
                    "placeholder_patterns": health.get('placeholder_patterns_loaded', 0)
                },
                "metrics": {
                    "cache_size": health.get('cache_size', 0),
                    "history_size": health.get('validation_history_size', 0)
                },
                "checked_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "validation_service",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
        )

@router.post("/rules/custom", status_code=status.HTTP_201_CREATED)
@rate_limit(requests=5, window=60)
async def add_custom_validation_rules(
    language: str = Body(..., description="Programming language"),
    rules: Dict[str, Any] = Body(..., description="Custom validation rules"),
    current_user: Dict = Depends(get_current_user),
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Add custom validation rules for specific languages.
    
    Allows users to define custom validation rules for
    specific programming languages or validation scenarios.
    """
    try:
        # In a real implementation, this would save rules to database
        # For now, we update the service rules temporarily
        validation_service._validation_rules[language] = rules
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Custom validation rules added successfully",
                "language": language,
                "rules_count": len(rules),
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add custom validation rules: {str(e)}"
        )