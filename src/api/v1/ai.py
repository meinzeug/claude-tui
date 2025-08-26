"""
AI Integration REST API Endpoints.

Provides comprehensive AI service integration:
- Claude Code integration for code generation
- Claude Flow orchestration for complex tasks
- AI response validation and quality assurance
- Performance monitoring and analytics
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ..dependencies.auth import get_current_user
from ..middleware.rate_limiting import rate_limit
from ...services.ai_service import AIService
from ...core.exceptions import (
    AIServiceError, ClaudeCodeError, ClaudeFlowError, ValidationError
)

# Initialize router
router = APIRouter()

# Enums for API
class OrchestrationStrategyEnum(str, Enum):
    """Orchestration strategy enumeration for API."""
    ADAPTIVE = "adaptive"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    BALANCED = "balanced"

class ValidationTypeEnum(str, Enum):
    """AI validation type enumeration for API."""
    CODE = "code"
    TEXT = "text"
    GENERAL = "general"
    JSON = "json"

# Pydantic Models
class CodeGenerationRequest(BaseModel):
    """Request model for AI code generation."""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Code generation prompt")
    language: str = Field(default="python", description="Target programming language")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    validate_response: bool = Field(default=True, description="Validate generated code")
    use_cache: bool = Field(default=True, description="Use response caching")
    
    @validator('language')
    def validate_language(cls, v):
        allowed_languages = ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go', 'rust', 'php', 'html', 'css']
        if v.lower() not in allowed_languages:
            raise ValueError(f'Language must be one of: {", ".join(allowed_languages)}')
        return v.lower()

class TaskOrchestrationRequest(BaseModel):
    """Request model for AI task orchestration."""
    task_description: str = Field(..., min_length=1, max_length=1000, description="Task description")
    requirements: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task requirements")
    agents: Optional[List[str]] = Field(default_factory=list, description="Specific agents to use")
    strategy: OrchestrationStrategyEnum = Field(default=OrchestrationStrategyEnum.ADAPTIVE, description="Orchestration strategy")

class AIValidationRequest(BaseModel):
    """Request model for AI response validation."""
    response: Dict[str, Any] = Field(..., description="AI response to validate")
    validation_type: ValidationTypeEnum = Field(default=ValidationTypeEnum.GENERAL, description="Validation type")
    criteria: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validation criteria")

class CodeGenerationResponse(BaseModel):
    """Response model for code generation."""
    code: str = Field(..., description="Generated code")
    language: str = Field(..., description="Programming language")
    metadata: Dict[str, Any] = Field(..., description="Generation metadata")
    validation: Optional[Dict[str, Any]] = Field(None, description="Validation results")
    cached: bool = Field(default=False, description="Whether result was cached")
    generated_at: str = Field(..., description="Generation timestamp")

class TaskOrchestrationResponse(BaseModel):
    """Response model for task orchestration."""
    task_id: str = Field(..., description="Orchestration task ID")
    status: str = Field(..., description="Orchestration status")
    agents_assigned: List[str] = Field(..., description="Assigned agents")
    strategy_used: str = Field(..., description="Orchestration strategy used")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    metadata: Dict[str, Any] = Field(..., description="Orchestration metadata")

class AIValidationResponse(BaseModel):
    """Response model for AI validation."""
    is_valid: bool = Field(..., description="Whether response is valid")
    score: float = Field(..., ge=0.0, le=1.0, description="Validation score")
    errors: List[str] = Field(..., description="Validation errors")
    warnings: List[str] = Field(..., description="Validation warnings")
    metadata: Dict[str, Any] = Field(..., description="Validation metadata")

class AIPerformanceResponse(BaseModel):
    """Response model for AI performance metrics."""
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    cache_size: int = Field(..., description="Current cache size")
    claude_code_available: bool = Field(..., description="Claude Code availability")
    claude_flow_available: bool = Field(..., description="Claude Flow availability")
    providers_count: int = Field(..., description="Number of available providers")

class AIHistoryResponse(BaseModel):
    """Response model for AI request history."""
    history: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int

# Dependency injection
async def get_ai_service() -> AIService:
    """Get AI service dependency."""
    service = AIService()
    await service.initialize()
    return service

# Routes
@router.post("/code/generate", response_model=CodeGenerationResponse, status_code=status.HTTP_200_OK)
@rate_limit(requests=10, window=60)  # 10 requests per minute
async def generate_code(
    generation_request: CodeGenerationRequest,
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Generate code using AI with validation and quality assurance.
    
    Generates code using Claude Code integration with:
    - Intelligent prompt processing
    - Language-specific optimization
    - Automatic validation and quality checking
    - Response caching for performance
    """
    try:
        result = await ai_service.generate_code(
            prompt=generation_request.prompt,
            language=generation_request.language,
            context=generation_request.context,
            validate_response=generation_request.validate_response,
            use_cache=generation_request.use_cache
        )
        
        return CodeGenerationResponse(
            code=result.get('code', ''),
            language=generation_request.language,
            metadata=result.get('metadata', {}),
            validation=result.get('validation'),
            cached=result.get('cached', False),
            generated_at=datetime.utcnow().isoformat()
        )
        
    except ClaudeCodeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Claude Code service error: {str(e)}"
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Code generation validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate code: {str(e)}"
        )

@router.post("/orchestrate", response_model=TaskOrchestrationResponse)
@rate_limit(requests=5, window=60)
async def orchestrate_task(
    orchestration_request: TaskOrchestrationRequest,
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Orchestrate complex tasks using Claude Flow AI agent coordination.
    
    Orchestrates multi-step tasks with:
    - Intelligent agent assignment
    - Dynamic workflow adaptation
    - Performance optimization
    - Progress monitoring and coordination
    """
    try:
        result = await ai_service.orchestrate_task(
            task_description=orchestration_request.task_description,
            requirements=orchestration_request.requirements,
            agents=orchestration_request.agents,
            strategy=orchestration_request.strategy.value
        )
        
        return TaskOrchestrationResponse(
            task_id=result.get('task_id', ''),
            status=result.get('status', 'initiated'),
            agents_assigned=result.get('agents', []),
            strategy_used=orchestration_request.strategy.value,
            estimated_completion=result.get('estimated_completion'),
            metadata=result.get('metadata', {})
        )
        
    except ClaudeFlowError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Claude Flow service error: {str(e)}"
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task orchestration validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to orchestrate task: {str(e)}"
        )

@router.post("/validate", response_model=AIValidationResponse)
@rate_limit(requests=30, window=60)
async def validate_ai_response(
    validation_request: AIValidationRequest,
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Validate AI-generated responses for quality and correctness.
    
    Validates AI responses with:
    - Content authenticity checking
    - Format and structure validation
    - Custom criteria evaluation
    - Quality score calculation
    """
    try:
        result = await ai_service.validate_response(
            response=validation_request.response,
            validation_type=validation_request.validation_type.value,
            criteria=validation_request.criteria
        )
        
        return AIValidationResponse(
            is_valid=result['is_valid'],
            score=result['score'],
            errors=result.get('errors', []),
            warnings=result.get('warnings', []),
            metadata=result.get('metadata', {})
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"AI response validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate AI response: {str(e)}"
        )

@router.get("/performance", response_model=AIPerformanceResponse)
@rate_limit(requests=20, window=60)
async def get_ai_performance_metrics(
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Get comprehensive AI service performance metrics and analytics.
    
    Returns detailed performance data including:
    - Request success rates and statistics
    - Provider availability status
    - Cache utilization metrics
    - Performance trends and insights
    """
    try:
        metrics = await ai_service.get_performance_metrics()
        
        return AIPerformanceResponse(
            total_requests=metrics['total_requests'],
            successful_requests=metrics['successful_requests'],
            success_rate=metrics['success_rate'],
            cache_size=metrics['cache_size'],
            claude_code_available=metrics['claude_code_available'],
            claude_flow_available=metrics['claude_flow_available'],
            providers_count=metrics['providers_count']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI performance metrics: {str(e)}"
        )

@router.get("/history", response_model=AIHistoryResponse)
@rate_limit(requests=20, window=60)
async def get_ai_request_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    operation_filter: Optional[str] = Query(None, description="Filter by operation type"),
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Get AI service request history with filtering and pagination.
    
    Returns historical AI service requests with:
    - Request and response summaries
    - Success/failure status
    - Timestamp and operation details
    - Performance metrics per request
    """
    try:
        history = await ai_service.get_request_history(
            limit=page * page_size,
            operation_filter=operation_filter
        )
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_history = history[start_idx:end_idx]
        
        return AIHistoryResponse(
            history=paginated_history,
            total=len(history),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI request history: {str(e)}"
        )

@router.post("/code/review")
@rate_limit(requests=10, window=60)
async def review_code_with_ai(
    code: str = Body(..., description="Code to review"),
    language: str = Body(default="python", description="Programming language"),
    review_criteria: Optional[Dict[str, Any]] = Body(default_factory=dict, description="Review criteria"),
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Review code using AI-powered analysis for quality and best practices.
    
    Provides comprehensive code review including:
    - Code quality assessment
    - Best practices validation
    - Security vulnerability detection
    - Performance optimization suggestions
    """
    try:
        # Generate a code review using AI
        review_prompt = f"""
        Please review the following {language} code for:
        1. Code quality and readability
        2. Best practices compliance
        3. Potential bugs or issues
        4. Performance considerations
        5. Security vulnerabilities
        
        Code to review:
        ```{language}
        {code}
        ```
        
        Provide detailed feedback with specific recommendations.
        """
        
        result = await ai_service.generate_code(
            prompt=review_prompt,
            language="markdown",  # Review response in markdown
            context={
                "type": "code_review",
                "original_language": language,
                "criteria": review_criteria
            },
            validate_response=True,
            use_cache=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "review": result.get('code', ''),
                "language": language,
                "criteria_applied": review_criteria,
                "reviewed_at": datetime.utcnow().isoformat(),
                "validation": result.get('validation'),
                "metadata": result.get('metadata', {})
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to review code: {str(e)}"
        )

@router.post("/code/refactor")
@rate_limit(requests=10, window=60)
async def refactor_code_with_ai(
    code: str = Body(..., description="Code to refactor"),
    language: str = Body(default="python", description="Programming language"),
    refactor_instructions: str = Body(..., description="Refactoring instructions"),
    preserve_functionality: bool = Body(default=True, description="Preserve original functionality"),
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Refactor code using AI-guided transformations.
    
    Performs intelligent code refactoring with:
    - Functionality preservation
    - Code quality improvement
    - Performance optimization
    - Best practices application
    """
    try:
        refactor_prompt = f"""
        Please refactor the following {language} code according to these instructions:
        {refactor_instructions}
        
        Requirements:
        - {'Preserve original functionality exactly' if preserve_functionality else 'Focus on improvements over functionality preservation'}
        - Improve code quality and readability
        - Follow {language} best practices
        - Add comments explaining significant changes
        
        Original code:
        ```{language}
        {code}
        ```
        
        Provide the refactored code with explanations of changes made.
        """
        
        result = await ai_service.generate_code(
            prompt=refactor_prompt,
            language=language,
            context={
                "type": "code_refactor",
                "original_code": code,
                "instructions": refactor_instructions,
                "preserve_functionality": preserve_functionality
            },
            validate_response=True,
            use_cache=False  # Don't cache refactoring results
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "refactored_code": result.get('code', ''),
                "original_code": code,
                "instructions_applied": refactor_instructions,
                "language": language,
                "preserve_functionality": preserve_functionality,
                "refactored_at": datetime.utcnow().isoformat(),
                "validation": result.get('validation'),
                "metadata": result.get('metadata', {})
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refactor code: {str(e)}"
        )

@router.delete("/cache")
@rate_limit(requests=5, window=60)
async def clear_ai_cache(
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Clear AI service response cache.
    
    Clears all cached AI responses to ensure fresh results
    for subsequent requests.
    """
    try:
        await ai_service.clear_cache()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "AI service cache cleared successfully",
                "cleared_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear AI cache: {str(e)}"
        )

@router.get("/providers/status")
@rate_limit(requests=30, window=60)
async def get_ai_providers_status(
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Get status of all AI service providers.
    
    Returns availability and health status for:
    - Claude Code integration
    - Claude Flow orchestration
    - Service connectivity and performance
    """
    try:
        health = await ai_service.health_check()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "providers": {
                    "claude_code": {
                        "available": health.get('claude_code_available', False),
                        "status": "online" if health.get('claude_code_available') else "offline"
                    },
                    "claude_flow": {
                        "available": health.get('claude_flow_available', False),
                        "status": "online" if health.get('claude_flow_available') else "offline"
                    }
                },
                "overall_status": health.get('status', 'unknown'),
                "cache_size": health.get('response_cache_size', 0),
                "request_history_size": health.get('request_history_size', 0),
                "checked_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "providers": {
                    "claude_code": {"available": False, "status": "error"},
                    "claude_flow": {"available": False, "status": "error"}
                },
                "overall_status": "unhealthy",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
        )

@router.post("/test/connection")
@rate_limit(requests=10, window=60)
async def test_ai_connections(
    current_user: Dict = Depends(get_current_user),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Test connections to all AI service providers.
    
    Performs connectivity tests to validate that all
    AI service integrations are functioning properly.
    """
    try:
        # Test connections to both providers
        claude_code_test = await ai_service._test_claude_code_connection()
        claude_flow_test = await ai_service._test_claude_flow_connection()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "connection_tests": {
                    "claude_code": {
                        "connected": claude_code_test,
                        "status": "success" if claude_code_test else "failed"
                    },
                    "claude_flow": {
                        "connected": claude_flow_test,
                        "status": "success" if claude_flow_test else "failed"
                    }
                },
                "overall_connectivity": claude_code_test or claude_flow_test,
                "tested_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test AI connections: {str(e)}"
        )