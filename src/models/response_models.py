"""
Pydantic Response Models for API Output Serialization
Provides consistent response formats and data serialization
"""
from typing import Optional, List, Dict, Any, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

T = TypeVar('T')


class ResponseStatus(str, Enum):
    """Response status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class BaseResponse(BaseModel, Generic[T]):
    """Base response model with common fields"""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    
    class Config:
        use_enum_values = True


class ErrorDetail(BaseModel):
    """Error detail model"""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response model"""
    status: ResponseStatus = Field(default=ResponseStatus.ERROR, description="Response status")
    message: str = Field(..., description="Error message")
    errors: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "status": "error",
                "message": "Validation failed",
                "errors": [
                    {
                        "field": "email",
                        "message": "Invalid email format",
                        "code": "INVALID_FORMAT"
                    }
                ],
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req-12345"
            }
        }


class PaginationMetadata(BaseModel):
    """Pagination metadata model"""
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response model"""
    status: ResponseStatus = Field(default=ResponseStatus.SUCCESS, description="Response status")
    message: str = Field(..., description="Response message")
    data: List[T] = Field(..., description="Response data items")
    pagination: PaginationMetadata = Field(..., description="Pagination information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        use_enum_values = True


class UserResponse(BaseModel):
    """User response model"""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    username: str = Field(..., description="Username")
    full_name: Optional[str] = Field(None, description="Full name")
    is_active: bool = Field(..., description="User active status")
    is_verified: bool = Field(..., description="Email verification status")
    is_superuser: bool = Field(..., description="Superuser status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    roles: List[str] = Field(default=[], description="User roles")
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "user@example.com",
                "username": "johndoe",
                "full_name": "John Doe",
                "is_active": True,
                "is_verified": True,
                "is_superuser": False,
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
                "last_login": "2024-01-01T12:00:00Z",
                "roles": ["USER", "DEVELOPER"]
            }
        }


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    expires_at: datetime = Field(..., description="Token expiration timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "expires_at": "2024-01-01T13:00:00Z"
            }
        }


class ProjectResponse(BaseModel):
    """Project response model"""
    id: str = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    project_type: str = Field(..., description="Project type")
    status: str = Field(..., description="Project status")
    is_public: bool = Field(..., description="Public visibility")
    is_archived: bool = Field(..., description="Archive status")
    owner_id: str = Field(..., description="Owner user ID")
    owner_name: Optional[str] = Field(None, description="Owner name")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    task_count: Optional[int] = Field(None, description="Number of tasks")
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "My Awesome Project",
                "description": "A comprehensive project description",
                "project_type": "web_application",
                "status": "active",
                "is_public": False,
                "is_archived": False,
                "owner_id": "123e4567-e89b-12d3-a456-426614174001",
                "owner_name": "John Doe",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
                "task_count": 5
            }
        }


class TaskResponse(BaseModel):
    """Task response model"""
    id: str = Field(..., description="Task ID")
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    status: str = Field(..., description="Task status")
    priority: str = Field(..., description="Task priority")
    project_id: str = Field(..., description="Project ID")
    project_name: Optional[str] = Field(None, description="Project name")
    assigned_to: Optional[str] = Field(None, description="Assigned user ID")
    assignee_name: Optional[str] = Field(None, description="Assignee name")
    due_date: Optional[datetime] = Field(None, description="Due date")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174002",
                "title": "Implement user authentication",
                "description": "Create secure JWT-based authentication system",
                "status": "in_progress",
                "priority": "high",
                "project_id": "123e4567-e89b-12d3-a456-426614174000",
                "project_name": "My Awesome Project",
                "assigned_to": "123e4567-e89b-12d3-a456-426614174001",
                "assignee_name": "John Doe",
                "due_date": "2024-12-31T23:59:59Z",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z"
            }
        }


class RoleResponse(BaseModel):
    """Role response model"""
    id: str = Field(..., description="Role ID")
    name: str = Field(..., description="Role name")
    description: str = Field(..., description="Role description")
    is_system_role: bool = Field(..., description="System role flag")
    permissions: List[str] = Field(default=[], description="Role permissions")
    user_count: Optional[int] = Field(None, description="Number of users with this role")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True


class PermissionResponse(BaseModel):
    """Permission response model"""
    id: str = Field(..., description="Permission ID")
    name: str = Field(..., description="Permission name")
    resource: str = Field(..., description="Resource type")
    action: str = Field(..., description="Action type")
    description: str = Field(..., description="Permission description")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        from_attributes = True


class AuditLogResponse(BaseModel):
    """Audit log response model"""
    id: str = Field(..., description="Audit log ID")
    user_id: Optional[str] = Field(None, description="User ID")
    username: Optional[str] = Field(None, description="Username")
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Resource type")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    result: str = Field(..., description="Action result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Timestamp")
    
    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    """Session response model"""
    id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    location: Optional[str] = Field(None, description="Geographic location")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    expires_at: datetime = Field(..., description="Session expiration time")
    is_active: bool = Field(..., description="Session active status")
    
    class Config:
        from_attributes = True


class APIKeyResponse(BaseModel):
    """API key response model"""
    id: str = Field(..., description="API key ID")
    name: str = Field(..., description="API key name")
    key_preview: str = Field(..., description="API key preview (masked)")
    permissions: List[str] = Field(default=[], description="API key permissions")
    is_active: bool = Field(..., description="API key active status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_used: Optional[datetime] = Field(None, description="Last used timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174003",
                "name": "Production API Key",
                "key_preview": "ck_abc123...xyz789",
                "permissions": ["api.read", "api.write"],
                "is_active": True,
                "created_at": "2024-01-01T12:00:00Z",
                "last_used": "2024-01-01T12:00:00Z",
                "expires_at": "2025-01-01T12:00:00Z"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")
    database: Optional[str] = Field(None, description="Database connection status")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00Z",
                "uptime": 3600.0,
                "database": "connected",
                "memory_usage": {
                    "used": "256MB",
                    "total": "512MB",
                    "percentage": 50.0
                }
            }
        }


class MetricsResponse(BaseModel):
    """Metrics response model"""
    requests_total: int = Field(..., description="Total requests")
    requests_per_minute: float = Field(..., description="Requests per minute")
    average_response_time: float = Field(..., description="Average response time in milliseconds")
    error_rate: float = Field(..., description="Error rate percentage")
    active_users: int = Field(..., description="Active users count")
    database_queries: int = Field(..., description="Database queries count")
    cache_hits: int = Field(..., description="Cache hits")
    cache_misses: int = Field(..., description="Cache misses")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "requests_total": 10000,
                "requests_per_minute": 167.5,
                "average_response_time": 250.5,
                "error_rate": 2.1,
                "active_users": 150,
                "database_queries": 5000,
                "cache_hits": 8500,
                "cache_misses": 1500,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class BulkActionResponse(BaseModel):
    """Bulk action response model"""
    action: str = Field(..., description="Performed action")
    total_items: int = Field(..., description="Total items processed")
    successful_items: int = Field(..., description="Successfully processed items")
    failed_items: int = Field(..., description="Failed items")
    errors: List[Dict[str, Any]] = Field(default=[], description="Error details for failed items")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "action": "delete",
                "total_items": 100,
                "successful_items": 98,
                "failed_items": 2,
                "errors": [
                    {
                        "id": "item-1",
                        "error": "Item not found"
                    },
                    {
                        "id": "item-2",
                        "error": "Permission denied"
                    }
                ],
                "processing_time": 1.25
            }
        }