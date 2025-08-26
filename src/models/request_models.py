"""
Pydantic Request Models for API Input Validation
Provides comprehensive input validation and sanitization
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, EmailStr
from datetime import datetime, date
from enum import Enum
import re
import bleach

class UserRegistrationRequest(BaseModel):
    """User registration request model"""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format"""
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v.lower().strip()
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @validator('full_name')
    def sanitize_full_name(cls, v):
        """Sanitize full name input"""
        if v is not None:
            # Remove HTML tags and sanitize
            v = bleach.clean(v.strip(), tags=[], strip=True)
            if len(v) == 0:
                return None
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "username": "johndoe",
                "password": "SecurePass123!",
                "full_name": "John Doe"
            }
        }


class UserLoginRequest(BaseModel):
    """User login request model"""
    identifier: str = Field(..., description="Email or username")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(default=False, description="Remember login")
    
    @validator('identifier')
    def validate_identifier(cls, v):
        """Sanitize login identifier"""
        return bleach.clean(v.strip().lower(), tags=[], strip=True)
    
    class Config:
        schema_extra = {
            "example": {
                "identifier": "user@example.com",
                "password": "SecurePass123!",
                "remember_me": false
            }
        }


class PasswordChangeRequest(BaseModel):
    """Password change request model"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    confirm_password: str = Field(..., description="Confirm new password")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @validator('confirm_password')
    def validate_passwords_match(cls, v, values):
        """Validate password confirmation"""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class TokenRefreshRequest(BaseModel):
    """Token refresh request model"""
    refresh_token: str = Field(..., description="Refresh token")


class UserUpdateRequest(BaseModel):
    """User update request model"""
    email: Optional[EmailStr] = Field(None, description="User email address")
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Username")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    is_active: Optional[bool] = Field(None, description="User active status")
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format"""
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9_]+$', v):
                raise ValueError('Username can only contain letters, numbers, and underscores')
            return v.lower().strip()
        return v
    
    @validator('full_name')
    def sanitize_full_name(cls, v):
        """Sanitize full name input"""
        if v is not None:
            v = bleach.clean(v.strip(), tags=[], strip=True)
            if len(v) == 0:
                return None
        return v


class ProjectStatus(str, Enum):
    """Project status enumeration"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"


class ProjectCreateRequest(BaseModel):
    """Project creation request model"""
    name: str = Field(..., min_length=1, max_length=100, description="Project name")
    description: Optional[str] = Field(None, max_length=1000, description="Project description")
    project_type: str = Field(default="general", max_length=50, description="Project type")
    is_public: bool = Field(default=False, description="Public project visibility")
    
    @validator('name')
    def sanitize_name(cls, v):
        """Sanitize project name"""
        v = bleach.clean(v.strip(), tags=[], strip=True)
        if not v:
            raise ValueError('Project name cannot be empty')
        return v
    
    @validator('description')
    def sanitize_description(cls, v):
        """Sanitize project description"""
        if v is not None:
            # Allow basic HTML tags in description but sanitize
            allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
            v = bleach.clean(v.strip(), tags=allowed_tags, strip=True)
            if len(v) == 0:
                return None
        return v
    
    @validator('project_type')
    def sanitize_project_type(cls, v):
        """Sanitize project type"""
        return bleach.clean(v.strip().lower(), tags=[], strip=True)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "My Awesome Project",
                "description": "A comprehensive project description",
                "project_type": "web_application",
                "is_public": false
            }
        }


class ProjectUpdateRequest(BaseModel):
    """Project update request model"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Project name")
    description: Optional[str] = Field(None, max_length=1000, description="Project description")
    status: Optional[ProjectStatus] = Field(None, description="Project status")
    is_public: Optional[bool] = Field(None, description="Public project visibility")
    is_archived: Optional[bool] = Field(None, description="Archive status")
    
    @validator('name')
    def sanitize_name(cls, v):
        """Sanitize project name"""
        if v is not None:
            v = bleach.clean(v.strip(), tags=[], strip=True)
            if not v:
                raise ValueError('Project name cannot be empty')
        return v
    
    @validator('description')
    def sanitize_description(cls, v):
        """Sanitize project description"""
        if v is not None:
            allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
            v = bleach.clean(v.strip(), tags=allowed_tags, strip=True)
            if len(v) == 0:
                return None
        return v


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskCreateRequest(BaseModel):
    """Task creation request model"""
    title: str = Field(..., min_length=1, max_length=200, description="Task title")
    description: Optional[str] = Field(None, max_length=2000, description="Task description")
    project_id: str = Field(..., description="Project ID")
    assigned_to: Optional[str] = Field(None, description="Assigned user ID")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    due_date: Optional[datetime] = Field(None, description="Due date")
    
    @validator('title')
    def sanitize_title(cls, v):
        """Sanitize task title"""
        v = bleach.clean(v.strip(), tags=[], strip=True)
        if not v:
            raise ValueError('Task title cannot be empty')
        return v
    
    @validator('description')
    def sanitize_description(cls, v):
        """Sanitize task description"""
        if v is not None:
            allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'code', 'pre']
            v = bleach.clean(v.strip(), tags=allowed_tags, strip=True)
            if len(v) == 0:
                return None
        return v
    
    @validator('due_date')
    def validate_due_date(cls, v):
        """Validate due date is in the future"""
        if v is not None and v < datetime.now():
            raise ValueError('Due date must be in the future')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Implement user authentication",
                "description": "Create secure JWT-based authentication system",
                "project_id": "123e4567-e89b-12d3-a456-426614174000",
                "priority": "high",
                "due_date": "2024-12-31T23:59:59"
            }
        }


class TaskUpdateRequest(BaseModel):
    """Task update request model"""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Task title")
    description: Optional[str] = Field(None, max_length=2000, description="Task description")
    status: Optional[TaskStatus] = Field(None, description="Task status")
    priority: Optional[TaskPriority] = Field(None, description="Task priority")
    assigned_to: Optional[str] = Field(None, description="Assigned user ID")
    due_date: Optional[datetime] = Field(None, description="Due date")
    
    @validator('title')
    def sanitize_title(cls, v):
        """Sanitize task title"""
        if v is not None:
            v = bleach.clean(v.strip(), tags=[], strip=True)
            if not v:
                raise ValueError('Task title cannot be empty')
        return v
    
    @validator('description')
    def sanitize_description(cls, v):
        """Sanitize task description"""
        if v is not None:
            allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'code', 'pre']
            v = bleach.clean(v.strip(), tags=allowed_tags, strip=True)
            if len(v) == 0:
                return None
        return v


class RoleCreateRequest(BaseModel):
    """Role creation request model"""
    name: str = Field(..., min_length=1, max_length=50, description="Role name")
    description: str = Field(..., max_length=200, description="Role description")
    permissions: List[str] = Field(default=[], description="List of permission names")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate and format role name"""
        v = bleach.clean(v.strip(), tags=[], strip=True)
        if not v:
            raise ValueError('Role name cannot be empty')
        # Convert to uppercase for consistency
        v = v.upper().replace(' ', '_')
        if not re.match(r'^[A-Z0-9_]+$', v):
            raise ValueError('Role name can only contain uppercase letters, numbers, and underscores')
        return v
    
    @validator('description')
    def sanitize_description(cls, v):
        """Sanitize role description"""
        return bleach.clean(v.strip(), tags=[], strip=True)


class RoleAssignmentRequest(BaseModel):
    """Role assignment request model"""
    user_id: str = Field(..., description="User ID")
    role_id: str = Field(..., description="Role ID")
    expires_at: Optional[datetime] = Field(None, description="Role expiration date")
    
    @validator('expires_at')
    def validate_expiration(cls, v):
        """Validate expiration date is in the future"""
        if v is not None and v < datetime.now():
            raise ValueError('Expiration date must be in the future')
        return v


class PaginationRequest(BaseModel):
    """Pagination request model"""
    skip: int = Field(default=0, ge=0, description="Number of records to skip")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of records to return")
    sort_by: Optional[str] = Field(None, max_length=50, description="Field to sort by")
    sort_order: Optional[str] = Field(default="asc", regex="^(asc|desc)$", description="Sort order")
    
    @validator('sort_by')
    def sanitize_sort_by(cls, v):
        """Sanitize sort field"""
        if v is not None:
            # Only allow alphanumeric characters and underscores
            v = re.sub(r'[^a-zA-Z0-9_]', '', v)
            if not v:
                return None
        return v


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default={}, description="Search filters")
    pagination: PaginationRequest = Field(default_factory=PaginationRequest)
    
    @validator('query')
    def sanitize_query(cls, v):
        """Sanitize search query"""
        # Remove HTML tags and limit special characters
        v = bleach.clean(v.strip(), tags=[], strip=True)
        # Remove potentially dangerous characters
        v = re.sub(r'[<>"\';\\]', '', v)
        if not v:
            raise ValueError('Search query cannot be empty')
        return v
    
    @validator('filters')
    def sanitize_filters(cls, v):
        """Sanitize filter values"""
        if v:
            sanitized = {}
            for key, value in v.items():
                # Sanitize key
                clean_key = re.sub(r'[^a-zA-Z0-9_]', '', str(key))
                if clean_key:
                    # Sanitize value
                    if isinstance(value, str):
                        clean_value = bleach.clean(str(value), tags=[], strip=True)
                        sanitized[clean_key] = clean_value
                    else:
                        sanitized[clean_key] = value
            return sanitized
        return {}


class BulkActionRequest(BaseModel):
    """Bulk action request model"""
    ids: List[str] = Field(..., min_items=1, max_items=100, description="List of resource IDs")
    action: str = Field(..., description="Action to perform")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Action parameters")
    
    @validator('action')
    def validate_action(cls, v):
        """Validate and sanitize action"""
        allowed_actions = ['delete', 'archive', 'activate', 'deactivate', 'update']
        v = bleach.clean(v.strip().lower(), tags=[], strip=True)
        if v not in allowed_actions:
            raise ValueError(f'Action must be one of: {", ".join(allowed_actions)}')
        return v
    
    @validator('parameters')
    def sanitize_parameters(cls, v):
        """Sanitize action parameters"""
        if v:
            sanitized = {}
            for key, value in v.items():
                clean_key = re.sub(r'[^a-zA-Z0-9_]', '', str(key))
                if clean_key and isinstance(value, (str, int, float, bool)):
                    if isinstance(value, str):
                        clean_value = bleach.clean(str(value), tags=[], strip=True)
                        sanitized[clean_key] = clean_value
                    else:
                        sanitized[clean_key] = value
            return sanitized
        return {}


class APIKeyCreateRequest(BaseModel):
    """API key creation request model"""
    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    permissions: List[str] = Field(default=[], description="List of permissions")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")
    
    @validator('name')
    def sanitize_name(cls, v):
        """Sanitize API key name"""
        v = bleach.clean(v.strip(), tags=[], strip=True)
        if not v:
            raise ValueError('API key name cannot be empty')
        return v
    
    @validator('expires_at')
    def validate_expiration(cls, v):
        """Validate expiration date"""
        if v is not None and v < datetime.now():
            raise ValueError('Expiration date must be in the future')
        return v