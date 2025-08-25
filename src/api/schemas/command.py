"""
Command schemas for request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class CommandType(str, Enum):
    """Command type enumeration."""
    CODE_GENERATION = "code_generation"
    REFACTORING = "refactoring"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    ANALYSIS = "analysis"


class CommandStatus(str, Enum):
    """Command status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CommandCreate(BaseModel):
    """Command creation schema."""
    command_type: CommandType
    description: str = Field(..., min_length=1, max_length=1000)
    prompt: str = Field(..., min_length=1, max_length=5000)
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "command_type": "code_generation",
                "description": "Generate a REST API endpoint for user management",
                "prompt": "Create a FastAPI endpoint that handles user CRUD operations with proper validation and error handling",
                "metadata": {
                    "language": "python",
                    "framework": "fastapi",
                    "complexity": "medium"
                },
                "tags": ["api", "crud", "users"]
            }
        }


class CommandExecute(BaseModel):
    """Command execution schema."""
    timeout: Optional[int] = Field(None, gt=0, le=3600)  # Max 1 hour
    priority: Optional[int] = Field(1, ge=1, le=5)
    context: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "timeout": 300,
                "priority": 2,
                "context": {
                    "project_path": "/path/to/project",
                    "target_file": "main.py"
                }
            }
        }


class CommandUpdate(BaseModel):
    """Command update schema."""
    description: Optional[str] = Field(None, min_length=1, max_length=1000)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "description": "Updated description",
                "tags": ["updated", "modified"],
                "metadata": {"updated": True}
            }
        }


class CommandResponse(BaseModel):
    """Command response schema."""
    id: int
    user_id: int
    command_type: CommandType
    description: str
    prompt: str
    status: CommandStatus
    result: Optional[str]
    error_message: Optional[str]
    execution_time: Optional[int]  # seconds
    metadata: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "command_type": "code_generation",
                "description": "Generate a REST API endpoint for user management",
                "prompt": "Create a FastAPI endpoint...",
                "status": "completed",
                "result": "Generated code here...",
                "error_message": None,
                "execution_time": 45,
                "metadata": {"language": "python"},
                "tags": ["api", "crud"],
                "created_at": "2023-01-01T00:00:00Z",
                "started_at": "2023-01-01T00:01:00Z",
                "completed_at": "2023-01-01T00:01:45Z"
            }
        }


class CommandList(BaseModel):
    """Command list response schema."""
    commands: List[CommandResponse]
    total: int
    page: int
    per_page: int
    
    class Config:
        schema_extra = {
            "example": {
                "commands": [],
                "total": 100,
                "page": 1,
                "per_page": 20
            }
        }


class CommandStats(BaseModel):
    """Command statistics schema."""
    total_commands: int
    completed_commands: int
    failed_commands: int
    running_commands: int
    average_execution_time: Optional[float]
    success_rate: float
    
    class Config:
        schema_extra = {
            "example": {
                "total_commands": 150,
                "completed_commands": 135,
                "failed_commands": 10,
                "running_commands": 5,
                "average_execution_time": 32.5,
                "success_rate": 0.9
            }
        }