"""
User schemas for request/response models.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=6, max_length=100)
    
    class Config:
        schema_extra = {
            "example": {
                "username": "developer",
                "email": "dev@example.com",
                "full_name": "John Developer",
                "password": "securepassword123",
                "bio": "Full-stack developer passionate about AI"
            }
        }


class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = Field(None, max_length=255)
    preferences: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "full_name": "John Senior Developer",
                "bio": "Senior full-stack developer with 5+ years experience",
                "avatar_url": "https://example.com/avatar.jpg",
                "preferences": {
                    "theme": "dark",
                    "language": "en",
                    "notifications": True
                }
            }
        }


class UserResponse(UserBase):
    """User response schema."""
    id: int
    is_active: bool
    is_superuser: bool
    avatar_url: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "username": "developer",
                "email": "dev@example.com",
                "full_name": "John Developer",
                "bio": "Full-stack developer passionate about AI",
                "is_active": True,
                "is_superuser": False,
                "avatar_url": "https://example.com/avatar.jpg",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-15T10:30:00Z",
                "last_login": "2023-01-15T10:30:00Z"
            }
        }


class UserProfile(BaseModel):
    """User profile schema with preferences."""
    id: int
    username: str
    email: EmailStr
    full_name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    preferences: Optional[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True


class UserList(BaseModel):
    """User list response schema."""
    users: list[UserResponse]
    total: int
    page: int
    per_page: int
    
    class Config:
        schema_extra = {
            "example": {
                "users": [],
                "total": 50,
                "page": 1,
                "per_page": 10
            }
        }