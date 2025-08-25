"""
Authentication schemas for request/response models.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    
    class Config:
        schema_extra = {
            "example": {
                "username": "developer",
                "password": "securepassword123"
            }
        }


class RegisterRequest(BaseModel):
    """User registration request schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)
    
    class Config:
        schema_extra = {
            "example": {
                "username": "newdeveloper",
                "email": "dev@example.com",
                "password": "securepassword123",
                "full_name": "John Developer"
            }
        }


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    refresh_token: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "refresh_token_here"
            }
        }


class TokenRefreshRequest(BaseModel):
    """Token refresh request schema."""
    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""
    email: EmailStr
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""
    token: str
    new_password: str = Field(..., min_length=6, max_length=100)
    
    class Config:
        schema_extra = {
            "example": {
                "token": "reset_token_here",
                "new_password": "newSecurePassword123"
            }
        }


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""
    current_password: str = Field(..., min_length=6, max_length=100)
    new_password: str = Field(..., min_length=6, max_length=100)
    revoke_other_sessions: bool = Field(default=True, description="Revoke other active sessions")
    
    class Config:
        schema_extra = {
            "example": {
                "current_password": "currentPassword123",
                "new_password": "newSecurePassword456",
                "revoke_other_sessions": True
            }
        }


class OAuthCallbackRequest(BaseModel):
    """OAuth callback request schema."""
    code: Optional[str] = Field(None, description="Authorization code")
    state: str = Field(..., description="CSRF state parameter")
    error: Optional[str] = Field(None, description="Error code")
    error_description: Optional[str] = Field(None, description="Error description")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "auth_code_here",
                "state": "csrf_state_token"
            }
        }


class SessionInfo(BaseModel):
    """Session information schema."""
    session_id: str = Field(..., description="Session ID")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent")
    created_at: str = Field(..., description="Creation timestamp")
    last_activity: str = Field(..., description="Last activity timestamp")
    expires_at: str = Field(..., description="Expiration timestamp")
    is_current: bool = Field(..., description="Is current session")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0...",
                "created_at": "2023-12-01T10:00:00Z",
                "last_activity": "2023-12-01T11:30:00Z",
                "expires_at": "2023-12-01T18:00:00Z",
                "is_current": True
            }
        }


class UserSessionsResponse(BaseModel):
    """User sessions response schema."""
    sessions: List[SessionInfo] = Field(..., description="Active sessions")
    
    class Config:
        schema_extra = {
            "example": {
                "sessions": [
                    {
                        "session_id": "session_123",
                        "ip_address": "192.168.1.100",
                        "user_agent": "Mozilla/5.0...",
                        "created_at": "2023-12-01T10:00:00Z",
                        "last_activity": "2023-12-01T11:30:00Z",
                        "expires_at": "2023-12-01T18:00:00Z",
                        "is_current": True
                    }
                ]
            }
        }