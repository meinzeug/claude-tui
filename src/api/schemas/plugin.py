"""
Plugin schemas for request/response models.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List
from datetime import datetime


class PluginCreate(BaseModel):
    """Plugin creation schema."""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field(..., min_length=1, max_length=20)
    author: Optional[str] = Field(None, max_length=100)
    category: Optional[str] = Field(None, max_length=50)
    tags: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    homepage_url: Optional[HttpUrl] = None
    repository_url: Optional[HttpUrl] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "ai-code-assistant",
                "display_name": "AI Code Assistant",
                "description": "Advanced AI-powered code generation and analysis plugin",
                "version": "1.0.0",
                "author": "Claude TIU Team",
                "category": "ai",
                "tags": ["ai", "code", "generation"],
                "config": {
                    "model": "claude-3",
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                "homepage_url": "https://example.com/plugin",
                "repository_url": "https://github.com/example/plugin"
            }
        }


class PluginUpdate(BaseModel):
    """Plugin update schema."""
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    version: Optional[str] = Field(None, min_length=1, max_length=20)
    author: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    category: Optional[str] = Field(None, max_length=50)
    tags: Optional[List[str]] = None
    homepage_url: Optional[HttpUrl] = None
    repository_url: Optional[HttpUrl] = None
    
    class Config:
        schema_extra = {
            "example": {
                "display_name": "Updated AI Code Assistant",
                "description": "Enhanced AI-powered code generation plugin with new features",
                "version": "1.1.0",
                "is_active": True,
                "config": {
                    "model": "claude-3.5",
                    "temperature": 0.8,
                    "max_tokens": 3000
                }
            }
        }


class PluginResponse(BaseModel):
    """Plugin response schema."""
    id: int
    name: str
    display_name: str
    description: Optional[str]
    version: str
    author: Optional[str]
    is_active: bool
    is_system_plugin: bool
    config: Optional[Dict[str, Any]]
    category: Optional[str]
    tags: Optional[List[str]]
    homepage_url: Optional[str]
    repository_url: Optional[str]
    install_path: Optional[str]
    checksum: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    installed_at: Optional[datetime]
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "name": "ai-code-assistant",
                "display_name": "AI Code Assistant",
                "description": "Advanced AI-powered code generation plugin",
                "version": "1.0.0",
                "author": "Claude TIU Team",
                "is_active": True,
                "is_system_plugin": False,
                "config": {"model": "claude-3"},
                "category": "ai",
                "tags": ["ai", "code"],
                "homepage_url": "https://example.com/plugin",
                "repository_url": "https://github.com/example/plugin",
                "install_path": "/plugins/ai-code-assistant",
                "checksum": "abc123...",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-15T10:30:00Z",
                "installed_at": "2023-01-01T00:05:00Z"
            }
        }


class PluginInstall(BaseModel):
    """Plugin installation schema."""
    source: str = Field(..., description="Plugin source (URL or path)")
    verify_checksum: bool = Field(True, description="Verify plugin checksum")
    auto_activate: bool = Field(True, description="Automatically activate after install")
    
    class Config:
        schema_extra = {
            "example": {
                "source": "https://github.com/example/plugin/releases/download/v1.0.0/plugin.zip",
                "verify_checksum": True,
                "auto_activate": True
            }
        }


class PluginList(BaseModel):
    """Plugin list response schema."""
    plugins: List[PluginResponse]
    total: int
    page: int
    per_page: int
    
    class Config:
        schema_extra = {
            "example": {
                "plugins": [],
                "total": 25,
                "page": 1,
                "per_page": 10
            }
        }


class PluginStatus(BaseModel):
    """Plugin status response schema."""
    plugin_id: int
    name: str
    is_active: bool
    is_loaded: bool
    last_error: Optional[str]
    health_check: Optional[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "plugin_id": 1,
                "name": "ai-code-assistant",
                "is_active": True,
                "is_loaded": True,
                "last_error": None,
                "health_check": {
                    "status": "healthy",
                    "response_time": 0.05,
                    "last_check": "2023-01-15T10:30:00Z"
                }
            }
        }