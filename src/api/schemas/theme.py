"""
Theme schemas for request/response models.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List
from datetime import datetime


class ColorScheme(BaseModel):
    """Color scheme definition."""
    primary: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
    secondary: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
    background: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
    surface: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
    text_primary: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
    text_secondary: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
    accent: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
    success: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
    warning: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")
    error: str = Field(..., regex="^#[0-9A-Fa-f]{6}$")


class Typography(BaseModel):
    """Typography settings."""
    font_family: str = Field(..., max_length=100)
    font_size_base: int = Field(..., ge=8, le=24)
    line_height: float = Field(..., ge=1.0, le=3.0)
    font_weight_normal: int = Field(..., ge=100, le=900)
    font_weight_bold: int = Field(..., ge=100, le=900)


class ThemeCreate(BaseModel):
    """Theme creation schema."""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field(..., min_length=1, max_length=20)
    author: Optional[str] = Field(None, max_length=100)
    category: Optional[str] = Field(None, max_length=50)
    color_scheme: ColorScheme
    typography: Optional[Typography] = None
    layout: Optional[Dict[str, Any]] = None
    animations: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    preview_image: Optional[HttpUrl] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "dark-professional",
                "display_name": "Dark Professional",
                "description": "A sleek dark theme for professional developers",
                "version": "1.0.0",
                "author": "Claude TIU Team",
                "category": "dark",
                "color_scheme": {
                    "primary": "#007ACC",
                    "secondary": "#4A90E2",
                    "background": "#1E1E1E",
                    "surface": "#252526",
                    "text_primary": "#CCCCCC",
                    "text_secondary": "#969696",
                    "accent": "#FF6B35",
                    "success": "#28A745",
                    "warning": "#FFC107",
                    "error": "#DC3545"
                },
                "typography": {
                    "font_family": "JetBrains Mono",
                    "font_size_base": 14,
                    "line_height": 1.5,
                    "font_weight_normal": 400,
                    "font_weight_bold": 600
                },
                "tags": ["dark", "professional", "coding"]
            }
        }


class ThemeUpdate(BaseModel):
    """Theme update schema."""
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    version: Optional[str] = Field(None, min_length=1, max_length=20)
    author: Optional[str] = Field(None, max_length=100)
    category: Optional[str] = Field(None, max_length=50)
    color_scheme: Optional[ColorScheme] = None
    typography: Optional[Typography] = None
    layout: Optional[Dict[str, Any]] = None
    animations: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    preview_image: Optional[HttpUrl] = None
    is_active: Optional[bool] = None
    
    class Config:
        schema_extra = {
            "example": {
                "display_name": "Updated Dark Professional",
                "description": "Enhanced dark theme with new features",
                "version": "1.1.0",
                "is_active": True
            }
        }


class ThemeResponse(BaseModel):
    """Theme response schema."""
    id: int
    name: str
    display_name: str
    description: Optional[str]
    version: str
    author: Optional[str]
    is_active: bool
    is_default: bool
    is_system_theme: bool
    color_scheme: Dict[str, str]
    typography: Optional[Dict[str, Any]]
    layout: Optional[Dict[str, Any]]
    animations: Optional[Dict[str, Any]]
    category: Optional[str]
    tags: Optional[List[str]]
    preview_image: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "name": "dark-professional",
                "display_name": "Dark Professional",
                "description": "A sleek dark theme for professional developers",
                "version": "1.0.0",
                "author": "Claude TIU Team",
                "is_active": True,
                "is_default": False,
                "is_system_theme": True,
                "color_scheme": {"primary": "#007ACC"},
                "typography": {"font_family": "JetBrains Mono"},
                "layout": None,
                "animations": None,
                "category": "dark",
                "tags": ["dark", "professional"],
                "preview_image": "https://example.com/preview.jpg",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-15T10:30:00Z"
            }
        }


class ThemeApply(BaseModel):
    """Theme application schema."""
    theme_id: int
    apply_to_all: bool = Field(False, description="Apply to all users")
    
    class Config:
        schema_extra = {
            "example": {
                "theme_id": 1,
                "apply_to_all": False
            }
        }


class ThemeList(BaseModel):
    """Theme list response schema."""
    themes: List[ThemeResponse]
    total: int
    page: int
    per_page: int
    
    class Config:
        schema_extra = {
            "example": {
                "themes": [],
                "total": 15,
                "page": 1,
                "per_page": 10
            }
        }