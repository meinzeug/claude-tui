"""
Theme model for UI theme management.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON
from sqlalchemy.sql import func
from .base import Base


class Theme(Base):
    """Theme model for UI customization."""
    
    __tablename__ = "themes"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Theme identity
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(20), nullable=False)
    author = Column(String(100), nullable=True)
    
    # Theme configuration
    is_active = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    is_system_theme = Column(Boolean, default=False)
    
    # Theme data
    color_scheme = Column(JSON, nullable=False)  # Color definitions
    typography = Column(JSON, nullable=True)  # Font and text settings
    layout = Column(JSON, nullable=True)  # Layout configuration
    animations = Column(JSON, nullable=True)  # Animation settings
    
    # Theme metadata
    category = Column(String(50), nullable=True)  # e.g., "dark", "light", "high-contrast"
    tags = Column(JSON, nullable=True)
    preview_image = Column(String(255), nullable=True)  # URL to preview image
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Theme(id={self.id}, name='{self.name}', version='{self.version}')>"