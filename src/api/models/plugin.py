"""
Plugin model for managing Claude TIU plugins.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base


class Plugin(Base):
    """Plugin model for extensibility system."""
    
    __tablename__ = "plugins"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Plugin identity
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(20), nullable=False)
    author = Column(String(100), nullable=True)
    
    # Plugin configuration
    is_active = Column(Boolean, default=True)
    is_system_plugin = Column(Boolean, default=False)
    config = Column(JSON, nullable=True)  # Plugin-specific configuration
    
    # Plugin metadata
    category = Column(String(50), nullable=True)  # e.g., "ai", "ui", "integration"
    tags = Column(JSON, nullable=True)
    homepage_url = Column(String(255), nullable=True)
    repository_url = Column(String(255), nullable=True)
    
    # Installation details
    install_path = Column(String(255), nullable=True)
    checksum = Column(String(64), nullable=True)  # SHA256 checksum
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    installed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Plugin(id={self.id}, name='{self.name}', version='{self.version}')>"