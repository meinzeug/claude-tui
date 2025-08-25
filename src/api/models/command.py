"""
Command model for storing AI command history and results.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from .base import Base


class CommandStatus(enum.Enum):
    """Command execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Command(Base):
    """Command model for AI command execution tracking."""
    
    __tablename__ = "commands"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Command details
    command_type = Column(String(50), nullable=False)  # e.g., "code_generation", "refactor"
    description = Column(Text, nullable=False)
    prompt = Column(Text, nullable=False)
    
    # Execution details
    status = Column(Enum(CommandStatus), default=CommandStatus.PENDING)
    result = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_time = Column(Integer, nullable=True)  # in seconds
    
    # Metadata
    metadata = Column(JSON, nullable=True)  # Additional command-specific data
    tags = Column(JSON, nullable=True)  # Command tags for organization
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", backref="commands")
    
    def __repr__(self):
        return f"<Command(id={self.id}, type='{self.command_type}', status='{self.status}')>"