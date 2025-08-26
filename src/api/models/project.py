"""
Project database model based on the database schema.
"""

from sqlalchemy import Column, String, Text, Boolean, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

from .base import Base, TimestampMixin


class Project(Base, TimestampMixin):
    """Project model following the database schema."""
    __tablename__ = "projects"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default='active', nullable=False)  # active, completed, archived, failed
    template_id = Column(String(255))
    configuration = Column(JSON, default=dict)
    project_metadata = Column(JSON, default=dict)
    real_progress = Column(Float, default=0.0)
    fake_progress = Column(Float, default=0.0)  
    authenticity_rate = Column(Float, default=100.0)
    quality_score = Column(Float, default=0.0)
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="projects")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    project_files = relationship("ProjectFile", back_populates="project", cascade="all, delete-orphan")
    validation_reports = relationship("ValidationReport", back_populates="project", cascade="all, delete-orphan")
    progress_metrics = relationship("ProgressMetric", back_populates="project", cascade="all, delete-orphan")


class Task(Base, TimestampMixin):
    """Task model following the database schema."""
    __tablename__ = "tasks"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default='pending', nullable=False)  # pending, in_progress, completed, failed, blocked
    priority = Column(String(20), default='medium', nullable=False)  # low, medium, high, critical
    real_progress = Column(Float, default=0.0)
    fake_progress = Column(Float, default=0.0)
    authenticity_score = Column(Float, default=100.0)
    ai_prompt = Column(Text)
    expected_outputs = Column(JSON, default=list)
    validation_criteria = Column(JSON, default=dict)
    estimated_duration = Column(Integer)  # in minutes
    actual_duration = Column(Integer)  # in minutes
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    project = relationship("Project", back_populates="tasks")
    dependencies = relationship("TaskDependency", 
                              foreign_keys="TaskDependency.task_id",
                              back_populates="task")
    dependents = relationship("TaskDependency",
                            foreign_keys="TaskDependency.depends_on_task_id", 
                            back_populates="depends_on_task")
    validations = relationship("TaskValidation", back_populates="task", cascade="all, delete-orphan")
    agent_assignments = relationship("AgentAssignment", back_populates="task", cascade="all, delete-orphan")


class TaskDependency(Base, TimestampMixin):
    """Task dependency model."""
    __tablename__ = "task_dependencies"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('tasks.id'), nullable=False)
    depends_on_task_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('tasks.id'), nullable=False)
    dependency_type = Column(String(50), default='blocks', nullable=False)  # blocks, requires, suggests
    
    # Relationships
    task = relationship("Task", foreign_keys=[task_id], back_populates="dependencies")
    depends_on_task = relationship("Task", foreign_keys=[depends_on_task_id], back_populates="dependents")


class ProjectFile(Base, TimestampMixin):
    """Project file model."""
    __tablename__ = "project_files"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    file_path = Column(String(1000), nullable=False)
    content = Column(Text)
    content_hash = Column(String(64))
    size_bytes = Column(Integer, default=0)
    status = Column(String(50), default='created', nullable=False)  # created, modified, validated, failed_validation, deleted
    validation_score = Column(Float, default=0.0)
    project_metadata = Column(JSON, default=dict)
    
    # Relationships
    project = relationship("Project", back_populates="project_files")


class ValidationReport(Base):
    """Validation report model."""
    __tablename__ = "validation_reports"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    validation_type = Column(String(50), nullable=False)  # quick, deep, comprehensive, scheduled
    overall_authenticity = Column(Float)
    real_progress = Column(Float)
    fake_progress = Column(Float)
    quality_metrics = Column(JSON, default=dict)
    issues_summary = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="validation_reports")
    issues = relationship("ValidationIssue", back_populates="validation_report", cascade="all, delete-orphan")


class ValidationIssue(Base, TimestampMixin):
    """Validation issue model."""
    __tablename__ = "validation_issues"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    validation_report_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('validation_reports.id'), nullable=False)
    task_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('tasks.id'))
    issue_type = Column(String(50), nullable=False)  # placeholder, empty_function, mock_data, broken_logic, security_risk
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    description = Column(Text, nullable=False)
    file_path = Column(String(1000))
    line_number = Column(Integer)
    auto_fix_available = Column(Boolean, default=False)
    suggested_fix = Column(Text)
    status = Column(String(50), default='open', nullable=False)  # open, in_progress, resolved, ignored
    resolved_at = Column(DateTime(timezone=True))
    
    # Relationships
    validation_report = relationship("ValidationReport", back_populates="issues")
    task = relationship("Task", back_populates="validations")


class TaskValidation(Base, TimestampMixin):
    """Task validation model (alias for ValidationIssue)."""
    __table__ = ValidationIssue.__table__
    
    # Relationships
    task = relationship("Task", back_populates="validations")


class Agent(Base, TimestampMixin):
    """Agent model."""
    __tablename__ = "agents"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_type = Column(String(100), nullable=False)  # backend-developer, frontend-developer, tester, etc.
    name = Column(String(255))
    capabilities = Column(JSON, default=list)
    status = Column(String(50), default='idle', nullable=False)  # idle, busy, offline, error
    configuration = Column(JSON, default=dict)
    coordination_hooks = Column(JSON, default=dict)
    last_active = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    assignments = relationship("AgentAssignment", back_populates="agent", cascade="all, delete-orphan")
    metrics = relationship("AgentMetric", back_populates="agent", cascade="all, delete-orphan")


class AgentAssignment(Base, TimestampMixin):
    """Agent assignment model."""
    __tablename__ = "agent_assignments"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    task_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('tasks.id'), nullable=False)
    status = Column(String(50), nullable=False)
    assignment_metadata = Column(JSON, default=dict)
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    agent = relationship("Agent", back_populates="assignments")
    task = relationship("Task", back_populates="agent_assignments")


class AgentMetric(Base):
    """Agent metric model."""
    __tablename__ = "agent_metrics"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    metric_type = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    project_metadata = Column(JSON, default=dict)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="metrics")


class ProgressMetric(Base):
    """Progress metric model."""
    __tablename__ = "progress_metrics"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    project_metadata = Column(JSON, default=dict)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="progress_metrics")


class Template(Base, TimestampMixin):
    """Template model."""
    __tablename__ = "templates"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    structure = Column(JSON, nullable=False)
    default_config = Column(JSON, default=dict)
    workflow_template = Column(JSON)
    is_public = Column(Boolean, default=False)
    created_by = Column(PostgreSQLUUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    creator = relationship("User", back_populates="templates")