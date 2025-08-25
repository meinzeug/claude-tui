"""
Project Data Models.

Defines the core project-related data structures including
Project, ProjectConfig, and ProjectTemplate models.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, validator


class ProjectStatus(Enum):
    """Project status enumeration."""
    CREATING = "creating"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    ERROR = "error"


class TemplateFileType(Enum):
    """Template file type enumeration."""
    STATIC = "static"  # Copy as-is
    TEMPLATE = "template"  # Process with Jinja2
    GENERATED = "generated"  # Generate with AI
    CONDITIONAL = "conditional"  # Include based on conditions


@dataclass
class TemplateFile:
    """Represents a file in a project template."""
    name: str
    path: str
    type: TemplateFileType
    content: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    ai_prompt: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class DirectoryStructure:
    """Represents a directory structure for project templates."""
    directories: List[str] = field(default_factory=list)
    files: List[TemplateFile] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'directories': self.directories,
            'files': [{
                'name': f.name,
                'path': f.path,
                'type': f.type.value,
                'content': f.content,
                'template_variables': f.template_variables,
                'conditions': f.conditions,
                'ai_prompt': f.ai_prompt,
                'dependencies': f.dependencies
            } for f in self.files]
        }


class ProjectTemplate(BaseModel):
    """Project template definition."""
    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Template structure
    structure: DirectoryStructure = Field(default_factory=DirectoryStructure)
    
    # Configuration
    default_variables: Dict[str, Any] = Field(default_factory=dict)
    required_variables: List[str] = Field(default_factory=list)
    optional_variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Dependencies and requirements
    dependencies: List[str] = Field(default_factory=list)
    dev_dependencies: List[str] = Field(default_factory=list)
    system_requirements: List[str] = Field(default_factory=list)
    
    # AI generation settings
    ai_generation_enabled: bool = True
    ai_validation_enabled: bool = True
    placeholder_tolerance: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    usage_count: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: str
        }
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Template name must be alphanumeric with hyphens or underscores')
        return v
    
    def get_variable_value(self, variable_name: str, user_values: Dict[str, Any]) -> Any:
        """Get the value for a template variable."""
        if variable_name in user_values:
            return user_values[variable_name]
        elif variable_name in self.default_variables:
            return self.default_variables[variable_name]
        elif variable_name in self.optional_variables:
            return self.optional_variables[variable_name]
        else:
            raise ValueError(f"Required variable '{variable_name}' not provided")
    
    def validate_variables(self, user_variables: Dict[str, Any]) -> Dict[str, str]:
        """Validate user-provided variables."""
        errors = {}
        
        # Check required variables
        for var in self.required_variables:
            if var not in user_variables:
                errors[var] = f"Required variable '{var}' is missing"
        
        # Validate variable types and constraints if defined
        # This could be expanded with more sophisticated validation
        
        return errors


class ProjectConfig(BaseModel):
    """Project configuration settings."""
    name: str
    template: str
    output_directory: Path
    
    # Template variables
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    
    # Development settings
    auto_validation: bool = True
    auto_completion: bool = True
    continuous_validation: bool = True
    
    # Quality settings
    code_quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    placeholder_detection_sensitivity: float = Field(default=0.95, ge=0.0, le=1.0)
    
    # AI settings
    ai_model_preferences: Dict[str, str] = Field(default_factory=dict)
    max_ai_retries: int = Field(default=3, ge=1, le=10)
    ai_timeout: int = Field(default=300, ge=30, le=3600)
    
    # Build and deployment
    build_command: Optional[str] = None
    test_command: Optional[str] = None
    deploy_command: Optional[str] = None
    
    # Backup and versioning
    backup_enabled: bool = True
    backup_interval_minutes: int = Field(default=30, ge=5, le=1440)
    version_control_integration: bool = True
    
    class Config:
        json_encoders = {
            Path: str
        }
    
    @validator('name')
    def validate_project_name(cls, v):
        if not v or not v.replace('-', '').replace('_', '').replace(' ', '').isalnum():
            raise ValueError('Project name must be alphanumeric with spaces, hyphens, or underscores')
        return v


class Project(BaseModel):
    """Represents a complete project instance."""
    
    # Basic information
    name: str
    path: Path
    config: ProjectConfig
    template: Optional[ProjectTemplate] = None
    
    # Status and metadata
    status: ProjectStatus = ProjectStatus.CREATING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    
    # Project statistics
    total_files: int = 0
    generated_files: int = 0
    validated_files: int = 0
    total_lines_of_code: int = 0
    
    # Quality metrics
    overall_quality_score: float = 0.0
    placeholder_count: int = 0
    validation_issues: List[str] = Field(default_factory=list)
    
    # Development progress
    completed_tasks: List[str] = Field(default_factory=list)
    pending_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    
    # Version control
    git_repository: Optional[str] = None
    current_branch: Optional[str] = None
    last_commit: Optional[str] = None
    
    # Custom metadata
    custom_properties: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: str
        }
    
    def update_timestamp(self) -> None:
        """Update the last updated timestamp."""
        self.updated_at = datetime.now()
    
    def update_access_time(self) -> None:
        """Update the last accessed timestamp."""
        self.last_accessed = datetime.now()
    
    def add_validation_issue(self, issue: str) -> None:
        """Add a validation issue to the project."""
        if issue not in self.validation_issues:
            self.validation_issues.append(issue)
            self.update_timestamp()
    
    def remove_validation_issue(self, issue: str) -> None:
        """Remove a validation issue from the project."""
        if issue in self.validation_issues:
            self.validation_issues.remove(issue)
            self.update_timestamp()
    
    def update_quality_score(self, score: float) -> None:
        """Update the overall quality score."""
        self.overall_quality_score = max(0.0, min(1.0, score))
        self.update_timestamp()
    
    def add_completed_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id in self.pending_tasks:
            self.pending_tasks.remove(task_id)
        if task_id in self.failed_tasks:
            self.failed_tasks.remove(task_id)
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        self.update_timestamp()
    
    def add_failed_task(self, task_id: str) -> None:
        """Mark a task as failed."""
        if task_id in self.pending_tasks:
            self.pending_tasks.remove(task_id)
        if task_id in self.completed_tasks:
            self.completed_tasks.remove(task_id)
        if task_id not in self.failed_tasks:
            self.failed_tasks.append(task_id)
        self.update_timestamp()
    
    def get_completion_percentage(self) -> float:
        """Calculate project completion percentage."""
        total_tasks = len(self.completed_tasks) + len(self.pending_tasks) + len(self.failed_tasks)
        if total_tasks == 0:
            return 0.0
        return (len(self.completed_tasks) / total_tasks) * 100.0
    
    def get_success_rate(self) -> float:
        """Calculate task success rate."""
        finished_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        if finished_tasks == 0:
            return 0.0
        return (len(self.completed_tasks) / finished_tasks) * 100.0
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Create a summary dictionary for display purposes."""
        return {
            'name': self.name,
            'path': str(self.path),
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'template': self.template.name if self.template else None,
            'completion_percentage': self.get_completion_percentage(),
            'success_rate': self.get_success_rate(),
            'quality_score': self.overall_quality_score,
            'total_files': self.total_files,
            'validation_issues': len(self.validation_issues),
            'placeholder_count': self.placeholder_count
        }
    
    def save_to_file(self, file_path: Path) -> None:
        """Save project metadata to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'Project':
        """Load project metadata from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert string paths back to Path objects
        if 'path' in data:
            data['path'] = Path(data['path'])
        if 'config' in data and 'output_directory' in data['config']:
            data['config']['output_directory'] = Path(data['config']['output_directory'])
        
        # Convert datetime strings back to datetime objects
        for date_field in ['created_at', 'updated_at', 'last_accessed']:
            if date_field in data and isinstance(data[date_field], str):
                data[date_field] = datetime.fromisoformat(data[date_field])
        
        return cls(**data)


@dataclass
class ProjectStats:
    """Statistics about a project."""
    file_count: int = 0
    directory_count: int = 0
    total_size_bytes: int = 0
    lines_of_code: int = 0
    generated_files: int = 0
    validated_files: int = 0
    placeholder_count: int = 0
    quality_score: float = 0.0
    last_modified: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'file_count': self.file_count,
            'directory_count': self.directory_count,
            'total_size_bytes': self.total_size_bytes,
            'lines_of_code': self.lines_of_code,
            'generated_files': self.generated_files,
            'validated_files': self.validated_files,
            'placeholder_count': self.placeholder_count,
            'quality_score': self.quality_score,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None
        }