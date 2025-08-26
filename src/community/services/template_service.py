"""
Template Service - Enhanced template management with inheritance and composition.
"""

import asyncio
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from jinja2 import Environment, FileSystemLoader, meta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.template import (
    Template, TemplateVersion, TemplateCreate, TemplateUpdate,
    TemplateInheritanceConfig, TemplateBuildResult, TemplateType
)
from ..models.validation import TemplateValidation
from ..repositories.template_repository import TemplateRepository
from ...core.exceptions import ValidationError, NotFoundError, PermissionError
from ...integrations.file_system import FileSystemManager

logger = logging.getLogger(__name__)


class TemplateService:
    """Enhanced template service with inheritance, composition, and validation."""
    
    def __init__(self, db: AsyncSession):
        """Initialize template service."""
        self.db = db
        self.repository = TemplateRepository(db)
        self.file_manager = FileSystemManager()
        self.jinja_env = Environment(loader=FileSystemLoader([]))
    
    async def create_template(
        self,
        template_data: TemplateCreate,
        author_id: UUID
    ) -> Template:
        """
        Create a new template with validation and processing.
        
        Args:
            template_data: Template creation data
            author_id: Author user ID
            
        Returns:
            Created template
        """
        try:
            # Generate unique slug
            slug = await self._generate_unique_slug(template_data.name)
            
            # Validate template structure
            await self._validate_template_structure(template_data.template_data)
            
            # Process template files
            processed_files = await self._process_template_files(template_data.template_files)
            
            # Create template instance
            template = Template(
                slug=slug,
                name=template_data.name,
                description=template_data.description,
                short_description=template_data.short_description,
                author_id=author_id,
                template_data=template_data.template_data,
                template_files=processed_files,
                template_config=template_data.template_config,
                template_type=template_data.template_type.value,
                complexity_level=template_data.complexity_level.value,
                tags=template_data.tags,
                categories=template_data.categories,
                frameworks=template_data.frameworks,
                languages=template_data.languages,
                license_type=template_data.license_type,
                is_public=template_data.is_public,
                dependencies=template_data.dependencies,
                requirements=template_data.requirements,
                published_at=datetime.utcnow() if template_data.is_public else None
            )
            
            # Calculate initial quality score
            quality_score = await self._calculate_initial_quality_score(template)
            template.quality_score = quality_score
            
            # Save template
            self.db.add(template)
            await self.db.commit()
            await self.db.refresh(template)
            
            # Create initial version
            await self._create_template_version(template, "Initial version")
            
            # Schedule validation if enabled
            await self._schedule_template_validation(template.id)
            
            logger.info(f"Created template '{template.name}' (ID: {template.id})")
            return template
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating template: {e}")
            raise ValidationError(f"Template creation failed: {str(e)}")
    
    async def update_template(
        self,
        template_id: UUID,
        template_data: TemplateUpdate,
        user_id: UUID
    ) -> Template:
        """
        Update an existing template.
        
        Args:
            template_id: Template ID
            template_data: Update data
            user_id: User performing update
            
        Returns:
            Updated template
        """
        try:
            # Get template
            template = await self.repository.get_template_by_id(template_id)
            if not template:
                raise NotFoundError("Template not found")
            
            # Check permissions
            if template.author_id != user_id:
                # Check if user is admin/moderator
                # This would be implemented based on your auth system
                raise PermissionError("Not authorized to update this template")
            
            # Track changes for versioning
            changes = []
            old_version = template.version
            
            # Update fields
            if template_data.name and template_data.name != template.name:
                changes.append(f"Name changed from '{template.name}' to '{template_data.name}'")
                template.name = template_data.name
            
            if template_data.description:
                template.description = template_data.description
                changes.append("Description updated")
            
            if template_data.short_description:
                template.short_description = template_data.short_description
            
            if template_data.template_data:
                await self._validate_template_structure(template_data.template_data)
                template.template_data = template_data.template_data
                changes.append("Template data updated")
            
            if template_data.template_files:
                processed_files = await self._process_template_files(template_data.template_files)
                template.template_files = processed_files
                changes.append("Template files updated")
            
            if template_data.template_config:
                template.template_config = template_data.template_config
                changes.append("Template configuration updated")
            
            if template_data.tags is not None:
                template.tags = template_data.tags
            
            if template_data.categories is not None:
                template.categories = template_data.categories
            
            if template_data.frameworks is not None:
                template.frameworks = template_data.frameworks
            
            if template_data.languages is not None:
                template.languages = template_data.languages
            
            if template_data.complexity_level:
                template.complexity_level = template_data.complexity_level.value
            
            if template_data.license_type:
                template.license_type = template_data.license_type
            
            if template_data.is_public is not None:
                template.is_public = template_data.is_public
                if template_data.is_public and not template.published_at:
                    template.published_at = datetime.utcnow()
            
            # Increment version if significant changes
            if changes:
                version_parts = template.version.split('.')
                patch_version = int(version_parts[2]) + 1
                template.version = f"{version_parts[0]}.{version_parts[1]}.{patch_version}"
                
                # Create new version record
                await self._create_template_version(template, f"Update: {', '.join(changes)}")
            
            # Recalculate quality score
            template.quality_score = await self._calculate_initial_quality_score(template)
            template.updated_at = datetime.utcnow()
            
            await self.db.commit()
            await self.db.refresh(template)
            
            logger.info(f"Updated template '{template.name}' (ID: {template.id})")
            return template
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating template: {e}")
            raise
    
    async def create_template_from_inheritance(
        self,
        inheritance_config: TemplateInheritanceConfig,
        author_id: UUID,
        template_name: str,
        description: str = ""
    ) -> Template:
        """
        Create a new template by inheriting from an existing template.
        
        Args:
            inheritance_config: Inheritance configuration
            author_id: Author user ID
            template_name: New template name
            description: New template description
            
        Returns:
            Created template
        """
        try:
            # Get parent template
            parent_template = await self.repository.get_template_by_id(inheritance_config.inherit_from)
            if not parent_template:
                raise NotFoundError("Parent template not found")
            
            if not parent_template.is_public:
                raise PermissionError("Cannot inherit from private template")
            
            # Start with parent template data
            template_data = parent_template.template_data.copy()
            template_files = parent_template.template_files.copy()
            template_config = parent_template.template_config.copy()
            
            # Apply inheritance logic
            if inheritance_config.merge_config:
                template_config.update(inheritance_config.custom_variables)
            else:
                template_config = inheritance_config.custom_variables
            
            # Override specific files
            if inheritance_config.override_files:
                template_files.update(inheritance_config.override_files)
            
            # Process template variables
            if inheritance_config.custom_variables:
                template_data = await self._apply_template_variables(
                    template_data, inheritance_config.custom_variables
                )
            
            # Create new template
            new_template_data = TemplateCreate(
                name=template_name,
                description=description,
                template_data=template_data,
                template_files=template_files,
                template_config=template_config,
                template_type=TemplateType(parent_template.template_type),
                tags=parent_template.tags.copy() if parent_template.tags else [],
                categories=parent_template.categories.copy() if parent_template.categories else [],
                frameworks=parent_template.frameworks.copy() if parent_template.frameworks else [],
                languages=parent_template.languages.copy() if parent_template.languages else [],
                dependencies=parent_template.dependencies.copy() if parent_template.dependencies else [],
                requirements=parent_template.requirements.copy() if parent_template.requirements else {}
            )
            
            # Create the template
            template = await self.create_template(new_template_data, author_id)
            
            # Set parent relationship
            template.parent_template_id = parent_template.id
            
            # Update parent fork count
            parent_template.fork_count += 1
            
            await self.db.commit()
            await self.db.refresh(template)
            
            logger.info(f"Created inherited template '{template_name}' from '{parent_template.name}'")
            return template
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating inherited template: {e}")
            raise
    
    async def build_template(
        self,
        template_id: UUID,
        build_config: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> TemplateBuildResult:
        """
        Build/render a template with given configuration.
        
        Args:
            template_id: Template ID
            build_config: Build configuration and variables
            output_path: Optional output path
            
        Returns:
            Build result
        """
        start_time = datetime.utcnow()
        
        try:
            # Get template
            template = await self.repository.get_template_by_id(template_id)
            if not template:
                raise NotFoundError("Template not found")
            
            # Create temporary build directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                build_path = output_path or temp_path / "build"
                build_path.mkdir(parents=True, exist_ok=True)
                
                files_generated = []
                warnings = []
                errors = []
                
                # Process each template file
                for file_path, file_content in template.template_files.items():
                    try:
                        # Apply Jinja2 templating
                        template_obj = self.jinja_env.from_string(file_content)
                        rendered_content = template_obj.render(**build_config)
                        
                        # Write file
                        output_file = build_path / file_path
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        output_file.write_text(rendered_content, encoding='utf-8')
                        
                        files_generated.append(str(file_path))
                        
                    except Exception as e:
                        error_msg = f"Error processing file '{file_path}': {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                
                # Copy static files (non-template files)
                static_files = build_config.get('static_files', {})
                for static_path, static_content in static_files.items():
                    try:
                        static_file = build_path / static_path
                        static_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        if isinstance(static_content, str):
                            static_file.write_text(static_content, encoding='utf-8')
                        else:
                            static_file.write_bytes(static_content)
                        
                        files_generated.append(str(static_path))
                        
                    except Exception as e:
                        warnings.append(f"Could not copy static file '{static_path}': {str(e)}")
                
                # Update template usage statistics
                template.usage_count += 1
                template.last_used_at = datetime.utcnow()
                await self.db.commit()
                
                # Calculate build metrics
                build_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Basic quality metrics
                quality_metrics = {
                    'files_generated': len(files_generated),
                    'template_variables_used': len([
                        var for var in build_config.keys() 
                        if var in str(template.template_files)
                    ]),
                    'build_time': build_time,
                    'success_rate': (len(files_generated) - len(errors)) / max(len(files_generated), 1)
                }
                
                return TemplateBuildResult(
                    success=len(errors) == 0,
                    template_id=template_id,
                    build_time=build_time,
                    files_generated=files_generated,
                    warnings=warnings,
                    errors=errors,
                    quality_metrics=quality_metrics
                )
                
        except Exception as e:
            logger.error(f"Error building template: {e}")
            build_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TemplateBuildResult(
                success=False,
                template_id=template_id,
                build_time=build_time,
                files_generated=[],
                warnings=[],
                errors=[str(e)],
                quality_metrics={}
            )
    
    async def get_template_variables(self, template_id: UUID) -> Dict[str, Any]:
        """
        Extract variables from template for UI generation.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template variables and their metadata
        """
        try:
            template = await self.repository.get_template_by_id(template_id)
            if not template:
                raise NotFoundError("Template not found")
            
            variables = {}
            
            # Extract variables from template files
            for file_path, file_content in template.template_files.items():
                try:
                    # Parse template to find variables
                    ast = self.jinja_env.parse(file_content)
                    template_vars = meta.find_undeclared_variables(ast)
                    
                    for var in template_vars:
                        if var not in variables:
                            variables[var] = {
                                'name': var,
                                'type': 'string',  # Default type
                                'required': True,
                                'description': f'Variable used in {file_path}',
                                'default': None,
                                'files': [file_path]
                            }
                        else:
                            variables[var]['files'].append(file_path)
                            
                except Exception as e:
                    logger.warning(f"Could not parse template file '{file_path}': {e}")
            
            # Enhance with config metadata if available
            config_vars = template.template_config.get('variables', {})
            for var_name, var_config in config_vars.items():
                if var_name in variables:
                    variables[var_name].update(var_config)
                else:
                    variables[var_name] = {
                        'name': var_name,
                        **var_config,
                        'files': []
                    }
            
            return variables
            
        except Exception as e:
            logger.error(f"Error getting template variables: {e}")
            raise
    
    async def validate_template_build(
        self,
        template_id: UUID,
        build_config: Dict[str, Any]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate template build configuration.
        
        Args:
            template_id: Template ID
            build_config: Build configuration
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        try:
            # Get required variables
            template_vars = await self.get_template_variables(template_id)
            
            errors = []
            warnings = []
            
            # Check required variables
            required_vars = {k: v for k, v in template_vars.items() if v.get('required', False)}
            for var_name, var_info in required_vars.items():
                if var_name not in build_config:
                    errors.append(f"Required variable '{var_name}' is missing")
                elif not build_config[var_name] and var_info.get('type') != 'boolean':
                    errors.append(f"Required variable '{var_name}' cannot be empty")
            
            # Type validation
            for var_name, value in build_config.items():
                if var_name in template_vars:
                    var_info = template_vars[var_name]
                    expected_type = var_info.get('type', 'string')
                    
                    if expected_type == 'integer' and not isinstance(value, int):
                        try:
                            int(value)
                        except (ValueError, TypeError):
                            errors.append(f"Variable '{var_name}' must be an integer")
                    
                    elif expected_type == 'boolean' and not isinstance(value, bool):
                        if str(value).lower() not in ['true', 'false', '1', '0']:
                            warnings.append(f"Variable '{var_name}' should be boolean")
                    
                    elif expected_type == 'list' and not isinstance(value, list):
                        warnings.append(f"Variable '{var_name}' should be a list")
            
            # Unused variables warning
            for var_name in build_config.keys():
                if var_name not in template_vars and not var_name.startswith('_'):
                    warnings.append(f"Variable '{var_name}' is not used in template")
            
            return len(errors) == 0, errors, warnings
            
        except Exception as e:
            logger.error(f"Error validating template build: {e}")
            return False, [str(e)], []
    
    async def _generate_unique_slug(self, name: str) -> str:
        """Generate unique slug from template name."""
        base_slug = name.lower().replace(' ', '-').replace('_', '-')
        # Remove special characters
        base_slug = ''.join(c for c in base_slug if c.isalnum() or c == '-')
        
        # Check uniqueness
        counter = 1
        slug = base_slug
        while await self.repository.get_template_by_slug(slug):
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        return slug
    
    async def _validate_template_structure(self, template_data: Dict[str, Any]) -> None:
        """Validate template structure and required fields."""
        required_fields = ['name', 'description']
        for field in required_fields:
            if field not in template_data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Additional validation logic here
        if 'directories' in template_data and not isinstance(template_data['directories'], list):
            raise ValidationError("'directories' must be a list")
        
        if 'files' in template_data and not isinstance(template_data['files'], dict):
            raise ValidationError("'files' must be a dictionary")
    
    async def _process_template_files(self, template_files: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate template files."""
        processed_files = {}
        
        for file_path, file_content in template_files.items():
            # Validate file path
            if '..' in file_path or file_path.startswith('/'):
                raise ValidationError(f"Invalid file path: {file_path}")
            
            # Process file content
            if isinstance(file_content, str):
                # Validate Jinja2 syntax
                try:
                    self.jinja_env.parse(file_content)
                    processed_files[file_path] = file_content
                except Exception as e:
                    raise ValidationError(f"Invalid template syntax in {file_path}: {str(e)}")
            else:
                processed_files[file_path] = file_content
        
        return processed_files
    
    async def _calculate_initial_quality_score(self, template: Template) -> float:
        """Calculate initial quality score based on template completeness."""
        score = 0.0
        max_score = 100.0
        
        # Basic completeness checks
        if template.description:
            score += 10
        if template.short_description:
            score += 5
        if template.tags:
            score += 10
        if template.categories:
            score += 10
        if template.template_files:
            score += 20
        if template.template_config:
            score += 10
        if template.dependencies:
            score += 5
        
        # File structure analysis
        if template.template_files:
            has_readme = any('readme' in path.lower() for path in template.template_files.keys())
            has_config = any(
                path.endswith(('.json', '.yaml', '.yml', '.toml')) 
                for path in template.template_files.keys()
            )
            
            if has_readme:
                score += 15
            if has_config:
                score += 10
        
        # License check
        if template.license_type and template.license_type != "":
            score += 5
        
        return min(score, max_score)
    
    async def _create_template_version(self, template: Template, changelog: str) -> TemplateVersion:
        """Create a new template version record."""
        version = TemplateVersion(
            template_id=template.id,
            version=template.version,
            changelog=changelog,
            template_data=template.template_data,
            template_files=template.template_files,
            template_config=template.template_config,
            is_stable=True
        )
        
        self.db.add(version)
        await self.db.commit()
        return version
    
    async def _schedule_template_validation(self, template_id: UUID) -> None:
        """Schedule template validation (would integrate with task queue)."""
        # This would integrate with your task queue system (Celery, RQ, etc.)
        logger.info(f"Scheduled validation for template {template_id}")
    
    async def _apply_template_variables(
        self,
        template_data: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply template variables to template data."""
        # Simple string replacement for now
        # Could be enhanced with more sophisticated templating
        data_str = json.dumps(template_data)
        
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            data_str = data_str.replace(placeholder, str(var_value))
        
        return json.loads(data_str)