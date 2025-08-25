"""
Community Integration - Integration layer for community features with existing Claude-TIU system.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from .services.marketplace_service import MarketplaceService
from .services.template_service import TemplateService
from ..core.project_manager import ProjectManager
from ..core.types import Project

logger = logging.getLogger(__name__)


class CommunityIntegration:
    """Integration layer for community features with core Claude-TIU system."""
    
    def __init__(self, db: AsyncSession, project_manager: ProjectManager):
        """
        Initialize community integration.
        
        Args:
            db: Database session
            project_manager: Core project manager
        """
        self.db = db
        self.project_manager = project_manager
        self.marketplace_service = MarketplaceService(db)
        self.template_service = TemplateService(db)
    
    async def create_template_from_project(
        self,
        project: Project,
        template_name: str,
        description: str,
        author_id: UUID,
        is_public: bool = True,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a marketplace template from an existing project.
        
        Args:
            project: Source project
            template_name: Template name
            description: Template description
            author_id: Author user ID
            is_public: Whether template should be public
            tags: Template tags
            categories: Template categories
            
        Returns:
            Created template data or None if failed
        """
        try:
            if not project.path or not project.path.exists():
                logger.error(f"Project path does not exist: {project.path}")
                return None
            
            # Extract project structure and files
            template_files = await self._extract_project_files(project.path)
            
            # Create template configuration from project config
            template_config = await self._create_template_config_from_project(project)
            
            # Prepare template data
            from .models.template import TemplateCreate, TemplateType, ComplexityLevel
            
            # Determine template type from project
            template_type = await self._determine_template_type(project)
            complexity_level = await self._determine_complexity_level(project)
            
            template_data = TemplateCreate(
                name=template_name,
                description=description,
                template_data={
                    'name': template_name,
                    'description': description,
                    'directories': await self._extract_directory_structure(project.path),
                    'files': template_files
                },
                template_files=template_files,
                template_config=template_config,
                template_type=template_type,
                complexity_level=complexity_level,
                tags=tags or [],
                categories=categories or [],
                frameworks=await self._detect_frameworks(project),
                languages=await self._detect_languages(project),
                is_public=is_public,
                dependencies=await self._extract_dependencies(project),
                requirements=await self._extract_requirements(project)
            )
            
            # Create template
            template = await self.template_service.create_template(template_data, author_id)
            
            logger.info(f"Created template '{template_name}' from project '{project.name}'")
            return template.to_dict()
            
        except Exception as e:
            logger.error(f"Failed to create template from project: {e}")
            return None
    
    async def apply_template_to_project(
        self,
        template_id: UUID,
        project_name: str,
        build_config: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Optional[Project]:
        """
        Apply a marketplace template to create a new project.
        
        Args:
            template_id: Template ID
            project_name: New project name
            build_config: Template build configuration
            output_path: Optional output path
            
        Returns:
            Created project or None if failed
        """
        try:
            # Build template
            build_result = await self.template_service.build_template(
                template_id, build_config, output_path
            )
            
            if not build_result.success:
                logger.error(f"Template build failed: {build_result.errors}")
                return None
            
            # Create project from built template
            project = await self.project_manager.create_project(
                name=project_name,
                template="community",  # Use community template type
                project_path=output_path
            )
            
            logger.info(f"Created project '{project_name}' from template {template_id}")
            return project
            
        except Exception as e:
            logger.error(f"Failed to apply template to project: {e}")
            return None
    
    async def get_recommended_templates_for_project(
        self,
        project: Project,
        user_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get template recommendations based on project characteristics.
        
        Args:
            project: Project for recommendations
            user_id: User ID for personalization
            limit: Number of recommendations
            
        Returns:
            List of recommended templates
        """
        try:
            # Analyze project to determine relevant filters
            project_languages = await self._detect_languages(project)
            project_frameworks = await self._detect_frameworks(project)
            
            # Get recommendations from marketplace
            recommendations = await self.marketplace_service.get_template_recommendations(
                user_id=user_id,
                limit=limit
            )
            
            # Filter and rank based on project characteristics
            relevant_recommendations = []
            for template in recommendations:
                relevance_score = await self._calculate_template_relevance(
                    template, project_languages, project_frameworks
                )
                
                if relevance_score > 0.3:  # Minimum relevance threshold
                    template['relevance_score'] = relevance_score
                    relevant_recommendations.append(template)
            
            # Sort by relevance score
            relevant_recommendations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return relevant_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recommendations for project: {e}")
            return []
    
    async def sync_project_with_template(
        self,
        project: Project,
        template_id: UUID,
        sync_config: Dict[str, Any]
    ) -> bool:
        """
        Sync project changes with template (for template authors).
        
        Args:
            project: Source project
            template_id: Template to update
            sync_config: Sync configuration
            
        Returns:
            True if sync successful
        """
        try:
            # Get template
            template = await self.template_service.repository.get_template_by_id(template_id)
            if not template:
                return False
            
            # Extract updated files from project
            updated_files = await self._extract_project_files(project.path)
            
            # Create update data
            from .models.template import TemplateUpdate
            
            update_data = TemplateUpdate(
                template_files=updated_files,
                template_config=await self._create_template_config_from_project(project)
            )
            
            # Update template (this will create a new version)
            await self.template_service.update_template(
                template_id, update_data, template.author_id
            )
            
            logger.info(f"Synced project '{project.name}' with template {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync project with template: {e}")
            return False
    
    async def _extract_project_files(self, project_path: Path) -> Dict[str, str]:
        """Extract files from project path for template creation."""
        files = {}
        
        # Common file patterns to include
        include_patterns = [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.vue", "*.html", "*.css",
            "*.json", "*.yaml", "*.yml", "*.toml", "*.md", "*.txt",
            "requirements.txt", "package.json", "Dockerfile", "docker-compose.yml",
            ".gitignore", "README.md", "LICENSE"
        ]
        
        # Exclude patterns
        exclude_patterns = [
            "__pycache__", "node_modules", ".git", ".venv", "venv",
            "*.pyc", "*.pyo", ".DS_Store"
        ]
        
        for pattern in include_patterns:
            for file_path in project_path.rglob(pattern):
                if file_path.is_file():
                    # Check if file should be excluded
                    should_exclude = any(
                        exclude in str(file_path) for exclude in exclude_patterns
                    )
                    
                    if not should_exclude:
                        try:
                            relative_path = file_path.relative_to(project_path)
                            files[str(relative_path)] = file_path.read_text(encoding='utf-8')
                        except (UnicodeDecodeError, PermissionError):
                            # Skip binary or inaccessible files
                            continue
        
        return files
    
    async def _extract_directory_structure(self, project_path: Path) -> List[str]:
        """Extract directory structure from project."""
        directories = []
        
        for item in project_path.rglob("*"):
            if item.is_dir():
                # Skip hidden and common exclusion directories
                if not any(part.startswith('.') or part in ['__pycache__', 'node_modules', 'venv'] 
                          for part in item.parts):
                    relative_path = item.relative_to(project_path)
                    directories.append(str(relative_path))
        
        return directories
    
    async def _create_template_config_from_project(self, project: Project) -> Dict[str, Any]:
        """Create template configuration from project."""
        config = {
            'version': '1.0.0',
            'source_project': {
                'id': str(project.id),
                'name': project.name,
                'created_at': project.created_at.isoformat() if project.created_at else None
            }
        }
        
        if project.config:
            config.update({
                'framework': project.config.framework,
                'features': getattr(project.config, 'features', [])
            })
        
        return config
    
    async def _determine_template_type(self, project: Project) -> 'TemplateType':
        """Determine template type from project characteristics."""
        from .models.template import TemplateType
        
        # Simple heuristics based on project structure
        if project.config and hasattr(project.config, 'framework'):
            framework = project.config.framework.lower()
            if framework in ['fastapi', 'django', 'flask']:
                return TemplateType.PROJECT
            elif framework in ['react', 'vue', 'angular']:
                return TemplateType.PROJECT
        
        return TemplateType.PROJECT  # Default
    
    async def _determine_complexity_level(self, project: Project) -> 'ComplexityLevel':
        """Determine complexity level from project characteristics."""
        from .models.template import ComplexityLevel
        
        # Simple heuristics based on file count and features
        if not project.path or not project.path.exists():
            return ComplexityLevel.BEGINNER
        
        file_count = sum(1 for _ in project.path.rglob("*.py"))  # Count Python files
        
        if file_count < 5:
            return ComplexityLevel.BEGINNER
        elif file_count < 20:
            return ComplexityLevel.INTERMEDIATE
        elif file_count < 50:
            return ComplexityLevel.ADVANCED
        else:
            return ComplexityLevel.EXPERT
    
    async def _detect_frameworks(self, project: Project) -> List[str]:
        """Detect frameworks used in project."""
        frameworks = []
        
        if project.config and hasattr(project.config, 'framework'):
            frameworks.append(project.config.framework)
        
        # Additional detection logic could be added here
        if project.path and project.path.exists():
            # Check for common framework files
            if (project.path / "package.json").exists():
                # Could parse package.json for framework dependencies
                frameworks.append("nodejs")
            
            if (project.path / "requirements.txt").exists():
                frameworks.append("python")
        
        return list(set(frameworks))  # Remove duplicates
    
    async def _detect_languages(self, project: Project) -> List[str]:
        """Detect programming languages used in project."""
        languages = []
        
        if not project.path or not project.path.exists():
            return languages
        
        # Map file extensions to languages
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.vue': 'vue',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        for file_path in project.path.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in extension_map:
                    languages.append(extension_map[suffix])
        
        return list(set(languages))  # Remove duplicates
    
    async def _extract_dependencies(self, project: Project) -> List[str]:
        """Extract project dependencies."""
        dependencies = []
        
        if not project.path or not project.path.exists():
            return dependencies
        
        # Python requirements.txt
        req_file = project.path / "requirements.txt"
        if req_file.exists():
            try:
                content = req_file.read_text()
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before version specifier)
                        package = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0]
                        dependencies.append(package.strip())
            except Exception:
                pass
        
        # Node.js package.json
        package_file = project.path / "package.json"
        if package_file.exists():
            try:
                import json
                content = json.loads(package_file.read_text())
                if 'dependencies' in content:
                    dependencies.extend(content['dependencies'].keys())
                if 'devDependencies' in content:
                    dependencies.extend(content['devDependencies'].keys())
            except Exception:
                pass
        
        return list(set(dependencies))  # Remove duplicates
    
    async def _extract_requirements(self, project: Project) -> Dict[str, Any]:
        """Extract project requirements and metadata."""
        requirements = {}
        
        if project.config:
            requirements.update({
                'framework': getattr(project.config, 'framework', None),
                'features': getattr(project.config, 'features', [])
            })
        
        return requirements
    
    async def _calculate_template_relevance(
        self,
        template: Dict[str, Any],
        project_languages: List[str],
        project_frameworks: List[str]
    ) -> float:
        """Calculate relevance score between template and project."""
        score = 0.0
        max_score = 1.0
        
        # Language match
        template_languages = template.get('languages', [])
        if template_languages and project_languages:
            language_overlap = len(set(template_languages) & set(project_languages))
            language_score = language_overlap / len(set(template_languages) | set(project_languages))
            score += language_score * 0.4
        
        # Framework match
        template_frameworks = template.get('frameworks', [])
        if template_frameworks and project_frameworks:
            framework_overlap = len(set(template_frameworks) & set(project_frameworks))
            framework_score = framework_overlap / len(set(template_frameworks) | set(project_frameworks))
            score += framework_score * 0.6
        
        return min(score, max_score)