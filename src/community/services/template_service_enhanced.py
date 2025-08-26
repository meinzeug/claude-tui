"""
Enhanced Template Service - Advanced template sharing with version control and collaboration.

Features:
- Git-based version control for templates
- Template forking and merging
- Collaborative editing
- Template collections and curation
- Advanced search and discovery
- Template analytics and usage tracking
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import git
from sqlalchemy import desc, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.template import Template, TemplateVersion, TemplateCollaboration
from ..models.marketplace import TemplateFork, TemplateStar, TemplateDownload, TemplateView
from ..models.user import UserProfile
from ..repositories.template_repository import TemplateRepository
from ...core.exceptions import ValidationError, NotFoundError, PermissionError
from ...security.input_validator import InputValidator
from ...integrations.git_advanced import AdvancedGitManager

logger = logging.getLogger(__name__)


class TemplateVersionControl:
    """Template version control system using Git."""
    
    def __init__(self, base_path: str = "/tmp/claude_tui_templates"):
        """Initialize version control system."""
        self.base_path = base_path
        self.git_manager = AdvancedGitManager()
        os.makedirs(base_path, exist_ok=True)
    
    async def initialize_template_repo(self, template_id: UUID) -> str:
        """Initialize Git repository for template."""
        repo_path = os.path.join(self.base_path, str(template_id))
        
        if os.path.exists(repo_path):
            return repo_path
        
        # Initialize repository
        repo = git.Repo.init(repo_path)
        
        # Create initial structure
        os.makedirs(os.path.join(repo_path, "src"), exist_ok=True)
        os.makedirs(os.path.join(repo_path, "docs"), exist_ok=True)
        os.makedirs(os.path.join(repo_path, "examples"), exist_ok=True)
        
        # Create template metadata file
        metadata = {
            "template_id": str(template_id),
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
        
        with open(os.path.join(repo_path, "template.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create README
        readme_content = f"""# Template {template_id}

This is a Claude-TUI template with version control support.

## Structure
- `src/` - Template source files
- `docs/` - Documentation
- `examples/` - Usage examples
- `template.json` - Template metadata

## Version Control
This template supports version control through Git integration.
"""
        
        with open(os.path.join(repo_path, "README.md"), "w") as f:
            f.write(readme_content)
        
        # Initial commit
        repo.index.add(["."])
        repo.index.commit("Initial template commit")
        
        return repo_path
    
    async def create_version(
        self, 
        template_id: UUID, 
        version: str, 
        changes: List[Dict[str, Any]], 
        commit_message: str,
        author_name: str,
        author_email: str
    ) -> str:
        """Create a new template version."""
        repo_path = await self.initialize_template_repo(template_id)
        repo = git.Repo(repo_path)
        
        # Apply changes
        for change in changes:
            file_path = os.path.join(repo_path, change["path"])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if change["operation"] == "create" or change["operation"] == "update":
                with open(file_path, "w") as f:
                    f.write(change["content"])
            elif change["operation"] == "delete":
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Update template metadata
        template_json_path = os.path.join(repo_path, "template.json")
        with open(template_json_path, "r") as f:
            metadata = json.load(f)
        
        metadata["version"] = version
        metadata["updated_at"] = datetime.utcnow().isoformat()
        
        with open(template_json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Commit changes
        repo.index.add(["."])
        
        # Configure author
        with repo.config_writer() as git_config:
            git_config.set_value("user", "name", author_name)
            git_config.set_value("user", "email", author_email)
        
        commit = repo.index.commit(commit_message)
        
        # Create version tag
        repo.create_tag(f"v{version}", message=f"Version {version}")
        
        return commit.hexsha
    
    async def get_version_history(self, template_id: UUID) -> List[Dict[str, Any]]:
        """Get template version history."""
        repo_path = os.path.join(self.base_path, str(template_id))
        
        if not os.path.exists(repo_path):
            return []
        
        repo = git.Repo(repo_path)
        
        history = []
        for commit in repo.iter_commits():
            history.append({
                "commit_hash": commit.hexsha,
                "message": commit.message,
                "author": commit.author.name,
                "email": commit.author.email,
                "committed_at": datetime.fromtimestamp(commit.committed_date),
                "files_changed": len(commit.stats.files),
                "additions": commit.stats.total["insertions"],
                "deletions": commit.stats.total["deletions"]
            })
        
        return history
    
    async def create_branch(self, template_id: UUID, branch_name: str, base_commit: str = "HEAD") -> str:
        """Create a new branch for collaborative editing."""
        repo_path = os.path.join(self.base_path, str(template_id))
        repo = git.Repo(repo_path)
        
        # Create new branch
        new_branch = repo.create_head(branch_name, base_commit)
        new_branch.checkout()
        
        return new_branch.name
    
    async def merge_branch(
        self, 
        template_id: UUID, 
        source_branch: str, 
        target_branch: str = "main",
        merge_message: str = "Merge collaborative changes"
    ) -> str:
        """Merge branches."""
        repo_path = os.path.join(self.base_path, str(template_id))
        repo = git.Repo(repo_path)
        
        # Checkout target branch
        repo.heads[target_branch].checkout()
        
        # Merge source branch
        merge_base = repo.merge_base(repo.heads[target_branch], repo.heads[source_branch])[0]
        repo.index.merge_tree(repo.heads[source_branch], base=merge_base)
        
        # Commit merge
        merge_commit = repo.index.commit(
            merge_message,
            parent_commits=(repo.heads[target_branch].commit, repo.heads[source_branch].commit)
        )
        
        return merge_commit.hexsha
    
    async def get_file_diff(self, template_id: UUID, file_path: str, commit1: str, commit2: str) -> str:
        """Get diff between two commits for a specific file."""
        repo_path = os.path.join(self.base_path, str(template_id))
        repo = git.Repo(repo_path)
        
        diff = repo.git.diff(commit1, commit2, file_path)
        return diff
    
    async def fork_template(self, source_template_id: UUID, target_template_id: UUID) -> str:
        """Fork a template repository."""
        source_path = os.path.join(self.base_path, str(source_template_id))
        target_path = os.path.join(self.base_path, str(target_template_id))
        
        if not os.path.exists(source_path):
            raise NotFoundError("Source template repository not found")
        
        # Clone repository
        repo = git.Repo(source_path)
        forked_repo = repo.clone(target_path)
        
        # Update metadata for fork
        template_json_path = os.path.join(target_path, "template.json")
        with open(template_json_path, "r") as f:
            metadata = json.load(f)
        
        metadata["template_id"] = str(target_template_id)
        metadata["forked_from"] = str(source_template_id)
        metadata["forked_at"] = datetime.utcnow().isoformat()
        
        with open(template_json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Commit fork metadata
        forked_repo.index.add(["template.json"])
        forked_repo.index.commit("Fork template")
        
        return target_path


class EnhancedTemplateService:
    """Enhanced template service with advanced sharing and collaboration."""
    
    def __init__(self, db: AsyncSession):
        """Initialize enhanced template service."""
        self.db = db
        self.repository = TemplateRepository(db)
        self.version_control = TemplateVersionControl()
        self.input_validator = InputValidator()
    
    async def create_template(
        self,
        author_id: UUID,
        template_data: Dict[str, Any],
        initial_files: List[Dict[str, Any]] = None
    ) -> UUID:
        """Create a new template with version control."""
        try:
            # Validate input data
            template_data = self.input_validator.sanitize_template_data(template_data)
            
            # Create template record
            template_id = uuid4()
            template = Template(
                id=template_id,
                author_id=author_id,
                name=template_data["name"],
                description=template_data["description"],
                short_description=template_data.get("short_description"),
                template_type=template_data["template_type"],
                complexity_level=template_data.get("complexity_level", "intermediate"),
                categories=template_data.get("categories", []),
                frameworks=template_data.get("frameworks", []),
                languages=template_data.get("languages", []),
                tags=template_data.get("tags", []),
                is_public=template_data.get("is_public", True),
                is_featured=False,
                version="1.0.0"
            )
            
            self.db.add(template)
            await self.db.commit()
            
            # Initialize version control
            await self.version_control.initialize_template_repo(template_id)
            
            # Create initial version with files
            if initial_files:
                author = await self.db.query(UserProfile).filter(
                    UserProfile.id == author_id
                ).first()
                
                await self.version_control.create_version(
                    template_id=template_id,
                    version="1.0.0",
                    changes=initial_files,
                    commit_message="Initial template creation",
                    author_name=author.full_name or author.username,
                    author_email=author.email
                )
                
                # Create version record
                version_record = TemplateVersion(
                    id=uuid4(),
                    template_id=template_id,
                    version="1.0.0",
                    description="Initial version",
                    author_id=author_id,
                    changes_summary=f"Created template with {len(initial_files)} files"
                )
                
                self.db.add(version_record)
                await self.db.commit()
            
            return template_id
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Template creation failed: {e}")
            raise ValidationError(f"Template creation failed: {str(e)}")
    
    async def create_template_version(
        self,
        template_id: UUID,
        author_id: UUID,
        version: str,
        description: str,
        changes: List[Dict[str, Any]],
        is_major: bool = False
    ) -> UUID:
        """Create a new template version."""
        try:
            # Validate ownership or collaboration rights
            await self._validate_template_access(template_id, author_id)
            
            # Get author info
            author = await self.db.query(UserProfile).filter(
                UserProfile.id == author_id
            ).first()
            
            # Create version in Git
            commit_message = f"Version {version}: {description}"
            commit_hash = await self.version_control.create_version(
                template_id=template_id,
                version=version,
                changes=changes,
                commit_message=commit_message,
                author_name=author.full_name or author.username,
                author_email=author.email
            )
            
            # Create version record
            version_id = uuid4()
            version_record = TemplateVersion(
                id=version_id,
                template_id=template_id,
                version=version,
                description=description,
                author_id=author_id,
                commit_hash=commit_hash,
                is_major=is_major,
                changes_summary=f"{len(changes)} files changed"
            )
            
            self.db.add(version_record)
            
            # Update template current version
            template = await self.repository.get_by_id(template_id)
            template.version = version
            template.updated_at = datetime.utcnow()
            
            await self.db.commit()
            
            return version_id
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Version creation failed: {e}")
            raise
    
    async def fork_template(
        self,
        source_template_id: UUID,
        forker_id: UUID,
        fork_name: str,
        fork_description: str,
        changes_description: Optional[str] = None
    ) -> UUID:
        """Fork a template for modification."""
        try:
            # Get source template
            source_template = await self.repository.get_by_id(source_template_id)
            if not source_template:
                raise NotFoundError("Source template not found")
            
            if not source_template.is_public:
                raise PermissionError("Cannot fork private template")
            
            # Create forked template
            fork_id = uuid4()
            forked_template = Template(
                id=fork_id,
                author_id=forker_id,
                name=fork_name,
                description=fork_description,
                short_description=source_template.short_description,
                template_type=source_template.template_type,
                complexity_level=source_template.complexity_level,
                categories=source_template.categories,
                frameworks=source_template.frameworks,
                languages=source_template.languages,
                tags=source_template.tags,
                is_public=True,
                is_fork=True,
                parent_template_id=source_template_id,
                version="1.0.0"
            )
            
            self.db.add(forked_template)
            
            # Create fork record
            fork_record = TemplateFork(
                id=uuid4(),
                original_template_id=source_template_id,
                forked_template_id=fork_id,
                user_id=forker_id,
                fork_reason="Template customization",
                changes_description=changes_description
            )
            
            self.db.add(fork_record)
            
            # Fork repository
            await self.version_control.fork_template(source_template_id, fork_id)
            
            # Update fork count
            source_template.fork_count += 1
            
            await self.db.commit()
            
            return fork_id
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Template fork failed: {e}")
            raise
    
    async def start_collaboration(
        self,
        template_id: UUID,
        collaborator_id: UUID,
        permission_level: str = "edit",
        branch_name: Optional[str] = None
    ) -> str:
        """Start collaborative editing session."""
        try:
            # Validate template access
            template = await self.repository.get_by_id(template_id)
            if not template:
                raise NotFoundError("Template not found")
            
            # Create collaboration record
            collaboration_id = uuid4()
            collaboration = TemplateCollaboration(
                id=collaboration_id,
                template_id=template_id,
                collaborator_id=collaborator_id,
                permission_level=permission_level,
                status="active"
            )
            
            self.db.add(collaboration)
            
            # Create collaboration branch
            if not branch_name:
                branch_name = f"collab-{collaborator_id}-{int(datetime.utcnow().timestamp())}"
            
            await self.version_control.create_branch(template_id, branch_name)
            
            collaboration.branch_name = branch_name
            await self.db.commit()
            
            return branch_name
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Collaboration start failed: {e}")
            raise
    
    async def merge_collaboration(
        self,
        template_id: UUID,
        collaboration_id: UUID,
        merge_message: str = "Merge collaborative changes"
    ) -> str:
        """Merge collaborative changes back to main branch."""
        try:
            # Get collaboration record
            collaboration = await self.db.query(TemplateCollaboration).filter(
                TemplateCollaboration.id == collaboration_id
            ).first()
            
            if not collaboration:
                raise NotFoundError("Collaboration not found")
            
            # Merge branch
            commit_hash = await self.version_control.merge_branch(
                template_id=template_id,
                source_branch=collaboration.branch_name,
                merge_message=merge_message
            )
            
            # Update collaboration status
            collaboration.status = "merged"
            collaboration.merged_at = datetime.utcnow()
            
            await self.db.commit()
            
            return commit_hash
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Collaboration merge failed: {e}")
            raise
    
    async def get_template_analytics(self, template_id: UUID, author_id: UUID) -> Dict[str, Any]:
        """Get comprehensive template analytics."""
        try:
            # Validate ownership
            await self._validate_template_ownership(template_id, author_id)
            
            # Get basic metrics
            template = await self.repository.get_by_id(template_id)
            
            # Download analytics (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_downloads = await self.db.query(func.count(TemplateDownload.id)).filter(
                and_(
                    TemplateDownload.template_id == template_id,
                    TemplateDownload.downloaded_at >= thirty_days_ago
                )
            ).scalar()
            
            # View analytics
            recent_views = await self.db.query(func.count(TemplateView.id)).filter(
                and_(
                    TemplateView.template_id == template_id,
                    TemplateView.viewed_at >= thirty_days_ago
                )
            ).scalar()
            
            # Geographic distribution (mock data for demo)
            geographic_data = {
                "US": 45,
                "EU": 30,
                "ASIA": 20,
                "OTHER": 5
            }
            
            # Version analytics
            version_history = await self.version_control.get_version_history(template_id)
            
            # Star analytics
            star_count = await self.db.query(func.count(TemplateStar.template_id)).filter(
                TemplateStar.template_id == template_id
            ).scalar()
            
            # Fork analytics
            fork_count = await self.db.query(func.count(TemplateFork.original_template_id)).filter(
                TemplateFork.original_template_id == template_id
            ).scalar()
            
            return {
                "basic_metrics": {
                    "total_downloads": template.download_count,
                    "total_views": template.view_count,
                    "total_stars": star_count,
                    "total_forks": fork_count,
                    "average_rating": template.average_rating,
                    "rating_count": template.rating_count
                },
                "recent_activity": {
                    "downloads_30d": recent_downloads,
                    "views_30d": recent_views,
                    "growth_rate": ((recent_downloads / max(template.download_count - recent_downloads, 1)) * 100) if template.download_count > 0 else 0
                },
                "geographic_distribution": geographic_data,
                "version_metrics": {
                    "total_versions": len(version_history),
                    "latest_version": template.version,
                    "recent_commits": version_history[:5]
                },
                "engagement_metrics": {
                    "fork_ratio": (fork_count / max(template.download_count, 1)) * 100,
                    "star_ratio": (star_count / max(template.view_count, 1)) * 100,
                    "conversion_rate": (template.download_count / max(template.view_count, 1)) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Analytics retrieval failed: {e}")
            raise
    
    async def export_template(self, template_id: UUID, version: Optional[str] = None) -> str:
        """Export template as downloadable package."""
        try:
            template = await self.repository.get_by_id(template_id)
            if not template:
                raise NotFoundError("Template not found")
            
            # Create temporary directory for export
            temp_dir = tempfile.mkdtemp()
            export_path = os.path.join(temp_dir, f"template_{template_id}.zip")
            
            # Get template repository path
            repo_path = os.path.join(self.version_control.base_path, str(template_id))
            
            if not os.path.exists(repo_path):
                raise NotFoundError("Template repository not found")
            
            # Create ZIP archive
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(repo_path):
                    # Skip .git directory
                    if '.git' in root:
                        continue
                        
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, repo_path)
                        zipf.write(file_path, arc_name)
                
                # Add template metadata
                metadata = {
                    "template_id": str(template_id),
                    "name": template.name,
                    "description": template.description,
                    "version": template.version,
                    "author": template.author.username if template.author else "Unknown",
                    "exported_at": datetime.utcnow().isoformat(),
                    "categories": template.categories,
                    "frameworks": template.frameworks,
                    "languages": template.languages,
                    "tags": template.tags
                }
                
                zipf.writestr("template_metadata.json", json.dumps(metadata, indent=2))
            
            return export_path
            
        except Exception as e:
            logger.error(f"Template export failed: {e}")
            raise
    
    async def import_template(
        self,
        author_id: UUID,
        template_package: str,
        template_name: Optional[str] = None
    ) -> UUID:
        """Import template from package file."""
        try:
            # Extract package
            temp_dir = tempfile.mkdtemp()
            
            with zipfile.ZipFile(template_package, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Read metadata
            metadata_path = os.path.join(temp_dir, "template_metadata.json")
            metadata = {}
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Create template
            template_data = {
                "name": template_name or metadata.get("name", "Imported Template"),
                "description": metadata.get("description", "Imported template"),
                "template_type": metadata.get("template_type", "general"),
                "categories": metadata.get("categories", []),
                "frameworks": metadata.get("frameworks", []),
                "languages": metadata.get("languages", []),
                "tags": metadata.get("tags", [])
            }
            
            # Extract files for version control
            initial_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file == "template_metadata.json":
                        continue
                        
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    initial_files.append({
                        "path": rel_path,
                        "operation": "create",
                        "content": content
                    })
            
            # Create template
            template_id = await self.create_template(
                author_id=author_id,
                template_data=template_data,
                initial_files=initial_files
            )
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            return template_id
            
        except Exception as e:
            logger.error(f"Template import failed: {e}")
            raise ValidationError(f"Template import failed: {str(e)}")
    
    async def _validate_template_access(self, template_id: UUID, user_id: UUID) -> bool:
        """Validate user access to template."""
        template = await self.repository.get_by_id(template_id)
        
        if not template:
            raise NotFoundError("Template not found")
        
        # Owner has full access
        if template.author_id == user_id:
            return True
        
        # Check collaboration permissions
        collaboration = await self.db.query(TemplateCollaboration).filter(
            and_(
                TemplateCollaboration.template_id == template_id,
                TemplateCollaboration.collaborator_id == user_id,
                TemplateCollaboration.status == "active"
            )
        ).first()
        
        if collaboration:
            return True
        
        raise PermissionError("Access denied to template")
    
    async def _validate_template_ownership(self, template_id: UUID, user_id: UUID) -> bool:
        """Validate template ownership."""
        template = await self.repository.get_by_id(template_id)
        
        if not template:
            raise NotFoundError("Template not found")
        
        if template.author_id != user_id:
            raise PermissionError("Not template owner")
        
        return True