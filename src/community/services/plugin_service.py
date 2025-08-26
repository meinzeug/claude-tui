"""
Plugin Service - Plugin management and marketplace operations.
"""

import asyncio
import hashlib
import logging
import os
import zipfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import aiohttp
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.plugin import (
    Plugin, PluginDependency, PluginInstall, PluginReview, PluginSecurityScan,
    PluginSearchFilters, PluginStatus
)
from ..models.rating import UserReputation
from ...core.exceptions import ValidationError, NotFoundError, PermissionError
from ...security.code_sandbox import SecureCodeSandbox as CodeSandbox

logger = logging.getLogger(__name__)


class PluginService:
    """Service for plugin management and marketplace operations."""
    
    def __init__(self, db: AsyncSession, storage_path: str = "/tmp/plugin_storage"):
        """Initialize plugin service."""
        self.db = db
        self.storage_path = storage_path
        self.code_sandbox = CodeSandbox()
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
    
    async def create_plugin(
        self,
        plugin_data: Dict[str, Any],
        author_id: UUID,
        plugin_file: Optional[bytes] = None
    ) -> Plugin:
        """Create a new plugin."""
        try:
            # Create plugin record
            plugin = Plugin(
                slug=plugin_data["slug"],
                name=plugin_data["name"],
                description=plugin_data.get("description"),
                short_description=plugin_data.get("short_description"),
                plugin_type=plugin_data["plugin_type"],
                version=plugin_data.get("version", "1.0.0"),
                author_id=author_id,
                homepage_url=plugin_data.get("homepage_url"),
                repository_url=plugin_data.get("repository_url"),
                documentation_url=plugin_data.get("documentation_url"),
                categories=plugin_data.get("categories", []),
                tags=plugin_data.get("tags", []),
                compatibility=plugin_data.get("compatibility", {}),
                dependencies=plugin_data.get("dependencies", []),
                permissions=plugin_data.get("permissions", []),
                install_command=plugin_data.get("install_command"),
                is_premium=plugin_data.get("is_premium", False),
                price=plugin_data.get("price", 0.0),
                license_type=plugin_data.get("license_type", "MIT")
            )
            
            # Handle plugin file upload
            if plugin_file:
                file_path = await self._store_plugin_file(plugin.id, plugin_file)
                plugin.download_url = file_path
                plugin.package_size = len(plugin_file)
                
                # Schedule security scan
                await self._schedule_security_scan(plugin.id, file_path)
            
            self.db.add(plugin)
            await self.db.commit()
            
            # Create dependencies
            if plugin_data.get("dependencies"):
                await self._create_dependencies(plugin.id, plugin_data["dependencies"])
            
            return plugin
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating plugin: {e}")
            raise ValidationError(f"Failed to create plugin: {str(e)}")
    
    async def search_plugins(
        self,
        filters: PluginSearchFilters,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Search plugins with filtering."""
        try:
            # Build base query
            query = self.db.query(Plugin).filter(
                and_(
                    Plugin.is_public == True,
                    Plugin.status == PluginStatus.PUBLISHED.value
                )
            )
            
            # Apply filters
            if filters.query:
                search_terms = filters.query.lower().split()
                for term in search_terms:
                    query = query.filter(
                        or_(
                            Plugin.name.ilike(f'%{term}%'),
                            Plugin.description.ilike(f'%{term}%'),
                            Plugin.short_description.ilike(f'%{term}%'),
                            Plugin.tags.op('?')(term)
                        )
                    )
            
            if filters.plugin_type:
                query = query.filter(Plugin.plugin_type == filters.plugin_type.value)
            
            if filters.categories:
                for category in filters.categories:
                    query = query.filter(Plugin.categories.op('?')(category))
            
            if filters.tags:
                for tag in filters.tags:
                    query = query.filter(Plugin.tags.op('?')(tag))
            
            if filters.is_featured is not None:
                query = query.filter(Plugin.is_featured == filters.is_featured)
            
            if filters.is_free is not None:
                query = query.filter(Plugin.is_premium != filters.is_free)
            
            if filters.is_verified is not None:
                query = query.filter(Plugin.is_verified == filters.is_verified)
            
            if filters.min_rating:
                query = query.filter(Plugin.average_rating >= filters.min_rating)
            
            if filters.max_rating:
                query = query.filter(Plugin.average_rating <= filters.max_rating)
            
            if filters.author_id:
                query = query.filter(Plugin.author_id == filters.author_id)
            
            # Count total results
            total_count = await query.count()
            
            # Apply sorting
            sort_column = getattr(Plugin, filters.sort_by, Plugin.updated_at)
            if filters.sort_order == "asc":
                query = query.order_by(sort_column)
            else:
                query = query.order_by(desc(sort_column))
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute query with relationships
            query = query.options(selectinload(Plugin.author))
            plugins = await query.all()
            
            # Convert to dictionaries
            plugin_dicts = []
            for plugin in plugins:
                plugin_dict = plugin.to_dict()
                plugin_dict['author'] = plugin.author.to_dict(include_private=False) if plugin.author else None
                plugin_dicts.append(plugin_dict)
            
            return plugin_dicts, total_count
            
        except Exception as e:
            logger.error(f"Error searching plugins: {e}")
            raise ValidationError(f"Search failed: {str(e)}")
    
    async def get_plugin_by_id(self, plugin_id: UUID) -> Optional[Plugin]:
        """Get plugin by ID."""
        try:
            query = self.db.query(Plugin).filter(Plugin.id == plugin_id)
            query = query.options(selectinload(Plugin.author))
            return await query.first()
            
        except Exception as e:
            logger.error(f"Error getting plugin by ID: {e}")
            raise
    
    async def install_plugin(
        self,
        plugin_id: UUID,
        user_id: UUID,
        installation_method: str = "manual"
    ) -> PluginInstall:
        """Install a plugin for a user."""
        try:
            # Get plugin
            plugin = await self.get_plugin_by_id(plugin_id)
            if not plugin:
                raise NotFoundError("Plugin not found")
            
            if not plugin.is_public or plugin.status != PluginStatus.PUBLISHED.value:
                raise PermissionError("Plugin is not available for installation")
            
            # Check if already installed
            existing_install = await self.db.query(PluginInstall).filter(
                and_(
                    PluginInstall.plugin_id == plugin_id,
                    PluginInstall.user_id == user_id,
                    PluginInstall.is_active == True
                )
            ).first()
            
            if existing_install:
                existing_install.last_used_at = datetime.utcnow()
                await self.db.commit()
                return existing_install
            
            # Create installation record
            install = PluginInstall(
                plugin_id=plugin_id,
                user_id=user_id,
                version_installed=plugin.version,
                installation_method=installation_method,
                client_info={}  # Could be populated with request info
            )
            
            self.db.add(install)
            
            # Update plugin statistics
            plugin.install_count += 1
            plugin.active_installs += 1
            
            await self.db.commit()
            
            return install
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error installing plugin: {e}")
            raise
    
    async def uninstall_plugin(self, plugin_id: UUID, user_id: UUID) -> bool:
        """Uninstall a plugin for a user."""
        try:
            # Find active installation
            install = await self.db.query(PluginInstall).filter(
                and_(
                    PluginInstall.plugin_id == plugin_id,
                    PluginInstall.user_id == user_id,
                    PluginInstall.is_active == True
                )
            ).first()
            
            if not install:
                return False
            
            # Mark as uninstalled
            install.is_active = False
            install.uninstalled_at = datetime.utcnow()
            
            # Update plugin statistics
            plugin = await self.get_plugin_by_id(plugin_id)
            if plugin and plugin.active_installs > 0:
                plugin.active_installs -= 1
            
            await self.db.commit()
            
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error uninstalling plugin: {e}")
            raise
    
    async def get_user_installed_plugins(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get plugins installed by a user."""
        try:
            query = self.db.query(PluginInstall).filter(
                and_(
                    PluginInstall.user_id == user_id,
                    PluginInstall.is_active == True
                )
            ).options(selectinload(PluginInstall.plugin))
            
            installs = await query.all()
            
            installed_plugins = []
            for install in installs:
                plugin_dict = install.plugin.to_dict()
                plugin_dict['installation_info'] = {
                    'installed_at': install.installed_at.isoformat(),
                    'version_installed': install.version_installed,
                    'last_used_at': install.last_used_at.isoformat() if install.last_used_at else None
                }
                installed_plugins.append(plugin_dict)
            
            return installed_plugins
            
        except Exception as e:
            logger.error(f"Error getting user installed plugins: {e}")
            raise
    
    async def check_for_updates(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Check for plugin updates for a user."""
        try:
            # Get user's installed plugins
            query = self.db.query(PluginInstall).filter(
                and_(
                    PluginInstall.user_id == user_id,
                    PluginInstall.is_active == True
                )
            ).options(selectinload(PluginInstall.plugin))
            
            installs = await query.all()
            
            updates_available = []
            for install in installs:
                plugin = install.plugin
                if self._version_compare(install.version_installed, plugin.version) < 0:
                    updates_available.append({
                        'plugin_id': str(plugin.id),
                        'plugin_name': plugin.name,
                        'current_version': install.version_installed,
                        'latest_version': plugin.version,
                        'update_available': True
                    })
            
            return updates_available
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            raise
    
    async def perform_security_scan(self, plugin_id: UUID) -> PluginSecurityScan:
        """Perform security scan on a plugin."""
        try:
            plugin = await self.get_plugin_by_id(plugin_id)
            if not plugin or not plugin.download_url:
                raise NotFoundError("Plugin or plugin file not found")
            
            # Create scan record
            scan = PluginSecurityScan(
                plugin_id=plugin_id,
                scan_version=plugin.version,
                scan_type="automated",
                status="pending"
            )
            
            self.db.add(scan)
            await self.db.commit()
            
            # Perform actual security scan
            scan_results = await self._run_security_scan(plugin.download_url)
            
            # Update scan results
            scan.status = "completed"
            scan.overall_score = scan_results["overall_score"]
            scan.risk_level = scan_results["risk_level"]
            scan.vulnerabilities = scan_results["vulnerabilities"]
            scan.security_issues = scan_results["security_issues"]
            scan.code_quality_issues = scan_results["code_quality_issues"]
            scan.scan_duration = scan_results["scan_duration"]
            scan.files_scanned = scan_results["files_scanned"]
            scan.completed_at = datetime.utcnow()
            
            # Update plugin security status
            plugin.security_scan_status = "completed"
            plugin.security_scan_date = datetime.utcnow()
            plugin.security_issues_count = len(scan_results["security_issues"])
            plugin.is_security_approved = scan_results["overall_score"] >= 80.0
            
            await self.db.commit()
            
            return scan
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error performing security scan: {e}")
            raise
    
    async def create_plugin_review(
        self,
        plugin_id: UUID,
        user_id: UUID,
        review_data: Dict[str, Any]
    ) -> PluginReview:
        """Create a plugin review."""
        try:
            # Check if user has already reviewed this plugin
            existing_review = await self.db.query(PluginReview).filter(
                and_(
                    PluginReview.plugin_id == plugin_id,
                    PluginReview.user_id == user_id
                )
            ).first()
            
            if existing_review:
                raise ValidationError("User has already reviewed this plugin")
            
            # Check if user has installed the plugin
            install = await self.db.query(PluginInstall).filter(
                and_(
                    PluginInstall.plugin_id == plugin_id,
                    PluginInstall.user_id == user_id
                )
            ).first()
            
            is_verified = install is not None
            
            # Create review
            review = PluginReview(
                plugin_id=plugin_id,
                user_id=user_id,
                rating=review_data["rating"],
                title=review_data.get("title"),
                content=review_data.get("content"),
                plugin_version_reviewed=review_data.get("plugin_version_reviewed"),
                is_verified_purchase=is_verified
            )
            
            self.db.add(review)
            
            # Update plugin rating statistics
            await self._update_plugin_rating_stats(plugin_id)
            
            await self.db.commit()
            
            return review
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating plugin review: {e}")
            raise
    
    async def _store_plugin_file(self, plugin_id: UUID, file_data: bytes) -> str:
        """Store plugin file and return path."""
        try:
            # Create file path
            filename = f"{plugin_id}.zip"
            file_path = os.path.join(self.storage_path, filename)
            
            # Validate it's a valid zip file
            try:
                with zipfile.ZipFile(io.BytesIO(file_data), 'r') as zip_file:
                    # Basic validation
                    if len(zip_file.namelist()) == 0:
                        raise ValidationError("Empty zip file")
            except zipfile.BadZipFile:
                raise ValidationError("Invalid zip file format")
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error storing plugin file: {e}")
            raise
    
    async def _schedule_security_scan(self, plugin_id: UUID, file_path: str) -> None:
        """Schedule security scan for a plugin."""
        # In a real implementation, this would queue the scan job
        # For now, we'll perform it immediately
        asyncio.create_task(self.perform_security_scan(plugin_id))
    
    async def _run_security_scan(self, file_path: str) -> Dict[str, Any]:
        """Run actual security scan on plugin file."""
        start_time = datetime.utcnow()
        
        try:
            # Use code sandbox for security analysis
            scan_results = await self.code_sandbox.analyze_code_archive(file_path)
            
            scan_duration = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "overall_score": scan_results.get("security_score", 50.0),
                "risk_level": self._determine_risk_level(scan_results.get("security_score", 50.0)),
                "vulnerabilities": scan_results.get("vulnerabilities", []),
                "security_issues": scan_results.get("security_issues", []),
                "code_quality_issues": scan_results.get("quality_issues", []),
                "scan_duration": int(scan_duration),
                "files_scanned": scan_results.get("files_analyzed", 0)
            }
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return {
                "overall_score": 0.0,
                "risk_level": "critical",
                "vulnerabilities": [{"type": "scan_error", "message": str(e)}],
                "security_issues": [],
                "code_quality_issues": [],
                "scan_duration": 0,
                "files_scanned": 0
            }
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on security score."""
        if score >= 80:
            return "low"
        elif score >= 60:
            return "medium"
        elif score >= 40:
            return "high"
        else:
            return "critical"
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare two semantic versions. Returns -1, 0, or 1."""
        def version_tuple(v):
            return tuple(map(int, (v.split("."))))
        
        v1_tuple = version_tuple(version1)
        v2_tuple = version_tuple(version2)
        
        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0
    
    async def _update_plugin_rating_stats(self, plugin_id: UUID) -> None:
        """Update plugin rating statistics."""
        try:
            # Get all reviews for the plugin
            reviews_query = self.db.query(PluginReview).filter(
                and_(
                    PluginReview.plugin_id == plugin_id,
                    PluginReview.is_approved == True
                )
            )
            
            reviews = await reviews_query.all()
            
            if not reviews:
                return
            
            # Calculate statistics
            total_reviews = len(reviews)
            total_rating = sum(review.rating for review in reviews)
            average_rating = total_rating / total_reviews
            
            # Update plugin
            plugin = await self.get_plugin_by_id(plugin_id)
            if plugin:
                plugin.rating_count = total_reviews
                plugin.average_rating = average_rating
            
        except Exception as e:
            logger.error(f"Error updating plugin rating stats: {e}")
    
    async def _create_dependencies(
        self,
        plugin_id: UUID,
        dependencies: List[str]
    ) -> None:
        """Create plugin dependencies."""
        try:
            for dep_name in dependencies:
                dependency = PluginDependency(
                    plugin_id=plugin_id,
                    dependency_name=dep_name,
                    dependency_type="required",
                    is_external=True
                )
                self.db.add(dependency)
            
            await self.db.commit()
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating dependencies: {e}")
            raise
