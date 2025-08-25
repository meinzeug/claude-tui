"""
Enhanced Plugin Service - Comprehensive plugin management with security scanning.
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import desc, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.plugin import (
    Plugin, PluginDependency, PluginInstall, PluginReview, 
    PluginSecurityScan, PluginStatus, PluginType
)
from ..models.user import UserProfile
from .cache_service import CacheService
from .moderation_service import ModerationService
from ...core.exceptions import ValidationError, NotFoundError, PermissionError
from ...core.logger import get_logger

logger = get_logger(__name__)


class SecurityScanner:
    """Security scanner for plugins with multiple detection methods."""
    
    def __init__(self):
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Security patterns to detect
        self.risk_patterns = {
            'high': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__',
                r'subprocess\.',
                r'os\.system',
                r'shell=True',
                r'rm\s+-rf',
                r'format\s*\(',  # String formatting vulnerabilities
            ],
            'medium': [
                r'input\s*\(',
                r'raw_input\s*\(',
                r'open\s*\(',
                r'urllib\.request',
                r'requests\.get',
                r'pickle\.',
                r'yaml\.load',
            ],
            'low': [
                r'print\s*\(',
                r'logging\.',
                r'sys\.',
                r'json\.',
            ]
        }
        
        # Dependency vulnerability database (simplified)
        self.vulnerable_deps = {
            'requests': ['2.25.1', '2.24.0'],  # Example vulnerable versions
            'pyyaml': ['5.3.1', '5.2'],
            'pillow': ['8.1.0', '8.0.1']
        }
    
    async def scan_plugin(
        self,
        plugin_id: UUID,
        plugin_content: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive security scan of a plugin.
        
        Args:
            plugin_id: Plugin ID
            plugin_content: Plugin source code content
            dependencies: List of plugin dependencies
            
        Returns:
            Security scan results
        """
        try:
            scan_results = {
                'plugin_id': str(plugin_id),
                'scan_timestamp': datetime.utcnow(),
                'overall_score': 100.0,
                'risk_level': 'low',
                'vulnerabilities': [],
                'security_issues': [],
                'code_quality_issues': [],
                'dependency_issues': [],
                'recommendations': []
            }
            
            # Code analysis if content provided
            if plugin_content:
                code_analysis = await self._analyze_code_security(plugin_content)
                scan_results.update(code_analysis)
            
            # Dependency analysis if dependencies provided
            if dependencies:
                dep_analysis = await self._analyze_dependencies(dependencies)
                scan_results['dependency_issues'] = dep_analysis
            
            # Calculate overall score and risk level
            scan_results['overall_score'] = self._calculate_security_score(scan_results)
            scan_results['risk_level'] = self._determine_risk_level(scan_results['overall_score'])
            
            # Generate recommendations
            scan_results['recommendations'] = self._generate_security_recommendations(scan_results)
            
            return scan_results
            
        except Exception as e:
            self.logger.error(f"Error scanning plugin {plugin_id}: {e}")
            return {
                'plugin_id': str(plugin_id),
                'scan_timestamp': datetime.utcnow(),
                'error': str(e),
                'overall_score': 0.0,
                'risk_level': 'unknown'
            }
    
    async def _analyze_code_security(self, code_content: str) -> Dict[str, Any]:
        """Analyze code for security vulnerabilities."""
        import re
        
        vulnerabilities = []
        security_issues = []
        code_quality_issues = []
        
        lines = code_content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for high-risk patterns
            for pattern in self.risk_patterns['high']:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    vulnerabilities.append({
                        'type': 'code_injection_risk',
                        'severity': 'high',
                        'line': line_num,
                        'pattern': pattern,
                        'description': f'Potential code injection vulnerability: {pattern}',
                        'recommendation': 'Avoid using dynamic code execution functions'
                    })
            
            # Check for medium-risk patterns
            for pattern in self.risk_patterns['medium']:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    security_issues.append({
                        'type': 'potential_vulnerability',
                        'severity': 'medium',
                        'line': line_num,
                        'pattern': pattern,
                        'description': f'Potential security risk: {pattern}',
                        'recommendation': 'Review usage and ensure proper input validation'
                    })
            
            # Code quality checks
            if len(line_stripped) > 120:
                code_quality_issues.append({
                    'type': 'line_too_long',
                    'severity': 'low',
                    'line': line_num,
                    'description': f'Line exceeds recommended length ({len(line_stripped)} chars)',
                    'recommendation': 'Break long lines for better readability'
                })
            
            # Check for hardcoded secrets/credentials
            if re.search(r'(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']', line_stripped, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'hardcoded_secret',
                    'severity': 'high',
                    'line': line_num,
                    'description': 'Potential hardcoded secret or credential',
                    'recommendation': 'Use environment variables or secure configuration for secrets'
                })
        
        return {
            'vulnerabilities': vulnerabilities,
            'security_issues': security_issues,
            'code_quality_issues': code_quality_issues
        }
    
    async def _analyze_dependencies(self, dependencies: List[str]) -> List[Dict[str, Any]]:
        """Analyze plugin dependencies for known vulnerabilities."""
        dependency_issues = []
        
        for dep in dependencies:
            # Parse dependency (name==version format)
            if '==' in dep:
                name, version = dep.split('==', 1)
            elif '>=' in dep:
                name, version = dep.split('>=', 1)
            else:
                name, version = dep, 'latest'
            
            name = name.strip().lower()
            
            # Check against vulnerability database
            if name in self.vulnerable_deps:
                vulnerable_versions = self.vulnerable_deps[name]
                if version in vulnerable_versions:
                    dependency_issues.append({
                        'dependency': dep,
                        'vulnerability': 'known_vulnerability',
                        'severity': 'high',
                        'description': f'Known vulnerability in {name} version {version}',
                        'recommendation': f'Update {name} to a secure version'
                    })
        
        return dependency_issues
    
    def _calculate_security_score(self, scan_results: Dict[str, Any]) -> float:
        """Calculate overall security score (0-100)."""
        base_score = 100.0
        
        # Deduct points for vulnerabilities
        for vuln in scan_results.get('vulnerabilities', []):
            if vuln.get('severity') == 'high':
                base_score -= 20
            elif vuln.get('severity') == 'medium':
                base_score -= 10
            else:
                base_score -= 5
        
        # Deduct points for security issues
        for issue in scan_results.get('security_issues', []):
            if issue.get('severity') == 'high':
                base_score -= 15
            elif issue.get('severity') == 'medium':
                base_score -= 8
            else:
                base_score -= 3
        
        # Deduct points for dependency issues
        for dep_issue in scan_results.get('dependency_issues', []):
            if dep_issue.get('severity') == 'high':
                base_score -= 25
            elif dep_issue.get('severity') == 'medium':
                base_score -= 12
        
        return max(0.0, base_score)
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on security score."""
        if score >= 80:
            return 'low'
        elif score >= 60:
            return 'medium'
        elif score >= 40:
            return 'high'
        else:
            return 'critical'
    
    def _generate_security_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        if scan_results.get('vulnerabilities'):
            recommendations.append("Review and fix high-severity security vulnerabilities")
        
        if scan_results.get('dependency_issues'):
            recommendations.append("Update vulnerable dependencies to secure versions")
        
        if scan_results.get('security_issues'):
            recommendations.append("Add proper input validation and sanitization")
        
        if scan_results['overall_score'] < 70:
            recommendations.append("Consider security code review before publication")
        
        return recommendations


class PluginService:
    """Enhanced plugin management service with comprehensive security features."""
    
    def __init__(self, session: AsyncSession, cache_service: Optional[CacheService] = None):
        """Initialize plugin service."""
        self.session = session
        self.cache = cache_service or CacheService()
        self.moderation = ModerationService(session)
        self.security_scanner = SecurityScanner()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Configuration
        self.approval_requirements = {
            'min_security_score': 70.0,
            'max_high_severity_issues': 0,
            'max_medium_severity_issues': 3,
            'require_code_review': True
        }
    
    async def create_plugin(
        self,
        author_id: UUID,
        plugin_data: Dict[str, Any],
        source_code: Optional[str] = None,
        auto_scan: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new plugin with comprehensive validation and security scanning.
        
        Args:
            author_id: Plugin author ID
            plugin_data: Plugin metadata
            source_code: Plugin source code for scanning
            auto_scan: Whether to perform automatic security scan
            
        Returns:
            Created plugin data
        """
        try:
            # Validate required fields
            required_fields = ['name', 'description', 'plugin_type', 'version']
            for field in required_fields:
                if field not in plugin_data or not plugin_data[field]:
                    raise ValidationError(f"Required field missing: {field}")
            
            # Validate plugin type
            if plugin_data['plugin_type'] not in [pt.value for pt in PluginType]:
                raise ValidationError(f"Invalid plugin type: {plugin_data['plugin_type']}")
            
            # Check for duplicate slug
            slug = plugin_data.get('slug') or self._generate_slug(plugin_data['name'])
            existing_plugin = await self.session.execute(
                text("SELECT id FROM plugins WHERE slug = :slug"),
                {'slug': slug}
            )
            
            if existing_plugin.first():
                raise ValidationError(f"Plugin with slug '{slug}' already exists")
            
            # Create plugin record
            plugin = Plugin(
                slug=slug,
                name=plugin_data['name'],
                description=plugin_data['description'],
                short_description=plugin_data.get('short_description'),
                plugin_type=plugin_data['plugin_type'],
                author_id=author_id,
                version=plugin_data['version'],
                homepage_url=plugin_data.get('homepage_url'),
                repository_url=plugin_data.get('repository_url'),
                documentation_url=plugin_data.get('documentation_url'),
                categories=plugin_data.get('categories', []),
                tags=plugin_data.get('tags', []),
                compatibility=plugin_data.get('compatibility', {}),
                dependencies=plugin_data.get('dependencies', []),
                permissions=plugin_data.get('permissions', []),
                download_url=plugin_data.get('download_url'),
                install_command=plugin_data.get('install_command'),
                package_size=plugin_data.get('package_size', 0),
                is_premium=plugin_data.get('is_premium', False),
                price=plugin_data.get('price', 0.0),
                license_type=plugin_data.get('license_type', 'MIT'),
                status=PluginStatus.DRAFT.value
            )
            
            self.session.add(plugin)
            await self.session.flush()  # Get ID without committing
            
            # Perform security scan if requested
            security_scan_results = None
            if auto_scan and (source_code or plugin_data.get('dependencies')):
                security_scan_results = await self._perform_security_scan(
                    plugin.id, source_code, plugin_data.get('dependencies', [])
                )
                
                # Update plugin security status
                plugin.security_scan_status = security_scan_results.get('risk_level', 'unknown')
                plugin.security_scan_date = datetime.utcnow()
                plugin.security_issues_count = len(
                    security_scan_results.get('vulnerabilities', []) + 
                    security_scan_results.get('security_issues', [])
                )
                plugin.is_security_approved = (
                    security_scan_results.get('overall_score', 0) >= self.approval_requirements['min_security_score']
                )
            
            # Add dependencies if specified
            if plugin_data.get('dependencies'):
                await self._create_plugin_dependencies(plugin.id, plugin_data['dependencies'])
            
            await self.session.commit()
            await self.session.refresh(plugin, ['author'])
            
            # Auto-moderation check
            moderation_result = await self.moderation.auto_moderate_content(
                content_type="plugin",
                content_id=plugin.id,
                content_text=f"{plugin.name} {plugin.description}",
                user_id=author_id
            )
            
            # Update status based on moderation and security scan
            if moderation_result['action'] == 'approve' and plugin.is_security_approved:
                plugin.status = PluginStatus.SUBMITTED.value
            elif moderation_result['action'] == 'reject':
                plugin.status = PluginStatus.REJECTED.value
            
            await self.session.commit()
            
            # Clear relevant caches
            await self.cache.delete_pattern(f"plugins:*")
            await self.cache.delete_pattern(f"user_plugins:{author_id}:*")
            
            result = plugin.to_dict(include_private=True)
            result['author'] = plugin.author.to_dict(include_private=False) if plugin.author else None
            
            if security_scan_results:
                result['security_scan'] = security_scan_results
            
            return result
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error creating plugin: {e}")
            raise
    
    async def update_plugin(
        self,
        plugin_id: UUID,
        author_id: UUID,
        updates: Dict[str, Any],
        rescan_security: bool = False
    ) -> Dict[str, Any]:
        """
        Update an existing plugin with validation and optional security rescan.
        
        Args:
            plugin_id: Plugin ID
            author_id: Plugin author ID (for authorization)
            updates: Update data
            rescan_security: Whether to perform new security scan
            
        Returns:
            Updated plugin data
        """
        try:
            # Get plugin
            plugin = await self.session.get(Plugin, plugin_id, options=[selectinload(Plugin.author)])
            if not plugin:
                raise NotFoundError("Plugin not found")
            
            # Verify ownership
            if plugin.author_id != author_id:
                raise PermissionError("Cannot update another user's plugin")
            
            # Check if plugin is in editable state
            if plugin.status in [PluginStatus.BANNED.value]:
                raise ValidationError("Cannot update banned plugin")
            
            # Apply updates
            updatable_fields = [
                'name', 'description', 'short_description', 'version',
                'homepage_url', 'repository_url', 'documentation_url',
                'categories', 'tags', 'compatibility', 'dependencies',
                'permissions', 'download_url', 'install_command',
                'package_size', 'price', 'license_type'
            ]
            
            version_changed = False
            dependencies_changed = False
            
            for field, value in updates.items():
                if field in updatable_fields and hasattr(plugin, field):
                    old_value = getattr(plugin, field)
                    setattr(plugin, field, value)
                    
                    if field == 'version' and old_value != value:
                        version_changed = True
                    elif field == 'dependencies' and old_value != value:
                        dependencies_changed = True
            
            plugin.updated_at = datetime.utcnow()
            
            # Update dependencies if changed
            if dependencies_changed or 'dependencies' in updates:
                await self._update_plugin_dependencies(plugin_id, updates.get('dependencies', []))
            
            # Perform security rescan if requested or if version/dependencies changed
            if rescan_security or version_changed or dependencies_changed:
                security_scan_results = await self._perform_security_scan(
                    plugin_id, None, plugin.dependencies or []
                )
                
                plugin.security_scan_status = security_scan_results.get('risk_level', 'unknown')
                plugin.security_scan_date = datetime.utcnow()
                plugin.security_issues_count = len(
                    security_scan_results.get('vulnerabilities', []) + 
                    security_scan_results.get('security_issues', [])
                )
                plugin.is_security_approved = (
                    security_scan_results.get('overall_score', 0) >= self.approval_requirements['min_security_score']
                )
                
                # May need re-approval if security status changed
                if not plugin.is_security_approved and plugin.status == PluginStatus.PUBLISHED.value:
                    plugin.status = PluginStatus.UNDER_REVIEW.value
            
            await self.session.commit()
            
            # Clear caches
            await self.cache.delete_pattern(f"plugin:{plugin_id}:*")
            await self.cache.delete_pattern(f"plugins:*")
            
            result = plugin.to_dict(include_private=True)
            result['author'] = plugin.author.to_dict(include_private=False)
            
            return result
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error updating plugin: {e}")
            raise
    
    async def install_plugin(
        self,
        plugin_id: UUID,
        user_id: UUID,
        installation_method: str = "manual",
        client_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Install a plugin for a user with tracking.
        
        Args:
            plugin_id: Plugin ID
            user_id: User ID
            installation_method: How the plugin was installed
            client_info: Client/platform information
            
        Returns:
            Installation confirmation data
        """
        try:
            # Verify plugin exists and is published
            plugin = await self.session.get(Plugin, plugin_id)
            if not plugin:
                raise NotFoundError("Plugin not found")
            
            if plugin.status != PluginStatus.PUBLISHED.value:
                raise ValidationError("Plugin is not available for installation")
            
            if not plugin.is_security_approved:
                raise ValidationError("Plugin has not passed security review")
            
            # Check if already installed
            existing_install = await self.session.execute(
                text("""
                    SELECT id FROM plugin_installs 
                    WHERE plugin_id = :pid AND user_id = :uid AND is_active = true
                """),
                {'pid': plugin_id, 'uid': user_id}
            )
            
            if existing_install.first():
                raise ValidationError("Plugin is already installed")
            
            # Create installation record
            install = PluginInstall(
                plugin_id=plugin_id,
                user_id=user_id,
                version_installed=plugin.version,
                installation_method=installation_method,
                client_info=client_info or {}
            )
            
            self.session.add(install)
            
            # Update plugin statistics
            plugin.install_count += 1
            plugin.active_installs += 1
            plugin.last_used_at = datetime.utcnow()
            
            await self.session.commit()
            
            # Clear caches
            await self.cache.delete_pattern(f"plugin:{plugin_id}:*")
            await self.cache.delete_pattern(f"user_plugins:{user_id}:*")
            
            return {
                'install_id': str(install.id),
                'plugin_id': str(plugin_id),
                'user_id': str(user_id),
                'version': plugin.version,
                'installed_at': install.installed_at.isoformat(),
                'status': 'installed'
            }
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error installing plugin: {e}")
            raise
    
    async def uninstall_plugin(
        self,
        plugin_id: UUID,
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        Uninstall a plugin for a user.
        
        Args:
            plugin_id: Plugin ID
            user_id: User ID
            
        Returns:
            Uninstallation confirmation data
        """
        try:
            # Find active installation
            install_result = await self.session.execute(
                text("""
                    SELECT id FROM plugin_installs 
                    WHERE plugin_id = :pid AND user_id = :uid AND is_active = true
                """),
                {'pid': plugin_id, 'uid': user_id}
            )
            
            install_row = install_result.first()
            if not install_row:
                raise NotFoundError("Plugin installation not found")
            
            # Update installation record
            await self.session.execute(
                text("""
                    UPDATE plugin_installs 
                    SET is_active = false, uninstalled_at = :now
                    WHERE id = :install_id
                """),
                {'install_id': install_row.id, 'now': datetime.utcnow()}
            )
            
            # Update plugin statistics
            plugin = await self.session.get(Plugin, plugin_id)
            if plugin and plugin.active_installs > 0:
                plugin.active_installs -= 1
            
            await self.session.commit()
            
            # Clear caches
            await self.cache.delete_pattern(f"plugin:{plugin_id}:*")
            await self.cache.delete_pattern(f"user_plugins:{user_id}:*")
            
            return {
                'plugin_id': str(plugin_id),
                'user_id': str(user_id),
                'uninstalled_at': datetime.utcnow().isoformat(),
                'status': 'uninstalled'
            }
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error uninstalling plugin: {e}")
            raise
    
    async def get_plugin_security_report(
        self,
        plugin_id: UUID,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Get security report for a plugin.
        
        Args:
            plugin_id: Plugin ID
            detailed: Whether to include detailed scan results
            
        Returns:
            Security report data
        """
        try:
            # Check cache
            cache_key = f"plugin_security:{plugin_id}:{detailed}"
            cached_report = await self.cache.get(cache_key)
            if cached_report:
                return cached_report
            
            # Get plugin
            plugin = await self.session.get(Plugin, plugin_id)
            if not plugin:
                raise NotFoundError("Plugin not found")
            
            # Get latest security scan
            scan_result = await self.session.execute(
                text("""
                    SELECT * FROM plugin_security_scans 
                    WHERE plugin_id = :pid 
                    ORDER BY started_at DESC 
                    LIMIT 1
                """),
                {'pid': plugin_id}
            )
            
            scan_row = scan_result.first()
            
            report = {
                'plugin_id': str(plugin_id),
                'plugin_name': plugin.name,
                'plugin_version': plugin.version,
                'security_status': plugin.security_scan_status,
                'is_security_approved': plugin.is_security_approved,
                'last_scan_date': plugin.security_scan_date.isoformat() if plugin.security_scan_date else None,
                'security_issues_count': plugin.security_issues_count,
            }
            
            if scan_row:
                report.update({
                    'overall_score': float(scan_row.overall_score or 0),
                    'risk_level': scan_row.risk_level,
                    'scan_status': scan_row.status,
                    'scan_duration': scan_row.scan_duration or 0,
                    'files_scanned': scan_row.files_scanned or 0
                })
                
                if detailed:
                    report.update({
                        'vulnerabilities': scan_row.vulnerabilities or [],
                        'security_issues': scan_row.security_issues or [],
                        'code_quality_issues': scan_row.code_quality_issues or []
                    })
            
            # Cache for 1 hour
            await self.cache.set(cache_key, report, ttl=3600)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error getting plugin security report: {e}")
            raise
    
    # Private helper methods
    
    def _generate_slug(self, name: str) -> str:
        """Generate URL-friendly slug from plugin name."""
        import re
        
        slug = name.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special characters
        slug = re.sub(r'[-\s]+', '-', slug)   # Replace spaces/multiple hyphens with single hyphen
        slug = slug.strip('-')                # Remove leading/trailing hyphens
        
        return slug[:100]  # Limit length
    
    async def _perform_security_scan(
        self,
        plugin_id: UUID,
        source_code: Optional[str],
        dependencies: List[str]
    ) -> Dict[str, Any]:
        """Perform comprehensive security scan of plugin."""
        try:
            # Run security scan
            scan_results = await self.security_scanner.scan_plugin(
                plugin_id=plugin_id,
                plugin_content=source_code,
                dependencies=dependencies
            )
            
            # Store scan results in database
            security_scan = PluginSecurityScan(
                plugin_id=plugin_id,
                scanner_name="internal",
                scan_type="automated",
                status="completed",
                overall_score=scan_results.get('overall_score', 0.0),
                risk_level=scan_results.get('risk_level', 'unknown'),
                vulnerabilities=scan_results.get('vulnerabilities', []),
                security_issues=scan_results.get('security_issues', []),
                code_quality_issues=scan_results.get('code_quality_issues', []),
                scan_duration=1,  # Placeholder
                files_scanned=1,  # Placeholder
                completed_at=datetime.utcnow()
            )
            
            self.session.add(security_scan)
            
            return scan_results
            
        except Exception as e:
            self.logger.error(f"Error performing security scan: {e}")
            return {
                'overall_score': 0.0,
                'risk_level': 'critical',
                'error': str(e)
            }
    
    async def _create_plugin_dependencies(
        self,
        plugin_id: UUID,
        dependencies: List[str]
    ) -> None:
        """Create plugin dependency records."""
        try:
            for dep in dependencies:
                # Parse dependency format (name==version)
                if '==' in dep:
                    name, version = dep.split('==', 1)
                elif '>=' in dep:
                    name, version = dep.split('>=', 1)
                else:
                    name, version = dep, None
                
                dependency = PluginDependency(
                    plugin_id=plugin_id,
                    dependency_name=name.strip(),
                    dependency_version=version.strip() if version else None,
                    dependency_type="required",
                    is_external=True
                )
                
                self.session.add(dependency)
                
        except Exception as e:
            self.logger.error(f"Error creating plugin dependencies: {e}")
    
    async def _update_plugin_dependencies(
        self,
        plugin_id: UUID,
        new_dependencies: List[str]
    ) -> None:
        """Update plugin dependencies."""
        try:
            # Delete existing dependencies
            await self.session.execute(
                text("DELETE FROM plugin_dependencies WHERE plugin_id = :pid"),
                {'pid': plugin_id}
            )
            
            # Create new dependencies
            await self._create_plugin_dependencies(plugin_id, new_dependencies)
            
        except Exception as e:
            self.logger.error(f"Error updating plugin dependencies: {e}")