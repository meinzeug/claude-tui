"""
Marketplace Service - Core marketplace business logic and operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import desc, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.template import Template, TemplateSearchFilters, TemplateStats
from ..models.marketplace import (
    FeaturedCollection, TemplateTag, TemplateDownload, 
    TemplateView, TemplateStar, TemplateFork, MarketplaceStats, MarketplaceTrends
)
from ..models.review import TemplateRating
from ..models.user import UserProfile
from ..repositories.marketplace_repository import MarketplaceRepository
from ...core.exceptions import ValidationError, NotFoundError

logger = logging.getLogger(__name__)


class MarketplaceService:
    """Service for marketplace operations and template discovery."""
    
    def __init__(self, db: AsyncSession):
        """Initialize marketplace service."""
        self.db = db
        self.repository = MarketplaceRepository(db)
    
    async def search_templates(
        self,
        filters: TemplateSearchFilters,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        """
        Search templates with advanced filtering and ranking.
        
        Args:
            filters: Search filters
            page: Page number
            page_size: Results per page
            
        Returns:
            Tuple of (templates, total_count, search_metadata)
        """
        try:
            # Build base query
            query = self.db.query(Template).filter(Template.is_public == True)
            
            # Apply filters
            if filters.query:
                search_terms = filters.query.lower().split()
                for term in search_terms:
                    query = query.filter(
                        or_(
                            Template.name.ilike(f'%{term}%'),
                            Template.description.ilike(f'%{term}%'),
                            Template.short_description.ilike(f'%{term}%'),
                            Template.tags.op('?')(term),
                            Template.categories.op('?')(term)
                        )
                    )
            
            if filters.template_type:
                query = query.filter(Template.template_type == filters.template_type.value)
            
            if filters.complexity_level:
                query = query.filter(Template.complexity_level == filters.complexity_level.value)
            
            if filters.categories:
                for category in filters.categories:
                    query = query.filter(Template.categories.op('?')(category))
            
            if filters.frameworks:
                for framework in filters.frameworks:
                    query = query.filter(Template.frameworks.op('?')(framework))
            
            if filters.languages:
                for language in filters.languages:
                    query = query.filter(Template.languages.op('?')(language))
            
            if filters.min_rating:
                query = query.filter(Template.average_rating >= filters.min_rating)
            
            if filters.max_rating:
                query = query.filter(Template.average_rating <= filters.max_rating)
            
            if filters.is_featured is not None:
                query = query.filter(Template.is_featured == filters.is_featured)
            
            if filters.is_free is not None:
                query = query.filter(Template.is_premium != filters.is_free)
            
            if filters.author_id:
                query = query.filter(Template.author_id == filters.author_id)
            
            if filters.created_after:
                query = query.filter(Template.created_at >= filters.created_after)
            
            if filters.updated_after:
                query = query.filter(Template.updated_at >= filters.updated_after)
            
            # Count total results
            total_count = await query.count()
            
            # Apply sorting
            sort_column = getattr(Template, filters.sort_by, Template.updated_at)
            if filters.sort_order == "asc":
                query = query.order_by(sort_column)
            else:
                query = query.order_by(desc(sort_column))
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute query with relationships
            query = query.options(selectinload(Template.author))
            templates = await query.all()
            
            # Convert to dict and add computed fields
            template_dicts = []
            for template in templates:
                template_dict = template.to_dict()
                template_dict['author'] = template.author.to_dict(include_private=False) if template.author else None
                template_dicts.append(template_dict)
            
            # Search metadata
            metadata = {
                'page': page,
                'page_size': page_size,
                'total_pages': (total_count + page_size - 1) // page_size,
                'has_next': page * page_size < total_count,
                'has_prev': page > 1,
                'filters_applied': len([f for f in [
                    filters.query, filters.template_type, filters.complexity_level,
                    filters.categories, filters.frameworks, filters.languages,
                    filters.min_rating, filters.max_rating, filters.is_featured,
                    filters.author_id, filters.created_after, filters.updated_after
                ] if f is not None and f != []]),
                'search_time': datetime.utcnow()
            }
            
            return template_dicts, total_count, metadata
            
        except Exception as e:
            logger.error(f"Error searching templates: {e}")
            raise ValidationError(f"Search failed: {str(e)}")
    
    async def get_featured_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get featured templates."""
        try:
            query = self.db.query(Template).filter(
                and_(Template.is_featured == True, Template.is_public == True)
            ).order_by(desc(Template.updated_at)).limit(limit)
            
            query = query.options(selectinload(Template.author))
            templates = await query.all()
            
            return [
                {**template.to_dict(), 'author': template.author.to_dict(include_private=False)}
                for template in templates
            ]
            
        except Exception as e:
            logger.error(f"Error getting featured templates: {e}")
            raise
    
    async def get_trending_templates(
        self,
        period_days: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get trending templates based on recent activity."""
        try:
            since_date = datetime.utcnow() - timedelta(days=period_days)
            
            # Calculate trending score based on recent downloads, views, and ratings
            trending_query = self.db.query(
                Template,
                func.coalesce(
                    func.sum(TemplateDownload.id), 0
                ).label('recent_downloads'),
                func.coalesce(
                    func.sum(TemplateView.id), 0
                ).label('recent_views'),
                (Template.average_rating * Template.rating_count).label('rating_score')
            ).outerjoin(
                TemplateDownload,
                and_(
                    Template.id == TemplateDownload.template_id,
                    TemplateDownload.downloaded_at >= since_date
                )
            ).outerjoin(
                TemplateView,
                and_(
                    Template.id == TemplateView.template_id,
                    TemplateView.viewed_at >= since_date
                )
            ).filter(
                Template.is_public == True
            ).group_by(Template.id).order_by(
                desc('recent_downloads'),
                desc('recent_views'),
                desc('rating_score')
            ).limit(limit)
            
            trending_query = trending_query.options(selectinload(Template.author))
            results = await trending_query.all()
            
            trending_templates = []
            for template, downloads, views, rating_score in results:
                template_dict = template.to_dict()
                template_dict['author'] = template.author.to_dict(include_private=False)
                template_dict['trending_metrics'] = {
                    'recent_downloads': downloads,
                    'recent_views': views,
                    'rating_score': float(rating_score or 0)
                }
                trending_templates.append(template_dict)
            
            return trending_templates
            
        except Exception as e:
            logger.error(f"Error getting trending templates: {e}")
            raise
    
    async def get_template_recommendations(
        self,
        user_id: Optional[UUID] = None,
        template_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get personalized template recommendations."""
        try:
            # If template_id is provided, find similar templates
            if template_id:
                template = await self.repository.get_template_by_id(template_id)
                if not template:
                    raise NotFoundError("Template not found")
                
                # Find templates with similar tags, categories, or frameworks
                similar_query = self.db.query(Template).filter(
                    and_(
                        Template.id != template_id,
                        Template.is_public == True,
                        or_(
                            # Similar categories
                            func.jsonb_path_exists(
                                Template.categories, 
                                f'$[*] ? (@ in {template.categories})'
                            ),
                            # Similar frameworks
                            func.jsonb_path_exists(
                                Template.frameworks,
                                f'$[*] ? (@ in {template.frameworks})'
                            ),
                            # Similar tags
                            func.jsonb_path_exists(
                                Template.tags,
                                f'$[*] ? (@ in {template.tags})'
                            ),
                            # Same complexity level
                            Template.complexity_level == template.complexity_level
                        )
                    )
                ).order_by(desc(Template.average_rating), desc(Template.download_count)).limit(limit)
                
            # If user_id is provided, personalized recommendations
            elif user_id:
                # Get user's preferences from their templates and activity
                user_templates = self.db.query(Template).filter(Template.author_id == user_id)
                user_downloads = self.db.query(TemplateDownload).filter(TemplateDownload.user_id == user_id)
                user_stars = self.db.query(TemplateStar).filter(TemplateStar.user_id == user_id)
                
                # Collect user's interests
                interests = {
                    'categories': set(),
                    'frameworks': set(),
                    'languages': set(),
                    'complexity_levels': set()
                }
                
                async for template in user_templates:
                    interests['categories'].update(template.categories or [])
                    interests['frameworks'].update(template.frameworks or [])
                    interests['languages'].update(template.languages or [])
                    interests['complexity_levels'].add(template.complexity_level)
                
                # Build recommendation query based on interests
                similar_query = self.db.query(Template).filter(
                    and_(
                        Template.author_id != user_id,
                        Template.is_public == True
                    )
                ).order_by(desc(Template.average_rating), desc(Template.download_count)).limit(limit)
                
            else:
                # General popular recommendations
                similar_query = self.db.query(Template).filter(
                    Template.is_public == True
                ).order_by(
                    desc(Template.average_rating),
                    desc(Template.download_count),
                    desc(Template.star_count)
                ).limit(limit)
            
            similar_query = similar_query.options(selectinload(Template.author))
            recommendations = await similar_query.all()
            
            return [
                {**template.to_dict(), 'author': template.author.to_dict(include_private=False)}
                for template in recommendations
            ]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise
    
    async def record_template_download(
        self,
        template_id: UUID,
        user_id: Optional[UUID] = None,
        download_type: str = "direct",
        project_name: Optional[str] = None,
        intended_use: Optional[str] = None
    ) -> None:
        """Record a template download."""
        try:
            # Create download record
            download = TemplateDownload(
                template_id=template_id,
                user_id=user_id,
                download_type=download_type,
                project_name=project_name,
                intended_use=intended_use
            )
            
            self.db.add(download)
            
            # Update template download count
            template = await self.repository.get_template_by_id(template_id)
            if template:
                template.download_count += 1
                template.last_used_at = datetime.utcnow()
            
            await self.db.commit()
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error recording download: {e}")
            raise
    
    async def record_template_view(
        self,
        template_id: UUID,
        user_id: Optional[UUID] = None,
        view_duration: Optional[int] = None,
        referrer: Optional[str] = None
    ) -> None:
        """Record a template view for analytics."""
        try:
            view = TemplateView(
                template_id=template_id,
                user_id=user_id,
                view_duration=view_duration,
                referrer=referrer
            )
            
            self.db.add(view)
            await self.db.commit()
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error recording view: {e}")
            raise
    
    async def star_template(self, template_id: UUID, user_id: UUID) -> bool:
        """Star/favorite a template."""
        try:
            # Check if already starred
            existing_star = await self.db.query(TemplateStar).filter(
                and_(
                    TemplateStar.template_id == template_id,
                    TemplateStar.user_id == user_id
                )
            ).first()
            
            if existing_star:
                return False  # Already starred
            
            # Create star record
            star = TemplateStar(
                template_id=template_id,
                user_id=user_id
            )
            
            self.db.add(star)
            
            # Update template star count
            template = await self.repository.get_template_by_id(template_id)
            if template:
                template.star_count += 1
            
            await self.db.commit()
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error starring template: {e}")
            raise
    
    async def unstar_template(self, template_id: UUID, user_id: UUID) -> bool:
        """Unstar/unfavorite a template."""
        try:
            # Find and remove star
            star = await self.db.query(TemplateStar).filter(
                and_(
                    TemplateStar.template_id == template_id,
                    TemplateStar.user_id == user_id
                )
            ).first()
            
            if not star:
                return False  # Not starred
            
            await self.db.delete(star)
            
            # Update template star count
            template = await self.repository.get_template_by_id(template_id)
            if template and template.star_count > 0:
                template.star_count -= 1
            
            await self.db.commit()
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error unstarring template: {e}")
            raise
    
    async def get_marketplace_stats(self) -> MarketplaceStats:
        """Get comprehensive marketplace statistics."""
        try:
            # Basic counts
            total_templates = await self.db.query(func.count(Template.id)).scalar()
            public_templates = await self.db.query(func.count(Template.id)).filter(
                Template.is_public == True
            ).scalar()
            featured_templates = await self.db.query(func.count(Template.id)).filter(
                Template.is_featured == True
            ).scalar()
            
            total_users = await self.db.query(func.count(UserProfile.id)).scalar()
            total_downloads = await self.db.query(func.count(TemplateDownload.id)).scalar()
            
            # Monthly active users (users who downloaded templates in last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            monthly_active_users = await self.db.query(
                func.count(func.distinct(TemplateDownload.user_id))
            ).filter(
                TemplateDownload.downloaded_at >= thirty_days_ago
            ).scalar()
            
            # Category distribution
            category_stats = {}
            templates_with_categories = await self.db.query(Template.categories).filter(
                Template.categories.isnot(None)
            ).all()
            
            for (categories,) in templates_with_categories:
                if categories:
                    for category in categories:
                        category_stats[category] = category_stats.get(category, 0) + 1
            
            # Language distribution
            language_stats = {}
            templates_with_languages = await self.db.query(Template.languages).filter(
                Template.languages.isnot(None)
            ).all()
            
            for (languages,) in templates_with_languages:
                if languages:
                    for language in languages:
                        language_stats[language] = language_stats.get(language, 0) + 1
            
            # Framework distribution
            framework_stats = {}
            templates_with_frameworks = await self.db.query(Template.frameworks).filter(
                Template.frameworks.isnot(None)
            ).all()
            
            for (frameworks,) in templates_with_frameworks:
                if frameworks:
                    for framework in frameworks:
                        framework_stats[framework] = framework_stats.get(framework, 0) + 1
            
            # Recent templates (last 7 days)
            recent_date = datetime.utcnow() - timedelta(days=7)
            recent_templates_query = self.db.query(Template).filter(
                and_(
                    Template.created_at >= recent_date,
                    Template.is_public == True
                )
            ).order_by(desc(Template.created_at)).limit(10)
            
            recent_templates_query = recent_templates_query.options(selectinload(Template.author))
            recent_templates = await recent_templates_query.all()
            
            # Top rated templates
            top_rated_query = self.db.query(Template).filter(
                and_(
                    Template.is_public == True,
                    Template.rating_count >= 5  # At least 5 ratings
                )
            ).order_by(desc(Template.average_rating), desc(Template.rating_count)).limit(10)
            
            top_rated_query = top_rated_query.options(selectinload(Template.author))
            top_rated_templates = await top_rated_query.all()
            
            return MarketplaceStats(
                total_templates=total_templates or 0,
                public_templates=public_templates or 0,
                featured_templates=featured_templates or 0,
                total_users=total_users or 0,
                total_downloads=total_downloads or 0,
                monthly_active_users=monthly_active_users or 0,
                categories=category_stats,
                languages=language_stats,
                frameworks=framework_stats,
                recent_templates=[
                    {**t.to_dict(), 'author': t.author.to_dict(include_private=False)}
                    for t in recent_templates
                ],
                trending_templates=await self.get_trending_templates(limit=10),
                top_rated_templates=[
                    {**t.to_dict(), 'author': t.author.to_dict(include_private=False)}
                    for t in top_rated_templates
                ]
            )
            
        except Exception as e:
            logger.error(f"Error getting marketplace stats: {e}")
            raise
    
    async def get_template_suggestions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get template search suggestions."""
        try:
            if not query or len(query) < 2:
                return []
            
            # Search in template names, descriptions, and tags
            suggestions_query = self.db.query(Template).filter(
                and_(
                    Template.is_public == True,
                    or_(
                        Template.name.ilike(f'%{query}%'),
                        Template.short_description.ilike(f'%{query}%'),
                        Template.tags.op('?')(query.lower())
                    )
                )
            ).order_by(desc(Template.download_count), desc(Template.star_count)).limit(limit)
            
            templates = await suggestions_query.all()
            
            suggestions = []
            for template in templates:
                suggestions.append({
                    'id': str(template.id),
                    'name': template.name,
                    'type': 'template',
                    'score': template.average_rating * template.rating_count,
                    'metadata': {
                        'author': template.author.username if template.author else None,
                        'downloads': template.download_count,
                        'rating': template.average_rating
                    }
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []