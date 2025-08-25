"""
Marketplace Repository - Enhanced data access layer for marketplace operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import desc, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.sql import select, update, delete

from ..models.template import Template, TemplateSearchFilters
from ..models.marketplace import (
    FeaturedCollection, TemplateTag, TemplateDownload, 
    TemplateView, TemplateStar, TemplateFork, TemplateMarketplace
)
from ..models.rating import TemplateRating, UserReputation
from ..models.user import UserProfile
from .base_repository import BaseRepository
from ...core.exceptions import NotFoundError, ValidationError

logger = logging.getLogger(__name__)


class MarketplaceRepository(BaseRepository):
    """Repository for marketplace-specific database operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize marketplace repository."""
        super().__init__(session)
        self.logger = logger.getChild(self.__class__.__name__)
    
    async def search_templates_advanced(
        self,
        filters: TemplateSearchFilters,
        page: int = 1,
        page_size: int = 20,
        enable_analytics: bool = True
    ) -> Tuple[List[Template], int, Dict[str, Any]]:
        """
        Advanced template search with ranking algorithms and analytics.
        
        Args:
            filters: Search filters
            page: Page number
            page_size: Results per page
            enable_analytics: Whether to track search analytics
            
        Returns:
            Tuple of (templates, total_count, search_metadata)
        """
        try:
            # Build base query with complex ranking
            query = select(Template).where(Template.is_public == True)
            
            # Full-text search with ranking
            if filters.query:
                search_terms = filters.query.lower().split()
                search_conditions = []
                
                for i, term in enumerate(search_terms):
                    term_weight = 1.0 / (i + 1)  # Diminishing weight for later terms
                    
                    # Name matches (highest weight)
                    name_match = Template.name.ilike(f'%{term}%') * 3.0 * term_weight
                    
                    # Description matches
                    desc_match = Template.description.ilike(f'%{term}%') * 1.0 * term_weight
                    
                    # Short description matches
                    short_desc_match = Template.short_description.ilike(f'%{term}%') * 1.5 * term_weight
                    
                    # Tag matches
                    tag_match = func.jsonb_path_exists(
                        Template.tags, f'$[*] ? (@ like_regex "{term}" flag "i")'
                    ) * 2.0 * term_weight
                    
                    # Category matches
                    category_match = func.jsonb_path_exists(
                        Template.categories, f'$[*] ? (@ like_regex "{term}" flag "i")'
                    ) * 1.8 * term_weight
                    
                    search_conditions.append(or_(
                        name_match, desc_match, short_desc_match, tag_match, category_match
                    ))
                
                if search_conditions:
                    query = query.where(and_(*search_conditions))
            
            # Apply filters with performance optimization
            filter_conditions = []
            
            if filters.template_type:
                filter_conditions.append(Template.template_type == filters.template_type.value)
            
            if filters.complexity_level:
                filter_conditions.append(Template.complexity_level == filters.complexity_level.value)
            
            if filters.categories:
                for category in filters.categories:
                    filter_conditions.append(
                        func.jsonb_path_exists(
                            Template.categories, f'$[*] ? (@ == "{category}")'
                        )
                    )
            
            if filters.frameworks:
                framework_condition = or_(*[
                    func.jsonb_path_exists(
                        Template.frameworks, f'$[*] ? (@ == "{framework}")'
                    ) for framework in filters.frameworks
                ])
                filter_conditions.append(framework_condition)
            
            if filters.languages:
                language_condition = or_(*[
                    func.jsonb_path_exists(
                        Template.languages, f'$[*] ? (@ == "{language}")'
                    ) for language in filters.languages
                ])
                filter_conditions.append(language_condition)
            
            if filters.min_rating:
                filter_conditions.append(Template.average_rating >= filters.min_rating)
            
            if filters.max_rating:
                filter_conditions.append(Template.average_rating <= filters.max_rating)
            
            if filters.is_featured is not None:
                filter_conditions.append(Template.is_featured == filters.is_featured)
            
            if filters.is_free is not None:
                filter_conditions.append(Template.is_premium != filters.is_free)
            
            if filters.author_id:
                filter_conditions.append(Template.author_id == filters.author_id)
            
            if filters.created_after:
                filter_conditions.append(Template.created_at >= filters.created_after)
            
            if filters.updated_after:
                filter_conditions.append(Template.updated_at >= filters.updated_after)
            
            if filter_conditions:
                query = query.where(and_(*filter_conditions))
            
            # Count total results
            count_query = select(func.count()).select_from(query.subquery())
            total_count = await self.session.scalar(count_query)
            
            # Apply advanced sorting with popularity boost
            if filters.sort_by == "relevance" and filters.query:
                # Relevance + popularity ranking
                query = query.order_by(
                    desc(Template.is_featured),  # Featured first
                    desc(Template.average_rating * func.log(Template.rating_count + 1)),  # Quality score
                    desc(func.log(Template.download_count + 1)),  # Popularity boost
                    desc(Template.updated_at)  # Freshness
                )
            elif filters.sort_by == "trending":
                # Calculate trending score based on recent activity
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                query = query.outerjoin(
                    TemplateDownload,
                    and_(
                        Template.id == TemplateDownload.template_id,
                        TemplateDownload.downloaded_at >= thirty_days_ago
                    )
                ).group_by(Template.id).order_by(
                    desc(func.count(TemplateDownload.id)),
                    desc(Template.average_rating),
                    desc(Template.star_count)
                )
            else:
                # Standard sorting
                sort_column = getattr(Template, filters.sort_by, Template.updated_at)
                if filters.sort_order == "asc":
                    query = query.order_by(sort_column)
                else:
                    query = query.order_by(desc(sort_column))
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute with eager loading
            query = query.options(
                selectinload(Template.author),
                selectinload(Template.ratings)
            )
            
            result = await self.session.execute(query)
            templates = result.scalars().all()
            
            # Analytics tracking
            search_metadata = {
                'page': page,
                'page_size': page_size,
                'total_count': total_count or 0,
                'total_pages': ((total_count or 0) + page_size - 1) // page_size,
                'has_next': page * page_size < (total_count or 0),
                'has_prev': page > 1,
                'filters_applied': len([f for f in [
                    filters.query, filters.template_type, filters.complexity_level,
                    filters.categories, filters.frameworks, filters.languages,
                    filters.min_rating, filters.max_rating, filters.is_featured,
                    filters.author_id, filters.created_after, filters.updated_after
                ] if f is not None and f != []]),
                'search_time': datetime.utcnow(),
                'performance_metrics': {
                    'result_count': len(templates),
                    'has_query': bool(filters.query),
                    'complex_filters': len(filter_conditions)
                }
            }
            
            # Track search analytics if enabled
            if enable_analytics and filters.query:
                await self._track_search_analytics(filters.query, total_count or 0, len(templates))
            
            return templates, total_count or 0, search_metadata
            
        except Exception as e:
            self.logger.error(f"Error in advanced template search: {e}")
            raise ValidationError(f"Search failed: {str(e)}")
    
    async def get_trending_templates(
        self, 
        period_days: int = 7, 
        limit: int = 20,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trending templates with sophisticated ranking algorithm.
        
        Args:
            period_days: Period to analyze for trending
            limit: Maximum number of results
            category_filter: Optional category filter
            
        Returns:
            List of trending templates with metrics
        """
        try:
            since_date = datetime.utcnow() - timedelta(days=period_days)
            
            # Build complex trending query with weighted metrics
            trending_query = select(
                Template,
                func.coalesce(func.count(TemplateDownload.id), 0).label('recent_downloads'),
                func.coalesce(func.count(TemplateView.id), 0).label('recent_views'),
                func.coalesce(func.count(TemplateStar.user_id), 0).label('recent_stars'),
                func.coalesce(func.count(TemplateRating.id), 0).label('recent_ratings'),
                (
                    # Trending score calculation
                    func.coalesce(func.count(TemplateDownload.id), 0) * 3.0 +  # Downloads weight: 3
                    func.coalesce(func.count(TemplateView.id), 0) * 0.1 +       # Views weight: 0.1
                    func.coalesce(func.count(TemplateStar.user_id), 0) * 2.0 +  # Stars weight: 2
                    func.coalesce(func.count(TemplateRating.id), 0) * 1.5 +     # Ratings weight: 1.5
                    Template.average_rating * Template.rating_count * 0.5        # Quality boost
                ).label('trending_score')
            ).select_from(Template).outerjoin(
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
            ).outerjoin(
                TemplateStar,
                and_(
                    Template.id == TemplateStar.template_id,
                    TemplateStar.starred_at >= since_date
                )
            ).outerjoin(
                TemplateRating,
                and_(
                    Template.id == TemplateRating.template_id,
                    TemplateRating.created_at >= since_date
                )
            ).where(
                Template.is_public == True
            )
            
            # Apply category filter if specified
            if category_filter:
                trending_query = trending_query.where(
                    func.jsonb_path_exists(
                        Template.categories, f'$[*] ? (@ == "{category_filter}")'
                    )
                )
            
            # Group and order by trending score
            trending_query = trending_query.group_by(Template.id).order_by(
                desc('trending_score'),
                desc(Template.average_rating),
                desc(Template.updated_at)
            ).limit(limit)
            
            # Execute with relationships
            trending_query = trending_query.options(
                selectinload(Template.author)
            )
            
            result = await self.session.execute(trending_query)
            trending_data = result.all()
            
            # Build response with metrics
            trending_templates = []
            for row in trending_data:
                template = row.Template
                template_dict = template.to_dict()
                template_dict['author'] = template.author.to_dict(include_private=False) if template.author else None
                template_dict['trending_metrics'] = {
                    'recent_downloads': row.recent_downloads,
                    'recent_views': row.recent_views,
                    'recent_stars': row.recent_stars,
                    'recent_ratings': row.recent_ratings,
                    'trending_score': float(row.trending_score),
                    'period_days': period_days
                }
                trending_templates.append(template_dict)
            
            return trending_templates
            
        except Exception as e:
            self.logger.error(f"Error getting trending templates: {e}")
            raise
    
    async def get_personalized_recommendations(
        self,
        user_id: UUID,
        limit: int = 20,
        exclude_authored: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get personalized template recommendations using collaborative filtering.
        
        Args:
            user_id: User ID for personalization
            limit: Maximum recommendations
            exclude_authored: Whether to exclude user's own templates
            
        Returns:
            List of recommended templates with reasoning
        """
        try:
            # Get user's interaction history
            user_downloads_query = select(TemplateDownload.template_id).where(
                TemplateDownload.user_id == user_id
            )
            user_stars_query = select(TemplateStar.template_id).where(
                TemplateStar.user_id == user_id
            )
            user_ratings_query = select(TemplateRating.template_id).where(
                TemplateRating.user_id == user_id
            )
            
            # Execute queries
            downloaded_templates = await self.session.scalars(user_downloads_query)
            starred_templates = await self.session.scalars(user_stars_query)
            rated_templates = await self.session.scalars(user_ratings_query)
            
            user_template_ids = set(list(downloaded_templates) + list(starred_templates) + list(rated_templates))
            
            if not user_template_ids:
                # Fallback to popular templates for new users
                return await self.get_popular_templates(limit=limit)
            
            # Get templates user interacted with to analyze preferences
            user_templates_query = select(Template).where(
                Template.id.in_(user_template_ids)
            )
            user_templates = await self.session.scalars(user_templates_query)
            user_templates_list = list(user_templates)
            
            # Analyze user preferences
            user_categories = set()
            user_frameworks = set()
            user_languages = set()
            user_complexity_levels = set()
            
            for template in user_templates_list:
                if template.categories:
                    user_categories.update(template.categories)
                if template.frameworks:
                    user_frameworks.update(template.frameworks)
                if template.languages:
                    user_languages.update(template.languages)
                if template.complexity_level:
                    user_complexity_levels.add(template.complexity_level)
            
            # Find similar users (collaborative filtering)
            similar_users_query = select(
                TemplateDownload.user_id,
                func.count(TemplateDownload.template_id).label('common_templates')
            ).where(
                and_(
                    TemplateDownload.template_id.in_(user_template_ids),
                    TemplateDownload.user_id != user_id
                )
            ).group_by(TemplateDownload.user_id).having(
                func.count(TemplateDownload.template_id) >= 2  # At least 2 common templates
            ).order_by(desc('common_templates')).limit(10)
            
            similar_users_result = await self.session.execute(similar_users_query)
            similar_users = [row.user_id for row in similar_users_result]
            
            # Get recommendations from similar users
            collaborative_recommendations = []
            if similar_users:
                similar_user_templates_query = select(
                    Template,
                    func.count(TemplateDownload.user_id).label('similar_user_count')
                ).select_from(Template).join(
                    TemplateDownload,
                    Template.id == TemplateDownload.template_id
                ).where(
                    and_(
                        TemplateDownload.user_id.in_(similar_users),
                        Template.id.notin_(user_template_ids),
                        Template.is_public == True
                    )
                )
                
                if exclude_authored:
                    similar_user_templates_query = similar_user_templates_query.where(
                        Template.author_id != user_id
                    )
                
                similar_user_templates_query = similar_user_templates_query.group_by(
                    Template.id
                ).order_by(
                    desc('similar_user_count'),
                    desc(Template.average_rating),
                    desc(Template.download_count)
                ).limit(limit // 2)
                
                similar_user_templates_query = similar_user_templates_query.options(
                    selectinload(Template.author)
                )
                
                result = await self.session.execute(similar_user_templates_query)
                for row in result:
                    template_dict = row.Template.to_dict()
                    template_dict['author'] = row.Template.author.to_dict(include_private=False) if row.Template.author else None
                    template_dict['recommendation_reason'] = 'collaborative_filtering'
                    template_dict['recommendation_score'] = row.similar_user_count
                    collaborative_recommendations.append(template_dict)
            
            # Content-based recommendations
            content_conditions = []
            
            # Match user's preferred categories
            if user_categories:
                category_conditions = [
                    func.jsonb_path_exists(
                        Template.categories, f'$[*] ? (@ == "{category}")'
                    ) for category in user_categories
                ]
                content_conditions.append(or_(*category_conditions))
            
            # Match user's preferred frameworks
            if user_frameworks:
                framework_conditions = [
                    func.jsonb_path_exists(
                        Template.frameworks, f'$[*] ? (@ == "{framework}")'
                    ) for framework in user_frameworks
                ]
                content_conditions.append(or_(*framework_conditions))
            
            # Match user's preferred languages
            if user_languages:
                language_conditions = [
                    func.jsonb_path_exists(
                        Template.languages, f'$[*] ? (@ == "{language}")'
                    ) for language in user_languages
                ]
                content_conditions.append(or_(*language_conditions))
            
            content_based_recommendations = []
            if content_conditions:
                content_query = select(Template).where(
                    and_(
                        or_(*content_conditions),
                        Template.id.notin_(user_template_ids),
                        Template.is_public == True
                    )
                )
                
                if exclude_authored:
                    content_query = content_query.where(Template.author_id != user_id)
                
                content_query = content_query.order_by(
                    desc(Template.average_rating),
                    desc(Template.download_count),
                    desc(Template.updated_at)
                ).limit(limit // 2)
                
                content_query = content_query.options(selectinload(Template.author))
                
                content_templates = await self.session.scalars(content_query)
                for template in content_templates:
                    template_dict = template.to_dict()
                    template_dict['author'] = template.author.to_dict(include_private=False) if template.author else None
                    template_dict['recommendation_reason'] = 'content_based'
                    template_dict['recommendation_score'] = template.average_rating * template.rating_count
                    content_based_recommendations.append(template_dict)
            
            # Combine recommendations and remove duplicates
            all_recommendations = collaborative_recommendations + content_based_recommendations
            seen_ids = set()
            unique_recommendations = []
            
            for rec in all_recommendations:
                if rec['id'] not in seen_ids:
                    seen_ids.add(rec['id'])
                    unique_recommendations.append(rec)
            
            # Sort by recommendation score and limit
            unique_recommendations.sort(
                key=lambda x: x.get('recommendation_score', 0), 
                reverse=True
            )
            
            return unique_recommendations[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting personalized recommendations: {e}")
            # Fallback to popular templates
            return await self.get_popular_templates(limit=limit)
    
    async def get_popular_templates(self, limit: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get popular templates based on multiple metrics.
        
        Args:
            limit: Maximum number of templates
            category: Optional category filter
            
        Returns:
            List of popular templates
        """
        try:
            query = select(Template).where(Template.is_public == True)
            
            if category:
                query = query.where(
                    func.jsonb_path_exists(
                        Template.categories, f'$[*] ? (@ == "{category}")'
                    )
                )
            
            # Order by popularity score (combination of ratings, downloads, stars)
            query = query.order_by(
                desc(Template.is_featured),
                desc(Template.average_rating * func.log(Template.rating_count + 1)),
                desc(func.log(Template.download_count + 1)),
                desc(Template.star_count),
                desc(Template.updated_at)
            ).limit(limit)
            
            query = query.options(selectinload(Template.author))
            
            templates = await self.session.scalars(query)
            
            return [
                {
                    **template.to_dict(),
                    'author': template.author.to_dict(include_private=False) if template.author else None,
                    'popularity_score': (
                        template.average_rating * template.rating_count +
                        template.download_count * 0.1 +
                        template.star_count * 2
                    )
                }
                for template in templates
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting popular templates: {e}")
            raise
    
    async def get_template_analytics(self, template_id: UUID, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive analytics for a template.
        
        Args:
            template_id: Template ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with analytics data
        """
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Downloads over time
            downloads_query = select(
                func.date(TemplateDownload.downloaded_at).label('date'),
                func.count(TemplateDownload.id).label('count')
            ).where(
                and_(
                    TemplateDownload.template_id == template_id,
                    TemplateDownload.downloaded_at >= since_date
                )
            ).group_by(func.date(TemplateDownload.downloaded_at)).order_by('date')
            
            downloads_result = await self.session.execute(downloads_query)
            downloads_by_date = {str(row.date): row.count for row in downloads_result}
            
            # Views over time
            views_query = select(
                func.date(TemplateView.viewed_at).label('date'),
                func.count(TemplateView.id).label('count')
            ).where(
                and_(
                    TemplateView.template_id == template_id,
                    TemplateView.viewed_at >= since_date
                )
            ).group_by(func.date(TemplateView.viewed_at)).order_by('date')
            
            views_result = await self.session.execute(views_query)
            views_by_date = {str(row.date): row.count for row in views_result}
            
            # Ratings distribution
            ratings_query = select(
                TemplateRating.overall_rating.label('rating'),
                func.count(TemplateRating.id).label('count')
            ).where(
                TemplateRating.template_id == template_id
            ).group_by(TemplateRating.overall_rating).order_by('rating')
            
            ratings_result = await self.session.execute(ratings_query)
            ratings_distribution = {row.rating: row.count for row in ratings_result}
            
            # Total metrics
            total_downloads = await self.session.scalar(
                select(func.count(TemplateDownload.id)).where(
                    TemplateDownload.template_id == template_id
                )
            )
            
            total_views = await self.session.scalar(
                select(func.count(TemplateView.id)).where(
                    TemplateView.template_id == template_id
                )
            )
            
            total_stars = await self.session.scalar(
                select(func.count(TemplateStar.user_id)).where(
                    TemplateStar.template_id == template_id
                )
            )
            
            return {
                'template_id': str(template_id),
                'period_days': days,
                'downloads': {
                    'total': total_downloads or 0,
                    'by_date': downloads_by_date
                },
                'views': {
                    'total': total_views or 0,
                    'by_date': views_by_date
                },
                'stars': {
                    'total': total_stars or 0
                },
                'ratings': {
                    'distribution': ratings_distribution
                },
                'conversion_rate': (
                    (total_downloads or 0) / (total_views or 1) * 100
                ),
                'engagement_score': (
                    (total_downloads or 0) * 3 +
                    (total_views or 0) * 0.1 +
                    (total_stars or 0) * 2
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error getting template analytics: {e}")
            raise
    
    async def _track_search_analytics(self, query: str, total_results: int, returned_results: int) -> None:
        """Track search analytics for improvement."""
        try:
            # This could be extended to store search analytics in a separate table
            # For now, we'll just log it
            self.logger.info(
                f"Search analytics - Query: {query[:50]}, "
                f"Total results: {total_results}, Returned: {returned_results}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to track search analytics: {e}")
    
    async def get_marketplace_health_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive marketplace health metrics.
        
        Returns:
            Dictionary with health metrics
        """
        try:
            # Content metrics
            total_templates = await self.session.scalar(
                select(func.count(Template.id))
            )
            public_templates = await self.session.scalar(
                select(func.count(Template.id)).where(Template.is_public == True)
            )
            featured_templates = await self.session.scalar(
                select(func.count(Template.id)).where(Template.is_featured == True)
            )
            
            # Activity metrics (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            recent_downloads = await self.session.scalar(
                select(func.count(TemplateDownload.id)).where(
                    TemplateDownload.downloaded_at >= thirty_days_ago
                )
            )
            
            recent_views = await self.session.scalar(
                select(func.count(TemplateView.id)).where(
                    TemplateView.viewed_at >= thirty_days_ago
                )
            )
            
            recent_ratings = await self.session.scalar(
                select(func.count(TemplateRating.id)).where(
                    TemplateRating.created_at >= thirty_days_ago
                )
            )
            
            # User engagement
            active_users = await self.session.scalar(
                select(func.count(func.distinct(TemplateDownload.user_id))).where(
                    TemplateDownload.downloaded_at >= thirty_days_ago
                )
            )
            
            # Quality metrics
            avg_rating = await self.session.scalar(
                select(func.avg(Template.average_rating)).where(
                    and_(Template.is_public == True, Template.rating_count > 0)
                )
            )
            
            high_quality_templates = await self.session.scalar(
                select(func.count(Template.id)).where(
                    and_(
                        Template.is_public == True,
                        Template.average_rating >= 4.0,
                        Template.rating_count >= 5
                    )
                )
            )
            
            return {
                'content_health': {
                    'total_templates': total_templates or 0,
                    'public_templates': public_templates or 0,
                    'featured_templates': featured_templates or 0,
                    'visibility_rate': (public_templates or 0) / max(total_templates or 1, 1) * 100
                },
                'activity_health': {
                    'recent_downloads': recent_downloads or 0,
                    'recent_views': recent_views or 0,
                    'recent_ratings': recent_ratings or 0,
                    'active_users': active_users or 0,
                    'conversion_rate': (recent_downloads or 0) / max(recent_views or 1, 1) * 100
                },
                'quality_health': {
                    'average_rating': float(avg_rating or 0),
                    'high_quality_templates': high_quality_templates or 0,
                    'quality_rate': (high_quality_templates or 0) / max(public_templates or 1, 1) * 100
                },
                'overall_health_score': min(
                    ((public_templates or 0) / max(total_templates or 1, 1) * 100) * 0.3 +
                    min((recent_downloads or 0) / 10, 100) * 0.4 +
                    float(avg_rating or 0) / 5 * 100 * 0.3,
                    100
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error getting marketplace health metrics: {e}")
            raise