"""
Enhanced Marketplace Service - AI-powered marketplace with advanced features.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from collections import defaultdict

from sqlalchemy import desc, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.template import Template, TemplateSearchFilters, TemplateStats
from ..models.marketplace import (
    FeaturedCollection, TemplateTag, TemplateDownload, 
    TemplateView, TemplateStar, TemplateFork, MarketplaceStats, MarketplaceTrends
)
from ..models.rating import TemplateRating, UserReputation
from ..models.user import UserProfile
from ..repositories.marketplace_repository import MarketplaceRepository
from .recommendation_engine import RecommendationEngine
from .cache_service import CacheService
from ...core.exceptions import ValidationError, NotFoundError
from ...core.logger import get_logger

logger = get_logger(__name__)


class EnhancedMarketplaceService:
    """
    Enhanced marketplace service with AI-powered recommendations,
    advanced analytics, and intelligent caching.
    """
    
    def __init__(self, session: AsyncSession, cache_service: Optional[CacheService] = None):
        """Initialize enhanced marketplace service."""
        self.session = session
        self.repository = MarketplaceRepository(session)
        self.recommendation_engine = RecommendationEngine(session)
        self.cache = cache_service or CacheService()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Configuration
        self.search_boost_factors = {
            'featured': 2.0,
            'verified_author': 1.5,
            'high_rating': 1.3,
            'recent_activity': 1.2
        }
    
    async def search_templates_advanced(
        self,
        filters: TemplateSearchFilters,
        user_id: Optional[UUID] = None,
        page: int = 1,
        page_size: int = 20,
        personalize: bool = True
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        """
        Advanced template search with AI-powered ranking and personalization.
        
        Args:
            filters: Search filters
            user_id: User ID for personalization
            page: Page number
            page_size: Results per page
            personalize: Whether to personalize results
            
        Returns:
            Tuple of (templates, total_count, search_metadata)
        """
        try:
            # Check cache first
            cache_key = f"search:{filters.query or 'all'}:{page}:{page_size}:{user_id or 'anon'}"
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Use repository's advanced search
            templates, total_count, metadata = await self.repository.search_templates_advanced(
                filters, page, page_size
            )
            
            # Apply personalization if user provided
            if personalize and user_id:
                templates = await self._personalize_search_results(
                    templates, user_id, filters.query
                )
            
            # Convert templates to dictionaries with enhanced data
            template_dicts = []
            for template in templates:
                template_dict = template.to_dict()
                
                # Add author information
                if hasattr(template, 'author') and template.author:
                    template_dict['author'] = template.author.to_dict(include_private=False)
                
                # Add enhanced metrics
                template_dict.update(await self._get_enhanced_template_metrics(template.id))
                
                # Add personalization score if applicable
                if personalize and user_id:
                    template_dict['personalization_score'] = await self._calculate_personalization_score(
                        template.id, user_id
                    )
                
                template_dicts.append(template_dict)
            
            # Cache results for 15 minutes
            result = (template_dicts, total_count, metadata)
            await self.cache.set(cache_key, result, ttl=900)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in advanced template search: {e}")
            raise ValidationError(f"Search failed: {str(e)}")
    
    async def get_personalized_recommendations(
        self,
        user_id: UUID,
        limit: int = 20,
        diversity_factor: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Get AI-powered personalized recommendations for a user.
        
        Args:
            user_id: User ID
            limit: Number of recommendations
            diversity_factor: Diversity factor for recommendations
            
        Returns:
            List of personalized template recommendations
        """
        try:
            # Check cache
            cache_key = f"user_recs:{user_id}:{limit}:{diversity_factor}"
            cached_recs = await self.cache.get(cache_key)
            if cached_recs:
                return cached_recs
            
            # Get recommendations from AI engine
            recommendations = await self.recommendation_engine.get_user_recommendations(
                user_id=user_id,
                item_type="template",
                limit=limit,
                diversity_factor=diversity_factor
            )
            
            # Enhance recommendations with marketplace data
            enhanced_recs = []
            for rec in recommendations:
                # Add real-time metrics
                rec.update(await self._get_enhanced_template_metrics(UUID(rec['id'])))
                
                # Add trending information
                rec['is_trending'] = await self._is_template_trending(UUID(rec['id']))
                
                # Add quality indicators
                rec['quality_score'] = await self._calculate_quality_score(UUID(rec['id']))
                
                enhanced_recs.append(rec)
            
            # Cache for 30 minutes
            await self.cache.set(cache_key, enhanced_recs, ttl=1800)
            
            return enhanced_recs
            
        except Exception as e:
            self.logger.error(f"Error getting personalized recommendations: {e}")
            # Fallback to popular templates
            return await self.get_popular_templates(limit=limit)
    
    async def get_trending_templates_with_analytics(
        self,
        period_days: int = 7,
        limit: int = 20,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trending templates with detailed analytics and predictions.
        
        Args:
            period_days: Period to analyze for trending
            limit: Number of results
            category: Optional category filter
            
        Returns:
            List of trending templates with analytics
        """
        try:
            # Check cache
            cache_key = f"trending:{period_days}:{limit}:{category or 'all'}"
            cached_trending = await self.cache.get(cache_key)
            if cached_trending:
                return cached_trending
            
            # Get trending from repository
            trending_templates = await self.repository.get_trending_templates(
                period_days=period_days,
                limit=limit,
                category_filter=category
            )
            
            # Enhance with predictive analytics
            enhanced_trending = []
            for template in trending_templates:
                # Add trend prediction
                template['trend_prediction'] = await self._predict_future_trend(
                    UUID(template['id']), period_days
                )
                
                # Add growth metrics
                template['growth_metrics'] = await self._calculate_growth_metrics(
                    UUID(template['id']), period_days
                )
                
                # Add market position
                template['market_position'] = await self._calculate_market_position(
                    UUID(template['id']), template.get('trending_metrics', {})
                )
                
                enhanced_trending.append(template)
            
            # Cache for 1 hour
            await self.cache.set(cache_key, enhanced_trending, ttl=3600)
            
            return enhanced_trending
            
        except Exception as e:
            self.logger.error(f"Error getting trending templates with analytics: {e}")
            raise
    
    async def get_marketplace_insights(
        self,
        time_period: str = "30d"  # 7d, 30d, 90d, 1y
    ) -> Dict[str, Any]:
        """
        Get comprehensive marketplace insights and analytics.
        
        Args:
            time_period: Time period for analysis
            
        Returns:
            Dictionary with marketplace insights
        """
        try:
            # Parse time period
            days_map = {'7d': 7, '30d': 30, '90d': 90, '1y': 365}
            days = days_map.get(time_period, 30)
            
            # Check cache
            cache_key = f"marketplace_insights:{time_period}"
            cached_insights = await self.cache.get(cache_key)
            if cached_insights:
                return cached_insights
            
            # Get base marketplace stats
            base_stats = await self.repository.get_marketplace_health_metrics()
            
            # Get advanced insights
            insights = {
                'overview': base_stats,
                'trends': await self._analyze_marketplace_trends(days),
                'user_behavior': await self._analyze_user_behavior(days),
                'content_quality': await self._analyze_content_quality(days),
                'growth_projections': await self._calculate_growth_projections(days),
                'recommendations': await self._generate_marketplace_recommendations(),
                'performance_metrics': await self._calculate_performance_metrics(days)
            }
            
            # Cache for 2 hours
            await self.cache.set(cache_key, insights, ttl=7200)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting marketplace insights: {e}")
            raise
    
    async def create_featured_collection(
        self,
        name: str,
        description: str,
        collection_type: str,
        curator_id: UUID,
        template_ids: Optional[List[UUID]] = None,
        auto_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new featured collection with intelligent template selection.
        
        Args:
            name: Collection name
            description: Collection description
            collection_type: Type of collection
            curator_id: Curator user ID
            template_ids: Manual template selection
            auto_criteria: Criteria for automatic selection
            
        Returns:
            Created collection data
        """
        try:
            # Create slug from name
            slug = name.lower().replace(' ', '-').replace('_', '-')
            
            # Validate template IDs or apply auto criteria
            if template_ids:
                # Validate templates exist and are public
                valid_templates = await self._validate_templates_for_collection(template_ids)
            elif auto_criteria:
                # Auto-select templates based on criteria
                valid_templates = await self._auto_select_templates(auto_criteria, limit=20)
            else:
                valid_templates = []
            
            # Create collection
            collection = FeaturedCollection(
                slug=slug,
                name=name,
                description=description,
                collection_type=collection_type,
                curator_id=curator_id,
                template_ids=[str(tid) for tid in valid_templates],
                template_count=len(valid_templates),
                is_dynamic=bool(auto_criteria)
            )
            
            self.session.add(collection)
            await self.session.commit()
            await self.session.refresh(collection)
            
            # Clear relevant caches
            await self.cache.delete_pattern("featured_collections:*")
            await self.cache.delete_pattern("marketplace_insights:*")
            
            return collection.to_dict()
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error creating featured collection: {e}")
            raise ValidationError(f"Failed to create collection: {str(e)}")
    
    async def update_featured_collection(
        self,
        collection_id: UUID,
        updates: Dict[str, Any],
        refresh_auto: bool = True
    ) -> Dict[str, Any]:
        """
        Update a featured collection with intelligent refresh.
        
        Args:
            collection_id: Collection ID
            updates: Update data
            refresh_auto: Whether to refresh auto-generated collections
            
        Returns:
            Updated collection data
        """
        try:
            # Get collection
            collection = await self.session.get(FeaturedCollection, collection_id)
            if not collection:
                raise NotFoundError("Collection not found")
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(collection, key):
                    setattr(collection, key, value)
            
            # Refresh auto-generated collections
            if collection.is_dynamic and refresh_auto and collection.auto_criteria:
                new_templates = await self._auto_select_templates(
                    collection.auto_criteria, 
                    limit=collection.max_templates
                )
                collection.template_ids = [str(tid) for tid in new_templates]
                collection.template_count = len(new_templates)
            
            collection.updated_at = datetime.utcnow()
            await self.session.commit()
            
            # Clear caches
            await self.cache.delete_pattern(f"collection:{collection_id}:*")
            await self.cache.delete_pattern("featured_collections:*")
            
            return collection.to_dict()
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error updating featured collection: {e}")
            raise
    
    async def track_user_interaction(
        self,
        user_id: UUID,
        template_id: UUID,
        interaction_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track user interactions for improved recommendations.
        
        Args:
            user_id: User ID
            template_id: Template ID
            interaction_type: Type of interaction
            metadata: Additional interaction metadata
        """
        try:
            # Record in appropriate table based on interaction type
            if interaction_type == "download":
                download = TemplateDownload(
                    template_id=template_id,
                    user_id=user_id,
                    download_type=metadata.get('download_type', 'direct') if metadata else 'direct',
                    project_name=metadata.get('project_name') if metadata else None,
                    intended_use=metadata.get('intended_use') if metadata else None
                )
                self.session.add(download)
                
                # Update template download count
                template = await self.session.get(Template, template_id)
                if template:
                    template.download_count += 1
                    template.last_used_at = datetime.utcnow()
            
            elif interaction_type == "view":
                view = TemplateView(
                    template_id=template_id,
                    user_id=user_id,
                    view_duration=metadata.get('duration') if metadata else None,
                    referrer=metadata.get('referrer') if metadata else None
                )
                self.session.add(view)
            
            elif interaction_type == "star":
                # Check if already starred
                existing_star = await self.session.get(TemplateStar, (user_id, template_id))
                if not existing_star:
                    star = TemplateStar(user_id=user_id, template_id=template_id)
                    self.session.add(star)
                    
                    # Update template star count
                    template = await self.session.get(Template, template_id)
                    if template:
                        template.star_count += 1
            
            await self.session.commit()
            
            # Update recommendation engine
            await self.recommendation_engine.update_user_preferences(
                user_id=user_id,
                interaction_type=interaction_type,
                item_id=template_id,
                item_type="template"
            )
            
            # Clear user-specific caches
            await self.cache.delete_pattern(f"user_recs:{user_id}:*")
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error tracking user interaction: {e}")
    
    async def get_template_analytics_detailed(
        self,
        template_id: UUID,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get detailed analytics for a specific template.
        
        Args:
            template_id: Template ID
            days: Number of days to analyze
            
        Returns:
            Detailed analytics data
        """
        try:
            # Check cache
            cache_key = f"template_analytics:{template_id}:{days}"
            cached_analytics = await self.cache.get(cache_key)
            if cached_analytics:
                return cached_analytics
            
            # Get base analytics from repository
            base_analytics = await self.repository.get_template_analytics(template_id, days)
            
            # Add advanced analytics
            advanced_analytics = {
                **base_analytics,
                'user_segments': await self._analyze_user_segments(template_id, days),
                'geographic_distribution': await self._analyze_geographic_usage(template_id, days),
                'usage_patterns': await self._analyze_usage_patterns(template_id, days),
                'competitive_analysis': await self._analyze_competitive_position(template_id),
                'optimization_suggestions': await self._generate_optimization_suggestions(template_id),
                'forecast': await self._forecast_template_performance(template_id, days)
            }
            
            # Cache for 1 hour
            await self.cache.set(cache_key, advanced_analytics, ttl=3600)
            
            return advanced_analytics
            
        except Exception as e:
            self.logger.error(f"Error getting detailed template analytics: {e}")
            raise
    
    # Private helper methods
    
    async def _personalize_search_results(
        self,
        templates: List[Template],
        user_id: UUID,
        query: Optional[str]
    ) -> List[Template]:
        """Personalize search results based on user preferences."""
        try:
            # Get user preferences
            user_downloads = await self.session.execute(
                text("""
                    SELECT template_id, COUNT(*) as interaction_count
                    FROM (
                        SELECT template_id FROM template_downloads WHERE user_id = :user_id
                        UNION ALL
                        SELECT template_id FROM template_ratings WHERE user_id = :user_id
                        UNION ALL
                        SELECT template_id FROM template_stars WHERE user_id = :user_id
                    ) AS interactions
                    GROUP BY template_id
                """),
                {'user_id': user_id}
            )
            
            interacted_templates = {row.template_id: row.interaction_count for row in user_downloads}
            
            # Calculate personalization scores and re-rank
            personalized_templates = []
            for template in templates:
                base_score = template.average_rating * template.rating_count
                
                # Boost based on similar templates user liked
                personalization_boost = 0
                if template.id in interacted_templates:
                    personalization_boost = interacted_templates[template.id] * 0.1
                
                # Additional boost for similar categories/frameworks
                # This would require more complex similarity calculation
                
                final_score = base_score + personalization_boost
                personalized_templates.append((template, final_score))
            
            # Sort by personalized score
            personalized_templates.sort(key=lambda x: x[1], reverse=True)
            
            return [template for template, _ in personalized_templates]
            
        except Exception as e:
            self.logger.warning(f"Error personalizing search results: {e}")
            return templates  # Return original order if personalization fails
    
    async def _get_enhanced_template_metrics(self, template_id: UUID) -> Dict[str, Any]:
        """Get enhanced metrics for a template."""
        try:
            # Get recent activity (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            recent_downloads = await self.session.scalar(
                text("SELECT COUNT(*) FROM template_downloads WHERE template_id = :id AND downloaded_at >= :date"),
                {'id': template_id, 'date': thirty_days_ago}
            )
            
            recent_views = await self.session.scalar(
                text("SELECT COUNT(*) FROM template_views WHERE template_id = :id AND viewed_at >= :date"),
                {'id': template_id, 'date': thirty_days_ago}
            )
            
            # Calculate momentum score
            momentum_score = (recent_downloads or 0) * 3 + (recent_views or 0) * 0.1
            
            return {
                'recent_downloads': recent_downloads or 0,
                'recent_views': recent_views or 0,
                'momentum_score': momentum_score,
                'activity_level': 'high' if momentum_score > 50 else 'medium' if momentum_score > 10 else 'low'
            }
            
        except Exception as e:
            self.logger.warning(f"Error getting enhanced template metrics: {e}")
            return {}
    
    async def _calculate_personalization_score(
        self,
        template_id: UUID,
        user_id: UUID
    ) -> float:
        """Calculate personalization score for a template and user."""
        try:
            # This would use the recommendation engine's similarity calculations
            # Simplified version here
            return 0.5  # Placeholder
            
        except Exception as e:
            self.logger.warning(f"Error calculating personalization score: {e}")
            return 0.0
    
    async def _is_template_trending(self, template_id: UUID) -> bool:
        """Check if a template is currently trending."""
        try:
            # Get recent activity and compare to baseline
            recent_activity = await self._get_enhanced_template_metrics(template_id)
            return recent_activity.get('activity_level') == 'high'
            
        except Exception as e:
            self.logger.warning(f"Error checking if template is trending: {e}")
            return False
    
    async def _calculate_quality_score(self, template_id: UUID) -> float:
        """Calculate quality score for a template."""
        try:
            template = await self.session.get(Template, template_id)
            if not template:
                return 0.0
            
            # Combine multiple quality factors
            rating_score = template.average_rating / 5.0 if template.average_rating else 0.0
            popularity_score = min(template.download_count / 1000, 1.0)  # Normalize
            engagement_score = min(template.star_count / 100, 1.0)  # Normalize
            
            quality_score = (
                rating_score * 0.5 +
                popularity_score * 0.3 +
                engagement_score * 0.2
            )
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality score: {e}")
            return 0.0
    
    # Additional private methods for analytics and insights would continue here...
    
    async def _analyze_marketplace_trends(self, days: int) -> Dict[str, Any]:
        """Analyze marketplace trends over the specified period."""
        return {'trending_categories': [], 'growth_rate': 0.0, 'top_creators': []}
    
    async def _analyze_user_behavior(self, days: int) -> Dict[str, Any]:
        """Analyze user behavior patterns."""
        return {'engagement_metrics': {}, 'user_journey': {}, 'retention_rates': {}}
    
    async def _analyze_content_quality(self, days: int) -> Dict[str, Any]:
        """Analyze content quality metrics."""
        return {'quality_distribution': {}, 'improvement_trends': {}, 'moderation_metrics': {}}
    
    async def _calculate_growth_projections(self, days: int) -> Dict[str, Any]:
        """Calculate growth projections."""
        return {'projected_downloads': 0, 'projected_templates': 0, 'confidence_intervals': {}}
    
    async def _generate_marketplace_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable marketplace recommendations."""
        return [{'type': 'content_gap', 'description': 'Consider adding more React templates'}]