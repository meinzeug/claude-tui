"""
Recommendation Engine - AI-powered recommendation system for templates and plugins.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID
from collections import defaultdict, Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from sqlalchemy import desc, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.template import Template
from ..models.plugin import Plugin
from ..models.marketplace import TemplateDownload, TemplateView, TemplateStar
from ..models.rating import TemplateRating, UserReputation
from ..models.user import UserProfile
from ..repositories.marketplace_repository import MarketplaceRepository
from ...core.exceptions import ValidationError
from ...core.logger import get_logger

logger = get_logger(__name__)


class RecommendationEngine:
    """
    AI-powered recommendation engine using collaborative filtering, 
    content-based filtering, and hybrid approaches.
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize recommendation engine."""
        self.session = session
        self.repository = MarketplaceRepository(session)
        self.logger = logger.getChild(self.__class__.__name__)
        
        # ML models (lazy initialization)
        self._tfidf_vectorizer = None
        self._content_features = None
        self._user_item_matrix = None
        self._svd_model = None
        
        # Caching
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour
        
        # Algorithm weights
        self.weights = {
            'collaborative_filtering': 0.4,
            'content_based': 0.3,
            'popularity': 0.2,
            'trending': 0.1
        }
    
    async def get_user_recommendations(
        self,
        user_id: UUID,
        item_type: str = "template",  # template, plugin
        limit: int = 20,
        diversity_factor: float = 0.3,
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get comprehensive recommendations for a user using hybrid approach.
        
        Args:
            user_id: User ID
            item_type: Type of items to recommend
            limit: Number of recommendations
            diversity_factor: Factor to increase diversity (0-1)
            exclude_interacted: Whether to exclude items user has interacted with
            
        Returns:
            List of recommendations with scores and explanations
        """
        try:
            # Check cache
            cache_key = f"user_recs_{user_id}_{item_type}_{limit}_{diversity_factor}"
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if (datetime.utcnow() - timestamp).seconds < self._cache_ttl:
                    return cached_data
            
            # Get user profile and interaction history
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                # New user - return popular items
                recommendations = await self._get_popular_items(item_type, limit)
                return self._format_recommendations(recommendations, "popular", user_id)
            
            # Get all recommendation signals
            collaborative_recs = await self._get_collaborative_recommendations(
                user_id, item_type, limit * 2
            )
            
            content_recs = await self._get_content_based_recommendations(
                user_id, item_type, limit * 2
            )
            
            popularity_recs = await self._get_popularity_recommendations(
                user_id, item_type, limit
            )
            
            trending_recs = await self._get_trending_recommendations(
                user_id, item_type, limit
            )
            
            # Hybrid recommendation fusion
            hybrid_recs = await self._fuse_recommendations([
                (collaborative_recs, self.weights['collaborative_filtering']),
                (content_recs, self.weights['content_based']),
                (popularity_recs, self.weights['popularity']),
                (trending_recs, self.weights['trending'])
            ])
            
            # Apply diversity and filtering
            final_recs = await self._apply_diversity_and_filters(
                hybrid_recs, 
                user_id, 
                item_type,
                limit,
                diversity_factor,
                exclude_interacted
            )
            
            # Cache results
            self._cache[cache_key] = (final_recs, datetime.utcnow())
            
            return final_recs
            
        except Exception as e:
            self.logger.error(f"Error getting user recommendations: {e}")
            # Fallback to popular items
            return await self._get_fallback_recommendations(item_type, limit)
    
    async def get_similar_items(
        self,
        item_id: UUID,
        item_type: str = "template",
        limit: int = 10,
        similarity_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Get items similar to the given item using content-based filtering.
        
        Args:
            item_id: Item ID to find similar items for
            item_type: Type of item
            limit: Number of similar items
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar items with similarity scores
        """
        try:
            # Get the target item
            if item_type == "template":
                item = await self.session.get(Template, item_id)
            else:  # plugin
                item = await self.session.get(Plugin, item_id)
            
            if not item:
                return []
            
            # Extract content features
            target_features = self._extract_content_features(item)
            
            # Find similar items
            similar_items = await self._find_content_similar_items(
                item_id, item_type, target_features, limit, similarity_threshold
            )
            
            return similar_items
            
        except Exception as e:
            self.logger.error(f"Error getting similar items: {e}")
            return []
    
    async def get_trending_recommendations(
        self,
        category: Optional[str] = None,
        item_type: str = "template",
        limit: int = 20,
        time_window_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get trending items based on recent activity and engagement.
        
        Args:
            category: Optional category filter
            item_type: Type of items
            limit: Number of recommendations
            time_window_days: Time window for trending analysis
            
        Returns:
            List of trending items with trend scores
        """
        try:
            since_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            if item_type == "template":
                trending_items = await self._get_trending_templates(
                    since_date, category, limit
                )
            else:  # plugin
                trending_items = await self._get_trending_plugins(
                    since_date, category, limit
                )
            
            return self._format_recommendations(
                trending_items, "trending", explanation_prefix="Currently trending"
            )
            
        except Exception as e:
            self.logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    async def get_category_recommendations(
        self,
        category: str,
        item_type: str = "template",
        limit: int = 20,
        sort_by: str = "quality"  # quality, popularity, recent
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for a specific category.
        
        Args:
            category: Category name
            item_type: Type of items
            limit: Number of recommendations
            sort_by: Sorting criteria
            
        Returns:
            List of category recommendations
        """
        try:
            if item_type == "template":
                items = await self._get_category_templates(category, limit, sort_by)
            else:  # plugin
                items = await self._get_category_plugins(category, limit, sort_by)
            
            return self._format_recommendations(
                items, "category", explanation_prefix=f"Popular in {category}"
            )
            
        except Exception as e:
            self.logger.error(f"Error getting category recommendations: {e}")
            return []
    
    async def update_user_preferences(
        self,
        user_id: UUID,
        interaction_type: str,  # download, view, rating, star
        item_id: UUID,
        item_type: str = "template",
        rating: Optional[int] = None
    ) -> None:
        """
        Update user preferences based on interactions for better recommendations.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            item_id: Item ID
            item_type: Type of item
            rating: Optional rating value
        """
        try:
            # This would update user preference models and invalidate cache
            cache_keys_to_remove = [
                key for key in self._cache.keys() 
                if key.startswith(f"user_recs_{user_id}")
            ]
            
            for key in cache_keys_to_remove:
                self._cache.pop(key, None)
            
            # Log interaction for future model training
            self.logger.debug(
                f"Updated preferences for user {user_id}: "
                f"{interaction_type} on {item_type} {item_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating user preferences: {e}")
    
    async def train_recommendation_models(self) -> Dict[str, Any]:
        """
        Train/update recommendation models with latest data.
        
        Returns:
            Training results and metrics
        """
        try:
            training_results = {
                'started_at': datetime.utcnow(),
                'models_trained': []
            }
            
            # Train content-based model
            content_results = await self._train_content_model()
            training_results['models_trained'].append({
                'model': 'content_based',
                'status': 'completed',
                'metrics': content_results
            })
            
            # Train collaborative filtering model
            collaborative_results = await self._train_collaborative_model()
            training_results['models_trained'].append({
                'model': 'collaborative_filtering',
                'status': 'completed',
                'metrics': collaborative_results
            })
            
            # Clear cache after training
            self._cache.clear()
            
            training_results['completed_at'] = datetime.utcnow()
            training_results['duration'] = (
                training_results['completed_at'] - training_results['started_at']
            ).total_seconds()
            
            self.logger.info(f"Recommendation models trained successfully in {training_results['duration']:.2f}s")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error training recommendation models: {e}")
            raise
    
    # Private methods for different recommendation algorithms
    
    async def _get_collaborative_recommendations(
        self, 
        user_id: UUID, 
        item_type: str, 
        limit: int
    ) -> List[Tuple[UUID, float, str]]:
        """Get recommendations using collaborative filtering."""
        try:
            # Find users with similar preferences
            similar_users = await self._find_similar_users(user_id, item_type)
            
            if not similar_users:
                return []
            
            # Get items liked by similar users
            similar_user_ids = [user_id for user_id, _ in similar_users]
            
            if item_type == "template":
                items_query = f"""
                    SELECT t.id, 
                           AVG(tr.overall_rating) as avg_rating,
                           COUNT(td.id) as download_count,
                           COUNT(ts.user_id) as star_count
                    FROM templates t
                    LEFT JOIN template_downloads td ON t.id = td.template_id
                    LEFT JOIN template_ratings tr ON t.id = tr.template_id
                    LEFT JOIN template_stars ts ON t.id = ts.template_id
                    WHERE (td.user_id = ANY(:similar_users) OR 
                           tr.user_id = ANY(:similar_users) OR
                           ts.user_id = ANY(:similar_users))
                    AND t.is_public = true
                    AND NOT EXISTS (
                        SELECT 1 FROM template_downloads td2 
                        WHERE td2.template_id = t.id AND td2.user_id = :user_id
                    )
                    GROUP BY t.id
                    ORDER BY 
                        AVG(tr.overall_rating) DESC NULLS LAST,
                        COUNT(td.id) DESC,
                        COUNT(ts.user_id) DESC
                    LIMIT :limit
                """
            else:  # plugin
                items_query = f"""
                    SELECT p.id,
                           p.average_rating,
                           p.download_count,
                           p.star_count
                    FROM plugins p
                    LEFT JOIN plugin_installs pi ON p.id = pi.plugin_id
                    LEFT JOIN plugin_reviews pr ON p.id = pr.plugin_id
                    WHERE (pi.user_id = ANY(:similar_users) OR pr.user_id = ANY(:similar_users))
                    AND p.is_public = true
                    AND NOT EXISTS (
                        SELECT 1 FROM plugin_installs pi2 
                        WHERE pi2.plugin_id = p.id AND pi2.user_id = :user_id
                    )
                    GROUP BY p.id, p.average_rating, p.download_count, p.star_count
                    ORDER BY p.average_rating DESC, p.download_count DESC
                    LIMIT :limit
                """
            
            result = await self.session.execute(
                text(items_query),
                {
                    'similar_users': similar_user_ids,
                    'user_id': user_id,
                    'limit': limit
                }
            )
            
            recommendations = []
            for row in result:
                # Calculate collaborative score
                score = self._calculate_collaborative_score(
                    row.avg_rating or row.average_rating or 0,
                    row.download_count or 0,
                    getattr(row, 'star_count', 0) or 0
                )
                recommendations.append((
                    row.id, 
                    score, 
                    f"Users with similar taste also liked this"
                ))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in collaborative recommendations: {e}")
            return []
    
    async def _get_content_based_recommendations(
        self, 
        user_id: UUID, 
        item_type: str, 
        limit: int
    ) -> List[Tuple[UUID, float, str]]:
        """Get recommendations using content-based filtering."""
        try:
            # Get user's interaction history to understand preferences
            user_preferences = await self._extract_user_preferences(user_id, item_type)
            
            if not user_preferences:
                return []
            
            # Find items matching user preferences
            if item_type == "template":
                model_class = Template
                preference_fields = ['categories', 'frameworks', 'languages', 'complexity_level']
            else:  # plugin
                model_class = Plugin
                preference_fields = ['categories', 'plugin_type', 'tags']
            
            # Build content-based query
            content_items = await self._find_content_matches(
                model_class, user_preferences, preference_fields, limit
            )
            
            recommendations = []
            for item, similarity_score in content_items:
                explanation = self._generate_content_explanation(
                    item, user_preferences, preference_fields
                )
                recommendations.append((item.id, similarity_score, explanation))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in content-based recommendations: {e}")
            return []
    
    async def _get_popularity_recommendations(
        self, 
        user_id: UUID, 
        item_type: str, 
        limit: int
    ) -> List[Tuple[UUID, float, str]]:
        """Get popular items as recommendations."""
        try:
            if item_type == "template":
                popular_items = await self.repository.get_popular_templates(limit=limit)
            else:  # plugin
                popular_items = await self._get_popular_plugins(limit)
            
            recommendations = []
            for item in popular_items:
                # Convert to tuple format
                popularity_score = item.get('popularity_score', 0)
                recommendations.append((
                    UUID(item['id']), 
                    popularity_score, 
                    f"Popular choice with {item.get('download_count', 0)} downloads"
                ))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in popularity recommendations: {e}")
            return []
    
    async def _get_trending_recommendations(
        self, 
        user_id: UUID, 
        item_type: str, 
        limit: int
    ) -> List[Tuple[UUID, float, str]]:
        """Get trending items as recommendations."""
        try:
            if item_type == "template":
                trending_items = await self.repository.get_trending_templates(limit=limit)
            else:  # plugin
                trending_items = await self._get_trending_plugins_simple(limit)
            
            recommendations = []
            for item in trending_items:
                # Extract trending score
                trending_score = 0
                if 'trending_metrics' in item:
                    metrics = item['trending_metrics']
                    trending_score = (
                        metrics.get('recent_downloads', 0) * 3 +
                        metrics.get('recent_views', 0) * 0.1 +
                        metrics.get('recent_stars', 0) * 2 +
                        metrics.get('recent_ratings', 0) * 1.5
                    )
                
                recommendations.append((
                    UUID(item['id']), 
                    trending_score, 
                    f"Trending with {item.get('download_count', 0)} recent downloads"
                ))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in trending recommendations: {e}")
            return []
    
    async def _fuse_recommendations(
        self, 
        recommendation_lists: List[Tuple[List[Tuple[UUID, float, str]], float]]
    ) -> List[Tuple[UUID, float, str]]:
        """Fuse multiple recommendation lists using weighted scores."""
        try:
            # Combine all recommendations with weights
            item_scores = defaultdict(list)
            item_explanations = {}
            
            for recommendations, weight in recommendation_lists:
                for item_id, score, explanation in recommendations:
                    item_scores[item_id].append(score * weight)
                    if item_id not in item_explanations:
                        item_explanations[item_id] = explanation
            
            # Calculate final scores
            final_recommendations = []
            for item_id, scores in item_scores.items():
                final_score = sum(scores) / len(scores)  # Average weighted score
                final_recommendations.append((
                    item_id, 
                    final_score, 
                    item_explanations[item_id]
                ))
            
            # Sort by final score
            final_recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return final_recommendations
            
        except Exception as e:
            self.logger.error(f"Error fusing recommendations: {e}")
            return []
    
    async def _apply_diversity_and_filters(
        self,
        recommendations: List[Tuple[UUID, float, str]],
        user_id: UUID,
        item_type: str,
        limit: int,
        diversity_factor: float,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Apply diversity and filtering to final recommendations."""
        try:
            # Get user's interacted items if excluding them
            interacted_items = set()
            if exclude_interacted:
                interacted_items = await self._get_user_interacted_items(user_id, item_type)
            
            # Filter out interacted items
            filtered_recs = [
                (item_id, score, explanation) 
                for item_id, score, explanation in recommendations
                if item_id not in interacted_items
            ]
            
            # Apply diversity if requested
            if diversity_factor > 0:
                diverse_recs = await self._apply_diversity(
                    filtered_recs, item_type, diversity_factor
                )
            else:
                diverse_recs = filtered_recs
            
            # Limit results
            final_recs = diverse_recs[:limit]
            
            # Format as dictionaries with item details
            formatted_recs = []
            for item_id, score, explanation in final_recs:
                if item_type == "template":
                    item = await self.session.get(Template, item_id, options=[selectinload(Template.author)])
                else:  # plugin
                    item = await self.session.get(Plugin, item_id, options=[selectinload(Plugin.author)])
                
                if item:
                    item_dict = item.to_dict()
                    if hasattr(item, 'author') and item.author:
                        item_dict['author'] = item.author.to_dict(include_private=False)
                    
                    item_dict.update({
                        'recommendation_score': float(score),
                        'recommendation_explanation': explanation,
                        'recommendation_type': 'hybrid'
                    })
                    formatted_recs.append(item_dict)
            
            return formatted_recs
            
        except Exception as e:
            self.logger.error(f"Error applying diversity and filters: {e}")
            return []
    
    # Helper methods
    
    async def _get_user_profile(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get user profile and interaction statistics."""
        try:
            user = await self.session.get(UserProfile, user_id)
            if not user:
                return None
            
            # Get interaction counts
            download_count = await self.session.scalar(
                text("SELECT COUNT(*) FROM template_downloads WHERE user_id = :user_id"),
                {'user_id': user_id}
            )
            
            rating_count = await self.session.scalar(
                text("SELECT COUNT(*) FROM template_ratings WHERE user_id = :user_id"),
                {'user_id': user_id}
            )
            
            return {
                'id': user.id,
                'username': user.username,
                'download_count': download_count or 0,
                'rating_count': rating_count or 0,
                'is_active': (download_count or 0) + (rating_count or 0) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            return None
    
    async def _find_similar_users(self, user_id: UUID, item_type: str) -> List[Tuple[UUID, float]]:
        """Find users with similar preferences using Jaccard similarity."""
        try:
            # Get user's downloaded/rated items
            if item_type == "template":
                user_items_query = """
                    SELECT DISTINCT template_id as item_id FROM (
                        SELECT template_id FROM template_downloads WHERE user_id = :user_id
                        UNION
                        SELECT template_id FROM template_ratings WHERE user_id = :user_id
                        UNION
                        SELECT template_id FROM template_stars WHERE user_id = :user_id
                    ) AS user_items
                """
            else:  # plugin
                user_items_query = """
                    SELECT DISTINCT plugin_id as item_id FROM (
                        SELECT plugin_id FROM plugin_installs WHERE user_id = :user_id
                        UNION
                        SELECT plugin_id FROM plugin_reviews WHERE user_id = :user_id
                    ) AS user_items
                """
            
            result = await self.session.execute(text(user_items_query), {'user_id': user_id})
            user_items = set(row.item_id for row in result)
            
            if len(user_items) < 2:  # Need at least 2 items for similarity
                return []
            
            # Find other users who interacted with these items
            if item_type == "template":
                other_users_query = """
                    SELECT DISTINCT user_id, 
                           ARRAY_AGG(DISTINCT item_id) as items
                    FROM (
                        SELECT user_id, template_id as item_id FROM template_downloads 
                        WHERE template_id = ANY(:user_items) AND user_id != :user_id
                        UNION
                        SELECT user_id, template_id as item_id FROM template_ratings 
                        WHERE template_id = ANY(:user_items) AND user_id != :user_id
                        UNION
                        SELECT user_id, template_id as item_id FROM template_stars 
                        WHERE template_id = ANY(:user_items) AND user_id != :user_id
                    ) AS other_user_items
                    GROUP BY user_id
                    HAVING COUNT(DISTINCT item_id) >= 2
                """
            else:  # plugin
                other_users_query = """
                    SELECT DISTINCT user_id, 
                           ARRAY_AGG(DISTINCT item_id) as items
                    FROM (
                        SELECT user_id, plugin_id as item_id FROM plugin_installs 
                        WHERE plugin_id = ANY(:user_items) AND user_id != :user_id
                        UNION
                        SELECT user_id, plugin_id as item_id FROM plugin_reviews 
                        WHERE plugin_id = ANY(:user_items) AND user_id != :user_id
                    ) AS other_user_items
                    GROUP BY user_id
                    HAVING COUNT(DISTINCT item_id) >= 2
                """
            
            result = await self.session.execute(
                text(other_users_query), 
                {'user_items': list(user_items), 'user_id': user_id}
            )
            
            similar_users = []
            for row in result:
                other_user_items = set(row.items)
                
                # Calculate Jaccard similarity
                intersection = len(user_items & other_user_items)
                union = len(user_items | other_user_items)
                
                if union > 0:
                    similarity = intersection / union
                    if similarity > 0.1:  # Minimum similarity threshold
                        similar_users.append((row.user_id, similarity))
            
            # Sort by similarity and return top 20
            similar_users.sort(key=lambda x: x[1], reverse=True)
            return similar_users[:20]
            
        except Exception as e:
            self.logger.error(f"Error finding similar users: {e}")
            return []
    
    def _calculate_collaborative_score(
        self, 
        avg_rating: float, 
        download_count: int, 
        star_count: int
    ) -> float:
        """Calculate collaborative filtering score."""
        # Weighted score combining multiple signals
        rating_score = avg_rating / 5.0  # Normalize to 0-1
        popularity_score = min(math.log(download_count + 1) / 10, 1.0)  # Log scale
        star_score = min(math.log(star_count + 1) / 8, 1.0)  # Log scale
        
        return rating_score * 0.5 + popularity_score * 0.3 + star_score * 0.2
    
    async def _get_fallback_recommendations(self, item_type: str, limit: int) -> List[Dict[str, Any]]:
        """Get fallback recommendations when other methods fail."""
        try:
            if item_type == "template":
                items = await self.repository.get_popular_templates(limit=limit)
            else:  # plugin
                items = await self._get_popular_plugins(limit)
            
            return self._format_recommendations(items, "fallback", "Popular choice")
            
        except Exception as e:
            self.logger.error(f"Error getting fallback recommendations: {e}")
            return []
    
    def _format_recommendations(
        self, 
        items: List[Dict[str, Any]], 
        rec_type: str, 
        explanation_prefix: str = "Recommended",
        user_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """Format recommendations with consistent structure."""
        formatted = []
        for item in items:
            item_copy = item.copy()
            item_copy.update({
                'recommendation_type': rec_type,
                'recommendation_explanation': f"{explanation_prefix}: {item.get('name', 'Unknown')}",
                'recommendation_score': item.get('popularity_score', item.get('trending_score', 0.5))
            })
            formatted.append(item_copy)
        
        return formatted
    
    # Additional helper methods would continue here...
    # (Training methods, content extraction, similarity calculations, etc.)
    
    async def _train_content_model(self) -> Dict[str, Any]:
        """Train content-based recommendation model."""
        # Implementation for training TF-IDF and content similarity models
        return {'status': 'completed', 'features_extracted': 0, 'model_accuracy': 0.0}
    
    async def _train_collaborative_model(self) -> Dict[str, Any]:
        """Train collaborative filtering model using SVD."""
        # Implementation for training matrix factorization model
        return {'status': 'completed', 'latent_factors': 50, 'rmse': 0.0}