"""
Enhanced Rating Service - Sophisticated rating system with weighted scoring and moderation.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from collections import defaultdict

from sqlalchemy import desc, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.rating import (
    TemplateRating, ReviewHelpfulness, ReviewReport, 
    UserReputation, ModerationQueue, ReviewStatus, ReportReason
)
from ..models.template import Template
from ..models.user import UserProfile
from .cache_service import CacheService
from .moderation_service import ModerationService
from ...core.exceptions import ValidationError, NotFoundError, PermissionError
from ...core.logger import get_logger

logger = get_logger(__name__)


class RatingService:
    """Service for rating and review management with moderation."""
    
    def __init__(self, db: AsyncSession):
        """Initialize rating service."""
        self.db = db
    
    async def create_rating(
        self,
        user_id: UUID,
        template_id: UUID,
        rating_data: Dict[str, Any]
    ) -> TemplateRating:
        """Create a new template rating/review."""
        try:
            # Check if user has already rated this template
            existing_rating = await self.db.query(TemplateRating).filter(
                and_(
                    TemplateRating.user_id == user_id,
                    TemplateRating.template_id == template_id
                )
            ).first()
            
            if existing_rating:
                raise ValidationError("User has already rated this template")
            
            # Verify template exists and is public
            template = await self.db.query(Template).filter(
                and_(
                    Template.id == template_id,
                    Template.is_public == True
                )
            ).first()
            
            if not template:
                raise NotFoundError("Template not found or not accessible")
            
            # Check if user has downloaded/used the template for verification
            download_query = await self.db.query(func.count()).filter(
                # Assuming TemplateDownload model exists
                and_(
                    # TemplateDownload.user_id == user_id,
                    # TemplateDownload.template_id == template_id
                )
            ).scalar()
            
            is_verified_usage = download_query > 0 if download_query else False
            
            # Create rating
            rating = TemplateRating(
                user_id=user_id,
                template_id=template_id,
                overall_rating=rating_data["overall_rating"],
                quality_rating=rating_data.get("quality_rating"),
                usability_rating=rating_data.get("usability_rating"),
                documentation_rating=rating_data.get("documentation_rating"),
                support_rating=rating_data.get("support_rating"),
                title=rating_data.get("title"),
                content=rating_data.get("content"),
                template_version_reviewed=rating_data.get("template_version_reviewed"),
                use_case=rating_data.get("use_case"),
                experience_level=rating_data.get("experience_level"),
                is_verified_usage=is_verified_usage,
                status=ReviewStatus.PENDING.value  # All reviews start as pending
            )
            
            self.db.add(rating)
            await self.db.commit()
            
            # Queue for moderation
            await self._queue_for_moderation(rating)
            
            # Update user reputation
            await self._update_user_reputation(user_id, "review_created")
            
            return rating
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating rating: {e}")
            raise
    
    async def update_rating(
        self,
        rating_id: UUID,
        user_id: UUID,
        update_data: Dict[str, Any]
    ) -> TemplateRating:
        """Update an existing rating."""
        try:
            rating = await self.db.query(TemplateRating).filter(
                and_(
                    TemplateRating.id == rating_id,
                    TemplateRating.user_id == user_id
                )
            ).first()
            
            if not rating:
                raise NotFoundError("Rating not found or not owned by user")
            
            # Update fields
            for field, value in update_data.items():
                if hasattr(rating, field) and value is not None:
                    setattr(rating, field, value)
            
            rating.updated_at = datetime.utcnow()
            rating.status = ReviewStatus.PENDING.value  # Re-queue for moderation
            
            await self.db.commit()
            
            # Re-queue for moderation
            await self._queue_for_moderation(rating)
            
            return rating
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating rating: {e}")
            raise
    
    async def get_template_ratings(
        self,
        template_id: UUID,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        include_pending: bool = False
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get ratings for a template."""
        try:
            # Build query
            query = self.db.query(TemplateRating).filter(
                TemplateRating.template_id == template_id
            )
            
            # Filter by status
            if not include_pending:
                query = query.filter(TemplateRating.status == ReviewStatus.APPROVED.value)
            
            # Count total
            total_count = await query.count()
            
            # Apply sorting
            sort_column = getattr(TemplateRating, sort_by, TemplateRating.created_at)
            if sort_order == "asc":
                query = query.order_by(sort_column)
            else:
                query = query.order_by(desc(sort_column))
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Include user information
            query = query.options(selectinload(TemplateRating.user))
            ratings = await query.all()
            
            # Convert to dictionaries
            rating_dicts = []
            for rating in ratings:
                rating_dict = rating.to_dict()
                rating_dict['user'] = rating.user.to_dict(include_private=False) if rating.user else None
                rating_dicts.append(rating_dict)
            
            return rating_dicts, total_count
            
        except Exception as e:
            logger.error(f"Error getting template ratings: {e}")
            raise
    
    async def get_rating_summary(self, template_id: UUID) -> Dict[str, Any]:
        """Get rating summary for a template."""
        try:
            # Get all approved ratings
            ratings = await self.db.query(TemplateRating).filter(
                and_(
                    TemplateRating.template_id == template_id,
                    TemplateRating.status == ReviewStatus.APPROVED.value
                )
            ).all()
            
            if not ratings:
                return {
                    'total_ratings': 0,
                    'average_rating': 0.0,
                    'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                    'detailed_averages': {}
                }
            
            # Calculate statistics
            total_ratings = len(ratings)
            total_overall = sum(r.overall_rating for r in ratings)
            average_rating = total_overall / total_ratings
            
            # Rating distribution
            distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for rating in ratings:
                distribution[rating.overall_rating] += 1
            
            # Detailed averages
            detailed_averages = {}
            for field in ['quality_rating', 'usability_rating', 'documentation_rating', 'support_rating']:
                values = [getattr(r, field) for r in ratings if getattr(r, field) is not None]
                if values:
                    detailed_averages[field] = sum(values) / len(values)
                else:
                    detailed_averages[field] = None
            
            return {
                'total_ratings': total_ratings,
                'average_rating': round(average_rating, 2),
                'rating_distribution': distribution,
                'detailed_averages': detailed_averages,
                'verified_ratings': len([r for r in ratings if r.is_verified_usage]),
                'recent_ratings': len([r for r in ratings if r.created_at >= datetime.utcnow() - timedelta(days=30)])
            }
            
        except Exception as e:
            logger.error(f"Error getting rating summary: {e}")
            raise
    
    async def vote_helpful(
        self,
        rating_id: UUID,
        user_id: UUID,
        is_helpful: bool
    ) -> bool:
        """Vote on review helpfulness."""
        try:
            # Check if user has already voted on this review
            existing_vote = await self.db.query(ReviewHelpfulness).filter(
                and_(
                    ReviewHelpfulness.review_id == rating_id,
                    ReviewHelpfulness.user_id == user_id
                )
            ).first()
            
            if existing_vote:
                # Update existing vote if different
                if existing_vote.is_helpful != is_helpful:
                    existing_vote.is_helpful = is_helpful
                    existing_vote.voted_at = datetime.utcnow()
                    
                    # Update helpfulness counts
                    await self._update_helpfulness_counts(rating_id)
                    await self.db.commit()
                    return True
                else:
                    return False  # Same vote, no change
            
            # Create new vote
            vote = ReviewHelpfulness(
                review_id=rating_id,
                user_id=user_id,
                is_helpful=is_helpful
            )
            
            self.db.add(vote)
            
            # Update helpfulness counts
            await self._update_helpfulness_counts(rating_id)
            
            await self.db.commit()
            
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error voting on helpfulness: {e}")
            raise
    
    async def report_review(
        self,
        review_id: UUID,
        reporter_id: UUID,
        reason: str,
        description: Optional[str] = None
    ) -> ReviewReport:
        """Report a review for moderation."""
        try:
            # Check if user has already reported this review
            existing_report = await self.db.query(ReviewReport).filter(
                and_(
                    ReviewReport.review_id == review_id,
                    ReviewReport.reporter_id == reporter_id
                )
            ).first()
            
            if existing_report:
                raise ValidationError("User has already reported this review")
            
            # Verify review exists
            review = await self.db.query(TemplateRating).filter(
                TemplateRating.id == review_id
            ).first()
            
            if not review:
                raise NotFoundError("Review not found")
            
            # Create report
            report = ReviewReport(
                review_id=review_id,
                reporter_id=reporter_id,
                reason=reason,
                description=description
            )
            
            self.db.add(report)
            
            # Queue review for priority moderation
            await self._queue_for_moderation(review, priority="high", reason="user_reported")
            
            await self.db.commit()
            
            return report
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error reporting review: {e}")
            raise
    
    async def get_user_reputation(self, user_id: UUID) -> UserReputation:
        """Get user's reputation information."""
        try:
            reputation = await self.db.query(UserReputation).filter(
                UserReputation.user_id == user_id
            ).first()
            
            if not reputation:
                # Create initial reputation entry
                reputation = UserReputation(user_id=user_id)
                self.db.add(reputation)
                await self.db.commit()
            
            return reputation
            
        except Exception as e:
            logger.error(f"Error getting user reputation: {e}")
            raise
    
    async def moderate_review(
        self,
        review_id: UUID,
        moderator_id: UUID,
        action: str,
        notes: Optional[str] = None
    ) -> TemplateRating:
        """Moderate a review (approve/reject/flag)."""
        try:
            review = await self.db.query(TemplateRating).filter(
                TemplateRating.id == review_id
            ).first()
            
            if not review:
                raise NotFoundError("Review not found")
            
            # Update review status
            if action == "approve":
                review.status = ReviewStatus.APPROVED.value
            elif action == "reject":
                review.status = ReviewStatus.REJECTED.value
            elif action == "flag":
                review.status = ReviewStatus.FLAGGED.value
            elif action == "hide":
                review.status = ReviewStatus.HIDDEN.value
            
            review.moderated_by = moderator_id
            review.moderated_at = datetime.utcnow()
            review.moderation_notes = notes
            
            # Update template rating statistics if approved
            if action == "approve":
                await self._update_template_rating_stats(review.template_id)
            
            await self.db.commit()
            
            return review
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error moderating review: {e}")
            raise
    
    async def _queue_for_moderation(
        self,
        rating: TemplateRating,
        priority: str = "medium",
        reason: str = "new_content"
    ) -> None:
        """Queue a rating for moderation."""
        try:
            # Check if already in queue
            existing = await self.db.query(ModerationQueue).filter(
                and_(
                    ModerationQueue.content_type == "review",
                    ModerationQueue.content_id == rating.id
                )
            ).first()
            
            if existing and existing.status == "pending":
                return  # Already queued
            
            # Create moderation entry
            moderation_entry = ModerationQueue(
                content_type="review",
                content_id=rating.id,
                priority=priority,
                reason=reason
            )
            
            # Run basic AI moderation checks
            ai_results = await self._run_ai_moderation(rating)
            moderation_entry.ai_spam_score = ai_results.get("spam_score", 0.0)
            moderation_entry.ai_toxicity_score = ai_results.get("toxicity_score", 0.0)
            moderation_entry.ai_recommendation = ai_results.get("recommendation", "needs_human_review")
            
            self.db.add(moderation_entry)
            
            # Auto-approve if AI is confident it's clean
            if (ai_results.get("spam_score", 1.0) < 0.1 and 
                ai_results.get("toxicity_score", 1.0) < 0.1 and
                ai_results.get("recommendation") == "approve"):
                
                rating.status = ReviewStatus.APPROVED.value
                moderation_entry.status = "completed"
                moderation_entry.resolution = "approved"
                moderation_entry.resolved_at = datetime.utcnow()
                
                # Update template stats
                await self._update_template_rating_stats(rating.template_id)
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error queuing for moderation: {e}")
            # Don't raise - moderation failure shouldn't block rating creation
    
    async def _run_ai_moderation(self, rating: TemplateRating) -> Dict[str, Any]:
        """Run AI-based moderation checks."""
        try:
            # Simple rule-based moderation for now
            # In production, this would use ML models
            
            spam_score = 0.0
            toxicity_score = 0.0
            
            content = (rating.title or "") + " " + (rating.content or "")
            content_lower = content.lower()
            
            # Basic spam detection
            spam_indicators = ['buy now', 'click here', 'free money', 'urgent', 'limited time']
            spam_count = sum(1 for indicator in spam_indicators if indicator in content_lower)
            spam_score = min(spam_count * 0.3, 1.0)
            
            # Basic toxicity detection
            toxic_words = ['hate', 'stupid', 'idiot', 'terrible', 'worst ever']
            toxic_count = sum(1 for word in toxic_words if word in content_lower)
            toxicity_score = min(toxic_count * 0.4, 1.0)
            
            # Determine recommendation
            if spam_score < 0.2 and toxicity_score < 0.2:
                recommendation = "approve"
            elif spam_score > 0.8 or toxicity_score > 0.8:
                recommendation = "reject"
            else:
                recommendation = "needs_human_review"
            
            return {
                "spam_score": spam_score,
                "toxicity_score": toxicity_score,
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Error in AI moderation: {e}")
            return {
                "spam_score": 0.5,
                "toxicity_score": 0.5,
                "recommendation": "needs_human_review"
            }
    
    async def _update_helpfulness_counts(self, rating_id: UUID) -> None:
        """Update helpfulness counts for a rating."""
        try:
            # Count helpful and not helpful votes
            helpful_count = await self.db.query(func.count(ReviewHelpfulness.id)).filter(
                and_(
                    ReviewHelpfulness.review_id == rating_id,
                    ReviewHelpfulness.is_helpful == True
                )
            ).scalar()
            
            not_helpful_count = await self.db.query(func.count(ReviewHelpfulness.id)).filter(
                and_(
                    ReviewHelpfulness.review_id == rating_id,
                    ReviewHelpfulness.is_helpful == False
                )
            ).scalar()
            
            # Update rating
            rating = await self.db.query(TemplateRating).filter(
                TemplateRating.id == rating_id
            ).first()
            
            if rating:
                rating.helpful_votes = helpful_count or 0
                rating.not_helpful_votes = not_helpful_count or 0
                rating.total_votes = (helpful_count or 0) + (not_helpful_count or 0)
                
                # Calculate helpfulness score
                if rating.total_votes > 0:
                    rating.helpfulness_score = (helpful_count or 0) / rating.total_votes
                else:
                    rating.helpfulness_score = 0.0
            
        except Exception as e:
            logger.error(f"Error updating helpfulness counts: {e}")
    
    async def _update_template_rating_stats(self, template_id: UUID) -> None:
        """Update template's rating statistics."""
        try:
            # Get all approved ratings for the template
            ratings = await self.db.query(TemplateRating).filter(
                and_(
                    TemplateRating.template_id == template_id,
                    TemplateRating.status == ReviewStatus.APPROVED.value
                )
            ).all()
            
            if not ratings:
                return
            
            # Calculate statistics
            total_ratings = len(ratings)
            total_score = sum(r.overall_rating for r in ratings)
            average_rating = total_score / total_ratings
            
            # Update template
            template = await self.db.query(Template).filter(
                Template.id == template_id
            ).first()
            
            if template:
                template.rating_count = total_ratings
                template.average_rating = average_rating
            
        except Exception as e:
            logger.error(f"Error updating template rating stats: {e}")
    
    async def _update_user_reputation(
        self,
        user_id: UUID,
        action: str,
        points: Optional[int] = None
    ) -> None:
        """Update user's reputation based on actions."""
        try:
            reputation = await self.get_user_reputation(user_id)
            
            # Define point values for different actions
            point_values = {
                "review_created": 5,
                "helpful_review": 10,
                "review_approved": 15,
                "review_rejected": -5,
                "reported_content": -2
            }
            
            points_to_add = points or point_values.get(action, 0)
            reputation.total_reputation += points_to_add
            
            if action == "review_created":
                reputation.total_reviews_written += 1
                reputation.reviewer_reputation += points_to_add
            
            # Update reputation calculations
            if reputation.total_reviews_written > 0:
                reputation.average_review_helpfulness = (
                    reputation.helpful_reviews_count / reputation.total_reviews_written
                )
            
            # Award badges based on milestones
            await self._check_and_award_badges(reputation)
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating user reputation: {e}")
    
    async def _check_and_award_badges(self, reputation: UserReputation) -> None:
        """Check and award badges based on reputation milestones."""
        try:
            badges = reputation.badges or []
            
            # Review badges
            if reputation.total_reviews_written >= 10 and "reviewer" not in badges:
                badges.append("reviewer")
            
            if reputation.total_reviews_written >= 50 and "expert_reviewer" not in badges:
                badges.append("expert_reviewer")
            
            if reputation.helpful_reviews_count >= 25 and "helpful_reviewer" not in badges:
                badges.append("helpful_reviewer")
            
            # Reputation badges
            if reputation.total_reputation >= 100 and "contributor" not in badges:
                badges.append("contributor")
            
            if reputation.total_reputation >= 500 and "valued_member" not in badges:
                badges.append("valued_member")
            
            if reputation.total_reputation >= 1000 and "community_leader" not in badges:
                badges.append("community_leader")
            
            reputation.badges = badges
            
            # Check for trusted reviewer status
            if (reputation.total_reviews_written >= 20 and
                reputation.average_review_helpfulness >= 0.8 and
                reputation.total_reputation >= 200):
                reputation.is_trusted_reviewer = True
            
        except Exception as e:
            logger.error(f"Error checking badges: {e}")
