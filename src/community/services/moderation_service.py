"""
Moderation Service - Comprehensive content moderation with AI detection and human review.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import desc, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.rating import ModerationQueue, ReviewReport, TemplateRating
from ..models.template import Template
from ..models.plugin import Plugin
from ..models.user import UserProfile
from .cache_service import CacheService
from ...core.exceptions import ValidationError, NotFoundError
from ...core.logger import get_logger

logger = get_logger(__name__)


class ModerationService:
    """Service for content moderation with AI assistance."""
    
    def __init__(self, db: AsyncSession):
        """Initialize moderation service."""
        self.db = db
        self._ai_moderation_enabled = True
        self._toxicity_threshold = 0.8
        self._spam_threshold = 0.7
    
    async def moderate_content(
        self,
        content_type: str,
        content_id: UUID,
        content_author_id: UUID,
        detection_method: str = "ai_auto",
        reporter_id: Optional[UUID] = None
    ) -> ContentModerationEntry:
        """Moderate content using AI and rule-based systems."""
        try:
            # Check if already being moderated
            existing = await self.db.query(ContentModerationEntry).filter(
                and_(
                    ContentModerationEntry.content_type == content_type,
                    ContentModerationEntry.content_id == content_id,
                    ContentModerationEntry.status.in_(["pending", "escalated"])
                )
            ).first()
            
            if existing:
                return existing
            
            # Create moderation entry
            moderation_entry = ContentModerationEntry(
                content_type=content_type,
                content_id=content_id,
                content_author_id=content_author_id,
                detection_method=detection_method,
                reporter_id=reporter_id
            )
            
            # Get content for analysis
            content_text = await self._extract_content_text(content_type, content_id)
            
            if content_text and self._ai_moderation_enabled:
                # Run AI moderation
                ai_results = await self._run_ai_analysis(content_text)
                
                moderation_entry.ai_confidence_score = ai_results.get("confidence_score", 0.0)
                moderation_entry.ai_spam_probability = ai_results.get("spam_probability", 0.0)
                moderation_entry.ai_toxicity_score = ai_results.get("toxicity_score", 0.0)
                moderation_entry.ai_adult_content_score = ai_results.get("adult_content_score", 0.0)
                moderation_entry.ai_violence_score = ai_results.get("violence_score", 0.0)
                moderation_entry.ai_hate_speech_score = ai_results.get("hate_speech_score", 0.0)
                moderation_entry.ai_recommendation = ai_results.get("recommendation", "escalate")
                
                moderation_entry.violations = ai_results.get("violations", [])
                moderation_entry.risk_factors = ai_results.get("risk_factors", [])
            
            # Apply moderation rules
            rule_results = await self._apply_moderation_rules(content_type, moderation_entry)
            
            # Determine priority
            moderation_entry.priority = self._calculate_priority(moderation_entry)
            
            self.db.add(moderation_entry)
            
            # Auto-action based on AI confidence and rules
            if rule_results.get("auto_action"):
                await self._execute_auto_action(moderation_entry, rule_results["auto_action"])
            elif (moderation_entry.ai_confidence_score > 0.9 and 
                  moderation_entry.ai_recommendation in ["approve", "reject"]):
                await self._execute_auto_action(moderation_entry, moderation_entry.ai_recommendation)
            else:
                # Queue for human review
                moderation_entry.status = "escalated"
            
            await self.db.commit()
            
            return moderation_entry
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error moderating content: {e}")
            raise
    
    async def human_moderate(
        self,
        moderation_id: UUID,
        moderator_id: UUID,
        action: str,
        reason: Optional[str] = None,
        notes: Optional[str] = None
    ) -> ContentModerationEntry:
        """Human moderation decision."""
        try:
            moderation_entry = await self.db.query(ContentModerationEntry).filter(
                ContentModerationEntry.id == moderation_id
            ).first()
            
            if not moderation_entry:
                raise NotFoundError("Moderation entry not found")
            
            # Update moderation entry
            moderation_entry.assigned_moderator_id = moderator_id
            moderation_entry.moderator_action = action
            moderation_entry.moderator_notes = notes
            moderation_entry.resolution_reason = reason
            moderation_entry.reviewed_at = datetime.utcnow()
            
            # Execute the action
            await self._execute_moderation_action(
                moderation_entry, 
                action, 
                moderator_id
            )
            
            # Update status
            moderation_entry.status = "resolved"
            moderation_entry.resolved_at = datetime.utcnow()
            
            await self.db.commit()
            
            return moderation_entry
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error in human moderation: {e}")
            raise
    
    async def submit_appeal(
        self,
        moderation_id: UUID,
        user_id: UUID,
        reason: str,
        additional_context: Optional[str] = None,
        evidence_urls: Optional[List[str]] = None
    ) -> ModerationAppeal:
        """Submit an appeal for a moderation decision."""
        try:
            # Verify moderation entry exists and user can appeal
            moderation_entry = await self.db.query(ContentModerationEntry).filter(
                ContentModerationEntry.id == moderation_id
            ).first()
            
            if not moderation_entry:
                raise NotFoundError("Moderation entry not found")
            
            if moderation_entry.content_author_id != user_id:
                raise PermissionError("Only content author can submit appeals")
            
            # Check if already appealed
            existing_appeal = await self.db.query(ModerationAppeal).filter(
                and_(
                    ModerationAppeal.moderation_id == moderation_id,
                    ModerationAppeal.user_id == user_id
                )
            ).first()
            
            if existing_appeal:
                raise ValidationError("Appeal already submitted for this moderation")
            
            # Create appeal
            appeal = ModerationAppeal(
                moderation_id=moderation_id,
                user_id=user_id,
                reason=reason,
                additional_context=additional_context,
                evidence_urls=evidence_urls or []
            )
            
            self.db.add(appeal)
            
            # Update moderation entry
            moderation_entry.appeal_count += 1
            moderation_entry.last_appeal_at = datetime.utcnow()
            
            await self.db.commit()
            
            return appeal
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error submitting appeal: {e}")
            raise
    
    async def process_appeal(
        self,
        appeal_id: UUID,
        reviewer_id: UUID,
        decision: str,
        notes: Optional[str] = None
    ) -> ModerationAppeal:
        """Process an appeal (approve/reject)."""
        try:
            appeal = await self.db.query(ModerationAppeal).filter(
                ModerationAppeal.id == appeal_id
            ).first()
            
            if not appeal:
                raise NotFoundError("Appeal not found")
            
            # Update appeal
            appeal.status = "reviewed"
            appeal.reviewed_by = reviewer_id
            appeal.review_notes = notes
            appeal.final_decision = decision
            appeal.reviewed_at = datetime.utcnow()
            
            # If appeal is approved, reverse the moderation decision
            if decision == "overturned":
                moderation_entry = await self.db.query(ContentModerationEntry).filter(
                    ContentModerationEntry.id == appeal.moderation_id
                ).first()
                
                if moderation_entry:
                    await self._reverse_moderation_action(moderation_entry)
            
            await self.db.commit()
            
            return appeal
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error processing appeal: {e}")
            raise
    
    async def get_moderation_queue(
        self,
        moderator_id: UUID,
        page: int = 1,
        page_size: int = 20,
        priority_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get moderation queue for a moderator."""
        try:
            # Build query for pending moderation items
            query = self.db.query(ContentModerationEntry).filter(
                ContentModerationEntry.status.in_(["pending", "escalated"])
            )
            
            # Apply filters
            if priority_filter:
                query = query.filter(ContentModerationEntry.priority == priority_filter)
            
            if content_type_filter:
                query = query.filter(ContentModerationEntry.content_type == content_type_filter)
            
            # Count total
            total_count = await query.count()
            
            # Order by priority and creation time
            priority_order = {
                "critical": 1,
                "high": 2,
                "medium": 3,
                "low": 4
            }
            
            query = query.order_by(
                func.case(priority_order, value=ContentModerationEntry.priority),
                ContentModerationEntry.created_at
            )
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            moderation_entries = await query.all()
            
            # Convert to dictionaries with content info
            queue_items = []
            for entry in moderation_entries:
                item_dict = entry.to_dict(include_sensitive=True)
                
                # Add content preview
                content_preview = await self._get_content_preview(
                    entry.content_type, 
                    entry.content_id
                )
                item_dict['content_preview'] = content_preview
                
                queue_items.append(item_dict)
            
            return queue_items, total_count
            
        except Exception as e:
            logger.error(f"Error getting moderation queue: {e}")
            raise
    
    async def get_moderation_stats(self) -> Dict[str, Any]:
        """Get moderation statistics."""
        try:
            now = datetime.utcnow()
            today = now.date()
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            # Basic counts
            total_items = await self.db.query(func.count(ContentModerationEntry.id)).scalar()
            pending_items = await self.db.query(func.count(ContentModerationEntry.id)).filter(
                ContentModerationEntry.status.in_(["pending", "escalated"])
            ).scalar()
            
            # Today's activity
            items_today = await self.db.query(func.count(ContentModerationEntry.id)).filter(
                func.date(ContentModerationEntry.created_at) == today
            ).scalar()
            
            resolved_today = await self.db.query(func.count(ContentModerationEntry.id)).filter(
                and_(
                    func.date(ContentModerationEntry.resolved_at) == today,
                    ContentModerationEntry.status == "resolved"
                )
            ).scalar()
            
            # Auto-moderation rate
            auto_moderated = await self.db.query(func.count(ContentModerationEntry.id)).filter(
                ContentModerationEntry.detection_method == "ai_auto"
            ).scalar()
            
            auto_moderation_rate = (auto_moderated / total_items * 100) if total_items > 0 else 0
            
            # Top violation types
            violation_stats = await self.db.query(
                func.jsonb_array_elements_text(ContentModerationEntry.violations).label('violation'),
                func.count().label('count')
            ).group_by('violation').order_by(desc('count')).limit(10).all()
            
            top_violations = [
                {"violation": v.violation, "count": v.count} 
                for v in violation_stats
            ]
            
            # Appeal statistics
            total_appeals = await self.db.query(func.count(ModerationAppeal.id)).scalar()
            pending_appeals = await self.db.query(func.count(ModerationAppeal.id)).filter(
                ModerationAppeal.status == "pending"
            ).scalar()
            
            # Average resolution time
            avg_resolution = await self.db.query(
                func.avg(
                    func.extract('epoch', ContentModerationEntry.resolved_at - ContentModerationEntry.created_at) / 3600
                )
            ).filter(
                and_(
                    ContentModerationEntry.resolved_at.isnot(None),
                    ContentModerationEntry.created_at >= month_ago
                )
            ).scalar()
            
            return {
                "total_items_moderated": total_items or 0,
                "pending_items": pending_items or 0,
                "items_created_today": items_today or 0,
                "items_resolved_today": resolved_today or 0,
                "auto_moderation_percentage": round(auto_moderation_rate, 2),
                "average_resolution_time_hours": round(avg_resolution or 0, 2),
                "top_violation_types": top_violations,
                "total_appeals": total_appeals or 0,
                "pending_appeals": pending_appeals or 0,
                "appeal_rate": round((total_appeals / total_items * 100) if total_items > 0 else 0, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting moderation stats: {e}")
            raise
    
    async def create_moderation_rule(
        self,
        rule_data: Dict[str, Any]
    ) -> ModerationRule:
        """Create a new moderation rule."""
        try:
            rule = ModerationRule(
                name=rule_data["name"],
                description=rule_data.get("description"),
                category=rule_data["category"],
                content_types=rule_data["content_types"],
                conditions=rule_data["conditions"],
                thresholds=rule_data.get("thresholds", {}),
                auto_action=rule_data.get("auto_action"),
                escalate_to_human=rule_data.get("escalate_to_human", False),
                priority=rule_data.get("priority", 1)
            )
            
            self.db.add(rule)
            await self.db.commit()
            
            return rule
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating moderation rule: {e}")
            raise
    
    async def _extract_content_text(self, content_type: str, content_id: UUID) -> Optional[str]:
        """Extract text content for analysis."""
        try:
            if content_type == "review":
                review = await self.db.query(TemplateRating).filter(
                    TemplateRating.id == content_id
                ).first()
                if review:
                    return (review.title or "") + " " + (review.content or "")
            
            elif content_type == "template":
                template = await self.db.query(Template).filter(
                    Template.id == content_id
                ).first()
                if template:
                    return template.name + " " + (template.description or "")
            
            elif content_type == "plugin":
                plugin = await self.db.query(Plugin).filter(
                    Plugin.id == content_id
                ).first()
                if plugin:
                    return plugin.name + " " + (plugin.description or "")
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting content text: {e}")
            return None
    
    async def _run_ai_analysis(self, content_text: str) -> Dict[str, Any]:
        """Run AI analysis on content."""
        try:
            # Simulate AI analysis - in production, this would call actual ML models
            content_lower = content_text.lower()
            
            # Spam detection
            spam_indicators = [
                'buy now', 'click here', 'free money', 'urgent', 'limited time',
                'act now', 'guarantee', 'no risk', 'special promotion', 'offer expires'
            ]
            spam_score = sum(1 for indicator in spam_indicators if indicator in content_lower)
            spam_probability = min(spam_score * 0.2, 1.0)
            
            # Toxicity detection
            toxic_patterns = [
                'hate', 'stupid', 'idiot', 'terrible', 'worst', 'sucks',
                'garbage', 'trash', 'useless', 'pathetic'
            ]
            toxicity_score = min(sum(1 for pattern in toxic_patterns if pattern in content_lower) * 0.25, 1.0)
            
            # Adult content detection
            adult_keywords = ['adult', 'explicit', 'nsfw', 'mature']
            adult_score = min(sum(1 for keyword in adult_keywords if keyword in content_lower) * 0.3, 1.0)
            
            # Violence detection
            violence_keywords = ['violence', 'kill', 'murder', 'death', 'harm']
            violence_score = min(sum(1 for keyword in violence_keywords if keyword in content_lower) * 0.4, 1.0)
            
            # Hate speech detection
            hate_keywords = ['discriminat', 'racist', 'sexist', 'homophobic']
            hate_score = min(sum(1 for keyword in hate_keywords if keyword in content_lower) * 0.5, 1.0)
            
            # Overall confidence
            max_score = max(spam_probability, toxicity_score, adult_score, violence_score, hate_score)
            confidence_score = min(max_score + 0.2, 1.0)  # Add base confidence
            
            # Determine violations
            violations = []
            if spam_probability > 0.5:
                violations.append(ViolationType.SPAM.value)
            if toxicity_score > 0.5:
                violations.append(ViolationType.OFFENSIVE_LANGUAGE.value)
            if adult_score > 0.3:
                violations.append(ViolationType.ADULT_CONTENT.value)
            if violence_score > 0.3:
                violations.append(ViolationType.VIOLENCE.value)
            if hate_score > 0.3:
                violations.append(ViolationType.HATE_SPEECH.value)
            
            # Recommendation
            if max_score > 0.8:
                recommendation = "reject"
            elif max_score < 0.2:
                recommendation = "approve"
            else:
                recommendation = "escalate"
            
            return {
                "confidence_score": confidence_score,
                "spam_probability": spam_probability,
                "toxicity_score": toxicity_score,
                "adult_content_score": adult_score,
                "violence_score": violence_score,
                "hate_speech_score": hate_score,
                "violations": violations,
                "risk_factors": [f"High {v}" for v in violations],
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {
                "confidence_score": 0.5,
                "spam_probability": 0.5,
                "toxicity_score": 0.5,
                "adult_content_score": 0.0,
                "violence_score": 0.0,
                "hate_speech_score": 0.0,
                "violations": [],
                "risk_factors": [],
                "recommendation": "escalate"
            }
    
    async def _apply_moderation_rules(
        self,
        content_type: str,
        moderation_entry: ContentModerationEntry
    ) -> Dict[str, Any]:
        """Apply configured moderation rules."""
        try:
            # Get active rules for this content type
            rules = await self.db.query(ModerationRule).filter(
                and_(
                    ModerationRule.is_active == True,
                    ModerationRule.content_types.op('?')(content_type)
                )
            ).order_by(ModerationRule.priority).all()
            
            triggered_rules = []
            auto_action = None
            
            for rule in rules:
                # Check if rule conditions are met
                if self._evaluate_rule_conditions(rule, moderation_entry):
                    triggered_rules.append(rule.name)
                    
                    # Update rule statistics
                    rule.times_triggered += 1
                    rule.last_triggered_at = datetime.utcnow()
                    
                    # Use the highest priority rule's action
                    if not auto_action and rule.auto_action:
                        auto_action = rule.auto_action
            
            return {
                "triggered_rules": triggered_rules,
                "auto_action": auto_action
            }
            
        except Exception as e:
            logger.error(f"Error applying moderation rules: {e}")
            return {"triggered_rules": [], "auto_action": None}
    
    def _evaluate_rule_conditions(self, rule: ModerationRule, entry: ContentModerationEntry) -> bool:
        """Evaluate if rule conditions are met."""
        try:
            conditions = rule.conditions
            thresholds = rule.thresholds or {}
            
            # Check score thresholds
            if "spam_threshold" in thresholds:
                if entry.ai_spam_probability < thresholds["spam_threshold"]:
                    return False
            
            if "toxicity_threshold" in thresholds:
                if entry.ai_toxicity_score < thresholds["toxicity_threshold"]:
                    return False
            
            if "confidence_threshold" in thresholds:
                if entry.ai_confidence_score < thresholds["confidence_threshold"]:
                    return False
            
            # Check violation requirements
            if "required_violations" in conditions:
                required = set(conditions["required_violations"])
                detected = set(entry.violations or [])
                if not required.issubset(detected):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating rule conditions: {e}")
            return False
    
    def _calculate_priority(self, entry: ContentModerationEntry) -> str:
        """Calculate moderation priority based on scores and factors."""
        try:
            max_score = max(
                entry.ai_spam_probability or 0,
                entry.ai_toxicity_score or 0,
                entry.ai_adult_content_score or 0,
                entry.ai_violence_score or 0,
                entry.ai_hate_speech_score or 0
            )
            
            if max_score > 0.9 or entry.detection_method == "user_report":
                return "critical"
            elif max_score > 0.7:
                return "high"
            elif max_score > 0.4:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Error calculating priority: {e}")
            return "medium"
    
    async def _execute_auto_action(self, entry: ContentModerationEntry, action: str) -> None:
        """Execute automatic moderation action."""
        try:
            entry.moderator_action = action
            entry.action_taken = f"auto_{action}"
            entry.status = "resolved"
            entry.resolved_at = datetime.utcnow()
            
            if action == "approve":
                await self._approve_content(entry.content_type, entry.content_id)
            elif action == "reject":
                await self._reject_content(entry.content_type, entry.content_id, "Automated moderation")
            elif action == "flag":
                await self._flag_content(entry.content_type, entry.content_id)
            
        except Exception as e:
            logger.error(f"Error executing auto action: {e}")
    
    async def _execute_moderation_action(
        self,
        entry: ContentModerationEntry,
        action: str,
        moderator_id: UUID
    ) -> None:
        """Execute human moderation action."""
        try:
            entry.action_taken = action
            
            if action == ModerationAction.APPROVE.value:
                await self._approve_content(entry.content_type, entry.content_id)
            elif action == ModerationAction.REJECT.value:
                await self._reject_content(entry.content_type, entry.content_id, entry.moderator_notes)
            elif action == ModerationAction.FLAG.value:
                await self._flag_content(entry.content_type, entry.content_id)
            elif action == ModerationAction.REMOVE.value:
                await self._remove_content(entry.content_type, entry.content_id)
            elif action == ModerationAction.BAN.value:
                await self._ban_user(entry.content_author_id, entry.moderator_notes)
            
        except Exception as e:
            logger.error(f"Error executing moderation action: {e}")
    
    async def _approve_content(self, content_type: str, content_id: UUID) -> None:
        """Approve content."""
        if content_type == "review":
            review = await self.db.query(TemplateRating).filter(TemplateRating.id == content_id).first()
            if review:
                review.status = "approved"
        elif content_type == "template":
            template = await self.db.query(Template).filter(Template.id == content_id).first()
            if template:
                template.is_public = True
        elif content_type == "plugin":
            plugin = await self.db.query(Plugin).filter(Plugin.id == content_id).first()
            if plugin:
                plugin.is_public = True
    
    async def _reject_content(self, content_type: str, content_id: UUID, reason: Optional[str]) -> None:
        """Reject content."""
        if content_type == "review":
            review = await self.db.query(TemplateRating).filter(TemplateRating.id == content_id).first()
            if review:
                review.status = "rejected"
                review.moderation_notes = reason
        elif content_type == "template":
            template = await self.db.query(Template).filter(Template.id == content_id).first()
            if template:
                template.is_public = False
        elif content_type == "plugin":
            plugin = await self.db.query(Plugin).filter(Plugin.id == content_id).first()
            if plugin:
                plugin.is_public = False
    
    async def _flag_content(self, content_type: str, content_id: UUID) -> None:
        """Flag content for further review."""
        if content_type == "review":
            review = await self.db.query(TemplateRating).filter(TemplateRating.id == content_id).first()
            if review:
                review.status = "flagged"
    
    async def _remove_content(self, content_type: str, content_id: UUID) -> None:
        """Remove content completely."""
        if content_type == "review":
            review = await self.db.query(TemplateRating).filter(TemplateRating.id == content_id).first()
            if review:
                review.status = "hidden"
    
    async def _ban_user(self, user_id: UUID, reason: Optional[str]) -> None:
        """Ban a user (would integrate with user management system)."""
        # This would integrate with the user management system
        logger.warning(f"User {user_id} should be banned. Reason: {reason}")
    
    async def _reverse_moderation_action(self, entry: ContentModerationEntry) -> None:
        """Reverse a moderation action (for appeals)."""
        try:
            if entry.action_taken in ["reject", "auto_reject"]:
                await self._approve_content(entry.content_type, entry.content_id)
            elif entry.action_taken in ["flag", "auto_flag"]:
                await self._approve_content(entry.content_type, entry.content_id)
            
            entry.status = "appealed"
            
        except Exception as e:
            logger.error(f"Error reversing moderation action: {e}")
    
    async def _get_content_preview(self, content_type: str, content_id: UUID) -> Dict[str, Any]:
        """Get a preview of content for moderation queue."""
        try:
            if content_type == "review":
                review = await self.db.query(TemplateRating).filter(
                    TemplateRating.id == content_id
                ).first()
                if review:
                    return {
                        "title": review.title,
                        "content_preview": (review.content or "")[:200] + "..." if review.content else None,
                        "rating": review.overall_rating
                    }
            
            elif content_type == "template":
                template = await self.db.query(Template).filter(
                    Template.id == content_id
                ).first()
                if template:
                    return {
                        "name": template.name,
                        "description_preview": (template.description or "")[:200] + "..." if template.description else None
                    }
            
            elif content_type == "plugin":
                plugin = await self.db.query(Plugin).filter(
                    Plugin.id == content_id
                ).first()
                if plugin:
                    return {
                        "name": plugin.name,
                        "description_preview": (plugin.description or "")[:200] + "..." if plugin.description else None,
                        "version": plugin.version
                    }
            
            return {"content_preview": "Content not available"}
            
        except Exception as e:
            logger.error(f"Error getting content preview: {e}")
            return {"content_preview": "Error loading content"}
