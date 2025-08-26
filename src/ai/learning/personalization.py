"""
Personalized AI Behavior System.

This module implements personalized AI behavior that adapts to individual user patterns,
preferences, and success metrics. It provides dynamic prompt generation, personalized
validation criteria, and adaptive AI responses based on learned user behaviors.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID
from pathlib import Path

from ...core.types import Task, ValidationResult, AITaskResult
from ...core.ai_interface import AIContext, ClaudeCodeRequest
from .pattern_engine import PatternRecognitionEngine, UserInteractionPattern


logger = logging.getLogger(__name__)


@dataclass
class PersonalizationProfile:
    """User personalization profile containing learned preferences and patterns."""
    user_id: str
    creation_date: datetime
    last_updated: datetime
    
    # Preferences
    preferred_coding_style: Dict[str, Any] = field(default_factory=dict)
    preferred_frameworks: List[str] = field(default_factory=list)
    preferred_patterns: List[str] = field(default_factory=list)
    communication_style: str = "balanced"  # concise, detailed, balanced
    
    # Success patterns
    successful_task_types: List[str] = field(default_factory=list)
    optimal_work_hours: List[int] = field(default_factory=list)
    productive_session_lengths: List[int] = field(default_factory=list)
    
    # Learning metrics
    learning_velocity: float = 1.0  # How quickly user adapts to feedback
    pattern_stability: float = 0.5  # How consistent user patterns are
    feedback_responsiveness: float = 0.8  # How well user responds to suggestions
    
    # Personalized settings
    validation_strictness: float = 0.7  # How strict validation should be
    detail_preference: float = 0.5  # How detailed responses should be
    example_preference: float = 0.6  # How many examples to include
    
    # Context-specific adaptations
    context_adaptations: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PersonalizedPromptTemplate:
    """Personalized prompt template adapted to user preferences."""
    template_id: str
    user_id: str
    base_template: str
    personalized_template: str
    effectiveness_score: float
    usage_count: int
    success_rate: float
    context_tags: List[str]
    created_at: datetime
    last_used: datetime


@dataclass
class AdaptationRule:
    """Rule for adapting AI behavior based on user patterns."""
    rule_id: str
    condition: Dict[str, Any]  # When to apply this rule
    adaptation: Dict[str, Any]  # What changes to make
    confidence: float
    effectiveness: float
    applies_to: List[str]  # Context types this rule applies to


class PersonalizedAIBehavior:
    """
    Personalized AI behavior system that adapts AI responses and behavior
    based on individual user patterns, preferences, and success metrics.
    """
    
    def __init__(
        self,
        pattern_engine: PatternRecognitionEngine,
        adaptation_rate: float = 0.1,
        min_interactions_for_personalization: int = 5
    ):
        """
        Initialize personalized AI behavior system.
        
        Args:
            pattern_engine: Pattern recognition engine for learning
            adaptation_rate: Rate of behavioral adaptation
            min_interactions_for_personalization: Minimum interactions needed for personalization
        """
        self.pattern_engine = pattern_engine
        self.adaptation_rate = adaptation_rate
        self.min_interactions_threshold = min_interactions_for_personalization
        
        # User profiles and templates
        self._user_profiles: Dict[str, PersonalizationProfile] = {}
        self._personalized_templates: Dict[str, List[PersonalizedPromptTemplate]] = {}
        self._adaptation_rules: Dict[str, List[AdaptationRule]] = {}
        
        # Base templates for personalization
        self._base_templates = self._load_base_templates()
        
        # Personalization cache
        self._personalization_cache: Dict[str, Any] = {}
        
        logger.info("Personalized AI behavior system initialized")
    
    async def get_or_create_profile(self, user_id: str) -> PersonalizationProfile:
        """Get existing profile or create new one for user."""
        if user_id not in self._user_profiles:
            profile = PersonalizationProfile(
                user_id=user_id,
                creation_date=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            # Initialize with defaults based on any existing patterns
            await self._initialize_profile_from_patterns(profile)
            
            self._user_profiles[user_id] = profile
            logger.info(f"Created new personalization profile for user {user_id}")
        
        return self._user_profiles[user_id]
    
    async def personalize_ai_request(
        self,
        user_id: str,
        base_request: ClaudeCodeRequest,
        task: Task,
        context: Dict[str, Any]
    ) -> ClaudeCodeRequest:
        """
        Personalize AI request based on user profile and learned patterns.
        
        Args:
            user_id: User identifier
            base_request: Base AI request to personalize
            task: Task being executed
            context: Additional context
            
        Returns:
            Personalized AI request
        """
        profile = await self.get_or_create_profile(user_id)
        
        # Check if we have enough data for personalization
        user_patterns = self.pattern_engine._user_patterns.get(user_id, [])
        if len(user_patterns) < self.min_interactions_threshold:
            logger.debug(f"Not enough interactions for user {user_id} personalization")
            return base_request
        
        # Personalize prompt
        personalized_prompt = await self._personalize_prompt(
            user_id, base_request.prompt, task, context, profile
        )
        
        # Adjust request parameters
        personalized_request = ClaudeCodeRequest(
            prompt=personalized_prompt,
            context=base_request.context,
            timeout=self._adjust_timeout(base_request.timeout, profile),
            max_retries=base_request.max_retries,
            format=base_request.format,
            enable_validation=base_request.enable_validation
        )
        
        # Add personalization metadata
        personalized_request.context.previous_context = self._build_personalized_context(
            profile, user_patterns, context
        )
        
        logger.debug(f"Personalized AI request for user {user_id}")
        return personalized_request
    
    async def adapt_validation_criteria(
        self,
        user_id: str,
        base_criteria: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt validation criteria based on user preferences and success patterns.
        
        Args:
            user_id: User identifier
            base_criteria: Base validation criteria
            context: Validation context
            
        Returns:
            Personalized validation criteria
        """
        profile = await self.get_or_create_profile(user_id)
        
        personalized_criteria = base_criteria.copy()
        
        # Adjust validation strictness
        if 'authenticity_threshold' in personalized_criteria:
            base_threshold = personalized_criteria['authenticity_threshold']
            adjusted_threshold = base_threshold * profile.validation_strictness
            personalized_criteria['authenticity_threshold'] = adjusted_threshold
        
        # Adjust based on user success patterns
        user_patterns = self.pattern_engine._user_patterns.get(user_id, [])
        recent_failures = [
            p for p in user_patterns
            if p.pattern_type in ['failure', 'validation_failure']
            and (datetime.utcnow() - p.timestamp).days <= 7
        ]
        
        if recent_failures:
            # Be more strict after recent failures
            personalized_criteria['placeholder_sensitivity'] = personalized_criteria.get(
                'placeholder_sensitivity', 1.0
            ) * 1.2
            personalized_criteria['quality_threshold'] = personalized_criteria.get(
                'quality_threshold', 0.7
            ) * 1.1
        
        # Context-specific adaptations
        task_type = context.get('task_type', 'general')
        if task_type in profile.context_adaptations:
            adaptations = profile.context_adaptations[task_type]
            for key, adjustment in adaptations.items():
                if key in personalized_criteria:
                    if isinstance(personalized_criteria[key], (int, float)):
                        personalized_criteria[key] *= adjustment
        
        logger.debug(f"Adapted validation criteria for user {user_id}")
        return personalized_criteria
    
    async def learn_from_user_feedback(
        self,
        user_id: str,
        task: Task,
        ai_result: AITaskResult,
        validation_result: ValidationResult,
        user_feedback: Dict[str, Any]
    ) -> None:
        """
        Learn from explicit user feedback to improve personalization.
        
        Args:
            user_id: User identifier
            task: Task that was executed
            ai_result: AI task result
            validation_result: Validation result
            user_feedback: User's feedback on the result
        """
        profile = await self.get_or_create_profile(user_id)
        
        # Extract feedback signals
        satisfaction_score = user_feedback.get('satisfaction', 0.5)
        feedback_type = user_feedback.get('type', 'general')  # positive, negative, suggestion
        specific_feedback = user_feedback.get('comments', '')
        
        # Update profile based on feedback
        await self._update_profile_from_feedback(
            profile, task, ai_result, validation_result, user_feedback
        )
        
        # Learn prompt effectiveness
        if hasattr(ai_result, 'metadata') and 'prompt_template' in ai_result.metadata:
            await self._update_template_effectiveness(
                user_id, ai_result.metadata['prompt_template'], satisfaction_score, user_feedback
            )
        
        # Create adaptation rules from negative feedback
        if feedback_type == 'negative' or satisfaction_score < 0.4:
            await self._create_adaptation_rules_from_feedback(
                user_id, task, ai_result, validation_result, user_feedback
            )
        
        # Update learning velocity based on feedback responsiveness
        self._update_learning_metrics(profile, user_feedback)
        
        profile.last_updated = datetime.utcnow()
        logger.info(f"Updated personalization from user {user_id} feedback")
    
    async def get_personalized_suggestions(
        self,
        user_id: str,
        current_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get personalized suggestions based on user patterns and current context.
        
        Args:
            user_id: User identifier
            current_context: Current working context
            
        Returns:
            List of personalized suggestions
        """
        profile = await self.get_or_create_profile(user_id)
        
        # Get base recommendations from pattern engine
        base_recommendations = await self.pattern_engine.generate_personalized_recommendations(
            user_id, current_context
        )
        
        # Enhance with profile-specific suggestions
        profile_suggestions = []
        
        # Time-based suggestions
        current_hour = datetime.utcnow().hour
        if profile.optimal_work_hours and current_hour not in profile.optimal_work_hours:
            closest_optimal = min(profile.optimal_work_hours, key=lambda x: abs(x - current_hour))
            profile_suggestions.append({
                "type": "timing",
                "category": "productivity_optimization",
                "message": f"Consider working at {closest_optimal}:00 for better productivity",
                "confidence": 0.7,
                "actionable": True,
                "suggested_actions": ["Schedule important tasks for optimal hours"]
            })
        
        # Framework preferences
        current_frameworks = current_context.get('frameworks', [])
        if profile.preferred_frameworks:
            missing_preferred = set(profile.preferred_frameworks) - set(current_frameworks)
            if missing_preferred:
                profile_suggestions.append({
                    "type": "preference",
                    "category": "framework_suggestion",
                    "message": f"Consider using your preferred frameworks: {', '.join(missing_preferred)}",
                    "confidence": 0.8,
                    "actionable": True,
                    "suggested_actions": [f"Integrate {fw} into current project" for fw in missing_preferred]
                })
        
        # Coding style suggestions
        if profile.preferred_coding_style:
            style_suggestions = self._generate_style_suggestions(
                profile.preferred_coding_style, current_context
            )
            profile_suggestions.extend(style_suggestions)
        
        # Communication style adaptations
        for suggestion in base_recommendations:
            if profile.communication_style == "concise":
                suggestion["message"] = self._make_message_concise(suggestion["message"])
            elif profile.communication_style == "detailed":
                suggestion = self._add_detailed_explanation(suggestion, profile)
        
        # Combine and prioritize
        all_suggestions = base_recommendations + profile_suggestions
        
        # Sort by confidence and relevance
        all_suggestions.sort(key=lambda x: x.get('confidence', 0.5), reverse=True)
        
        return all_suggestions[:8]  # Return top 8 suggestions
    
    async def measure_personalization_effectiveness(
        self,
        user_id: str,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Measure how effective personalization has been for a user.
        
        Args:
            user_id: User identifier
            time_window_days: Time window for analysis
            
        Returns:
            Personalization effectiveness metrics
        """
        profile = self._user_profiles.get(user_id)
        if not profile:
            return {"error": "No profile found for user"}
        
        # Get user patterns in time window
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
        user_patterns = self.pattern_engine._user_patterns.get(user_id, [])
        recent_patterns = [p for p in user_patterns if p.timestamp >= cutoff_date]
        
        if not recent_patterns:
            return {"error": "No recent patterns for analysis"}
        
        # Calculate effectiveness metrics
        personalized_interactions = [
            p for p in recent_patterns
            if p.context.get('personalized', False)
        ]
        non_personalized_interactions = [
            p for p in recent_patterns
            if not p.context.get('personalized', False)
        ]
        
        effectiveness_metrics = {
            'total_interactions': len(recent_patterns),
            'personalized_interactions': len(personalized_interactions),
            'personalization_rate': len(personalized_interactions) / len(recent_patterns) if recent_patterns else 0,
            'profile_age_days': (datetime.utcnow() - profile.creation_date).days,
            'last_updated_days_ago': (datetime.utcnow() - profile.last_updated).days
        }
        
        if personalized_interactions and non_personalized_interactions:
            # Compare personalized vs non-personalized success rates
            personalized_success_rate = sum(
                p.success_rate for p in personalized_interactions
            ) / len(personalized_interactions)
            
            non_personalized_success_rate = sum(
                p.success_rate for p in non_personalized_interactions
            ) / len(non_personalized_interactions)
            
            effectiveness_metrics.update({
                'personalized_success_rate': personalized_success_rate,
                'non_personalized_success_rate': non_personalized_success_rate,
                'improvement_from_personalization': personalized_success_rate - non_personalized_success_rate,
                'effectiveness_score': profile.feedback_responsiveness * profile.pattern_stability
            })
        
        # Template effectiveness
        user_templates = self._personalized_templates.get(user_id, [])
        if user_templates:
            avg_template_effectiveness = sum(t.effectiveness_score for t in user_templates) / len(user_templates)
            effectiveness_metrics['template_effectiveness'] = avg_template_effectiveness
        
        # Adaptation rules effectiveness
        user_rules = self._adaptation_rules.get(user_id, [])
        if user_rules:
            avg_rule_effectiveness = sum(r.effectiveness for r in user_rules) / len(user_rules)
            effectiveness_metrics['adaptation_rule_effectiveness'] = avg_rule_effectiveness
        
        return effectiveness_metrics
    
    def _load_base_templates(self) -> Dict[str, str]:
        """Load base prompt templates for personalization."""
        return {
            'code_generation': """
You are an expert software developer. Generate complete, functional code based on the following requirements:

{requirements}

IMPORTANT GUIDELINES:
1. Write complete, working code without placeholders
2. Follow best practices for the specified language/framework
3. Include proper error handling and validation
4. Add clear comments and documentation
5. Ensure code is production-ready

Context:
{context}

Please provide a complete implementation.
""",
            
            'code_review': """
Please review the following code for quality, correctness, and best practices:

{code}

Context:
{context}

Provide a detailed review covering:
1. Code quality and maintainability
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Improvement suggestions

Be constructive and specific in your feedback.
""",
            
            'debugging': """
Help debug the following issue:

Problem: {problem}
Code: {code}
Error/Context: {error_context}

Please:
1. Identify the root cause of the issue
2. Explain why it's happening
3. Provide a corrected version
4. Suggest ways to prevent similar issues

Focus on providing a working solution with clear explanations.
""",
            
            'architecture_design': """
Design a software architecture for the following requirements:

{requirements}

Consider:
1. Scalability and performance
2. Maintainability and modularity
3. Security and reliability
4. Technology stack recommendations
5. Deployment considerations

Context: {context}

Provide a comprehensive architectural design with rationale.
"""
        }
    
    async def _initialize_profile_from_patterns(self, profile: PersonalizationProfile) -> None:
        """Initialize profile preferences from existing patterns."""
        user_patterns = self.pattern_engine._user_patterns.get(profile.user_id, [])
        
        if not user_patterns:
            return
        
        # Analyze successful patterns to infer preferences
        success_patterns = [
            p for p in user_patterns
            if p.pattern_type in ['success', 'validation_success']
            and p.confidence > 0.7
        ]
        
        if success_patterns:
            # Extract optimal work hours
            work_hours = [p.features.get('timestamp_hour') for p in success_patterns]
            work_hours = [h for h in work_hours if h is not None]
            if work_hours:
                # Find most common work hours
                from collections import Counter
                hour_counts = Counter(work_hours)
                profile.optimal_work_hours = [
                    hour for hour, count in hour_counts.most_common(3)
                ]
            
            # Extract successful task types
            task_types = [p.features.get('task_type') for p in success_patterns]
            task_types = [t for t in task_types if t and t != 'unknown']
            if task_types:
                from collections import Counter
                type_counts = Counter(task_types)
                profile.successful_task_types = [
                    task_type for task_type, count in type_counts.most_common(5)
                ]
            
            # Calculate learning velocity based on pattern evolution
            sorted_patterns = sorted(success_patterns, key=lambda x: x.timestamp)
            if len(sorted_patterns) >= 3:
                early_success_rate = sum(p.success_rate for p in sorted_patterns[:len(sorted_patterns)//2])
                late_success_rate = sum(p.success_rate for p in sorted_patterns[len(sorted_patterns)//2:])
                
                if early_success_rate > 0:
                    improvement_rate = late_success_rate / early_success_rate
                    profile.learning_velocity = min(max(improvement_rate, 0.5), 2.0)
    
    async def _personalize_prompt(
        self,
        user_id: str,
        base_prompt: str,
        task: Task,
        context: Dict[str, Any],
        profile: PersonalizationProfile
    ) -> str:
        """Personalize prompt based on user preferences and patterns."""
        # Determine prompt type
        prompt_type = self._classify_prompt_type(base_prompt, task)
        
        # Get base template
        base_template = self._base_templates.get(prompt_type, base_prompt)
        
        # Check for existing personalized template
        user_templates = self._personalized_templates.get(user_id, [])
        existing_template = None
        
        for template in user_templates:
            if prompt_type in template.context_tags:
                existing_template = template
                break
        
        if existing_template and existing_template.effectiveness_score > 0.7:
            # Use existing effective template
            personalized_prompt = existing_template.personalized_template
        else:
            # Create new personalized template
            personalized_prompt = await self._create_personalized_template(
                user_id, base_template, profile, task, context, prompt_type
            )
        
        # Apply real-time adaptations
        personalized_prompt = self._apply_real_time_adaptations(
            personalized_prompt, profile, context
        )
        
        return personalized_prompt
    
    async def _create_personalized_template(
        self,
        user_id: str,
        base_template: str,
        profile: PersonalizationProfile,
        task: Task,
        context: Dict[str, Any],
        prompt_type: str
    ) -> str:
        """Create personalized template based on user profile."""
        personalized_template = base_template
        
        # Adjust communication style
        if profile.communication_style == "concise":
            personalized_template = self._make_template_concise(personalized_template)
        elif profile.communication_style == "detailed":
            personalized_template = self._make_template_detailed(personalized_template)
        
        # Add preferred frameworks/patterns if applicable
        if profile.preferred_frameworks:
            framework_preference = f"\nPreferred frameworks/technologies: {', '.join(profile.preferred_frameworks)}"
            personalized_template += framework_preference
        
        # Add coding style preferences
        if profile.preferred_coding_style:
            style_instructions = self._generate_style_instructions(profile.preferred_coding_style)
            personalized_template += f"\n\nCoding style preferences:\n{style_instructions}"
        
        # Adjust example inclusion based on preference
        if profile.example_preference > 0.7:
            personalized_template += "\n\nPlease include specific examples in your response."
        elif profile.example_preference < 0.3:
            personalized_template += "\n\nFocus on the solution without extensive examples."
        
        # Save personalized template
        new_template = PersonalizedPromptTemplate(
            template_id=f"{user_id}_{prompt_type}_{len(self._personalized_templates.get(user_id, []))}",
            user_id=user_id,
            base_template=base_template,
            personalized_template=personalized_template,
            effectiveness_score=0.5,  # Initial score
            usage_count=1,
            success_rate=0.0,  # Will be updated based on results
            context_tags=[prompt_type],
            created_at=datetime.utcnow(),
            last_used=datetime.utcnow()
        )
        
        if user_id not in self._personalized_templates:
            self._personalized_templates[user_id] = []
        self._personalized_templates[user_id].append(new_template)
        
        return personalized_template
    
    def _classify_prompt_type(self, prompt: str, task: Task) -> str:
        """Classify the type of prompt being used."""
        prompt_lower = prompt.lower()
        task_desc_lower = task.description.lower()
        
        if any(keyword in prompt_lower or keyword in task_desc_lower 
               for keyword in ['generate', 'create', 'implement', 'build', 'develop']):
            return 'code_generation'
        elif any(keyword in prompt_lower or keyword in task_desc_lower
                 for keyword in ['review', 'analyze', 'check', 'examine']):
            return 'code_review'
        elif any(keyword in prompt_lower or keyword in task_desc_lower
                 for keyword in ['debug', 'fix', 'error', 'issue', 'problem']):
            return 'debugging'
        elif any(keyword in prompt_lower or keyword in task_desc_lower
                 for keyword in ['architecture', 'design', 'structure', 'system']):
            return 'architecture_design'
        else:
            return 'code_generation'  # Default
    
    def _adjust_timeout(self, base_timeout: int, profile: PersonalizationProfile) -> int:
        """Adjust timeout based on user patterns."""
        # Users with higher learning velocity might need more time for complex tasks
        adjustment_factor = 0.8 + (profile.learning_velocity * 0.4)
        return int(base_timeout * adjustment_factor)
    
    def _build_personalized_context(
        self,
        profile: PersonalizationProfile,
        user_patterns: List[UserInteractionPattern],
        current_context: Dict[str, Any]
    ) -> str:
        """Build personalized context string."""
        context_parts = []
        
        # Add successful patterns context
        if profile.successful_task_types:
            context_parts.append(f"User excels at: {', '.join(profile.successful_task_types)}")
        
        # Add preferred frameworks
        if profile.preferred_frameworks:
            context_parts.append(f"Preferred frameworks: {', '.join(profile.preferred_frameworks)}")
        
        # Add recent success insights
        recent_successes = [
            p for p in user_patterns
            if p.pattern_type in ['success', 'validation_success']
            and (datetime.utcnow() - p.timestamp).days <= 7
        ]
        
        if recent_successes:
            avg_recent_success = sum(p.success_rate for p in recent_successes) / len(recent_successes)
            context_parts.append(f"Recent success rate: {avg_recent_success:.1%}")
        
        return " | ".join(context_parts)
    
    async def _update_profile_from_feedback(
        self,
        profile: PersonalizationProfile,
        task: Task,
        ai_result: AITaskResult,
        validation_result: ValidationResult,
        user_feedback: Dict[str, Any]
    ) -> None:
        """Update profile based on user feedback."""
        satisfaction = user_feedback.get('satisfaction', 0.5)
        feedback_type = user_feedback.get('type', 'general')
        
        # Update communication style preference
        if 'too_verbose' in user_feedback.get('comments', '').lower():
            profile.communication_style = 'concise'
        elif 'more_detail' in user_feedback.get('comments', '').lower():
            profile.communication_style = 'detailed'
        
        # Update validation strictness based on feedback
        if feedback_type == 'false_positive' or 'too_strict' in user_feedback.get('comments', ''):
            profile.validation_strictness *= 0.9  # Be less strict
        elif feedback_type == 'missed_issue' or 'not_strict_enough' in user_feedback.get('comments', ''):
            profile.validation_strictness *= 1.1  # Be more strict
        
        # Update detail preference
        if satisfaction > 0.8:
            # Successful interaction - reinforce current preferences
            detail_level = len(ai_result.generated_content or '') / 1000  # Rough metric
            profile.detail_preference = (
                profile.detail_preference * 0.9 + 
                min(detail_level, 1.0) * 0.1
            )
        
        # Update context-specific adaptations
        task_type = getattr(task, 'task_type', 'general')
        if task_type not in profile.context_adaptations:
            profile.context_adaptations[task_type] = {}
        
        context_adaptation = profile.context_adaptations[task_type]
        
        if satisfaction > 0.7:
            # Reinforce successful patterns
            if validation_result.authenticity_score > 80:
                context_adaptation['validation_threshold'] = context_adaptation.get(
                    'validation_threshold', 1.0
                ) * 0.95  # Slightly lower threshold for this context
        elif satisfaction < 0.4:
            # Adjust patterns that led to poor outcomes
            context_adaptation['validation_threshold'] = context_adaptation.get(
                'validation_threshold', 1.0
            ) * 1.05  # Higher threshold for this context
        
        # Clamp values to reasonable ranges
        profile.validation_strictness = max(0.3, min(1.5, profile.validation_strictness))
        profile.detail_preference = max(0.1, min(1.0, profile.detail_preference))
    
    async def _update_template_effectiveness(
        self,
        user_id: str,
        template_id: str,
        satisfaction_score: float,
        feedback: Dict[str, Any]
    ) -> None:
        """Update template effectiveness based on usage feedback."""
        user_templates = self._personalized_templates.get(user_id, [])
        
        for template in user_templates:
            if template.template_id == template_id or template.base_template in template_id:
                # Update effectiveness score
                template.usage_count += 1
                template.last_used = datetime.utcnow()
                
                # Update success rate (weighted average)
                weight = 1.0 / template.usage_count
                template.success_rate = (
                    template.success_rate * (1 - weight) +
                    satisfaction_score * weight
                )
                
                # Update effectiveness score based on success rate and usage
                usage_factor = min(template.usage_count / 10.0, 1.0)
                template.effectiveness_score = template.success_rate * usage_factor
                
                break
    
    async def _create_adaptation_rules_from_feedback(
        self,
        user_id: str,
        task: Task,
        ai_result: AITaskResult,
        validation_result: ValidationResult,
        feedback: Dict[str, Any]
    ) -> None:
        """Create adaptation rules from negative feedback."""
        # Identify what went wrong
        issues = []
        if validation_result.authenticity_score < 70:
            issues.append('low_authenticity')
        if len(validation_result.issues) > 3:
            issues.append('many_issues')
        if ai_result.execution_time > 300:  # 5 minutes
            issues.append('slow_execution')
        
        if not issues:
            return
        
        # Create condition based on current context
        condition = {
            'task_type': getattr(task, 'task_type', 'general'),
            'priority': task.priority.value if hasattr(task, 'priority') else 'medium',
            'issues_present': issues
        }
        
        # Create adaptation based on feedback
        adaptation = {}
        
        if 'low_authenticity' in issues:
            adaptation['increase_validation_strictness'] = 1.2
            adaptation['add_authenticity_check'] = True
        
        if 'many_issues' in issues:
            adaptation['increase_quality_threshold'] = 1.1
            adaptation['enable_detailed_review'] = True
        
        if 'slow_execution' in issues:
            adaptation['optimize_for_speed'] = True
            adaptation['reduce_complexity'] = True
        
        # Create adaptation rule
        rule = AdaptationRule(
            rule_id=f"{user_id}_rule_{len(self._adaptation_rules.get(user_id, []))}",
            condition=condition,
            adaptation=adaptation,
            confidence=0.7,  # Initial confidence
            effectiveness=0.0,  # Will be updated based on results
            applies_to=[getattr(task, 'task_type', 'general')]
        )
        
        if user_id not in self._adaptation_rules:
            self._adaptation_rules[user_id] = []
        self._adaptation_rules[user_id].append(rule)
    
    def _update_learning_metrics(
        self,
        profile: PersonalizationProfile,
        feedback: Dict[str, Any]
    ) -> None:
        """Update learning metrics based on feedback responsiveness."""
        satisfaction = feedback.get('satisfaction', 0.5)
        
        # Update feedback responsiveness
        profile.feedback_responsiveness = (
            profile.feedback_responsiveness * 0.9 +
            satisfaction * 0.1
        )
        
        # Update pattern stability (how consistent user patterns are)
        # This is a simplified metric - in practice would analyze pattern variance
        if satisfaction > 0.7:
            profile.pattern_stability = min(profile.pattern_stability + 0.05, 1.0)
        elif satisfaction < 0.3:
            profile.pattern_stability = max(profile.pattern_stability - 0.05, 0.1)
    
    def _make_template_concise(self, template: str) -> str:
        """Make template more concise."""
        # Remove verbose instructions
        concise_template = template.replace(
            "Please provide a complete implementation.", 
            "Provide implementation."
        )
        concise_template = concise_template.replace(
            "Be constructive and specific in your feedback.",
            "Be specific."
        )
        
        # Reduce instruction redundancy
        lines = concise_template.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Skip overly verbose lines
            if len(line) > 100 and any(word in line.lower() for word in ['please', 'ensure', 'make sure']):
                continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _make_template_detailed(self, template: str) -> str:
        """Make template more detailed."""
        detailed_additions = [
            "\nAdditional considerations:",
            "- Explain your reasoning and decision-making process",
            "- Provide alternative approaches where applicable", 
            "- Include relevant documentation and references",
            "- Consider edge cases and error scenarios"
        ]
        
        return template + '\n'.join(detailed_additions)
    
    def _generate_style_instructions(self, coding_style: Dict[str, Any]) -> str:
        """Generate coding style instructions from preferences."""
        instructions = []
        
        for aspect, preference in coding_style.items():
            if aspect == 'indentation' and preference:
                instructions.append(f"- Use {preference} for indentation")
            elif aspect == 'naming_convention' and preference:
                instructions.append(f"- Follow {preference} naming convention")
            elif aspect == 'comment_style' and preference:
                instructions.append(f"- Use {preference} comment style")
            elif aspect == 'line_length' and isinstance(preference, int):
                instructions.append(f"- Keep lines under {preference} characters")
        
        return '\n'.join(instructions) if instructions else "Follow clean code principles"
    
    def _apply_real_time_adaptations(
        self,
        prompt: str,
        profile: PersonalizationProfile,
        context: Dict[str, Any]
    ) -> str:
        """Apply real-time adaptations to prompt."""
        adapted_prompt = prompt
        
        # Add context-specific adaptations
        task_type = context.get('task_type', 'general')
        if task_type in profile.context_adaptations:
            adaptations = profile.context_adaptations[task_type]
            
            if adaptations.get('enable_detailed_review'):
                adapted_prompt += "\n\nPerform detailed code review and quality analysis."
            
            if adaptations.get('optimize_for_speed'):
                adapted_prompt += "\n\nPrioritize execution speed and efficiency."
        
        return adapted_prompt
    
    def _generate_style_suggestions(
        self,
        preferred_style: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate coding style suggestions."""
        suggestions = []
        
        # Check if current context conflicts with preferred style
        current_indentation = context.get('indentation_style')
        preferred_indentation = preferred_style.get('indentation')
        
        if (current_indentation and preferred_indentation and 
            current_indentation != preferred_indentation):
            suggestions.append({
                "type": "style",
                "category": "coding_consistency",
                "message": f"Consider using {preferred_indentation} indentation (your preference)",
                "confidence": 0.8,
                "actionable": True,
                "suggested_actions": [f"Configure editor to use {preferred_indentation}"]
            })
        
        return suggestions
    
    def _make_message_concise(self, message: str) -> str:
        """Make a message more concise."""
        # Simple heuristic: if message is > 100 chars, try to shorten
        if len(message) <= 100:
            return message
        
        # Remove filler words and phrases
        concise_message = message.replace("Consider ", "").replace("You might want to ", "")
        concise_message = concise_message.replace(" in order to", " to")
        
        # Truncate if still too long
        if len(concise_message) > 120:
            concise_message = concise_message[:117] + "..."
        
        return concise_message
    
    def _add_detailed_explanation(
        self,
        suggestion: Dict[str, Any],
        profile: PersonalizationProfile
    ) -> Dict[str, Any]:
        """Add detailed explanation to suggestion."""
        if profile.communication_style == "detailed":
            # Add explanation field
            suggestion["detailed_explanation"] = (
                f"Based on your usage patterns and preferences, this suggestion "
                f"aims to improve your development efficiency and code quality."
            )
            
            # Expand suggested actions with rationale
            if "suggested_actions" in suggestion:
                expanded_actions = []
                for action in suggestion["suggested_actions"]:
                    expanded_actions.append(f"{action} (improves workflow efficiency)")
                suggestion["suggested_actions"] = expanded_actions
        
        return suggestion
    
    async def export_personalization_data(
        self,
        user_id: str,
        include_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Export user's personalization data."""
        if user_id not in self._user_profiles:
            return {"error": "No profile found for user"}
        
        profile = self._user_profiles[user_id]
        
        export_data = {
            "profile": {
                "user_id": profile.user_id,
                "creation_date": profile.creation_date.isoformat(),
                "last_updated": profile.last_updated.isoformat(),
                "communication_style": profile.communication_style,
                "successful_task_types": profile.successful_task_types,
                "optimal_work_hours": profile.optimal_work_hours,
                "learning_velocity": profile.learning_velocity,
                "pattern_stability": profile.pattern_stability,
                "feedback_responsiveness": profile.feedback_responsiveness,
                "validation_strictness": profile.validation_strictness,
                "detail_preference": profile.detail_preference,
                "example_preference": profile.example_preference
            },
            "templates": {
                "count": len(self._personalized_templates.get(user_id, [])),
                "templates": [
                    {
                        "template_id": t.template_id,
                        "effectiveness_score": t.effectiveness_score,
                        "usage_count": t.usage_count,
                        "success_rate": t.success_rate,
                        "context_tags": t.context_tags,
                        "created_at": t.created_at.isoformat(),
                        "last_used": t.last_used.isoformat()
                    }
                    for t in self._personalized_templates.get(user_id, [])
                ]
            },
            "adaptation_rules": {
                "count": len(self._adaptation_rules.get(user_id, [])),
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "confidence": r.confidence,
                        "effectiveness": r.effectiveness,
                        "applies_to": r.applies_to
                    }
                    for r in self._adaptation_rules.get(user_id, [])
                ]
            },
            "export_metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "include_sensitive": include_sensitive
            }
        }
        
        if include_sensitive:
            # Include actual template content and rule details
            export_data["sensitive_data"] = {
                "template_content": [
                    {
                        "template_id": t.template_id,
                        "personalized_template": t.personalized_template
                    }
                    for t in self._personalized_templates.get(user_id, [])
                ],
                "adaptation_rule_details": [
                    {
                        "rule_id": r.rule_id,
                        "condition": r.condition,
                        "adaptation": r.adaptation
                    }
                    for r in self._adaptation_rules.get(user_id, [])
                ]
            }
        
        return export_data