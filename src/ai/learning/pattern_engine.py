"""
Advanced Pattern Recognition Engine for AI Learning.

This module implements sophisticated pattern recognition algorithms
that learn from user interactions, validation feedback, and development
success patterns to improve AI behavior over time.
"""

import asyncio
import json
import logging
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
import hashlib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ...core.types import Task, ValidationResult, AITaskResult, ProgressMetrics
from ...core.logger import get_logger


logger = get_logger(__name__)


@dataclass
class UserInteractionPattern:
    """Represents a learned user interaction pattern."""
    pattern_id: str
    user_id: str
    pattern_type: str  # 'success', 'failure', 'preference', 'workflow'
    features: Dict[str, Any]
    success_rate: float
    frequency: int
    confidence: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class PromptEffectivenessMetrics:
    """Metrics for prompt effectiveness analysis."""
    prompt_template: str
    success_count: int
    failure_count: int
    average_validation_score: float
    average_execution_time: float
    user_satisfaction: float
    context_categories: List[str]
    improvement_suggestions: List[str]


@dataclass 
class DevelopmentSuccessPattern:
    """Pattern representing successful development practices."""
    pattern_name: str
    description: str
    conditions: Dict[str, Any]
    success_indicators: Dict[str, Any]
    frequency_score: float
    effectiveness_score: float
    applicable_contexts: List[str]
    learned_from: List[str]  # user_ids or team_ids


class PatternRecognitionEngine:
    """
    Advanced pattern recognition engine that learns from user interactions,
    validation feedback, and development outcomes to identify successful patterns.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        pattern_confidence_threshold: float = 0.7,
        enable_clustering: bool = True,
        max_patterns_per_user: int = 100
    ):
        """
        Initialize pattern recognition engine.
        
        Args:
            learning_rate: Rate at which patterns are updated
            pattern_confidence_threshold: Minimum confidence for pattern validity
            enable_clustering: Whether to use clustering for pattern discovery
            max_patterns_per_user: Maximum patterns stored per user
        """
        self.learning_rate = learning_rate
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.enable_clustering = enable_clustering
        self.max_patterns_per_user = max_patterns_per_user
        
        # Pattern storage
        self._user_patterns: Dict[str, List[UserInteractionPattern]] = defaultdict(list)
        self._prompt_effectiveness: Dict[str, PromptEffectivenessMetrics] = {}
        self._success_patterns: List[DevelopmentSuccessPattern] = []
        self._failure_patterns: List[DevelopmentSuccessPattern] = []
        
        # Learning models
        self._text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._pattern_clusters = {}
        self._feature_weights: Dict[str, float] = {}
        
        # Analytics
        self._pattern_evolution: List[Dict[str, Any]] = []
        self._learning_metrics: Dict[str, Any] = {}
        
        logger.info("Pattern recognition engine initialized")
    
    async def learn_from_interaction(
        self,
        user_id: str,
        task: Task,
        result: Union[AITaskResult, ValidationResult],
        context: Dict[str, Any]
    ) -> UserInteractionPattern:
        """
        Learn patterns from user interactions with AI system.
        
        Args:
            user_id: User identifier
            task: Task that was executed
            result: Result of the task execution
            context: Additional context information
            
        Returns:
            Learned interaction pattern
        """
        # Extract features from interaction
        features = await self._extract_interaction_features(task, result, context)
        
        # Determine pattern type and success
        pattern_type, success_rate = self._classify_interaction_outcome(result)
        
        # Create interaction pattern
        pattern = UserInteractionPattern(
            pattern_id=self._generate_pattern_id(user_id, task.name, features),
            user_id=user_id,
            pattern_type=pattern_type,
            features=features,
            success_rate=success_rate,
            frequency=1,
            confidence=self._calculate_initial_confidence(result),
            timestamp=datetime.utcnow(),
            context=context
        )
        
        # Update or create pattern
        await self._update_user_pattern(pattern)
        
        # Update prompt effectiveness if applicable
        if hasattr(result, 'metadata') and 'prompt_template' in result.metadata:
            await self._update_prompt_effectiveness(
                result.metadata['prompt_template'],
                result,
                context
            )
        
        # Trigger pattern discovery if needed
        if len(self._user_patterns[user_id]) % 10 == 0:
            await self._discover_new_patterns(user_id)
        
        logger.debug(f"Learned new pattern for user {user_id}: {pattern.pattern_type}")
        return pattern
    
    async def learn_from_validation_feedback(
        self,
        user_id: str,
        validation_result: ValidationResult,
        ai_task_result: AITaskResult,
        feedback_context: Dict[str, Any]
    ) -> None:
        """
        Learn from validation feedback to improve pattern recognition.
        
        Args:
            user_id: User identifier
            validation_result: Validation result
            ai_task_result: Original AI task result
            feedback_context: Feedback context
        """
        # Extract validation-specific features
        features = {
            'validation_score': validation_result.authenticity_score,
            'real_progress': validation_result.real_progress,
            'fake_progress': validation_result.fake_progress,
            'issue_count': len(validation_result.issues),
            'authenticity_rate': validation_result.authenticity_rate,
            'quality_score': getattr(validation_result, 'quality_score', 0.0)
        }
        
        # Add AI task features
        if ai_task_result:
            features.update({
                'execution_time': ai_task_result.execution_time,
                'generated_content_length': len(ai_task_result.generated_content or ''),
                'files_modified_count': len(ai_task_result.files_modified),
                'ai_confidence': getattr(ai_task_result, 'confidence', 0.0)
            })
        
        # Determine if this represents a success or failure pattern
        success_threshold = 80.0
        is_success = validation_result.authenticity_score >= success_threshold
        
        pattern_type = 'validation_success' if is_success else 'validation_failure'
        
        # Create pattern
        pattern = UserInteractionPattern(
            pattern_id=self._generate_pattern_id(user_id, 'validation', features),
            user_id=user_id,
            pattern_type=pattern_type,
            features=features,
            success_rate=validation_result.authenticity_score / 100.0,
            frequency=1,
            confidence=0.8,  # High confidence in validation feedback
            timestamp=datetime.utcnow(),
            context=feedback_context
        )
        
        await self._update_user_pattern(pattern)
        
        # Update success/failure pattern libraries
        if is_success:
            await self._update_success_patterns(features, feedback_context)
        else:
            await self._update_failure_patterns(features, validation_result.issues)
        
        logger.debug(f"Learned validation pattern for user {user_id}: {pattern_type}")
    
    async def identify_success_patterns(
        self,
        user_id: Optional[str] = None,
        context_filter: Optional[Dict[str, Any]] = None,
        min_frequency: int = 3
    ) -> List[DevelopmentSuccessPattern]:
        """
        Identify patterns that lead to successful development outcomes.
        
        Args:
            user_id: Filter by specific user
            context_filter: Filter by context attributes
            min_frequency: Minimum frequency for pattern validity
            
        Returns:
            List of identified success patterns
        """
        if user_id:
            user_patterns = self._user_patterns.get(user_id, [])
            success_patterns = [
                p for p in user_patterns 
                if p.pattern_type in ['success', 'validation_success'] 
                and p.frequency >= min_frequency
                and p.confidence >= self.pattern_confidence_threshold
            ]
        else:
            # Aggregate patterns across all users
            all_patterns = []
            for patterns in self._user_patterns.values():
                all_patterns.extend(patterns)
            
            success_patterns = [
                p for p in all_patterns
                if p.pattern_type in ['success', 'validation_success']
                and p.frequency >= min_frequency
                and p.confidence >= self.pattern_confidence_threshold
            ]
        
        # Convert to development success patterns
        dev_patterns = []
        for pattern in success_patterns:
            dev_pattern = DevelopmentSuccessPattern(
                pattern_name=f"Success Pattern {pattern.pattern_id[:8]}",
                description=self._generate_pattern_description(pattern),
                conditions=pattern.features,
                success_indicators={
                    'success_rate': pattern.success_rate,
                    'frequency': pattern.frequency,
                    'confidence': pattern.confidence
                },
                frequency_score=min(pattern.frequency / 10.0, 1.0),
                effectiveness_score=pattern.success_rate,
                applicable_contexts=self._extract_applicable_contexts(pattern),
                learned_from=[pattern.user_id]
            )
            dev_patterns.append(dev_pattern)
        
        # Sort by effectiveness
        dev_patterns.sort(key=lambda x: x.effectiveness_score * x.frequency_score, reverse=True)
        
        return dev_patterns
    
    async def predict_task_success_probability(
        self,
        user_id: str,
        task: Task,
        context: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Predict probability of task success based on learned patterns.
        
        Args:
            user_id: User identifier
            task: Task to predict success for
            context: Task context
            
        Returns:
            Tuple of (success_probability, contributing_factors)
        """
        # Extract task features
        task_features = await self._extract_task_features(task, context)
        
        # Get user patterns
        user_patterns = self._user_patterns.get(user_id, [])
        
        if not user_patterns:
            return 0.5, ["No historical patterns available"]
        
        # Find similar patterns
        similar_patterns = await self._find_similar_patterns(
            task_features, user_patterns, similarity_threshold=0.6
        )
        
        if not similar_patterns:
            return 0.5, ["No similar patterns found"]
        
        # Calculate weighted success probability
        total_weight = 0
        weighted_success = 0
        contributing_factors = []
        
        for pattern, similarity in similar_patterns:
            weight = similarity * pattern.confidence * np.log(1 + pattern.frequency)
            weighted_success += pattern.success_rate * weight
            total_weight += weight
            
            contributing_factors.append(
                f"{pattern.pattern_type} (similarity: {similarity:.2f}, "
                f"success_rate: {pattern.success_rate:.2f})"
            )
        
        success_probability = weighted_success / total_weight if total_weight > 0 else 0.5
        
        return success_probability, contributing_factors
    
    async def generate_personalized_recommendations(
        self,
        user_id: str,
        current_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations based on user patterns.
        
        Args:
            user_id: User identifier
            current_context: Current context for recommendations
            
        Returns:
            List of personalized recommendations
        """
        user_patterns = self._user_patterns.get(user_id, [])
        
        if not user_patterns:
            return [{"type": "info", "message": "No personalized patterns available yet"}]
        
        recommendations = []
        
        # Analyze recent failures
        recent_failures = [
            p for p in user_patterns
            if p.pattern_type in ['failure', 'validation_failure']
            and (datetime.utcnow() - p.timestamp).days <= 7
        ]
        
        if recent_failures:
            failure_features = Counter()
            for pattern in recent_failures:
                for feature, value in pattern.features.items():
                    if isinstance(value, (int, float)):
                        failure_features[feature] += 1
            
            # Most common failure patterns
            for feature, count in failure_features.most_common(3):
                recommendations.append({
                    "type": "warning",
                    "category": "pattern_analysis",
                    "message": f"Recent failures often involve {feature}. Consider focusing on this area.",
                    "confidence": min(count / len(recent_failures), 1.0),
                    "actionable": True,
                    "suggested_actions": [
                        f"Review and improve {feature} handling",
                        f"Add validation for {feature}",
                        f"Consider alternative approaches for {feature}"
                    ]
                })
        
        # Identify successful patterns to replicate
        success_patterns = [
            p for p in user_patterns
            if p.pattern_type in ['success', 'validation_success']
            and p.confidence >= self.pattern_confidence_threshold
        ]
        
        if success_patterns:
            # Most effective recent successes
            recent_successes = sorted(
                [p for p in success_patterns if (datetime.utcnow() - p.timestamp).days <= 14],
                key=lambda x: x.success_rate * x.confidence,
                reverse=True
            )[:3]
            
            for pattern in recent_successes:
                recommendations.append({
                    "type": "success",
                    "category": "pattern_replication",
                    "message": f"Your recent success with {pattern.pattern_type} had {pattern.success_rate:.1%} success rate",
                    "confidence": pattern.confidence,
                    "actionable": True,
                    "suggested_actions": [
                        "Apply similar approach to current tasks",
                        "Document this successful pattern",
                        "Share pattern with team if appropriate"
                    ],
                    "pattern_features": dict(list(pattern.features.items())[:5])  # Top 5 features
                })
        
        # Context-specific recommendations
        context_recs = await self._generate_context_recommendations(
            user_id, current_context, user_patterns
        )
        recommendations.extend(context_recs)
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    async def _extract_interaction_features(
        self,
        task: Task,
        result: Union[AITaskResult, ValidationResult],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract features from user interaction."""
        features = {
            'task_type': getattr(task, 'task_type', 'unknown'),
            'task_priority': task.priority.value if hasattr(task, 'priority') else 'medium',
            'task_description_length': len(task.description),
            'estimated_duration': getattr(task, 'estimated_duration', 0) or 0,
            'dependency_count': len(getattr(task, 'dependencies', set())),
            'context_size': len(context),
            'timestamp_hour': datetime.utcnow().hour,
            'timestamp_day_of_week': datetime.utcnow().weekday()
        }
        
        # Add result-specific features
        if isinstance(result, AITaskResult):
            features.update({
                'execution_time': result.execution_time,
                'generated_content_length': len(result.generated_content or ''),
                'files_modified_count': len(result.files_modified),
                'success': result.success,
                'validation_score': getattr(result, 'validation_score', 0.0)
            })
        elif isinstance(result, ValidationResult):
            features.update({
                'authenticity_score': result.authenticity_score,
                'real_progress': result.real_progress,
                'fake_progress': result.fake_progress,
                'issue_count': len(result.issues),
                'is_authentic': result.is_authentic
            })
        
        # Add context features
        for key, value in context.items():
            if isinstance(value, (int, float, bool)):
                features[f'context_{key}'] = value
            elif isinstance(value, str) and len(value) < 100:
                features[f'context_{key}_length'] = len(value)
        
        return features
    
    async def _extract_task_features(
        self,
        task: Task,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract features from a task for similarity matching."""
        features = {
            'task_type': getattr(task, 'task_type', 'unknown'),
            'priority': task.priority.value if hasattr(task, 'priority') else 'medium',
            'description_length': len(task.description),
            'estimated_duration': getattr(task, 'estimated_duration', 0) or 0,
            'dependency_count': len(getattr(task, 'dependencies', set())),
            'has_ai_prompt': bool(getattr(task, 'ai_prompt', None)),
            'ai_prompt_length': len(getattr(task, 'ai_prompt', '') or ''),
            'context_size': len(context)
        }
        
        # Add text-based features using TF-IDF
        text_content = f"{task.name} {task.description} {getattr(task, 'ai_prompt', '') or ''}"
        try:
            # Fit or transform text content
            if hasattr(self._text_vectorizer, 'vocabulary_') and self._text_vectorizer.vocabulary_:
                text_vector = self._text_vectorizer.transform([text_content])
            else:
                # Need some text to fit the vectorizer
                if len(self._user_patterns) > 0:
                    all_texts = []
                    for patterns in self._user_patterns.values():
                        for pattern in patterns[-10:]:  # Last 10 patterns
                            if 'description' in pattern.context:
                                all_texts.append(str(pattern.context['description']))
                    
                    if all_texts:
                        all_texts.append(text_content)
                        self._text_vectorizer.fit(all_texts)
                        text_vector = self._text_vectorizer.transform([text_content])
                    else:
                        text_vector = None
                else:
                    text_vector = None
            
            if text_vector is not None:
                # Add top TF-IDF features
                feature_names = self._text_vectorizer.get_feature_names_out()
                text_features = text_vector.toarray()[0]
                top_indices = np.argsort(text_features)[-10:]  # Top 10 features
                
                for idx in top_indices:
                    if text_features[idx] > 0:
                        features[f'text_{feature_names[idx]}'] = float(text_features[idx])
        
        except Exception as e:
            logger.warning(f"Error extracting text features: {e}")
        
        return features
    
    def _classify_interaction_outcome(
        self,
        result: Union[AITaskResult, ValidationResult]
    ) -> Tuple[str, float]:
        """Classify interaction outcome and calculate success rate."""
        if isinstance(result, AITaskResult):
            if result.success:
                success_rate = getattr(result, 'validation_score', 90.0) / 100.0
                return 'success', success_rate
            else:
                return 'failure', 0.0
        elif isinstance(result, ValidationResult):
            success_rate = result.authenticity_score / 100.0
            if result.is_authentic:
                return 'validation_success', success_rate
            else:
                return 'validation_failure', success_rate
        
        return 'unknown', 0.5
    
    def _calculate_initial_confidence(
        self,
        result: Union[AITaskResult, ValidationResult]
    ) -> float:
        """Calculate initial confidence in pattern."""
        base_confidence = 0.3
        
        if isinstance(result, AITaskResult):
            if result.success:
                confidence = base_confidence + 0.4
                # Boost confidence based on validation score
                validation_score = getattr(result, 'validation_score', 0.0)
                confidence += (validation_score / 100.0) * 0.3
            else:
                confidence = base_confidence
        elif isinstance(result, ValidationResult):
            # High confidence in validation results
            confidence = 0.6 + (result.authenticity_score / 100.0) * 0.3
        else:
            confidence = base_confidence
        
        return min(confidence, 1.0)
    
    def _generate_pattern_id(
        self,
        user_id: str,
        task_name: str,
        features: Dict[str, Any]
    ) -> str:
        """Generate unique pattern ID based on features."""
        # Create hash from key features
        feature_str = json.dumps(features, sort_keys=True, default=str)
        content = f"{user_id}:{task_name}:{feature_str}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def _update_user_pattern(self, new_pattern: UserInteractionPattern) -> None:
        """Update or create user pattern."""
        user_patterns = self._user_patterns[new_pattern.user_id]
        
        # Check for existing similar pattern
        existing_pattern = None
        for pattern in user_patterns:
            if pattern.pattern_id == new_pattern.pattern_id:
                existing_pattern = pattern
                break
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.frequency += 1
            existing_pattern.timestamp = new_pattern.timestamp
            
            # Update success rate using learning rate
            old_success = existing_pattern.success_rate
            new_success = new_pattern.success_rate
            existing_pattern.success_rate = (
                old_success * (1 - self.learning_rate) +
                new_success * self.learning_rate
            )
            
            # Update confidence
            frequency_boost = min(existing_pattern.frequency / 10.0, 0.3)
            existing_pattern.confidence = min(
                existing_pattern.confidence + frequency_boost,
                1.0
            )
        else:
            # Add new pattern
            user_patterns.append(new_pattern)
            
            # Maintain max patterns per user
            if len(user_patterns) > self.max_patterns_per_user:
                # Remove oldest low-confidence patterns
                user_patterns.sort(key=lambda x: (x.confidence, x.timestamp))
                self._user_patterns[new_pattern.user_id] = user_patterns[-self.max_patterns_per_user:]
    
    async def _update_prompt_effectiveness(
        self,
        prompt_template: str,
        result: AITaskResult,
        context: Dict[str, Any]
    ) -> None:
        """Update prompt effectiveness metrics."""
        if prompt_template not in self._prompt_effectiveness:
            self._prompt_effectiveness[prompt_template] = PromptEffectivenessMetrics(
                prompt_template=prompt_template,
                success_count=0,
                failure_count=0,
                average_validation_score=0.0,
                average_execution_time=0.0,
                user_satisfaction=0.0,
                context_categories=[],
                improvement_suggestions=[]
            )
        
        metrics = self._prompt_effectiveness[prompt_template]
        
        if result.success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
        
        # Update averages
        total_uses = metrics.success_count + metrics.failure_count
        validation_score = getattr(result, 'validation_score', 0.0)
        
        metrics.average_validation_score = (
            (metrics.average_validation_score * (total_uses - 1) + validation_score) / total_uses
        )
        
        metrics.average_execution_time = (
            (metrics.average_execution_time * (total_uses - 1) + result.execution_time) / total_uses
        )
        
        # Update context categories
        context_type = context.get('type', 'unknown')
        if context_type not in metrics.context_categories:
            metrics.context_categories.append(context_type)
    
    async def _discover_new_patterns(self, user_id: str) -> None:
        """Discover new patterns using clustering."""
        if not self.enable_clustering:
            return
        
        user_patterns = self._user_patterns.get(user_id, [])
        if len(user_patterns) < 5:
            return
        
        try:
            # Prepare feature vectors for clustering
            feature_vectors = []
            pattern_indices = []
            
            for i, pattern in enumerate(user_patterns):
                # Convert features to numeric vector
                vector = []
                for key in sorted(pattern.features.keys()):
                    value = pattern.features[key]
                    if isinstance(value, (int, float)):
                        vector.append(value)
                    elif isinstance(value, bool):
                        vector.append(1.0 if value else 0.0)
                    else:
                        vector.append(hash(str(value)) % 1000 / 1000.0)  # Hash to [0,1]
                
                if vector:
                    feature_vectors.append(vector)
                    pattern_indices.append(i)
            
            if len(feature_vectors) >= 3:
                # Perform clustering
                n_clusters = min(5, len(feature_vectors) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(feature_vectors)
                
                # Analyze clusters for new patterns
                for cluster_id in range(n_clusters):
                    cluster_patterns = [
                        user_patterns[pattern_indices[i]]
                        for i, label in enumerate(cluster_labels)
                        if label == cluster_id
                    ]
                    
                    if len(cluster_patterns) >= 2:
                        # Extract common features in this cluster
                        await self._extract_cluster_pattern(cluster_id, cluster_patterns, user_id)
        
        except Exception as e:
            logger.warning(f"Pattern discovery failed for user {user_id}: {e}")
    
    async def _extract_cluster_pattern(
        self,
        cluster_id: int,
        cluster_patterns: List[UserInteractionPattern],
        user_id: str
    ) -> None:
        """Extract common pattern from cluster."""
        # Calculate cluster statistics
        success_rates = [p.success_rate for p in cluster_patterns]
        avg_success_rate = np.mean(success_rates)
        
        # Find common features
        common_features = {}
        for feature_name in cluster_patterns[0].features.keys():
            values = []
            for pattern in cluster_patterns:
                if feature_name in pattern.features:
                    value = pattern.features[feature_name]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if values and len(values) >= len(cluster_patterns) * 0.8:
                common_features[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Create meta-pattern
        if common_features and avg_success_rate > 0.6:
            meta_pattern_id = f"cluster_{user_id}_{cluster_id}"
            
            # Store cluster analysis
            self._pattern_clusters[meta_pattern_id] = {
                'cluster_id': cluster_id,
                'user_id': user_id,
                'pattern_count': len(cluster_patterns),
                'avg_success_rate': avg_success_rate,
                'common_features': common_features,
                'discovered_at': datetime.utcnow(),
                'confidence': min(len(cluster_patterns) / 10.0, 1.0)
            }
    
    async def _find_similar_patterns(
        self,
        task_features: Dict[str, Any],
        user_patterns: List[UserInteractionPattern],
        similarity_threshold: float = 0.5
    ) -> List[Tuple[UserInteractionPattern, float]]:
        """Find patterns similar to given task features."""
        similar_patterns = []
        
        for pattern in user_patterns:
            similarity = self._calculate_feature_similarity(
                task_features, pattern.features
            )
            
            if similarity >= similarity_threshold:
                similar_patterns.append((pattern, similarity))
        
        # Sort by similarity
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        return similar_patterns[:10]  # Top 10 similar patterns
    
    def _calculate_feature_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two feature sets."""
        # Get common feature keys
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity using normalized difference
                if val1 == 0 and val2 == 0:
                    sim = 1.0
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        sim = 1.0 - abs(val1 - val2) / max_val
                    else:
                        sim = 1.0
            elif isinstance(val1, bool) and isinstance(val2, bool):
                sim = 1.0 if val1 == val2 else 0.0
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity using character overlap
                if val1 == val2:
                    sim = 1.0
                else:
                    chars1 = set(val1.lower())
                    chars2 = set(val2.lower())
                    if chars1 or chars2:
                        sim = len(chars1 & chars2) / len(chars1 | chars2)
                    else:
                        sim = 1.0
            else:
                # Default equality check
                sim = 1.0 if val1 == val2 else 0.0
            
            similarities.append(sim)
        
        # Weight by feature importance if available
        weights = [self._feature_weights.get(key, 1.0) for key in common_keys]
        weighted_similarity = np.average(similarities, weights=weights)
        
        # Penalty for missing features
        total_features = len(set(features1.keys()) | set(features2.keys()))
        feature_coverage = len(common_keys) / total_features if total_features > 0 else 1.0
        
        return weighted_similarity * feature_coverage
    
    async def _update_success_patterns(
        self,
        features: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Update library of success patterns."""
        # Look for existing similar success pattern
        for pattern in self._success_patterns:
            similarity = self._calculate_feature_similarity(features, pattern.conditions)
            if similarity > 0.8:
                # Update existing pattern
                pattern.frequency_score = min(pattern.frequency_score + 0.1, 1.0)
                pattern.effectiveness_score = (
                    pattern.effectiveness_score * 0.9 +
                    (features.get('validation_score', 0.0) / 100.0) * 0.1
                )
                return
        
        # Create new success pattern
        success_pattern = DevelopmentSuccessPattern(
            pattern_name=f"Success Pattern {len(self._success_patterns) + 1}",
            description="Auto-discovered success pattern",
            conditions=features.copy(),
            success_indicators={
                'validation_score': features.get('validation_score', 0.0),
                'real_progress': features.get('real_progress', 0.0)
            },
            frequency_score=0.1,
            effectiveness_score=features.get('validation_score', 0.0) / 100.0,
            applicable_contexts=[context.get('task_type', 'general')],
            learned_from=['pattern_discovery']
        )
        
        self._success_patterns.append(success_pattern)
        
        # Keep only top patterns
        if len(self._success_patterns) > 50:
            self._success_patterns.sort(
                key=lambda x: x.effectiveness_score * x.frequency_score,
                reverse=True
            )
            self._success_patterns = self._success_patterns[:50]
    
    async def _update_failure_patterns(
        self,
        features: Dict[str, Any],
        issues: List[Any]
    ) -> None:
        """Update library of failure patterns."""
        failure_pattern = DevelopmentSuccessPattern(
            pattern_name=f"Failure Pattern {len(self._failure_patterns) + 1}",
            description=f"Failure with {len(issues)} issues",
            conditions=features.copy(),
            success_indicators={'issue_count': len(issues)},
            frequency_score=0.1,
            effectiveness_score=0.0,  # Failure has 0 effectiveness
            applicable_contexts=['failure_prevention'],
            learned_from=['validation_feedback']
        )
        
        self._failure_patterns.append(failure_pattern)
        
        # Keep recent failure patterns for analysis
        if len(self._failure_patterns) > 30:
            self._failure_patterns = self._failure_patterns[-30:]
    
    def _generate_pattern_description(self, pattern: UserInteractionPattern) -> str:
        """Generate human-readable pattern description."""
        desc_parts = [f"Pattern type: {pattern.pattern_type}"]
        
        if pattern.success_rate > 0.8:
            desc_parts.append("High success rate")
        elif pattern.success_rate > 0.6:
            desc_parts.append("Moderate success rate")
        else:
            desc_parts.append("Low success rate")
        
        # Add key feature insights
        key_features = sorted(
            pattern.features.items(),
            key=lambda x: abs(hash(str(x[1]))) % 100,  # Pseudo-importance
            reverse=True
        )[:3]
        
        for key, value in key_features:
            if isinstance(value, (int, float)):
                desc_parts.append(f"{key}: {value:.2f}")
            else:
                desc_parts.append(f"{key}: {value}")
        
        return " | ".join(desc_parts)
    
    def _extract_applicable_contexts(self, pattern: UserInteractionPattern) -> List[str]:
        """Extract applicable contexts from pattern."""
        contexts = []
        
        if pattern.pattern_type in ['success', 'validation_success']:
            contexts.append('high_confidence_tasks')
        
        task_type = pattern.features.get('task_type', 'unknown')
        if task_type != 'unknown':
            contexts.append(f'{task_type}_tasks')
        
        priority = pattern.features.get('task_priority', 'unknown')
        if priority != 'unknown':
            contexts.append(f'{priority}_priority')
        
        return contexts
    
    async def _generate_context_recommendations(
        self,
        user_id: str,
        current_context: Dict[str, Any],
        user_patterns: List[UserInteractionPattern]
    ) -> List[Dict[str, Any]]:
        """Generate context-specific recommendations."""
        recommendations = []
        
        # Time-based patterns
        current_hour = datetime.utcnow().hour
        hour_patterns = [
            p for p in user_patterns
            if p.features.get('timestamp_hour') == current_hour
        ]
        
        if hour_patterns:
            avg_success = np.mean([p.success_rate for p in hour_patterns])
            if avg_success > 0.8:
                recommendations.append({
                    "type": "info",
                    "category": "temporal_pattern",
                    "message": f"You typically perform well at this hour ({avg_success:.1%} success rate)",
                    "confidence": 0.7,
                    "actionable": False
                })
            elif avg_success < 0.5:
                recommendations.append({
                    "type": "warning", 
                    "category": "temporal_pattern",
                    "message": f"Consider avoiding complex tasks at this hour (low success rate: {avg_success:.1%})",
                    "confidence": 0.6,
                    "actionable": True,
                    "suggested_actions": ["Schedule complex tasks for better times", "Take breaks more frequently"]
                })
        
        return recommendations
    
    async def get_pattern_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive pattern analytics."""
        if user_id:
            patterns = self._user_patterns.get(user_id, [])
            total_users = 1
        else:
            patterns = []
            for user_patterns in self._user_patterns.values():
                patterns.extend(user_patterns)
            total_users = len(self._user_patterns)
        
        if not patterns:
            return {
                'total_patterns': 0,
                'users': total_users,
                'pattern_types': {},
                'success_analysis': {},
                'temporal_analysis': {}
            }
        
        # Pattern type distribution
        pattern_types = Counter(p.pattern_type for p in patterns)
        
        # Success analysis
        success_patterns = [p for p in patterns if p.pattern_type in ['success', 'validation_success']]
        failure_patterns = [p for p in patterns if p.pattern_type in ['failure', 'validation_failure']]
        
        success_analysis = {
            'success_count': len(success_patterns),
            'failure_count': len(failure_patterns),
            'success_rate': len(success_patterns) / len(patterns) if patterns else 0,
            'avg_success_score': np.mean([p.success_rate for p in success_patterns]) if success_patterns else 0,
            'avg_confidence': np.mean([p.confidence for p in success_patterns]) if success_patterns else 0
        }
        
        # Temporal analysis
        recent_patterns = [
            p for p in patterns
            if (datetime.utcnow() - p.timestamp).days <= 7
        ]
        
        temporal_analysis = {
            'recent_patterns': len(recent_patterns),
            'recent_success_rate': (
                len([p for p in recent_patterns if p.pattern_type in ['success', 'validation_success']]) /
                len(recent_patterns) if recent_patterns else 0
            ),
            'learning_velocity': len(recent_patterns) / 7,  # Patterns per day
            'pattern_evolution': self._analyze_pattern_evolution(patterns)
        }
        
        return {
            'total_patterns': len(patterns),
            'users': total_users,
            'pattern_types': dict(pattern_types),
            'success_analysis': success_analysis,
            'temporal_analysis': temporal_analysis,
            'cluster_analysis': self._get_cluster_analytics(),
            'prompt_effectiveness': self._get_prompt_analytics()
        }
    
    def _analyze_pattern_evolution(self, patterns: List[UserInteractionPattern]) -> Dict[str, Any]:
        """Analyze how patterns have evolved over time."""
        if not patterns:
            return {'trend': 'no_data'}
        
        # Sort by timestamp
        sorted_patterns = sorted(patterns, key=lambda x: x.timestamp)
        
        # Split into time windows
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        recent_week = [p for p in sorted_patterns if p.timestamp >= week_ago]
        recent_month = [p for p in sorted_patterns if p.timestamp >= month_ago]
        older = [p for p in sorted_patterns if p.timestamp < month_ago]
        
        evolution = {
            'recent_week_count': len(recent_week),
            'recent_month_count': len(recent_month), 
            'older_count': len(older)
        }
        
        if recent_week and older:
            recent_success = np.mean([
                p.success_rate for p in recent_week
                if p.pattern_type in ['success', 'validation_success']
            ]) if recent_week else 0
            
            older_success = np.mean([
                p.success_rate for p in older
                if p.pattern_type in ['success', 'validation_success']
            ]) if older else 0
            
            evolution['success_improvement'] = recent_success - older_success
            evolution['trend'] = 'improving' if recent_success > older_success else 'declining'
        else:
            evolution['trend'] = 'insufficient_data'
        
        return evolution
    
    def _get_cluster_analytics(self) -> Dict[str, Any]:
        """Get cluster analysis analytics."""
        return {
            'total_clusters': len(self._pattern_clusters),
            'clusters_by_confidence': {
                'high': len([c for c in self._pattern_clusters.values() if c['confidence'] > 0.7]),
                'medium': len([c for c in self._pattern_clusters.values() if 0.3 <= c['confidence'] <= 0.7]),
                'low': len([c for c in self._pattern_clusters.values() if c['confidence'] < 0.3])
            }
        }
    
    def _get_prompt_analytics(self) -> Dict[str, Any]:
        """Get prompt effectiveness analytics."""
        if not self._prompt_effectiveness:
            return {'total_prompts': 0}
        
        total_uses = sum(
            m.success_count + m.failure_count
            for m in self._prompt_effectiveness.values()
        )
        
        avg_success_rate = np.mean([
            m.success_count / (m.success_count + m.failure_count)
            for m in self._prompt_effectiveness.values()
            if (m.success_count + m.failure_count) > 0
        ]) if self._prompt_effectiveness else 0
        
        return {
            'total_prompts': len(self._prompt_effectiveness),
            'total_uses': total_uses,
            'average_success_rate': avg_success_rate,
            'top_performing_prompts': self._get_top_prompts(3)
        }
    
    def _get_top_prompts(self, limit: int) -> List[Dict[str, Any]]:
        """Get top performing prompts."""
        prompt_scores = []
        
        for template, metrics in self._prompt_effectiveness.items():
            total_uses = metrics.success_count + metrics.failure_count
            if total_uses >= 3:  # Minimum usage threshold
                success_rate = metrics.success_count / total_uses
                score = success_rate * np.log(1 + total_uses)  # Frequency boost
                
                prompt_scores.append({
                    'template': template[:100] + '...' if len(template) > 100 else template,
                    'success_rate': success_rate,
                    'total_uses': total_uses,
                    'score': score,
                    'avg_validation_score': metrics.average_validation_score
                })
        
        prompt_scores.sort(key=lambda x: x['score'], reverse=True)
        return prompt_scores[:limit]
    
    async def export_patterns(
        self,
        user_id: Optional[str] = None,
        format: str = 'json'
    ) -> Union[str, Dict[str, Any]]:
        """Export learned patterns for backup or sharing."""
        if user_id:
            patterns_data = {
                'user_patterns': {user_id: [
                    self._pattern_to_dict(p) for p in self._user_patterns.get(user_id, [])
                ]},
                'clusters': {
                    k: v for k, v in self._pattern_clusters.items()
                    if v.get('user_id') == user_id
                }
            }
        else:
            patterns_data = {
                'user_patterns': {
                    uid: [self._pattern_to_dict(p) for p in patterns]
                    for uid, patterns in self._user_patterns.items()
                },
                'clusters': self._pattern_clusters.copy(),
                'success_patterns': [
                    self._success_pattern_to_dict(p) for p in self._success_patterns
                ],
                'prompt_effectiveness': {
                    template: {
                        'success_count': metrics.success_count,
                        'failure_count': metrics.failure_count,
                        'average_validation_score': metrics.average_validation_score,
                        'average_execution_time': metrics.average_execution_time
                    }
                    for template, metrics in self._prompt_effectiveness.items()
                }
            }
        
        patterns_data['export_metadata'] = {
            'exported_at': datetime.utcnow().isoformat(),
            'version': '1.0',
            'total_patterns': sum(len(patterns) for patterns in self._user_patterns.values())
        }
        
        if format == 'json':
            return json.dumps(patterns_data, indent=2, default=str)
        else:
            return patterns_data
    
    def _pattern_to_dict(self, pattern: UserInteractionPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization."""
        return {
            'pattern_id': pattern.pattern_id,
            'user_id': pattern.user_id,
            'pattern_type': pattern.pattern_type,
            'features': pattern.features,
            'success_rate': pattern.success_rate,
            'frequency': pattern.frequency,
            'confidence': pattern.confidence,
            'timestamp': pattern.timestamp.isoformat(),
            'context': pattern.context
        }
    
    def _success_pattern_to_dict(self, pattern: DevelopmentSuccessPattern) -> Dict[str, Any]:
        """Convert success pattern to dictionary."""
        return {
            'pattern_name': pattern.pattern_name,
            'description': pattern.description,
            'conditions': pattern.conditions,
            'success_indicators': pattern.success_indicators,
            'frequency_score': pattern.frequency_score,
            'effectiveness_score': pattern.effectiveness_score,
            'applicable_contexts': pattern.applicable_contexts,
            'learned_from': pattern.learned_from
        }