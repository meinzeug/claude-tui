"""
Federated Learning System for AI Personalization.

This module implements privacy-preserving federated learning that allows teams
and organizations to collaboratively improve AI behavior while preserving
individual privacy. It enables pattern sharing, collaborative learning, and
best practice propagation across teams.
"""

import asyncio
import hashlib
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
from pathlib import Path
import hmac
import base64
from cryptography.fernet import Fernet
from collections import defaultdict, Counter

from .pattern_engine import PatternRecognitionEngine, UserInteractionPattern, DevelopmentSuccessPattern
from .personalization import PersonalizationProfile


logger = logging.getLogger(__name__)


@dataclass
class FederatedNode:
    """Represents a node in the federated learning network."""
    node_id: str
    organization: str
    node_type: str  # 'individual', 'team', 'organization'
    public_key: str
    capabilities: List[str]
    trust_score: float
    last_sync: datetime
    contribution_score: float
    privacy_level: str  # 'strict', 'moderate', 'open'
    

@dataclass
class PrivacyPreservingPattern:
    """Pattern with privacy-preserving transformations applied."""
    pattern_hash: str
    generalized_features: Dict[str, Any]
    success_metrics: Dict[str, float]
    confidence_level: float
    anonymized_metadata: Dict[str, Any]
    contribution_node: str
    created_at: datetime
    privacy_level: str
    usage_permissions: List[str]


@dataclass
class FederatedLearningRound:
    """Represents one round of federated learning."""
    round_id: str
    start_time: datetime
    end_time: Optional[datetime]
    participating_nodes: List[str]
    patterns_contributed: int
    patterns_received: int
    learning_objectives: List[str]
    privacy_budget: float
    consensus_threshold: float
    status: str  # 'active', 'completed', 'failed'


@dataclass
class CollaborativeInsight:
    """Insight derived from collaborative learning across nodes."""
    insight_id: str
    insight_type: str  # 'pattern', 'antipattern', 'best_practice', 'warning'
    description: str
    supporting_evidence: List[str]
    confidence: float
    applicability: List[str]  # Context where this insight applies
    learned_from_nodes: int
    privacy_safe: bool
    actionable_recommendations: List[str]


class PrivacyPreservingEncoder:
    """Handles privacy-preserving encoding of patterns and data."""
    
    def __init__(self, privacy_level: str = 'moderate'):
        """
        Initialize privacy encoder.
        
        Args:
            privacy_level: Level of privacy protection ('strict', 'moderate', 'open')
        """
        self.privacy_level = privacy_level
        self._generalization_rules = self._load_generalization_rules()
        self._differential_privacy_epsilon = {
            'strict': 0.1,
            'moderate': 1.0,
            'open': 10.0
        }.get(privacy_level, 1.0)
    
    async def encode_pattern(
        self,
        pattern: UserInteractionPattern,
        node_context: Dict[str, Any]
    ) -> PrivacyPreservingPattern:
        """
        Encode pattern with privacy-preserving transformations.
        
        Args:
            pattern: Pattern to encode
            node_context: Context information from the contributing node
            
        Returns:
            Privacy-preserving encoded pattern
        """
        # Apply generalization to features
        generalized_features = await self._generalize_features(pattern.features)
        
        # Add differential privacy noise
        if self.privacy_level in ['strict', 'moderate']:
            generalized_features = self._add_differential_privacy_noise(generalized_features)
        
        # Create anonymized metadata
        anonymized_metadata = self._anonymize_metadata(pattern.context, node_context)
        
        # Generate pattern hash for deduplication
        pattern_content = json.dumps(generalized_features, sort_keys=True)
        pattern_hash = hashlib.sha256(pattern_content.encode()).hexdigest()
        
        # Extract success metrics without revealing specifics
        success_metrics = {
            'success_rate_bin': self._bin_success_rate(pattern.success_rate),
            'confidence_level': self._bin_confidence(pattern.confidence),
            'frequency_bin': self._bin_frequency(pattern.frequency)
        }
        
        return PrivacyPreservingPattern(
            pattern_hash=pattern_hash,
            generalized_features=generalized_features,
            success_metrics=success_metrics,
            confidence_level=pattern.confidence,
            anonymized_metadata=anonymized_metadata,
            contribution_node=self._hash_node_id(node_context.get('node_id', 'unknown')),
            created_at=datetime.utcnow(),
            privacy_level=self.privacy_level,
            usage_permissions=node_context.get('usage_permissions', ['research'])
        )
    
    async def _generalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply generalization rules to features."""
        generalized = {}
        
        for feature_name, value in features.items():
            if feature_name in self._generalization_rules:
                rule = self._generalization_rules[feature_name]
                generalized[feature_name] = self._apply_generalization_rule(value, rule)
            else:
                # Default generalization based on data type
                generalized[feature_name] = self._default_generalize(value)
        
        return generalized
    
    def _apply_generalization_rule(self, value: Any, rule: Dict[str, Any]) -> Any:
        """Apply specific generalization rule."""
        rule_type = rule.get('type', 'range')
        
        if rule_type == 'range' and isinstance(value, (int, float)):
            # Bin numeric values into ranges
            bins = rule.get('bins', [0, 1, 5, 10, 50, 100, float('inf')])
            for i, threshold in enumerate(bins[1:]):
                if value <= threshold:
                    return f"range_{i}"
            return f"range_{len(bins)-1}"
        
        elif rule_type == 'category' and isinstance(value, str):
            # Map specific values to broader categories
            mapping = rule.get('mapping', {})
            return mapping.get(value, 'other')
        
        elif rule_type == 'suppress':
            # Suppress sensitive features
            return None
        
        else:
            return self._default_generalize(value)
    
    def _default_generalize(self, value: Any) -> Any:
        """Apply default generalization based on privacy level."""
        if self.privacy_level == 'strict':
            # Aggressive generalization
            if isinstance(value, (int, float)):
                return 'numeric_value'
            elif isinstance(value, str):
                return 'string_value'
            elif isinstance(value, bool):
                return 'boolean_value'
            else:
                return 'other_value'
        
        elif self.privacy_level == 'moderate':
            # Moderate generalization
            if isinstance(value, (int, float)):
                # Bin into broad ranges
                if value < 1:
                    return 'very_low'
                elif value < 10:
                    return 'low'
                elif value < 100:
                    return 'medium'
                else:
                    return 'high'
            else:
                return value  # Keep as-is for moderate privacy
        
        else:  # open
            return value  # No generalization for open privacy
    
    def _add_differential_privacy_noise(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Add differential privacy noise to numeric features."""
        noisy_features = features.copy()
        
        for feature_name, value in features.items():
            if isinstance(value, (int, float)):
                # Add Laplacian noise for differential privacy
                sensitivity = 1.0  # Assume unit sensitivity
                noise_scale = sensitivity / self._differential_privacy_epsilon
                noise = np.random.laplace(0, noise_scale)
                
                noisy_features[feature_name] = max(0, value + noise)  # Ensure non-negative
        
        return noisy_features
    
    def _anonymize_metadata(
        self,
        pattern_context: Dict[str, Any],
        node_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create anonymized metadata."""
        anonymized = {}
        
        # Only include non-identifying metadata
        safe_fields = ['task_type', 'framework', 'language', 'context_size']
        
        for field in safe_fields:
            if field in pattern_context:
                anonymized[field] = pattern_context[field]
        
        # Add aggregated temporal information
        if 'timestamp' in pattern_context:
            timestamp = pattern_context['timestamp']
            if isinstance(timestamp, datetime):
                # Only include hour of day and day of week (not specific date)
                anonymized['hour_of_day'] = timestamp.hour
                anonymized['day_of_week'] = timestamp.weekday()
        
        return anonymized
    
    def _bin_success_rate(self, success_rate: float) -> str:
        """Bin success rate into categories."""
        if success_rate >= 0.9:
            return 'excellent'
        elif success_rate >= 0.7:
            return 'good'
        elif success_rate >= 0.5:
            return 'moderate'
        else:
            return 'poor'
    
    def _bin_confidence(self, confidence: float) -> str:
        """Bin confidence into categories."""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _bin_frequency(self, frequency: int) -> str:
        """Bin frequency into categories."""
        if frequency >= 20:
            return 'very_frequent'
        elif frequency >= 10:
            return 'frequent'
        elif frequency >= 5:
            return 'moderate'
        else:
            return 'rare'
    
    def _hash_node_id(self, node_id: str) -> str:
        """Create privacy-preserving hash of node ID."""
        return hashlib.sha256(f"{node_id}:{self.privacy_level}".encode()).hexdigest()[:16]
    
    def _load_generalization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load feature generalization rules."""
        return {
            'execution_time': {
                'type': 'range',
                'bins': [0, 1, 5, 30, 300, 3600, float('inf')]
            },
            'task_type': {
                'type': 'category',
                'mapping': {
                    'code_generation': 'development',
                    'code_review': 'review',
                    'debugging': 'maintenance',
                    'testing': 'quality',
                    'documentation': 'documentation'
                }
            },
            'user_id': {
                'type': 'suppress'  # Never share user IDs
            },
            'file_path': {
                'type': 'suppress'  # Never share specific file paths
            },
            'project_path': {
                'type': 'suppress'  # Never share project paths
            }
        }


class FederatedLearningCoordinator:
    """Coordinates federated learning across multiple nodes."""
    
    def __init__(
        self,
        node_id: str,
        organization: str,
        privacy_level: str = 'moderate'
    ):
        """
        Initialize federated learning coordinator.
        
        Args:
            node_id: Unique identifier for this node
            organization: Organization this node belongs to
            privacy_level: Privacy protection level
        """
        self.node_id = node_id
        self.organization = organization
        self.privacy_encoder = PrivacyPreservingEncoder(privacy_level)
        
        # Network state
        self._trusted_nodes: Dict[str, FederatedNode] = {}
        self._learning_rounds: Dict[str, FederatedLearningRound] = {}
        self._shared_patterns: Dict[str, PrivacyPreservingPattern] = {}
        self._collaborative_insights: List[CollaborativeInsight] = []
        
        # Node information
        self._node_info = FederatedNode(
            node_id=node_id,
            organization=organization,
            node_type='individual',  # Can be updated
            public_key=self._generate_node_key(),
            capabilities=['pattern_sharing', 'collaborative_learning'],
            trust_score=1.0,
            last_sync=datetime.utcnow(),
            contribution_score=0.0,
            privacy_level=privacy_level
        )
        
        # Consensus and aggregation
        self._consensus_threshold = 0.7
        self._min_nodes_for_insight = 3
        
        logger.info(f"Federated learning coordinator initialized for node {node_id}")
    
    async def join_federation(
        self,
        federation_endpoint: str,
        credentials: Dict[str, str]
    ) -> bool:
        """
        Join a federated learning network.
        
        Args:
            federation_endpoint: Network endpoint to join
            credentials: Authentication credentials
            
        Returns:
            Success status
        """
        try:
            # In a real implementation, this would connect to the federation network
            # For now, we simulate the joining process
            
            # Authenticate and register node
            registration_data = {
                'node_id': self.node_id,
                'organization': self.organization,
                'capabilities': self._node_info.capabilities,
                'privacy_level': self._node_info.privacy_level,
                'public_key': self._node_info.public_key
            }
            
            # Simulate successful registration
            await asyncio.sleep(1)  # Simulate network delay
            
            logger.info(f"Node {self.node_id} successfully joined federation at {federation_endpoint}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to join federation: {e}")
            return False
    
    async def contribute_patterns(
        self,
        pattern_engine: PatternRecognitionEngine,
        user_patterns: Optional[Dict[str, List[UserInteractionPattern]]] = None,
        max_patterns: int = 50
    ) -> Dict[str, Any]:
        """
        Contribute patterns to federated learning network.
        
        Args:
            pattern_engine: Local pattern engine
            user_patterns: Specific patterns to contribute (optional)
            max_patterns: Maximum number of patterns to contribute
            
        Returns:
            Contribution summary
        """
        if user_patterns is None:
            user_patterns = pattern_engine._user_patterns
        
        contributed_patterns = []
        contribution_summary = {
            'patterns_contributed': 0,
            'patterns_rejected': 0,
            'privacy_level': self._node_info.privacy_level,
            'contribution_timestamp': datetime.utcnow().isoformat()
        }
        
        # Select patterns to contribute based on quality and privacy
        patterns_to_contribute = self._select_patterns_for_contribution(
            user_patterns, max_patterns
        )
        
        for user_id, patterns in patterns_to_contribute.items():
            for pattern in patterns:
                try:
                    # Apply privacy-preserving encoding
                    encoded_pattern = await self.privacy_encoder.encode_pattern(
                        pattern,
                        {
                            'node_id': self.node_id,
                            'organization': self.organization,
                            'usage_permissions': ['research', 'improvement']
                        }
                    )
                    
                    # Add to shared patterns
                    self._shared_patterns[encoded_pattern.pattern_hash] = encoded_pattern
                    contributed_patterns.append(encoded_pattern)
                    contribution_summary['patterns_contributed'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to encode pattern {pattern.pattern_id}: {e}")
                    contribution_summary['patterns_rejected'] += 1
        
        # Update contribution score
        self._node_info.contribution_score += len(contributed_patterns) * 0.1
        
        # Simulate broadcasting to network
        await self._broadcast_patterns(contributed_patterns)
        
        logger.info(f"Contributed {len(contributed_patterns)} patterns to federation")
        return contribution_summary
    
    async def receive_federated_insights(
        self,
        source_patterns: Optional[List[PrivacyPreservingPattern]] = None
    ) -> List[CollaborativeInsight]:
        """
        Process received patterns and generate collaborative insights.
        
        Args:
            source_patterns: Patterns received from other nodes
            
        Returns:
            Generated collaborative insights
        """
        if source_patterns is None:
            source_patterns = list(self._shared_patterns.values())
        
        insights = []
        
        # Cluster patterns to find common themes
        pattern_clusters = await self._cluster_patterns(source_patterns)
        
        for cluster_id, cluster_patterns in pattern_clusters.items():
            if len(cluster_patterns) >= self._min_nodes_for_insight:
                # Generate insights from cluster
                cluster_insights = await self._generate_cluster_insights(
                    cluster_id, cluster_patterns
                )
                insights.extend(cluster_insights)
        
        # Identify anti-patterns (common failure modes)
        anti_patterns = await self._identify_anti_patterns(source_patterns)
        insights.extend(anti_patterns)
        
        # Generate best practice recommendations
        best_practices = await self._generate_best_practices(source_patterns)
        insights.extend(best_practices)
        
        # Store insights
        self._collaborative_insights.extend(insights)
        
        logger.info(f"Generated {len(insights)} collaborative insights")
        return insights
    
    async def start_learning_round(
        self,
        learning_objectives: List[str],
        privacy_budget: float = 1.0,
        duration_hours: int = 24
    ) -> str:
        """
        Start a new federated learning round.
        
        Args:
            learning_objectives: Objectives for this learning round
            privacy_budget: Privacy budget allocation
            duration_hours: Duration of learning round
            
        Returns:
            Learning round ID
        """
        round_id = str(uuid4())
        
        learning_round = FederatedLearningRound(
            round_id=round_id,
            start_time=datetime.utcnow(),
            end_time=None,
            participating_nodes=[self.node_id],
            patterns_contributed=0,
            patterns_received=0,
            learning_objectives=learning_objectives,
            privacy_budget=privacy_budget,
            consensus_threshold=self._consensus_threshold,
            status='active'
        )
        
        self._learning_rounds[round_id] = learning_round
        
        # Schedule round completion
        asyncio.create_task(
            self._complete_learning_round(round_id, duration_hours)
        )
        
        logger.info(f"Started learning round {round_id} with objectives: {learning_objectives}")
        return round_id
    
    async def get_federated_recommendations(
        self,
        context: Dict[str, Any],
        user_profile: Optional[PersonalizationProfile] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on federated learning insights.
        
        Args:
            context: Current context for recommendations
            user_profile: User's personalization profile
            
        Returns:
            Federated recommendations
        """
        recommendations = []
        
        # Apply collaborative insights to current context
        relevant_insights = [
            insight for insight in self._collaborative_insights
            if self._is_insight_applicable(insight, context)
        ]
        
        for insight in relevant_insights[:5]:  # Top 5 insights
            recommendation = {
                'type': 'federated_insight',
                'category': insight.insight_type,
                'message': insight.description,
                'confidence': insight.confidence,
                'source': 'collaborative_learning',
                'supporting_evidence': len(insight.supporting_evidence),
                'learned_from_nodes': insight.learned_from_nodes,
                'actionable': len(insight.actionable_recommendations) > 0,
                'suggested_actions': insight.actionable_recommendations[:3]
            }
            
            # Adjust recommendation based on user profile
            if user_profile:
                recommendation = self._personalize_federated_recommendation(
                    recommendation, user_profile
                )
            
            recommendations.append(recommendation)
        
        # Add pattern-based recommendations
        pattern_recommendations = await self._get_pattern_based_recommendations(
            context, user_profile
        )
        recommendations.extend(pattern_recommendations)
        
        return recommendations
    
    def _select_patterns_for_contribution(
        self,
        user_patterns: Dict[str, List[UserInteractionPattern]],
        max_patterns: int
    ) -> Dict[str, List[UserInteractionPattern]]:
        """Select high-quality patterns for contribution."""
        selected_patterns = {}
        total_selected = 0
        
        for user_id, patterns in user_patterns.items():
            if total_selected >= max_patterns:
                break
            
            # Filter patterns based on quality criteria
            quality_patterns = [
                p for p in patterns
                if p.confidence >= 0.7 and
                   p.frequency >= 3 and
                   p.success_rate >= 0.6 and
                   p.pattern_type in ['success', 'validation_success']
            ]
            
            # Sort by effectiveness score
            quality_patterns.sort(
                key=lambda x: x.confidence * x.success_rate * np.log(1 + x.frequency),
                reverse=True
            )
            
            # Select top patterns for this user
            user_quota = min(len(quality_patterns), max_patterns // len(user_patterns) + 1)
            selected_for_user = quality_patterns[:user_quota]
            
            if selected_for_user:
                selected_patterns[user_id] = selected_for_user
                total_selected += len(selected_for_user)
        
        return selected_patterns
    
    async def _broadcast_patterns(
        self,
        patterns: List[PrivacyPreservingPattern]
    ) -> None:
        """Simulate broadcasting patterns to federation network."""
        # In real implementation, this would send patterns to other nodes
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Update local metrics
        for pattern in patterns:
            self._shared_patterns[pattern.pattern_hash] = pattern
    
    async def _cluster_patterns(
        self,
        patterns: List[PrivacyPreservingPattern]
    ) -> Dict[str, List[PrivacyPreservingPattern]]:
        """Cluster patterns to identify common themes."""
        clusters = defaultdict(list)
        
        # Simple clustering based on feature similarity
        for pattern in patterns:
            # Create cluster key from generalized features
            cluster_features = []
            for key, value in pattern.generalized_features.items():
                if isinstance(value, str):
                    cluster_features.append(f"{key}:{value}")
            
            cluster_key = "|".join(sorted(cluster_features[:5]))  # Use top 5 features
            clusters[cluster_key].append(pattern)
        
        # Filter clusters with minimum size
        filtered_clusters = {
            key: patterns for key, patterns in clusters.items()
            if len(patterns) >= self._min_nodes_for_insight
        }
        
        return filtered_clusters
    
    async def _generate_cluster_insights(
        self,
        cluster_id: str,
        patterns: List[PrivacyPreservingPattern]
    ) -> List[CollaborativeInsight]:
        """Generate insights from a cluster of patterns."""
        insights = []
        
        # Analyze success metrics across cluster
        success_rates = [
            self._decode_success_rate_bin(p.success_metrics.get('success_rate_bin', 'moderate'))
            for p in patterns
        ]
        
        avg_success_rate = np.mean(success_rates)
        
        # Extract common features
        feature_counts = Counter()
        for pattern in patterns:
            for feature, value in pattern.generalized_features.items():
                feature_counts[f"{feature}:{value}"] += 1
        
        common_features = [
            feature for feature, count in feature_counts.most_common(5)
            if count >= len(patterns) * 0.7  # Present in 70% of patterns
        ]
        
        if avg_success_rate >= 0.7 and common_features:
            # Success pattern insight
            insight = CollaborativeInsight(
                insight_id=f"success_pattern_{cluster_id}",
                insight_type='pattern',
                description=f"High success pattern identified with features: {', '.join(common_features)}",
                supporting_evidence=[p.pattern_hash for p in patterns],
                confidence=min(avg_success_rate, 0.95),
                applicability=self._extract_applicability_contexts(patterns),
                learned_from_nodes=len(set(p.contribution_node for p in patterns)),
                privacy_safe=True,
                actionable_recommendations=[
                    f"Apply pattern with features: {feature}" for feature in common_features[:3]
                ]
            )
            insights.append(insight)
        
        return insights
    
    async def _identify_anti_patterns(
        self,
        patterns: List[PrivacyPreservingPattern]
    ) -> List[CollaborativeInsight]:
        """Identify common failure patterns (anti-patterns)."""
        anti_patterns = []
        
        # Find patterns with poor success rates
        poor_patterns = [
            p for p in patterns
            if self._decode_success_rate_bin(p.success_metrics.get('success_rate_bin', 'moderate')) < 0.4
        ]
        
        if len(poor_patterns) >= self._min_nodes_for_insight:
            # Analyze common features in poor patterns
            feature_counts = Counter()
            for pattern in poor_patterns:
                for feature, value in pattern.generalized_features.items():
                    feature_counts[f"{feature}:{value}"] += 1
            
            common_failure_features = [
                feature for feature, count in feature_counts.most_common(3)
                if count >= len(poor_patterns) * 0.6
            ]
            
            if common_failure_features:
                anti_pattern = CollaborativeInsight(
                    insight_id=f"anti_pattern_{uuid4().hex[:8]}",
                    insight_type='antipattern',
                    description=f"Common failure pattern identified: {', '.join(common_failure_features)}",
                    supporting_evidence=[p.pattern_hash for p in poor_patterns],
                    confidence=0.8,
                    applicability=['failure_prevention'],
                    learned_from_nodes=len(set(p.contribution_node for p in poor_patterns)),
                    privacy_safe=True,
                    actionable_recommendations=[
                        f"Avoid: {feature}" for feature in common_failure_features[:2]
                    ]
                )
                anti_patterns.append(anti_pattern)
        
        return anti_patterns
    
    async def _generate_best_practices(
        self,
        patterns: List[PrivacyPreservingPattern]
    ) -> List[CollaborativeInsight]:
        """Generate best practice recommendations."""
        best_practices = []
        
        # Find consistently successful patterns across different contexts
        excellent_patterns = [
            p for p in patterns
            if (self._decode_success_rate_bin(p.success_metrics.get('success_rate_bin', 'moderate')) >= 0.8 and
                p.success_metrics.get('confidence_level') == 'high')
        ]
        
        if len(excellent_patterns) >= self._min_nodes_for_insight:
            # Extract best practice features
            practice_features = Counter()
            contexts = Counter()
            
            for pattern in excellent_patterns:
                for feature, value in pattern.generalized_features.items():
                    practice_features[f"{feature}:{value}"] += 1
                
                # Extract context information
                for context_key, context_value in pattern.anonymized_metadata.items():
                    contexts[f"{context_key}:{context_value}"] += 1
            
            # Generate best practice insight
            top_practices = [
                feature for feature, count in practice_features.most_common(3)
                if count >= len(excellent_patterns) * 0.5
            ]
            
            if top_practices:
                best_practice = CollaborativeInsight(
                    insight_id=f"best_practice_{uuid4().hex[:8]}",
                    insight_type='best_practice',
                    description=f"Best practices identified: {', '.join(top_practices)}",
                    supporting_evidence=[p.pattern_hash for p in excellent_patterns],
                    confidence=0.9,
                    applicability=list(set(self._extract_applicability_contexts(excellent_patterns))),
                    learned_from_nodes=len(set(p.contribution_node for p in excellent_patterns)),
                    privacy_safe=True,
                    actionable_recommendations=[
                        f"Adopt: {practice}" for practice in top_practices
                    ]
                )
                best_practices.append(best_practice)
        
        return best_practices
    
    def _decode_success_rate_bin(self, bin_label: str) -> float:
        """Decode success rate bin to numeric value."""
        bin_mapping = {
            'excellent': 0.95,
            'good': 0.8,
            'moderate': 0.6,
            'poor': 0.3
        }
        return bin_mapping.get(bin_label, 0.5)
    
    def _extract_applicability_contexts(
        self,
        patterns: List[PrivacyPreservingPattern]
    ) -> List[str]:
        """Extract contexts where patterns are applicable."""
        contexts = set()
        
        for pattern in patterns:
            metadata = pattern.anonymized_metadata
            
            # Extract task types
            if 'task_type' in metadata:
                contexts.add(metadata['task_type'])
            
            # Extract frameworks
            if 'framework' in metadata:
                contexts.add(f"framework_{metadata['framework']}")
            
            # Extract languages
            if 'language' in metadata:
                contexts.add(f"language_{metadata['language']}")
        
        return list(contexts)
    
    def _is_insight_applicable(
        self,
        insight: CollaborativeInsight,
        context: Dict[str, Any]
    ) -> bool:
        """Check if insight is applicable to current context."""
        if not insight.applicability:
            return True  # General applicability
        
        # Check if context matches any applicability criteria
        context_str = json.dumps(context, sort_keys=True).lower()
        
        for applicability_criterion in insight.applicability:
            if applicability_criterion.lower() in context_str:
                return True
        
        return False
    
    def _personalize_federated_recommendation(
        self,
        recommendation: Dict[str, Any],
        user_profile: PersonalizationProfile
    ) -> Dict[str, Any]:
        """Personalize federated recommendation based on user profile."""
        # Adjust message style based on communication preference
        if user_profile.communication_style == 'concise':
            recommendation['message'] = recommendation['message'][:100] + "..."
        elif user_profile.communication_style == 'detailed':
            recommendation['detailed_explanation'] = (
                "This recommendation is based on collaborative learning from multiple teams "
                "and has been validated across different contexts."
            )
        
        # Adjust confidence based on user's typical successful contexts
        if any(ctx in user_profile.successful_task_types 
               for ctx in recommendation.get('applicability', [])):
            recommendation['confidence'] = min(recommendation['confidence'] * 1.1, 1.0)
        
        return recommendation
    
    async def _get_pattern_based_recommendations(
        self,
        context: Dict[str, Any],
        user_profile: Optional[PersonalizationProfile]
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on federated patterns."""
        recommendations = []
        
        # Find patterns similar to current context
        relevant_patterns = []
        for pattern in self._shared_patterns.values():
            if self._is_pattern_contextually_relevant(pattern, context):
                relevant_patterns.append(pattern)
        
        if relevant_patterns:
            # Analyze successful patterns for recommendations
            successful_patterns = [
                p for p in relevant_patterns
                if self._decode_success_rate_bin(p.success_metrics.get('success_rate_bin', 'moderate')) >= 0.7
            ]
            
            if successful_patterns:
                # Extract common success factors
                success_factors = Counter()
                for pattern in successful_patterns:
                    for feature, value in pattern.generalized_features.items():
                        success_factors[f"{feature}: {value}"] += 1
                
                top_factors = success_factors.most_common(3)
                
                for factor, count in top_factors:
                    confidence = min(count / len(relevant_patterns), 1.0)
                    
                    recommendations.append({
                        'type': 'pattern_based',
                        'category': 'success_factor',
                        'message': f"Teams often succeed when {factor}",
                        'confidence': confidence,
                        'source': 'federated_patterns',
                        'supporting_patterns': count,
                        'actionable': True,
                        'suggested_actions': [f"Consider applying: {factor}"]
                    })
        
        return recommendations[:3]  # Limit to top 3
    
    def _is_pattern_contextually_relevant(
        self,
        pattern: PrivacyPreservingPattern,
        context: Dict[str, Any]
    ) -> bool:
        """Check if pattern is contextually relevant."""
        pattern_context = pattern.anonymized_metadata
        
        # Check for context overlap
        overlap_score = 0
        total_contexts = 0
        
        for key, value in context.items():
            if key in pattern_context:
                total_contexts += 1
                if pattern_context[key] == value:
                    overlap_score += 1
        
        # Require at least 30% context overlap
        return (overlap_score / total_contexts) >= 0.3 if total_contexts > 0 else False
    
    async def _complete_learning_round(
        self,
        round_id: str,
        duration_hours: int
    ) -> None:
        """Complete learning round after specified duration."""
        await asyncio.sleep(duration_hours * 3600)  # Wait for duration
        
        if round_id in self._learning_rounds:
            learning_round = self._learning_rounds[round_id]
            learning_round.end_time = datetime.utcnow()
            learning_round.status = 'completed'
            
            logger.info(f"Learning round {round_id} completed")
    
    def _generate_node_key(self) -> str:
        """Generate public key for node identification."""
        # In real implementation, this would generate actual cryptographic keys
        key_material = f"{self.node_id}:{self.organization}:{datetime.utcnow().timestamp()}"
        return base64.b64encode(hashlib.sha256(key_material.encode()).digest()).decode()
    
    async def get_federation_analytics(self) -> Dict[str, Any]:
        """Get analytics about federation participation."""
        total_patterns_contributed = len([
            p for p in self._shared_patterns.values()
            if p.contribution_node == self._hash_node_id(self.node_id)
        ])
        
        total_insights_generated = len(self._collaborative_insights)
        
        recent_rounds = [
            r for r in self._learning_rounds.values()
            if (datetime.utcnow() - r.start_time).days <= 30
        ]
        
        return {
            'node_info': {
                'node_id': self.node_id,
                'organization': self.organization,
                'trust_score': self._node_info.trust_score,
                'contribution_score': self._node_info.contribution_score,
                'privacy_level': self._node_info.privacy_level
            },
            'contribution_metrics': {
                'patterns_contributed': total_patterns_contributed,
                'insights_generated': total_insights_generated,
                'learning_rounds_participated': len(recent_rounds),
                'trusted_nodes': len(self._trusted_nodes)
            },
            'privacy_metrics': {
                'privacy_budget_used': sum(r.privacy_budget for r in recent_rounds),
                'privacy_preserving_patterns': len([
                    p for p in self._shared_patterns.values()
                    if p.privacy_level in ['strict', 'moderate']
                ])
            },
            'collaboration_metrics': {
                'unique_contributing_nodes': len(set(
                    p.contribution_node for p in self._shared_patterns.values()
                )),
                'cross_organization_insights': len([
                    i for i in self._collaborative_insights
                    if i.learned_from_nodes > 1
                ])
            }
        }
    
    def _hash_node_id(self, node_id: str) -> str:
        """Hash node ID for privacy."""
        return hashlib.sha256(f"{node_id}:privacy".encode()).hexdigest()[:16]


class FederatedLearningSystem:
    """
    Main federated learning system that integrates with pattern recognition
    and personalization to enable collaborative learning while preserving privacy.
    """
    
    def __init__(
        self,
        pattern_engine: PatternRecognitionEngine,
        node_id: str,
        organization: str,
        privacy_level: str = 'moderate'
    ):
        """
        Initialize federated learning system.
        
        Args:
            pattern_engine: Local pattern recognition engine
            node_id: Unique node identifier
            organization: Organization name
            privacy_level: Privacy protection level
        """
        self.pattern_engine = pattern_engine
        self.coordinator = FederatedLearningCoordinator(
            node_id, organization, privacy_level
        )
        
        # Integration state
        self._learning_active = False
        self._sync_interval_hours = 24
        self._last_sync = datetime.utcnow()
        
        logger.info(f"Federated learning system initialized for {organization}")
    
    async def enable_federated_learning(
        self,
        federation_config: Dict[str, Any]
    ) -> bool:
        """
        Enable federated learning with specified configuration.
        
        Args:
            federation_config: Federation configuration
            
        Returns:
            Success status
        """
        try:
            # Join federation network
            success = await self.coordinator.join_federation(
                federation_config.get('endpoint', 'localhost:8080'),
                federation_config.get('credentials', {})
            )
            
            if success:
                self._learning_active = True
                
                # Start periodic synchronization
                asyncio.create_task(self._periodic_sync())
                
                logger.info("Federated learning enabled successfully")
                return True
            
        except Exception as e:
            logger.error(f"Failed to enable federated learning: {e}")
            return False
        
        return False
    
    async def contribute_to_collective_learning(
        self,
        learning_objectives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Contribute patterns to collective learning network.
        
        Args:
            learning_objectives: Specific learning objectives
            
        Returns:
            Contribution results
        """
        if not self._learning_active:
            return {'error': 'Federated learning not enabled'}
        
        # Contribute patterns from local engine
        contribution_result = await self.coordinator.contribute_patterns(
            self.pattern_engine,
            max_patterns=100
        )
        
        # Start learning round if objectives specified
        if learning_objectives:
            round_id = await self.coordinator.start_learning_round(
                learning_objectives,
                privacy_budget=1.0
            )
            contribution_result['learning_round_id'] = round_id
        
        return contribution_result
    
    async def receive_collective_insights(self) -> List[CollaborativeInsight]:
        """Receive and process collective insights from network."""
        if not self._learning_active:
            return []
        
        # Process shared patterns to generate insights
        insights = await self.coordinator.receive_federated_insights()
        
        # Apply insights to improve local pattern engine
        await self._apply_insights_to_local_engine(insights)
        
        return insights
    
    async def get_federated_recommendations(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations enhanced with federated learning insights.
        
        Args:
            user_id: User identifier
            context: Current context
            
        Returns:
            Enhanced recommendations
        """
        if not self._learning_active:
            # Fallback to local recommendations only
            return await self.pattern_engine.generate_personalized_recommendations(
                user_id, context
            )
        
        # Get federated recommendations
        federated_recs = await self.coordinator.get_federated_recommendations(
            context
        )
        
        # Combine with local recommendations
        local_recs = await self.pattern_engine.generate_personalized_recommendations(
            user_id, context
        )
        
        # Merge and prioritize recommendations
        all_recommendations = federated_recs + local_recs
        
        # Deduplicate and sort by confidence
        unique_recs = {}
        for rec in all_recommendations:
            key = rec.get('message', '')[:50]  # Use message prefix as key
            if key not in unique_recs or rec.get('confidence', 0) > unique_recs[key].get('confidence', 0):
                unique_recs[key] = rec
        
        sorted_recs = sorted(
            unique_recs.values(),
            key=lambda x: x.get('confidence', 0) * (1.1 if x.get('source') == 'collaborative_learning' else 1.0),
            reverse=True
        )
        
        return sorted_recs[:10]  # Top 10 recommendations
    
    async def measure_federated_impact(
        self,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Measure impact of federated learning on local performance.
        
        Args:
            time_window_days: Time window for analysis
            
        Returns:
            Impact analysis results
        """
        # Get federation analytics
        federation_analytics = await self.coordinator.get_federation_analytics()
        
        # Analyze local pattern improvements
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
        
        # Count patterns learned from federation vs local
        federated_insights = len([
            i for i in self.coordinator._collaborative_insights
            if i.learned_from_nodes > 1
        ])
        
        local_patterns = sum(
            len(patterns) for patterns in self.pattern_engine._user_patterns.values()
        )
        
        impact_metrics = {
            'federation_analytics': federation_analytics,
            'local_vs_federated': {
                'local_patterns': local_patterns,
                'federated_insights': federated_insights,
                'enhancement_ratio': federated_insights / local_patterns if local_patterns > 0 else 0
            },
            'learning_acceleration': self._calculate_learning_acceleration(),
            'knowledge_diversity': self._calculate_knowledge_diversity(),
            'privacy_preservation_score': self._calculate_privacy_score()
        }
        
        return impact_metrics
    
    async def _periodic_sync(self) -> None:
        """Periodic synchronization with federation network."""
        while self._learning_active:
            try:
                # Wait for sync interval
                await asyncio.sleep(self._sync_interval_hours * 3600)
                
                # Contribute recent patterns
                await self.contribute_to_collective_learning(['pattern_sharing'])
                
                # Receive new insights
                await self.receive_collective_insights()
                
                self._last_sync = datetime.utcnow()
                logger.info("Periodic federation sync completed")
                
            except Exception as e:
                logger.error(f"Periodic sync failed: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _apply_insights_to_local_engine(
        self,
        insights: List[CollaborativeInsight]
    ) -> None:
        """Apply federated insights to improve local pattern engine."""
        for insight in insights:
            if insight.insight_type == 'best_practice' and insight.confidence > 0.8:
                # Update feature weights based on best practices
                for recommendation in insight.actionable_recommendations:
                    if ':' in recommendation:
                        feature_name = recommendation.split(':')[0].replace('Adopt: ', '')
                        if feature_name in self.pattern_engine._feature_weights:
                            self.pattern_engine._feature_weights[feature_name] *= 1.1
                        else:
                            self.pattern_engine._feature_weights[feature_name] = 1.1
            
            elif insight.insight_type == 'antipattern' and insight.confidence > 0.7:
                # Reduce weights for anti-pattern features
                for recommendation in insight.actionable_recommendations:
                    if ':' in recommendation:
                        feature_name = recommendation.split(':')[0].replace('Avoid: ', '')
                        if feature_name in self.pattern_engine._feature_weights:
                            self.pattern_engine._feature_weights[feature_name] *= 0.9
                        else:
                            self.pattern_engine._feature_weights[feature_name] = 0.9
    
    def _calculate_learning_acceleration(self) -> float:
        """Calculate learning acceleration from federated participation."""
        # Simple metric based on insights received vs time active
        days_active = (datetime.utcnow() - self.coordinator._node_info.last_sync).days or 1
        insights_per_day = len(self.coordinator._collaborative_insights) / days_active
        
        # Normalize to 0-1 scale (assumes 1 insight per day is baseline)
        return min(insights_per_day, 1.0)
    
    def _calculate_knowledge_diversity(self) -> float:
        """Calculate diversity of knowledge gained from federation."""
        if not self.coordinator._collaborative_insights:
            return 0.0
        
        # Count unique insight types and contributing nodes
        insight_types = set(i.insight_type for i in self.coordinator._collaborative_insights)
        contributing_nodes = set(i.learned_from_nodes for i in self.coordinator._collaborative_insights)
        
        # Diversity score based on variety of sources and types
        type_diversity = len(insight_types) / 4  # 4 possible types
        node_diversity = min(len(contributing_nodes) / 10, 1.0)  # Up to 10 nodes
        
        return (type_diversity + node_diversity) / 2
    
    def _calculate_privacy_score(self) -> float:
        """Calculate privacy preservation score."""
        privacy_levels = {'strict': 1.0, 'moderate': 0.7, 'open': 0.3}
        base_score = privacy_levels.get(self.coordinator._node_info.privacy_level, 0.5)
        
        # Adjust based on patterns shared with privacy protection
        if self.coordinator._shared_patterns:
            protected_patterns = sum(
                1 for p in self.coordinator._shared_patterns.values()
                if p.privacy_level in ['strict', 'moderate']
            )
            protection_ratio = protected_patterns / len(self.coordinator._shared_patterns)
            return base_score * protection_ratio
        
        return base_score