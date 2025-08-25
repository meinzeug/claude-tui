"""
Learning Analytics Dashboard and Insights System.

This module provides comprehensive analytics and insights for AI learning
and personalization systems, including pattern analysis, learning metrics,
performance tracking, and predictive analytics for continuous improvement.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from uuid import UUID
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .pattern_engine import PatternRecognitionEngine, UserInteractionPattern
from .personalization import PersonalizedAIBehavior, PersonalizationProfile
from .federated import FederatedLearningSystem, CollaborativeInsight


logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Core learning metrics for analysis."""
    user_id: str
    time_period: str
    total_interactions: int
    successful_interactions: int
    success_rate: float
    average_confidence: float
    learning_velocity: float
    pattern_stability: float
    personalization_effectiveness: float
    federated_contributions: int
    insights_received: int
    recommendation_adoption_rate: float


@dataclass 
class PatternAnalysis:
    """Analysis results for patterns."""
    pattern_type: str
    frequency_distribution: Dict[str, int]
    success_rate_by_context: Dict[str, float]
    temporal_trends: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    outlier_patterns: List[str]
    emerging_patterns: List[str]
    declining_patterns: List[str]


@dataclass
class PersonalizationInsight:
    """Insight derived from personalization analysis."""
    insight_id: str
    user_id: str
    insight_type: str  # 'improvement', 'trend', 'recommendation', 'alert'
    title: str
    description: str
    impact_score: float
    confidence: float
    actionable_steps: List[str]
    supporting_data: Dict[str, Any]
    generated_at: datetime


@dataclass
class LearningReport:
    """Comprehensive learning analytics report."""
    report_id: str
    report_type: str  # 'user', 'team', 'organization'
    time_range: Tuple[datetime, datetime]
    summary_metrics: LearningMetrics
    pattern_analyses: List[PatternAnalysis]
    personalization_insights: List[PersonalizationInsight]
    federated_impact: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    visualizations: List[str]  # Paths to generated charts
    generated_at: datetime


class LearningAnalytics:
    """
    Comprehensive analytics system for AI learning and personalization.
    
    Provides deep insights into learning patterns, user behavior, system
    performance, and opportunities for improvement through advanced
    analytics and machine learning techniques.
    """
    
    def __init__(
        self,
        pattern_engine: PatternRecognitionEngine,
        personalized_behavior: PersonalizedAIBehavior,
        federated_system: Optional[FederatedLearningSystem] = None,
        analytics_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize learning analytics system.
        
        Args:
            pattern_engine: Pattern recognition engine
            personalized_behavior: Personalized AI behavior system
            federated_system: Federated learning system (optional)
            analytics_config: Analytics configuration
        """
        self.pattern_engine = pattern_engine
        self.personalized_behavior = personalized_behavior
        self.federated_system = federated_system
        
        # Configuration
        self.config = analytics_config or self._default_config()
        
        # Analytics data storage
        self._learning_metrics_history: Dict[str, List[LearningMetrics]] = defaultdict(list)
        self._pattern_analyses_cache: Dict[str, PatternAnalysis] = {}
        self._insight_history: List[PersonalizationInsight] = []
        self._report_cache: Dict[str, LearningReport] = {}
        
        # Analysis state
        self._last_analysis_time: Dict[str, datetime] = {}
        self._trend_analysis_window_days = 30
        self._anomaly_detection_threshold = 2.0
        
        # Visualization settings
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("Learning analytics system initialized")
    
    async def generate_user_analytics(
        self,
        user_id: str,
        time_range_days: int = 30,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analytics for a specific user.
        
        Args:
            user_id: User identifier
            time_range_days: Time range for analysis
            include_predictions: Whether to include predictive analytics
            
        Returns:
            Comprehensive user analytics
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_range_days)
        
        # Get user patterns in time range
        user_patterns = self.pattern_engine._user_patterns.get(user_id, [])
        time_filtered_patterns = [
            p for p in user_patterns
            if start_date <= p.timestamp <= end_date
        ]
        
        if not time_filtered_patterns:
            return {
                'error': 'No patterns found for user in specified time range',
                'user_id': user_id,
                'time_range_days': time_range_days
            }
        
        # Calculate core metrics
        core_metrics = await self._calculate_user_metrics(user_id, time_filtered_patterns)
        
        # Analyze learning progression
        learning_progression = await self._analyze_learning_progression(user_id, time_filtered_patterns)
        
        # Pattern analysis
        pattern_analysis = await self._analyze_user_patterns(user_id, time_filtered_patterns)
        
        # Personalization effectiveness
        personalization_analysis = await self._analyze_personalization_effectiveness(user_id)
        
        # Success factor analysis
        success_factors = await self._identify_success_factors(user_id, time_filtered_patterns)
        
        # Generate insights
        insights = await self._generate_user_insights(user_id, core_metrics, pattern_analysis)
        
        # Predictive analytics
        predictions = {}
        if include_predictions:
            predictions = await self._generate_user_predictions(user_id, time_filtered_patterns)
        
        # Comparative analysis (vs other users - anonymized)
        comparative_analysis = await self._generate_comparative_analysis(user_id, core_metrics)
        
        user_analytics = {
            'user_id': user_id,
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': time_range_days
            },
            'core_metrics': core_metrics,
            'learning_progression': learning_progression,
            'pattern_analysis': pattern_analysis,
            'personalization_analysis': personalization_analysis,
            'success_factors': success_factors,
            'insights': insights,
            'predictions': predictions,
            'comparative_analysis': comparative_analysis,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return user_analytics
    
    async def generate_team_analytics(
        self,
        team_users: List[str],
        time_range_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate team-wide analytics across multiple users.
        
        Args:
            team_users: List of user IDs in the team
            time_range_days: Time range for analysis
            
        Returns:
            Team analytics report
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_range_days)
        
        # Collect team data
        team_patterns = []
        team_metrics = []
        
        for user_id in team_users:
            user_patterns = self.pattern_engine._user_patterns.get(user_id, [])
            time_filtered = [
                p for p in user_patterns
                if start_date <= p.timestamp <= end_date
            ]
            
            if time_filtered:
                team_patterns.extend(time_filtered)
                user_metrics = await self._calculate_user_metrics(user_id, time_filtered)
                team_metrics.append(user_metrics)
        
        if not team_patterns:
            return {'error': 'No team patterns found in specified time range'}
        
        # Aggregate team metrics
        aggregated_metrics = await self._aggregate_team_metrics(team_metrics)
        
        # Analyze team collaboration patterns
        collaboration_analysis = await self._analyze_team_collaboration(team_users, team_patterns)
        
        # Identify team knowledge gaps
        knowledge_gaps = await self._identify_team_knowledge_gaps(team_users, team_patterns)
        
        # Team performance trends
        performance_trends = await self._analyze_team_performance_trends(team_users, time_range_days)
        
        # Best practices within team
        team_best_practices = await self._identify_team_best_practices(team_patterns)
        
        # Federated learning impact (if available)
        federated_impact = {}
        if self.federated_system:
            federated_impact = await self._analyze_team_federated_impact(team_users)
        
        team_analytics = {
            'team_composition': {
                'user_count': len(team_users),
                'active_users': len([u for u in team_users if u in self.pattern_engine._user_patterns]),
                'analysis_period_days': time_range_days
            },
            'aggregated_metrics': aggregated_metrics,
            'collaboration_analysis': collaboration_analysis,
            'knowledge_gaps': knowledge_gaps,
            'performance_trends': performance_trends,
            'team_best_practices': team_best_practices,
            'federated_impact': federated_impact,
            'recommendations': await self._generate_team_recommendations(
                team_users, aggregated_metrics, knowledge_gaps
            ),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return team_analytics
    
    async def generate_learning_insights(
        self,
        scope: str = 'all',  # 'user', 'team', 'organization', 'all'
        focus_areas: Optional[List[str]] = None
    ) -> List[PersonalizationInsight]:
        """
        Generate actionable learning insights across different scopes.
        
        Args:
            scope: Scope of analysis
            focus_areas: Specific areas to focus on
            
        Returns:
            List of generated insights
        """
        insights = []
        focus_areas = focus_areas or ['patterns', 'personalization', 'collaboration', 'trends']
        
        # Pattern-based insights
        if 'patterns' in focus_areas:
            pattern_insights = await self._generate_pattern_insights(scope)
            insights.extend(pattern_insights)
        
        # Personalization effectiveness insights
        if 'personalization' in focus_areas:
            personalization_insights = await self._generate_personalization_insights(scope)
            insights.extend(personalization_insights)
        
        # Collaboration insights (if federated system available)
        if 'collaboration' in focus_areas and self.federated_system:
            collaboration_insights = await self._generate_collaboration_insights()
            insights.extend(collaboration_insights)
        
        # Trend analysis insights
        if 'trends' in focus_areas:
            trend_insights = await self._generate_trend_insights(scope)
            insights.extend(trend_insights)
        
        # System performance insights
        if 'performance' in focus_areas:
            performance_insights = await self._generate_performance_insights()
            insights.extend(performance_insights)
        
        # Sort by impact score and confidence
        insights.sort(key=lambda x: x.impact_score * x.confidence, reverse=True)
        
        # Store insights in history
        self._insight_history.extend(insights)
        
        # Keep history manageable
        if len(self._insight_history) > 1000:
            self._insight_history = self._insight_history[-500:]
        
        logger.info(f"Generated {len(insights)} learning insights for scope: {scope}")
        return insights
    
    async def create_learning_dashboard(
        self,
        user_id: Optional[str] = None,
        team_users: Optional[List[str]] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive learning dashboard with visualizations.
        
        Args:
            user_id: Specific user for individual dashboard
            team_users: Team users for team dashboard
            output_path: Path to save dashboard files
            
        Returns:
            Dashboard data and visualization paths
        """
        dashboard_data = {}
        visualization_paths = []
        
        if output_path is None:
            output_path = Path("dashboards") / f"dashboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        if user_id:
            # Individual user dashboard
            user_analytics = await self.generate_user_analytics(user_id)
            dashboard_data['user_analytics'] = user_analytics
            
            # Create user visualizations
            user_viz_paths = await self._create_user_visualizations(
                user_id, user_analytics, output_path
            )
            visualization_paths.extend(user_viz_paths)
        
        elif team_users:
            # Team dashboard
            team_analytics = await self.generate_team_analytics(team_users)
            dashboard_data['team_analytics'] = team_analytics
            
            # Create team visualizations
            team_viz_paths = await self._create_team_visualizations(
                team_users, team_analytics, output_path
            )
            visualization_paths.extend(team_viz_paths)
        
        else:
            # System-wide dashboard
            system_analytics = await self._generate_system_analytics()
            dashboard_data['system_analytics'] = system_analytics
            
            # Create system visualizations
            system_viz_paths = await self._create_system_visualizations(
                system_analytics, output_path
            )
            visualization_paths.extend(system_viz_paths)
        
        # Generate insights for dashboard
        insights = await self.generate_learning_insights(
            scope='user' if user_id else 'team' if team_users else 'all'
        )
        dashboard_data['insights'] = [
            {
                'title': insight.title,
                'description': insight.description,
                'impact_score': insight.impact_score,
                'confidence': insight.confidence,
                'actionable_steps': insight.actionable_steps
            }
            for insight in insights[:10]  # Top 10 insights
        ]
        
        # Create summary HTML dashboard
        html_path = await self._create_html_dashboard(
            dashboard_data, visualization_paths, output_path
        )
        
        dashboard_result = {
            'dashboard_data': dashboard_data,
            'visualization_paths': visualization_paths,
            'html_dashboard_path': str(html_path),
            'generated_at': datetime.utcnow().isoformat(),
            'dashboard_type': 'user' if user_id else 'team' if team_users else 'system'
        }
        
        logger.info(f"Created learning dashboard at {output_path}")
        return dashboard_result
    
    async def _calculate_user_metrics(
        self,
        user_id: str,
        patterns: List[UserInteractionPattern]
    ) -> LearningMetrics:
        """Calculate core learning metrics for user."""
        if not patterns:
            return LearningMetrics(
                user_id=user_id,
                time_period="",
                total_interactions=0,
                successful_interactions=0,
                success_rate=0.0,
                average_confidence=0.0,
                learning_velocity=0.0,
                pattern_stability=0.0,
                personalization_effectiveness=0.0,
                federated_contributions=0,
                insights_received=0,
                recommendation_adoption_rate=0.0
            )
        
        # Basic metrics
        total_interactions = len(patterns)
        successful_patterns = [
            p for p in patterns
            if p.pattern_type in ['success', 'validation_success'] and p.success_rate >= 0.7
        ]
        successful_interactions = len(successful_patterns)
        success_rate = successful_interactions / total_interactions if total_interactions > 0 else 0.0
        
        # Confidence metrics
        average_confidence = np.mean([p.confidence for p in patterns])
        
        # Learning velocity (improvement over time)
        learning_velocity = self._calculate_learning_velocity(patterns)
        
        # Pattern stability (consistency of patterns)
        pattern_stability = self._calculate_pattern_stability(patterns)
        
        # Personalization effectiveness
        personalization_effectiveness = await self._calculate_personalization_effectiveness(user_id)
        
        # Federated learning metrics
        federated_contributions = 0
        insights_received = 0
        if self.federated_system:
            federation_analytics = await self.federated_system.coordinator.get_federation_analytics()
            federated_contributions = federation_analytics['contribution_metrics']['patterns_contributed']
            insights_received = federation_analytics['contribution_metrics']['insights_generated']
        
        # Recommendation adoption rate
        recommendation_adoption_rate = await self._calculate_recommendation_adoption_rate(user_id)
        
        return LearningMetrics(
            user_id=user_id,
            time_period=f"{patterns[0].timestamp.date()} to {patterns[-1].timestamp.date()}",
            total_interactions=total_interactions,
            successful_interactions=successful_interactions,
            success_rate=success_rate,
            average_confidence=average_confidence,
            learning_velocity=learning_velocity,
            pattern_stability=pattern_stability,
            personalization_effectiveness=personalization_effectiveness,
            federated_contributions=federated_contributions,
            insights_received=insights_received,
            recommendation_adoption_rate=recommendation_adoption_rate
        )
    
    def _calculate_learning_velocity(self, patterns: List[UserInteractionPattern]) -> float:
        """Calculate learning velocity from pattern progression."""
        if len(patterns) < 5:
            return 0.0
        
        # Sort patterns by timestamp
        sorted_patterns = sorted(patterns, key=lambda x: x.timestamp)
        
        # Split into early and late periods
        midpoint = len(sorted_patterns) // 2
        early_patterns = sorted_patterns[:midpoint]
        late_patterns = sorted_patterns[midpoint:]
        
        # Calculate average success rates
        early_success = np.mean([p.success_rate for p in early_patterns])
        late_success = np.mean([p.success_rate for p in late_patterns])
        
        # Learning velocity is improvement rate
        improvement = late_success - early_success
        
        # Normalize to 0-1 scale
        return max(0.0, min(improvement * 2, 1.0))  # Scale so 0.5 improvement = 1.0 velocity
    
    def _calculate_pattern_stability(self, patterns: List[UserInteractionPattern]) -> float:
        """Calculate pattern stability (consistency)."""
        if len(patterns) < 3:
            return 0.0
        
        # Calculate variance in success rates
        success_rates = [p.success_rate for p in patterns]
        success_variance = np.var(success_rates)
        
        # Calculate variance in confidence scores
        confidences = [p.confidence for p in patterns]
        confidence_variance = np.var(confidences)
        
        # Stability is inverse of variance (higher variance = lower stability)
        stability = 1.0 - (success_variance + confidence_variance) / 2
        
        return max(0.0, min(stability, 1.0))
    
    async def _calculate_personalization_effectiveness(self, user_id: str) -> float:
        """Calculate effectiveness of personalization for user."""
        try:
            effectiveness_metrics = await self.personalized_behavior.measure_personalization_effectiveness(
                user_id, time_window_days=30
            )
            
            if 'improvement_from_personalization' in effectiveness_metrics:
                improvement = effectiveness_metrics['improvement_from_personalization']
                return max(0.0, min(improvement, 1.0))
            
            return 0.5  # Default if no data available
        
        except Exception:
            return 0.5
    
    async def _calculate_recommendation_adoption_rate(self, user_id: str) -> float:
        """Calculate rate of recommendation adoption."""
        # This would track how often user follows recommendations
        # For now, return a placeholder based on user pattern success
        user_patterns = self.pattern_engine._user_patterns.get(user_id, [])
        
        if not user_patterns:
            return 0.0
        
        # Simple heuristic: users with improving patterns likely adopt recommendations
        recent_patterns = [
            p for p in user_patterns
            if (datetime.utcnow() - p.timestamp).days <= 14
        ]
        
        if not recent_patterns:
            return 0.5
        
        avg_recent_success = np.mean([p.success_rate for p in recent_patterns])
        return min(avg_recent_success, 1.0)
    
    async def _analyze_learning_progression(
        self,
        user_id: str,
        patterns: List[UserInteractionPattern]
    ) -> Dict[str, Any]:
        """Analyze user's learning progression over time."""
        if len(patterns) < 3:
            return {'error': 'Insufficient data for progression analysis'}
        
        # Sort patterns by timestamp
        sorted_patterns = sorted(patterns, key=lambda x: x.timestamp)
        
        # Create time series data
        timestamps = [p.timestamp for p in sorted_patterns]
        success_rates = [p.success_rate for p in sorted_patterns]
        confidence_scores = [p.confidence for p in sorted_patterns]
        
        # Calculate trends
        success_trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
        confidence_trend = np.polyfit(range(len(confidence_scores)), confidence_scores, 1)[0]
        
        # Identify learning phases
        learning_phases = self._identify_learning_phases(sorted_patterns)
        
        # Calculate learning rate (how quickly user improves)
        learning_rate = self._calculate_learning_rate(sorted_patterns)
        
        progression_analysis = {
            'total_time_span_days': (timestamps[-1] - timestamps[0]).days,
            'pattern_frequency': len(patterns) / max((timestamps[-1] - timestamps[0]).days, 1),
            'success_trend': {
                'direction': 'improving' if success_trend > 0.01 else 'declining' if success_trend < -0.01 else 'stable',
                'slope': success_trend,
                'current_success_rate': success_rates[-1],
                'initial_success_rate': success_rates[0],
                'improvement': success_rates[-1] - success_rates[0]
            },
            'confidence_trend': {
                'direction': 'improving' if confidence_trend > 0.01 else 'declining' if confidence_trend < -0.01 else 'stable',
                'slope': confidence_trend,
                'current_confidence': confidence_scores[-1],
                'initial_confidence': confidence_scores[0]
            },
            'learning_phases': learning_phases,
            'learning_rate': learning_rate,
            'consistency_score': self._calculate_pattern_stability(patterns)
        }
        
        return progression_analysis
    
    def _identify_learning_phases(
        self,
        sorted_patterns: List[UserInteractionPattern]
    ) -> List[Dict[str, Any]]:
        """Identify distinct learning phases."""
        if len(sorted_patterns) < 10:
            return [{'phase': 'initial', 'pattern_count': len(sorted_patterns)}]
        
        phases = []
        
        # Split into segments and analyze each
        segment_size = max(len(sorted_patterns) // 5, 3)  # At least 5 segments, min 3 patterns each
        
        for i in range(0, len(sorted_patterns), segment_size):
            segment = sorted_patterns[i:i + segment_size]
            if len(segment) < 2:
                continue
            
            avg_success = np.mean([p.success_rate for p in segment])
            avg_confidence = np.mean([p.confidence for p in segment])
            
            # Classify phase
            if avg_success >= 0.8 and avg_confidence >= 0.8:
                phase_type = 'mastery'
            elif avg_success >= 0.6:
                phase_type = 'competent'
            elif avg_success >= 0.4:
                phase_type = 'developing'
            else:
                phase_type = 'struggling'
            
            phases.append({
                'phase': phase_type,
                'start_date': segment[0].timestamp.isoformat(),
                'end_date': segment[-1].timestamp.isoformat(),
                'pattern_count': len(segment),
                'avg_success_rate': avg_success,
                'avg_confidence': avg_confidence
            })
        
        return phases
    
    def _calculate_learning_rate(
        self,
        sorted_patterns: List[UserInteractionPattern]
    ) -> float:
        """Calculate learning rate (improvement speed)."""
        if len(sorted_patterns) < 5:
            return 0.0
        
        # Calculate rolling average improvements
        window_size = 5
        improvements = []
        
        for i in range(window_size, len(sorted_patterns)):
            current_window = sorted_patterns[i-window_size:i]
            previous_window = sorted_patterns[i-window_size*2:i-window_size] if i >= window_size*2 else []
            
            if previous_window:
                current_avg = np.mean([p.success_rate for p in current_window])
                previous_avg = np.mean([p.success_rate for p in previous_window])
                improvement = current_avg - previous_avg
                improvements.append(improvement)
        
        if improvements:
            # Learning rate is average improvement per window
            return np.mean(improvements)
        
        return 0.0
    
    async def _analyze_user_patterns(
        self,
        user_id: str,
        patterns: List[UserInteractionPattern]
    ) -> PatternAnalysis:
        """Analyze user patterns in detail."""
        if not patterns:
            return PatternAnalysis(
                pattern_type="",
                frequency_distribution={},
                success_rate_by_context={},
                temporal_trends={},
                correlation_matrix={},
                outlier_patterns=[],
                emerging_patterns=[],
                declining_patterns=[]
            )
        
        # Pattern type distribution
        pattern_types = [p.pattern_type for p in patterns]
        frequency_distribution = dict(Counter(pattern_types))
        
        # Success rate by context
        success_by_context = {}
        context_patterns = defaultdict(list)
        
        for pattern in patterns:
            context = pattern.features.get('task_type', 'unknown')
            context_patterns[context].append(pattern.success_rate)
        
        for context, success_rates in context_patterns.items():
            success_by_context[context] = np.mean(success_rates)
        
        # Temporal trends
        temporal_trends = await self._analyze_temporal_trends(patterns)
        
        # Feature correlation analysis
        correlation_matrix = await self._calculate_feature_correlations(patterns)
        
        # Identify outlier patterns
        outlier_patterns = self._identify_outlier_patterns(patterns)
        
        # Emerging and declining patterns
        emerging_patterns, declining_patterns = self._identify_pattern_trends(patterns)
        
        return PatternAnalysis(
            pattern_type='user_analysis',
            frequency_distribution=frequency_distribution,
            success_rate_by_context=success_by_context,
            temporal_trends=temporal_trends,
            correlation_matrix=correlation_matrix,
            outlier_patterns=outlier_patterns,
            emerging_patterns=emerging_patterns,
            declining_patterns=declining_patterns
        )
    
    async def _analyze_temporal_trends(
        self,
        patterns: List[UserInteractionPattern]
    ) -> Dict[str, float]:
        """Analyze temporal trends in patterns."""
        if len(patterns) < 5:
            return {}
        
        # Group by time periods
        hourly_success = defaultdict(list)
        daily_success = defaultdict(list)
        
        for pattern in patterns:
            hour = pattern.timestamp.hour
            day = pattern.timestamp.weekday()
            
            hourly_success[hour].append(pattern.success_rate)
            daily_success[day].append(pattern.success_rate)
        
        # Calculate average success by time period
        hourly_trends = {
            f"hour_{hour}": np.mean(success_rates)
            for hour, success_rates in hourly_success.items()
            if len(success_rates) >= 2
        }
        
        daily_trends = {
            f"day_{day}": np.mean(success_rates)
            for day, success_rates in daily_success.items()
            if len(success_rates) >= 2
        }
        
        return {**hourly_trends, **daily_trends}
    
    async def _calculate_feature_correlations(
        self,
        patterns: List[UserInteractionPattern]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between features and success."""
        if len(patterns) < 10:
            return {}
        
        # Extract numeric features
        numeric_features = defaultdict(list)
        success_rates = []
        
        for pattern in patterns:
            success_rates.append(pattern.success_rate)
            
            for feature_name, feature_value in pattern.features.items():
                if isinstance(feature_value, (int, float)):
                    numeric_features[feature_name].append(feature_value)
        
        # Calculate correlations
        correlations = {}
        
        for feature_name, feature_values in numeric_features.items():
            if len(feature_values) == len(success_rates):
                try:
                    correlation = np.corrcoef(feature_values, success_rates)[0, 1]
                    if not np.isnan(correlation):
                        correlations[feature_name] = {
                            'success_correlation': correlation,
                            'strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
                        }
                except Exception:
                    continue
        
        return correlations
    
    def _identify_outlier_patterns(
        self,
        patterns: List[UserInteractionPattern]
    ) -> List[str]:
        """Identify outlier patterns using statistical methods."""
        if len(patterns) < 10:
            return []
        
        success_rates = [p.success_rate for p in patterns]
        confidence_scores = [p.confidence for p in patterns]
        
        # Calculate z-scores
        success_mean = np.mean(success_rates)
        success_std = np.std(success_rates)
        
        confidence_mean = np.mean(confidence_scores)
        confidence_std = np.std(confidence_scores)
        
        outliers = []
        
        for i, pattern in enumerate(patterns):
            success_z = abs((pattern.success_rate - success_mean) / success_std) if success_std > 0 else 0
            confidence_z = abs((pattern.confidence - confidence_mean) / confidence_std) if confidence_std > 0 else 0
            
            # Pattern is outlier if either metric has high z-score
            if success_z > self._anomaly_detection_threshold or confidence_z > self._anomaly_detection_threshold:
                outliers.append(pattern.pattern_id)
        
        return outliers
    
    def _identify_pattern_trends(
        self,
        patterns: List[UserInteractionPattern]
    ) -> Tuple[List[str], List[str]]:
        """Identify emerging and declining patterns."""
        if len(patterns) < 10:
            return [], []
        
        # Sort patterns by timestamp
        sorted_patterns = sorted(patterns, key=lambda x: x.timestamp)
        
        # Split into early and late periods
        split_point = len(sorted_patterns) // 2
        early_patterns = sorted_patterns[:split_point]
        late_patterns = sorted_patterns[split_point:]
        
        # Count pattern types in each period
        early_types = Counter(p.pattern_type for p in early_patterns)
        late_types = Counter(p.pattern_type for p in late_patterns)
        
        # Identify trends
        emerging_patterns = []
        declining_patterns = []
        
        all_types = set(early_types.keys()) | set(late_types.keys())
        
        for pattern_type in all_types:
            early_count = early_types.get(pattern_type, 0)
            late_count = late_types.get(pattern_type, 0)
            
            # Calculate relative change
            if early_count > 0:
                change_ratio = (late_count - early_count) / early_count
                
                if change_ratio > 0.5:  # 50% increase
                    emerging_patterns.append(pattern_type)
                elif change_ratio < -0.5:  # 50% decrease
                    declining_patterns.append(pattern_type)
            elif late_count > 0:
                # New pattern type emerged
                emerging_patterns.append(pattern_type)
        
        return emerging_patterns, declining_patterns
    
    async def _analyze_personalization_effectiveness(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Analyze effectiveness of personalization for user."""
        try:
            effectiveness_metrics = await self.personalized_behavior.measure_personalization_effectiveness(
                user_id, time_window_days=30
            )
            
            return {
                'overall_effectiveness': effectiveness_metrics.get('effectiveness_score', 0.5),
                'personalization_rate': effectiveness_metrics.get('personalization_rate', 0.0),
                'improvement_from_personalization': effectiveness_metrics.get('improvement_from_personalization', 0.0),
                'template_effectiveness': effectiveness_metrics.get('template_effectiveness', 0.5),
                'adaptation_rule_effectiveness': effectiveness_metrics.get('adaptation_rule_effectiveness', 0.5),
                'profile_maturity_days': effectiveness_metrics.get('profile_age_days', 0)
            }
        
        except Exception as e:
            logger.warning(f"Error analyzing personalization effectiveness for {user_id}: {e}")
            return {
                'overall_effectiveness': 0.5,
                'error': 'Unable to analyze personalization effectiveness'
            }
    
    async def _identify_success_factors(
        self,
        user_id: str,
        patterns: List[UserInteractionPattern]
    ) -> Dict[str, Any]:
        """Identify factors that contribute to user success."""
        if len(patterns) < 5:
            return {}
        
        # Separate successful and unsuccessful patterns
        successful_patterns = [p for p in patterns if p.success_rate >= 0.7]
        unsuccessful_patterns = [p for p in patterns if p.success_rate < 0.4]
        
        if not successful_patterns:
            return {'message': 'No highly successful patterns found'}
        
        success_factors = {}
        
        # Analyze feature differences
        successful_features = defaultdict(list)
        unsuccessful_features = defaultdict(list)
        
        for pattern in successful_patterns:
            for feature, value in pattern.features.items():
                if isinstance(value, (int, float)):
                    successful_features[feature].append(value)
        
        for pattern in unsuccessful_patterns:
            for feature, value in pattern.features.items():
                if isinstance(value, (int, float)):
                    unsuccessful_features[feature].append(value)
        
        # Find features with significant differences
        significant_factors = {}
        
        for feature in successful_features:
            if feature in unsuccessful_features:
                success_mean = np.mean(successful_features[feature])
                fail_mean = np.mean(unsuccessful_features[feature])
                
                if abs(success_mean - fail_mean) > 0.1:  # Significant difference threshold
                    significant_factors[feature] = {
                        'success_average': success_mean,
                        'failure_average': fail_mean,
                        'difference': success_mean - fail_mean,
                        'impact': 'positive' if success_mean > fail_mean else 'negative'
                    }
        
        # Identify contextual success factors
        contextual_factors = {}
        
        # Task types with highest success rates
        task_success = defaultdict(list)
        for pattern in patterns:
            task_type = pattern.features.get('task_type', 'unknown')
            task_success[task_type].append(pattern.success_rate)
        
        for task_type, success_rates in task_success.items():
            if len(success_rates) >= 3:  # Minimum sample size
                avg_success = np.mean(success_rates)
                if avg_success >= 0.7:
                    contextual_factors[f'task_type_{task_type}'] = {
                        'average_success_rate': avg_success,
                        'pattern_count': len(success_rates)
                    }
        
        return {
            'significant_factors': significant_factors,
            'contextual_factors': contextual_factors,
            'success_pattern_count': len(successful_patterns),
            'total_pattern_count': len(patterns),
            'overall_success_rate': len(successful_patterns) / len(patterns)
        }
    
    async def _generate_user_insights(
        self,
        user_id: str,
        metrics: LearningMetrics,
        pattern_analysis: PatternAnalysis
    ) -> List[PersonalizationInsight]:
        """Generate actionable insights for user."""
        insights = []
        
        # Learning velocity insights
        if metrics.learning_velocity > 0.7:
            insights.append(PersonalizationInsight(
                insight_id=f"{user_id}_velocity_high",
                user_id=user_id,
                insight_type='improvement',
                title='Excellent Learning Progress',
                description=f'Your learning velocity is {metrics.learning_velocity:.2f}, showing rapid skill development.',
                impact_score=0.8,
                confidence=0.9,
                actionable_steps=[
                    'Continue current learning practices',
                    'Consider taking on more challenging tasks',
                    'Share successful strategies with team'
                ],
                supporting_data={'learning_velocity': metrics.learning_velocity},
                generated_at=datetime.utcnow()
            ))
        elif metrics.learning_velocity < 0.3:
            insights.append(PersonalizationInsight(
                insight_id=f"{user_id}_velocity_low",
                user_id=user_id,
                insight_type='alert',
                title='Learning Progress Could Improve',
                description=f'Learning velocity is {metrics.learning_velocity:.2f}. Consider adjusting approach.',
                impact_score=0.7,
                confidence=0.8,
                actionable_steps=[
                    'Review recent patterns for improvement opportunities',
                    'Seek feedback on current work',
                    'Try different learning strategies'
                ],
                supporting_data={'learning_velocity': metrics.learning_velocity},
                generated_at=datetime.utcnow()
            ))
        
        # Success rate insights
        if metrics.success_rate < 0.5:
            insights.append(PersonalizationInsight(
                insight_id=f"{user_id}_success_low",
                user_id=user_id,
                insight_type='alert',
                title='Success Rate Below Target',
                description=f'Current success rate is {metrics.success_rate:.1%}. Focus on pattern improvement.',
                impact_score=0.9,
                confidence=0.85,
                actionable_steps=[
                    'Analyze patterns with high success rates',
                    'Focus on contexts where you perform well',
                    'Consider additional training or resources'
                ],
                supporting_data={'success_rate': metrics.success_rate},
                generated_at=datetime.utcnow()
            ))
        
        # Pattern stability insights
        if metrics.pattern_stability < 0.4:
            insights.append(PersonalizationInsight(
                insight_id=f"{user_id}_stability_low",
                user_id=user_id,
                insight_type='recommendation',
                title='Inconsistent Performance Patterns',
                description='Your performance varies significantly. Consider standardizing your approach.',
                impact_score=0.6,
                confidence=0.7,
                actionable_steps=[
                    'Identify factors contributing to variability',
                    'Develop consistent workflows',
                    'Track what works best for you'
                ],
                supporting_data={'pattern_stability': metrics.pattern_stability},
                generated_at=datetime.utcnow()
            ))
        
        # Personalization effectiveness insights
        if metrics.personalization_effectiveness > 0.7:
            insights.append(PersonalizationInsight(
                insight_id=f"{user_id}_personalization_effective",
                user_id=user_id,
                insight_type='improvement',
                title='Personalization Working Well',
                description='AI personalization is significantly improving your outcomes.',
                impact_score=0.7,
                confidence=0.8,
                actionable_steps=[
                    'Continue providing feedback to improve personalization',
                    'Explore advanced personalization features',
                    'Consider sharing successful patterns'
                ],
                supporting_data={'personalization_effectiveness': metrics.personalization_effectiveness},
                generated_at=datetime.utcnow()
            ))
        
        return insights
    
    async def _generate_user_predictions(
        self,
        user_id: str,
        patterns: List[UserInteractionPattern]
    ) -> Dict[str, Any]:
        """Generate predictive analytics for user."""
        if len(patterns) < 10:
            return {'error': 'Insufficient data for predictions'}
        
        # Sort patterns by timestamp
        sorted_patterns = sorted(patterns, key=lambda x: x.timestamp)
        
        # Predict future success rate based on trend
        success_rates = [p.success_rate for p in sorted_patterns]
        time_indices = list(range(len(success_rates)))
        
        try:
            # Linear trend prediction
            coeffs = np.polyfit(time_indices, success_rates, 1)
            trend_slope = coeffs[0]
            
            # Predict next 5 interactions
            future_predictions = []
            for i in range(1, 6):
                predicted_success = coeffs[1] + coeffs[0] * (len(success_rates) + i)
                predicted_success = max(0.0, min(1.0, predicted_success))  # Clamp to [0,1]
                future_predictions.append(predicted_success)
            
            # Confidence in prediction based on trend consistency
            recent_trend_consistency = self._calculate_trend_consistency(success_rates[-10:])
            
            predictions = {
                'future_success_rates': future_predictions,
                'trend_direction': 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable',
                'trend_strength': abs(trend_slope),
                'prediction_confidence': recent_trend_consistency,
                'expected_performance': {
                    'next_week': np.mean(future_predictions[:2]),
                    'next_month': np.mean(future_predictions),
                },
                'risk_factors': self._identify_risk_factors(sorted_patterns),
                'opportunities': self._identify_opportunities(sorted_patterns)
            }
            
            return predictions
        
        except Exception as e:
            logger.warning(f"Error generating predictions for {user_id}: {e}")
            return {'error': 'Unable to generate predictions'}
    
    def _calculate_trend_consistency(self, values: List[float]) -> float:
        """Calculate consistency of trend in values."""
        if len(values) < 5:
            return 0.5
        
        # Calculate local trends in sliding windows
        window_size = 3
        local_trends = []
        
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            trend = np.polyfit(range(window_size), window, 1)[0]
            local_trends.append(trend)
        
        # Consistency is inverse of variance in local trends
        if len(local_trends) > 1:
            trend_variance = np.var(local_trends)
            consistency = 1.0 / (1.0 + trend_variance)
            return min(consistency, 1.0)
        
        return 0.5
    
    def _identify_risk_factors(
        self,
        sorted_patterns: List[UserInteractionPattern]
    ) -> List[Dict[str, Any]]:
        """Identify potential risk factors for future performance."""
        risk_factors = []
        
        # Recent declining performance
        if len(sorted_patterns) >= 5:
            recent_patterns = sorted_patterns[-5:]
            recent_success = [p.success_rate for p in recent_patterns]
            
            if len(recent_success) >= 3:
                recent_trend = np.polyfit(range(len(recent_success)), recent_success, 1)[0]
                if recent_trend < -0.1:  # Declining trend
                    risk_factors.append({
                        'factor': 'declining_performance',
                        'description': 'Recent performance shows declining trend',
                        'severity': 'medium' if recent_trend > -0.2 else 'high',
                        'recommendation': 'Review recent work patterns and seek feedback'
                    })
        
        # Low pattern stability
        if len(sorted_patterns) >= 10:
            stability = self._calculate_pattern_stability(sorted_patterns[-10:])
            if stability < 0.4:
                risk_factors.append({
                    'factor': 'inconsistent_performance',
                    'description': 'Performance varies significantly between tasks',
                    'severity': 'medium',
                    'recommendation': 'Focus on developing consistent work processes'
                })
        
        # Lack of variety in successful patterns
        successful_types = set(
            p.pattern_type for p in sorted_patterns
            if p.success_rate >= 0.7
        )
        
        if len(successful_types) < 2:
            risk_factors.append({
                'factor': 'limited_success_patterns',
                'description': 'Success limited to narrow range of task types',
                'severity': 'low',
                'recommendation': 'Experiment with different approaches and task types'
            })
        
        return risk_factors
    
    def _identify_opportunities(
        self,
        sorted_patterns: List[UserInteractionPattern]
    ) -> List[Dict[str, Any]]:
        """Identify opportunities for improvement."""
        opportunities = []
        
        # High confidence but low success rate patterns
        high_confidence_low_success = [
            p for p in sorted_patterns
            if p.confidence >= 0.8 and p.success_rate < 0.6
        ]
        
        if high_confidence_low_success:
            opportunities.append({
                'opportunity': 'confidence_calibration',
                'description': 'High confidence with lower success suggests calibration opportunity',
                'impact': 'medium',
                'recommendation': 'Review assumptions and validation criteria'
            })
        
        # Improving trend with room for growth
        if len(sorted_patterns) >= 5:
            recent_patterns = sorted_patterns[-5:]
            recent_trend = np.polyfit(
                range(len(recent_patterns)),
                [p.success_rate for p in recent_patterns],
                1
            )[0]
            
            current_success = recent_patterns[-1].success_rate
            
            if recent_trend > 0.05 and current_success < 0.9:
                opportunities.append({
                    'opportunity': 'momentum_building',
                    'description': 'Positive trend with potential for continued growth',
                    'impact': 'high',
                    'recommendation': 'Maintain current approach and consider stretch goals'
                })
        
        # Underutilized personalization features
        profile = self.personalized_behavior._user_profiles.get(sorted_patterns[0].user_id)
        if profile and profile.feedback_responsiveness > 0.8:
            opportunities.append({
                'opportunity': 'advanced_personalization',
                'description': 'High responsiveness to feedback suggests readiness for advanced features',
                'impact': 'medium',
                'recommendation': 'Explore advanced personalization and collaboration features'
            })
        
        return opportunities
    
    async def _generate_comparative_analysis(
        self,
        user_id: str,
        user_metrics: LearningMetrics
    ) -> Dict[str, Any]:
        """Generate comparative analysis (anonymized)."""
        # Calculate aggregate metrics across all users
        all_users_metrics = []
        
        for other_user_id in self.pattern_engine._user_patterns:
            if other_user_id == user_id:
                continue
            
            other_patterns = self.pattern_engine._user_patterns[other_user_id]
            if len(other_patterns) >= 5:  # Minimum for comparison
                other_metrics = await self._calculate_user_metrics(other_user_id, other_patterns)
                all_users_metrics.append(other_metrics)
        
        if not all_users_metrics:
            return {'message': 'No comparative data available'}
        
        # Calculate percentiles
        success_rates = [m.success_rate for m in all_users_metrics]
        learning_velocities = [m.learning_velocity for m in all_users_metrics]
        pattern_stabilities = [m.pattern_stability for m in all_users_metrics]
        
        def calculate_percentile(value: float, distribution: List[float]) -> float:
            return np.mean([v <= value for v in distribution]) * 100
        
        comparative_analysis = {
            'user_percentiles': {
                'success_rate': calculate_percentile(user_metrics.success_rate, success_rates),
                'learning_velocity': calculate_percentile(user_metrics.learning_velocity, learning_velocities),
                'pattern_stability': calculate_percentile(user_metrics.pattern_stability, pattern_stabilities)
            },
            'peer_group_averages': {
                'success_rate': np.mean(success_rates),
                'learning_velocity': np.mean(learning_velocities),
                'pattern_stability': np.mean(pattern_stabilities)
            },
            'relative_performance': {
                'success_rate': 'above_average' if user_metrics.success_rate > np.mean(success_rates) else 'below_average',
                'learning_velocity': 'above_average' if user_metrics.learning_velocity > np.mean(learning_velocities) else 'below_average',
                'pattern_stability': 'above_average' if user_metrics.pattern_stability > np.mean(pattern_stabilities) else 'below_average'
            },
            'comparison_cohort_size': len(all_users_metrics)
        }
        
        return comparative_analysis
    
    def _default_config(self) -> Dict[str, Any]:
        """Default analytics configuration."""
        return {
            'trend_analysis_window_days': 30,
            'anomaly_detection_threshold': 2.0,
            'min_patterns_for_analysis': 5,
            'confidence_threshold': 0.7,
            'visualization_style': 'seaborn',
            'report_cache_duration_hours': 24,
            'enable_predictive_analytics': True,
            'enable_comparative_analysis': True
        }
    
    async def _generate_pattern_insights(self, scope: str) -> List[PersonalizationInsight]:
        """Generate insights based on pattern analysis."""
        insights = []
        # Implementation would analyze patterns across specified scope
        # For now, return placeholder
        return insights
    
    async def _generate_personalization_insights(self, scope: str) -> List[PersonalizationInsight]:
        """Generate insights about personalization effectiveness.""" 
        insights = []
        # Implementation would analyze personalization effectiveness
        # For now, return placeholder
        return insights
    
    async def _generate_collaboration_insights(self) -> List[PersonalizationInsight]:
        """Generate insights about federated learning collaboration."""
        insights = []
        # Implementation would analyze federated learning patterns
        # For now, return placeholder
        return insights
    
    async def _generate_trend_insights(self, scope: str) -> List[PersonalizationInsight]:
        """Generate insights about trends and patterns."""
        insights = []
        # Implementation would analyze temporal trends
        # For now, return placeholder
        return insights
    
    async def _generate_performance_insights(self) -> List[PersonalizationInsight]:
        """Generate insights about system performance."""
        insights = []
        # Implementation would analyze system performance metrics
        # For now, return placeholder  
        return insights
    
    async def _aggregate_team_metrics(self, team_metrics: List[LearningMetrics]) -> Dict[str, Any]:
        """Aggregate metrics across team members."""
        if not team_metrics:
            return {}
        
        return {
            'team_size': len(team_metrics),
            'average_success_rate': np.mean([m.success_rate for m in team_metrics]),
            'average_learning_velocity': np.mean([m.learning_velocity for m in team_metrics]),
            'average_pattern_stability': np.mean([m.pattern_stability for m in team_metrics]),
            'total_interactions': sum(m.total_interactions for m in team_metrics),
            'total_successful_interactions': sum(m.successful_interactions for m in team_metrics),
            'team_success_rate': sum(m.successful_interactions for m in team_metrics) / sum(m.total_interactions for m in team_metrics) if sum(m.total_interactions for m in team_metrics) > 0 else 0.0
        }
    
    async def _analyze_team_collaboration(self, team_users: List[str], team_patterns: List[UserInteractionPattern]) -> Dict[str, Any]:
        """Analyze collaboration patterns within team."""
        # Placeholder implementation
        return {'collaboration_score': 0.7, 'knowledge_sharing_index': 0.6}
    
    async def _identify_team_knowledge_gaps(self, team_users: List[str], team_patterns: List[UserInteractionPattern]) -> List[Dict[str, Any]]:
        """Identify knowledge gaps within team.""" 
        # Placeholder implementation
        return [{'area': 'advanced_algorithms', 'gap_score': 0.3}]
    
    async def _analyze_team_performance_trends(self, team_users: List[str], time_range_days: int) -> Dict[str, Any]:
        """Analyze team performance trends over time."""
        # Placeholder implementation
        return {'trend': 'improving', 'velocity': 0.05}
    
    async def _identify_team_best_practices(self, team_patterns: List[UserInteractionPattern]) -> List[Dict[str, Any]]:
        """Identify best practices from team patterns."""
        # Placeholder implementation  
        return [{'practice': 'code_review', 'effectiveness': 0.85}]
    
    async def _analyze_team_federated_impact(self, team_users: List[str]) -> Dict[str, Any]:
        """Analyze federated learning impact on team."""
        # Placeholder implementation
        return {'federated_improvement': 0.15, 'knowledge_diversity': 0.8}
    
    async def _generate_team_recommendations(self, team_users: List[str], metrics: Dict[str, Any], gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate team-specific recommendations.""" 
        # Placeholder implementation
        return [{'recommendation': 'Increase knowledge sharing sessions', 'impact': 'high'}]
    
    async def _generate_system_analytics(self) -> Dict[str, Any]:
        """Generate system-wide analytics."""
        # Placeholder implementation
        return {'system_health': 0.9, 'total_users': 100, 'total_patterns': 5000}
    
    async def _create_user_visualizations(self, user_id: str, analytics: Dict[str, Any], output_path: Path) -> List[str]:
        """Create visualizations for user analytics."""
        viz_paths = []
        
        # Success rate trend chart
        try:
            plt.figure(figsize=(10, 6))
            # Placeholder data - in real implementation would use actual pattern data
            dates = pd.date_range('2024-01-01', periods=30, freq='D')
            success_rates = np.random.normal(0.7, 0.1, 30)
            
            plt.plot(dates, success_rates, marker='o')
            plt.title(f'Success Rate Trend - User {user_id}')
            plt.xlabel('Date')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            
            viz_path = output_path / f'user_{user_id}_success_trend.png'
            plt.savefig(viz_path)
            plt.close()
            viz_paths.append(str(viz_path))
        except Exception as e:
            logger.warning(f"Failed to create success trend chart: {e}")
        
        return viz_paths
    
    async def _create_team_visualizations(self, team_users: List[str], analytics: Dict[str, Any], output_path: Path) -> List[str]:
        """Create visualizations for team analytics."""
        # Placeholder implementation
        return []
    
    async def _create_system_visualizations(self, analytics: Dict[str, Any], output_path: Path) -> List[str]:
        """Create system-wide visualizations."""
        # Placeholder implementation  
        return []
    
    async def _create_html_dashboard(self, data: Dict[str, Any], viz_paths: List[str], output_path: Path) -> Path:
        """Create HTML dashboard from analytics data."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Learning Analytics Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .insight {{ background: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Learning Analytics Dashboard</h1>
            <p>Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Key Insights</h2>
            {self._format_insights_html(data.get('insights', []))}
            
            <h2>Visualizations</h2>
            {self._format_visualizations_html(viz_paths)}
        </body>
        </html>
        """
        
        html_path = output_path / 'dashboard.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _format_insights_html(self, insights: List[Dict[str, Any]]) -> str:
        """Format insights for HTML display."""
        html_parts = []
        
        for insight in insights[:5]:  # Top 5 insights
            html_parts.append(f"""
            <div class="insight">
                <h3>{insight.get('title', 'Insight')}</h3>
                <p>{insight.get('description', '')}</p>
                <p><strong>Impact Score:</strong> {insight.get('impact_score', 0):.2f}</p>
                <p><strong>Confidence:</strong> {insight.get('confidence', 0):.2f}</p>
            </div>
            """)
        
        return ''.join(html_parts)
    
    def _format_visualizations_html(self, viz_paths: List[str]) -> str:
        """Format visualizations for HTML display."""
        html_parts = []
        
        for viz_path in viz_paths:
            filename = Path(viz_path).name
            html_parts.append(f"""
            <div class="chart">
                <img src="{filename}" alt="Visualization" style="max-width: 100%; height: auto;">
            </div>
            """)
        
        return ''.join(html_parts)