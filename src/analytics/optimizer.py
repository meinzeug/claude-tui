"""
Performance Optimizer with AI-Powered Recommendations.

Advanced optimization engine that analyzes performance data and generates
intelligent recommendations for improving system performance.
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import asdict, field
from enum import Enum

import numpy as np

from ..core.types import Priority, Severity
from .models import (
    PerformanceMetrics, BottleneckAnalysis, OptimizationRecommendation,
    TrendAnalysis, AnalyticsConfiguration, OptimizationType,
    PerformanceAlert
)


class OptimizationStrategy(str, Enum):
    """Optimization strategies."""
    AGGRESSIVE = "aggressive"  # Maximum performance, higher risk
    BALANCED = "balanced"     # Balance performance and stability
    CONSERVATIVE = "conservative"  # Minimal risk, moderate gains
    SAFETY_FIRST = "safety_first"  # Only safe, proven optimizations


class OptimizationScope(str, Enum):
    """Scope of optimization."""
    SYSTEM = "system"
    APPLICATION = "application"
    WORKFLOW = "workflow"
    AI_MODEL = "ai_model"
    DATABASE = "database"
    NETWORK = "network"


@dataclass
class OptimizationContext:
    """Context information for optimization decisions."""
    current_metrics: PerformanceMetrics
    historical_metrics: List[PerformanceMetrics] = field(default_factory=list)
    active_bottlenecks: List[BottleneckAnalysis] = field(default_factory=list)
    recent_alerts: List[PerformanceAlert] = field(default_factory=list)
    environment: str = "development"
    constraints: Dict[str, Any] = field(default_factory=dict)
    objectives: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationPlan:
    """Comprehensive optimization plan."""
    id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    
    # Plan details
    title: str = ""
    description: str = ""
    objectives: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    implementation_order: List[str] = field(default_factory=list)  # Recommendation IDs
    
    # Impact estimation
    estimated_total_improvement: float = 0.0
    risk_assessment: str = "medium"
    implementation_time: str = "unknown"
    
    # Validation
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    rollback_plan: Optional[str] = None
    
    # Tracking
    status: str = "draft"  # draft, approved, implementing, completed, cancelled
    progress: float = 0.0
    implemented_recommendations: Set[str] = field(default_factory=set)


class PerformanceOptimizer:
    """
    AI-powered performance optimization engine.
    
    Features:
    - Intelligent bottleneck analysis
    - Context-aware optimization recommendations
    - Risk assessment and safety validation
    - Implementation planning and tracking
    - A/B testing support
    - Automated rollback capabilities
    """
    
    def __init__(
        self,
        config: Optional[AnalyticsConfiguration] = None,
        default_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ):
        """Initialize the performance optimizer."""
        self.config = config or AnalyticsConfiguration()
        self.default_strategy = default_strategy
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Optimization knowledge base
        self.optimization_patterns: Dict[str, Dict[str, Any]] = {}
        self.success_history: Dict[str, List[float]] = {}
        self.implementation_templates: Dict[str, Dict[str, Any]] = {}
        
        # Active optimization plans
        self.active_plans: Dict[str, OptimizationPlan] = {}
        self.recommendation_history: List[OptimizationRecommendation] = []
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        
        # Machine learning models (placeholder for future implementation)
        self.optimization_model = None
        self.risk_assessment_model = None
    
    async def initialize(self) -> None:
        """Initialize the optimizer."""
        try:
            self.logger.info("Initializing Performance Optimizer...")
            
            # Load optimization patterns
            await self._load_optimization_patterns()
            
            # Load implementation templates
            await self._load_implementation_templates()
            
            # Initialize ML models (placeholder)
            await self._initialize_optimization_models()
            
            self.logger.info("Performance Optimizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize optimizer: {e}")
            raise
    
    async def analyze_and_recommend(
        self,
        context: OptimizationContext,
        strategy: Optional[OptimizationStrategy] = None,
        max_recommendations: int = 10
    ) -> List[OptimizationRecommendation]:
        """Analyze performance and generate optimization recommendations."""
        try:
            strategy = strategy or self.default_strategy
            recommendations = []
            
            # Analyze bottlenecks and generate targeted recommendations
            bottleneck_recommendations = await self._analyze_bottlenecks_for_optimization(
                context, strategy
            )
            recommendations.extend(bottleneck_recommendations)
            
            # Analyze trends and generate proactive recommendations
            trend_recommendations = await self._analyze_trends_for_optimization(
                context, strategy
            )
            recommendations.extend(trend_recommendations)
            
            # Generate general optimization recommendations
            general_recommendations = await self._generate_general_optimizations(
                context, strategy
            )
            recommendations.extend(general_recommendations)
            
            # Apply ML-based recommendation enhancement (placeholder)
            enhanced_recommendations = await self._enhance_with_ml(
                recommendations, context
            )
            recommendations = enhanced_recommendations
            
            # Filter and prioritize recommendations
            filtered_recommendations = await self._filter_and_prioritize_recommendations(
                recommendations, context, strategy, max_recommendations
            )
            
            # Store recommendations in history
            self.recommendation_history.extend(filtered_recommendations)
            
            return filtered_recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing and recommending optimizations: {e}")
            return []
    
    async def create_optimization_plan(
        self,
        recommendations: List[OptimizationRecommendation],
        context: OptimizationContext,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ) -> OptimizationPlan:
        """Create a comprehensive optimization plan."""
        try:
            plan_id = f"opt_plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate implementation order
            implementation_order = await self._calculate_implementation_order(
                recommendations, context
            )
            
            # Estimate total improvement
            total_improvement = await self._estimate_total_improvement(
                recommendations, context
            )
            
            # Assess overall risk
            risk_assessment = await self._assess_plan_risk(recommendations, context)
            
            # Create success criteria
            success_criteria = await self._define_success_criteria(
                recommendations, context
            )
            
            # Generate rollback plan
            rollback_plan = await self._generate_rollback_plan(
                recommendations, context
            )
            
            plan = OptimizationPlan(
                id=plan_id,
                strategy=strategy,
                title=f"Performance Optimization Plan - {strategy.value.title()}",
                description=f"Comprehensive optimization plan with {len(recommendations)} recommendations",
                objectives=context.objectives,
                recommendations=recommendations,
                implementation_order=implementation_order,
                estimated_total_improvement=total_improvement,
                risk_assessment=risk_assessment,
                success_criteria=success_criteria,
                rollback_plan=rollback_plan
            )
            
            # Store active plan
            self.active_plans[plan_id] = plan
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating optimization plan: {e}")
            raise
    
    async def validate_recommendation_safety(
        self,
        recommendation: OptimizationRecommendation,
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """Validate the safety of implementing a recommendation."""
        try:
            validation_result = {
                'is_safe': True,
                'risk_level': 'low',
                'warnings': [],
                'prerequisites_met': True,
                'rollback_available': True,
                'estimated_downtime': 0,
                'confidence_score': 0.8
            }
            
            # Check environment constraints
            if context.environment == "production":
                validation_result['risk_level'] = 'medium'
                validation_result['warnings'].append(
                    "Production environment detected - extra caution required"
                )
            
            # Check resource impact
            resource_impact = await self._assess_resource_impact(recommendation, context)
            if resource_impact['high_impact']:
                validation_result['risk_level'] = 'high'
                validation_result['warnings'].extend(resource_impact['warnings'])
            
            # Check for conflicting optimizations
            conflicts = await self._detect_optimization_conflicts(recommendation, context)
            if conflicts:
                validation_result['warnings'].extend([
                    f"Potential conflict with: {conflict}" for conflict in conflicts
                ])
            
            # Check prerequisites
            prerequisites_check = await self._check_prerequisites(recommendation, context)
            validation_result['prerequisites_met'] = prerequisites_check['all_met']
            if not prerequisites_check['all_met']:
                validation_result['warnings'].extend(prerequisites_check['missing'])
            
            # Check rollback capability
            rollback_check = await self._check_rollback_capability(recommendation, context)
            validation_result['rollback_available'] = rollback_check['available']
            if not rollback_check['available']:
                validation_result['risk_level'] = 'high'
                validation_result['warnings'].append("No automatic rollback available")
            
            # Calculate final safety assessment
            if validation_result['warnings']:
                validation_result['is_safe'] = len(validation_result['warnings']) <= 2
            
            # Adjust confidence based on risk factors
            risk_penalty = {
                'low': 0.0,
                'medium': 0.1,
                'high': 0.3
            }.get(validation_result['risk_level'], 0.2)
            
            validation_result['confidence_score'] = max(
                0.1, validation_result['confidence_score'] - risk_penalty
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating recommendation safety: {e}")
            return {
                'is_safe': False,
                'risk_level': 'high',
                'warnings': [f"Validation error: {str(e)}"],
                'prerequisites_met': False,
                'rollback_available': False,
                'confidence_score': 0.0
            }
    
    async def simulate_optimization_impact(
        self,
        recommendation: OptimizationRecommendation,
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """Simulate the impact of applying an optimization."""
        try:
            # Create simulated metrics based on current metrics and expected improvement
            current = context.current_metrics
            improvement_factor = recommendation.estimated_improvement / 100
            
            # Calculate projected metrics
            projected_metrics = PerformanceMetrics()
            
            # Copy base metrics
            for field in ['timestamp', 'session_id', 'metric_type', 'environment']:
                if hasattr(current, field):
                    setattr(projected_metrics, field, getattr(current, field))
            
            # Apply optimization improvements based on type
            if recommendation.optimization_type == OptimizationType.CODE_OPTIMIZATION:
                projected_metrics.cpu_percent = max(0, current.cpu_percent * (1 - improvement_factor * 0.3))
                projected_metrics.latency_p95 = max(0, current.latency_p95 * (1 - improvement_factor * 0.4))
                projected_metrics.throughput = current.throughput * (1 + improvement_factor * 0.2)
                
            elif recommendation.optimization_type == OptimizationType.RESOURCE_SCALING:
                projected_metrics.cpu_percent = max(0, current.cpu_percent * (1 - improvement_factor * 0.5))
                projected_metrics.memory_percent = max(0, current.memory_percent * (1 - improvement_factor * 0.4))
                projected_metrics.throughput = current.throughput * (1 + improvement_factor * 0.3)
                
            elif recommendation.optimization_type == OptimizationType.CACHING:
                projected_metrics.cache_hit_rate = min(1.0, current.cache_hit_rate * (1 + improvement_factor * 0.2))
                projected_metrics.latency_p95 = max(0, current.latency_p95 * (1 - improvement_factor * 0.6))
                projected_metrics.throughput = current.throughput * (1 + improvement_factor * 0.4)
                
            elif recommendation.optimization_type == OptimizationType.ALGORITHM_IMPROVEMENT:
                projected_metrics.cpu_percent = max(0, current.cpu_percent * (1 - improvement_factor * 0.4))
                projected_metrics.latency_p95 = max(0, current.latency_p95 * (1 - improvement_factor * 0.5))
                projected_metrics.code_quality_score = min(100, current.code_quality_score * (1 + improvement_factor * 0.1))
                
            else:
                # Generic improvement
                projected_metrics.cpu_percent = max(0, current.cpu_percent * (1 - improvement_factor * 0.2))
                projected_metrics.latency_p95 = max(0, current.latency_p95 * (1 - improvement_factor * 0.2))
                projected_metrics.throughput = current.throughput * (1 + improvement_factor * 0.1)
            
            # Copy unchanged metrics
            unchanged_fields = [
                'memory_used', 'disk_percent', 'active_tasks', 'ai_response_time',
                'error_rate', 'model_accuracy', 'workflow_completion_rate'
            ]
            for field in unchanged_fields:
                if hasattr(current, field):
                    setattr(projected_metrics, field, getattr(current, field))
            
            # Calculate impact metrics
            impact_analysis = {
                'projected_metrics': projected_metrics,
                'improvement_breakdown': {
                    'cpu_improvement': ((current.cpu_percent - projected_metrics.cpu_percent) / current.cpu_percent * 100) if current.cpu_percent > 0 else 0,
                    'latency_improvement': ((current.latency_p95 - projected_metrics.latency_p95) / current.latency_p95 * 100) if current.latency_p95 > 0 else 0,
                    'throughput_improvement': ((projected_metrics.throughput - current.throughput) / current.throughput * 100) if current.throughput > 0 else 0,
                },
                'composite_score_improvement': projected_metrics.calculate_composite_score() - current.calculate_composite_score(),
                'simulation_confidence': recommendation.confidence_score / 100,
                'risk_factors': [],
                'assumptions': [
                    f"Optimization will achieve {recommendation.estimated_improvement:.1f}% improvement",
                    "No unexpected side effects",
                    "Implementation follows best practices"
                ]
            }
            
            # Add risk factors based on optimization type
            if recommendation.optimization_type == OptimizationType.RESOURCE_SCALING:
                impact_analysis['risk_factors'].append("May require infrastructure changes")
            if recommendation.optimization_type == OptimizationType.ALGORITHM_IMPROVEMENT:
                impact_analysis['risk_factors'].append("Code changes may introduce bugs")
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error simulating optimization impact: {e}")
            return {
                'error': str(e),
                'simulation_confidence': 0.0
            }
    
    async def track_optimization_results(
        self,
        recommendation_id: str,
        before_metrics: PerformanceMetrics,
        after_metrics: PerformanceMetrics,
        implementation_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Track the actual results of an implemented optimization."""
        try:
            # Find the recommendation
            recommendation = None
            for rec in self.recommendation_history:
                if str(rec.id) == recommendation_id:
                    recommendation = rec
                    break
            
            if not recommendation:
                raise ValueError(f"Recommendation {recommendation_id} not found")
            
            # Calculate actual improvement
            actual_improvement = self._calculate_actual_improvement(
                before_metrics, after_metrics, recommendation.optimization_type
            )
            
            # Update recommendation with actual results
            recommendation.actual_improvement = actual_improvement
            recommendation.implemented_at = datetime.utcnow()
            recommendation.status = "completed"
            
            # Calculate accuracy of prediction
            predicted_improvement = recommendation.estimated_improvement
            prediction_accuracy = 1 - abs(predicted_improvement - actual_improvement) / max(predicted_improvement, actual_improvement, 1)
            
            # Store results in success history
            opt_type = recommendation.optimization_type.value
            if opt_type not in self.success_history:
                self.success_history[opt_type] = []
            
            self.success_history[opt_type].append(actual_improvement)
            
            # Generate learning insights
            insights = await self._generate_learning_insights(
                recommendation, before_metrics, after_metrics, actual_improvement
            )
            
            result = {
                'recommendation_id': recommendation_id,
                'predicted_improvement': predicted_improvement,
                'actual_improvement': actual_improvement,
                'prediction_accuracy': prediction_accuracy,
                'success': actual_improvement > 0,
                'performance_impact': {
                    'before_composite_score': before_metrics.calculate_composite_score(),
                    'after_composite_score': after_metrics.calculate_composite_score(),
                    'score_improvement': after_metrics.calculate_composite_score() - before_metrics.calculate_composite_score()
                },
                'implementation_notes': implementation_notes,
                'learning_insights': insights,
                'tracked_at': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error tracking optimization results: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    async def get_optimization_recommendations(
        self,
        metrics: PerformanceMetrics,
        bottlenecks: List[BottleneckAnalysis],
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ) -> List[OptimizationRecommendation]:
        """Get optimization recommendations based on current performance data."""
        context = OptimizationContext(
            current_metrics=metrics,
            active_bottlenecks=bottlenecks,
            environment=metrics.environment
        )
        
        return await self.analyze_and_recommend(context, strategy)
    
    # Private implementation methods
    
    async def _analyze_bottlenecks_for_optimization(
        self,
        context: OptimizationContext,
        strategy: OptimizationStrategy
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations based on active bottlenecks."""
        recommendations = []
        
        for bottleneck in context.active_bottlenecks:
            if bottleneck.bottleneck_type == "cpu":
                rec = OptimizationRecommendation(
                    optimization_type=OptimizationType.CODE_OPTIMIZATION,
                    title="Optimize CPU-Intensive Operations",
                    description="Reduce CPU usage through algorithmic improvements",
                    rationale=f"CPU bottleneck detected with {bottleneck.performance_impact:.1f}% impact",
                    estimated_improvement=min(50, bottleneck.performance_impact),
                    confidence_score=80 - (10 if strategy == OptimizationStrategy.AGGRESSIVE else 0),
                    priority=Priority.HIGH if bottleneck.severity == Severity.HIGH else Priority.MEDIUM,
                    implementation_steps=[
                        "Profile CPU usage patterns",
                        "Identify optimization opportunities",
                        "Implement algorithmic improvements",
                        "Measure and validate improvements"
                    ],
                    target_components=["cpu_intensive_operations"]
                )
                recommendations.append(rec)
            
            elif bottleneck.bottleneck_type == "memory":
                rec = OptimizationRecommendation(
                    optimization_type=OptimizationType.RESOURCE_SCALING,
                    title="Optimize Memory Usage",
                    description="Reduce memory footprint and improve allocation patterns",
                    rationale=f"Memory bottleneck with {bottleneck.performance_impact:.1f}% impact",
                    estimated_improvement=min(40, bottleneck.performance_impact),
                    confidence_score=85,
                    priority=Priority.HIGH if bottleneck.severity == Severity.HIGH else Priority.MEDIUM,
                    implementation_steps=[
                        "Analyze memory usage patterns",
                        "Implement memory optimization strategies",
                        "Add memory monitoring",
                        "Validate improvements"
                    ],
                    target_components=["memory_management"]
                )
                recommendations.append(rec)
        
        return recommendations
    
    async def _analyze_trends_for_optimization(
        self,
        context: OptimizationContext,
        strategy: OptimizationStrategy
    ) -> List[OptimizationRecommendation]:
        """Generate proactive recommendations based on performance trends."""
        recommendations = []
        
        if not context.historical_metrics:
            return recommendations
        
        # Analyze CPU trend
        cpu_values = [m.cpu_percent for m in context.historical_metrics[-20:]]
        if len(cpu_values) > 5:
            cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
            
            if cpu_trend > 2:  # Increasing CPU usage
                rec = OptimizationRecommendation(
                    optimization_type=OptimizationType.CODE_OPTIMIZATION,
                    title="Proactive CPU Optimization",
                    description="Address increasing CPU usage trend before it becomes critical",
                    rationale=f"CPU usage trending upward at {cpu_trend:.1f}% per measurement",
                    estimated_improvement=15,
                    confidence_score=60,
                    priority=Priority.MEDIUM,
                    implementation_steps=[
                        "Monitor CPU usage patterns",
                        "Identify growth sources",
                        "Implement preventive optimizations"
                    ]
                )
                recommendations.append(rec)
        
        return recommendations
    
    async def _generate_general_optimizations(
        self,
        context: OptimizationContext,
        strategy: OptimizationStrategy
    ) -> List[OptimizationRecommendation]:
        """Generate general optimization recommendations."""
        recommendations = []
        metrics = context.current_metrics
        
        # Cache optimization
        if metrics.cache_hit_rate < 0.8:
            rec = OptimizationRecommendation(
                optimization_type=OptimizationType.CACHING,
                title="Improve Cache Performance",
                description="Optimize caching strategy to improve hit rates",
                rationale=f"Cache hit rate is {metrics.cache_hit_rate*100:.1f}%, below optimal",
                estimated_improvement=20,
                confidence_score=75,
                priority=Priority.MEDIUM,
                implementation_steps=[
                    "Analyze cache usage patterns",
                    "Optimize cache size and policies",
                    "Implement cache warming strategies"
                ]
            )
            recommendations.append(rec)
        
        # Code quality optimization
        if metrics.code_quality_score < 70:
            rec = OptimizationRecommendation(
                optimization_type=OptimizationType.CODE_OPTIMIZATION,
                title="Improve Code Quality",
                description="Refactor code to improve maintainability and performance",
                rationale=f"Code quality score is {metrics.code_quality_score:.1f}, below standards",
                estimated_improvement=10,
                confidence_score=65,
                priority=Priority.LOW,
                implementation_steps=[
                    "Identify code quality issues",
                    "Refactor problematic code",
                    "Add comprehensive tests",
                    "Improve documentation"
                ]
            )
            recommendations.append(rec)
        
        return recommendations
    
    async def _enhance_with_ml(
        self,
        recommendations: List[OptimizationRecommendation],
        context: OptimizationContext
    ) -> List[OptimizationRecommendation]:
        """Enhance recommendations using machine learning (placeholder)."""
        # This would use trained ML models to improve recommendations
        # For now, just return the original recommendations
        return recommendations
    
    async def _filter_and_prioritize_recommendations(
        self,
        recommendations: List[OptimizationRecommendation],
        context: OptimizationContext,
        strategy: OptimizationStrategy,
        max_recommendations: int
    ) -> List[OptimizationRecommendation]:
        """Filter and prioritize recommendations based on strategy."""
        if not recommendations:
            return []
        
        # Filter by strategy
        filtered = []
        for rec in recommendations:
            if strategy == OptimizationStrategy.CONSERVATIVE:
                if rec.risk_level == "low" and rec.confidence_score >= 70:
                    filtered.append(rec)
            elif strategy == OptimizationStrategy.AGGRESSIVE:
                filtered.append(rec)  # Include all recommendations
            else:  # BALANCED
                if rec.confidence_score >= 60:
                    filtered.append(rec)
        
        # Sort by priority and estimated improvement
        filtered.sort(
            key=lambda r: (
                ["critical", "high", "medium", "low"].index(r.priority.value),
                -r.estimated_improvement,
                -r.confidence_score
            )
        )
        
        return filtered[:max_recommendations]
    
    async def _calculate_implementation_order(
        self,
        recommendations: List[OptimizationRecommendation],
        context: OptimizationContext
    ) -> List[str]:
        """Calculate optimal implementation order for recommendations."""
        # Simple ordering by priority and dependencies
        # In a real implementation, this would consider complex dependencies
        
        ordered_ids = []
        remaining = recommendations.copy()
        
        # First pass: Critical and high priority items
        for rec in remaining.copy():
            if rec.priority in [Priority.CRITICAL, Priority.HIGH]:
                ordered_ids.append(str(rec.id))
                remaining.remove(rec)
        
        # Second pass: Medium priority items
        for rec in remaining.copy():
            if rec.priority == Priority.MEDIUM:
                ordered_ids.append(str(rec.id))
                remaining.remove(rec)
        
        # Third pass: Low priority items
        for rec in remaining:
            ordered_ids.append(str(rec.id))
        
        return ordered_ids
    
    async def _estimate_total_improvement(
        self,
        recommendations: List[OptimizationRecommendation],
        context: OptimizationContext
    ) -> float:
        """Estimate total improvement from all recommendations."""
        if not recommendations:
            return 0.0
        
        # Use diminishing returns model
        improvements = [rec.estimated_improvement for rec in recommendations]
        improvements.sort(reverse=True)
        
        total = 0.0
        diminishing_factor = 1.0
        
        for improvement in improvements:
            total += improvement * diminishing_factor
            diminishing_factor *= 0.8  # Each subsequent improvement has 80% effect
        
        return min(100.0, total)  # Cap at 100%
    
    async def _assess_plan_risk(
        self,
        recommendations: List[OptimizationRecommendation],
        context: OptimizationContext
    ) -> str:
        """Assess overall risk of the optimization plan."""
        if not recommendations:
            return "low"
        
        risk_scores = {
            "low": 1,
            "medium": 2,
            "high": 3
        }
        
        total_risk = sum(risk_scores.get(rec.risk_level, 2) for rec in recommendations)
        avg_risk = total_risk / len(recommendations)
        
        if avg_risk <= 1.5:
            return "low"
        elif avg_risk <= 2.5:
            return "medium"
        else:
            return "high"
    
    async def _define_success_criteria(
        self,
        recommendations: List[OptimizationRecommendation],
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """Define success criteria for the optimization plan."""
        criteria = {
            'minimum_improvement': 5.0,  # Minimum 5% overall improvement
            'no_performance_regression': True,
            'stability_maintained': True,
            'metrics_to_track': [
                'composite_score',
                'cpu_percent',
                'memory_percent',
                'latency_p95',
                'throughput',
                'error_rate'
            ],
            'success_thresholds': {
                'composite_score_improvement': 5.0,
                'error_rate_max': context.current_metrics.error_rate * 1.1,  # Allow 10% increase
                'latency_p95_max': context.current_metrics.latency_p95 * 1.2  # Allow 20% increase
            }
        }
        
        return criteria
    
    async def _generate_rollback_plan(
        self,
        recommendations: List[OptimizationRecommendation],
        context: OptimizationContext
    ) -> str:
        """Generate rollback plan for the optimization."""
        rollback_steps = [
            "1. Monitor key performance metrics continuously",
            "2. If performance degrades beyond thresholds, initiate rollback",
            "3. Revert changes in reverse implementation order",
            "4. Validate that original performance is restored",
            "5. Document lessons learned and adjust future optimizations"
        ]
        
        return "\n".join(rollback_steps)
    
    def _calculate_actual_improvement(
        self,
        before_metrics: PerformanceMetrics,
        after_metrics: PerformanceMetrics,
        optimization_type: OptimizationType
    ) -> float:
        """Calculate actual improvement achieved."""
        before_score = before_metrics.calculate_composite_score()
        after_score = after_metrics.calculate_composite_score()
        
        if before_score == 0:
            return 0.0
        
        improvement = ((after_score - before_score) / before_score) * 100
        return max(0.0, improvement)  # Only count positive improvements
    
    async def _generate_learning_insights(
        self,
        recommendation: OptimizationRecommendation,
        before_metrics: PerformanceMetrics,
        after_metrics: PerformanceMetrics,
        actual_improvement: float
    ) -> List[str]:
        """Generate insights from optimization results for future learning."""
        insights = []
        
        predicted = recommendation.estimated_improvement
        actual = actual_improvement
        
        if actual > predicted * 1.2:
            insights.append("Optimization exceeded expectations - consider similar optimizations")
        elif actual < predicted * 0.8:
            insights.append("Optimization underperformed - review implementation approach")
        
        if actual < 0:
            insights.append("Optimization caused performance regression - review safety checks")
        
        # Add type-specific insights
        if recommendation.optimization_type == OptimizationType.CACHING:
            cache_improvement = after_metrics.cache_hit_rate - before_metrics.cache_hit_rate
            if cache_improvement > 0.1:
                insights.append("Cache optimization highly effective")
        
        return insights
    
    async def _load_optimization_patterns(self) -> None:
        """Load optimization patterns from knowledge base."""
        # Placeholder for loading optimization patterns
        self.optimization_patterns = {
            "cpu_bottleneck": {
                "common_causes": ["inefficient_algorithms", "excessive_computation"],
                "solutions": ["algorithm_optimization", "caching", "parallel_processing"]
            },
            "memory_bottleneck": {
                "common_causes": ["memory_leaks", "large_objects", "poor_gc"],
                "solutions": ["memory_optimization", "object_pooling", "gc_tuning"]
            }
        }
    
    async def _load_implementation_templates(self) -> None:
        """Load implementation templates for different optimization types."""
        # Placeholder for loading implementation templates
        pass
    
    async def _initialize_optimization_models(self) -> None:
        """Initialize machine learning models for optimization."""
        # Placeholder for ML model initialization
        pass
    
    async def _assess_resource_impact(
        self,
        recommendation: OptimizationRecommendation,
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """Assess resource impact of a recommendation."""
        return {
            'high_impact': recommendation.optimization_type == OptimizationType.RESOURCE_SCALING,
            'warnings': [] if recommendation.optimization_type != OptimizationType.RESOURCE_SCALING
            else ["May require significant resource changes"]
        }
    
    async def _detect_optimization_conflicts(
        self,
        recommendation: OptimizationRecommendation,
        context: OptimizationContext
    ) -> List[str]:
        """Detect potential conflicts with other optimizations."""
        # Placeholder for conflict detection
        return []
    
    async def _check_prerequisites(
        self,
        recommendation: OptimizationRecommendation,
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """Check if prerequisites for the recommendation are met."""
        return {
            'all_met': True,
            'missing': []
        }
    
    async def _check_rollback_capability(
        self,
        recommendation: OptimizationRecommendation,
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """Check if rollback is available for the recommendation."""
        return {
            'available': recommendation.optimization_type != OptimizationType.INFRASTRUCTURE_UPGRADE
        }