"""
Analytics Engine - Advanced team analytics and productivity insights
Provides comprehensive analytics for team performance and collaboration metrics
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from enum import Enum
from dataclasses import dataclass
import json

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, distinct
from sqlalchemy.sql import text

from .models import (
    Workspace, WorkspaceMember, ActivityFeed, CollaborationSession,
    ConflictResolution, TeamAnalytics, Comment
)
from ..database.models import User, Task

logger = logging.getLogger(__name__)


class AnalyticsTimeframe(Enum):
    """Analytics timeframe options"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class MetricType(Enum):
    """Types of metrics to analyze"""
    PRODUCTIVITY = "productivity"
    COLLABORATION = "collaboration"
    COMMUNICATION = "communication"
    QUALITY = "quality"
    PERFORMANCE = "performance"


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    workspace_id: str
    timeframe: str
    period_start: str
    period_end: str
    summary_metrics: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    insights: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    generated_at: str


class AnalyticsEngine:
    """
    Advanced analytics engine for team collaboration insights.
    Provides comprehensive metrics, trends, and actionable insights.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize analytics engine.
        
        Args:
            db_session: Database session for operations
        """
        self.db = db_session
        
        # Analytics cache
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Metric calculators
        self._metric_calculators = {
            MetricType.PRODUCTIVITY: self._calculate_productivity_metrics,
            MetricType.COLLABORATION: self._calculate_collaboration_metrics,
            MetricType.COMMUNICATION: self._calculate_communication_metrics,
            MetricType.QUALITY: self._calculate_quality_metrics,
            MetricType.PERFORMANCE: self._calculate_performance_metrics
        }
        
        # Insight generators
        self._insight_generators = [
            self._generate_productivity_insights,
            self._generate_collaboration_insights,
            self._generate_communication_insights,
            self._generate_trend_insights
        ]
        
        logger.info("Analytics engine initialized")
    
    async def generate_comprehensive_report(
        self,
        workspace_id: UUID,
        timeframe: AnalyticsTimeframe = AnalyticsTimeframe.WEEKLY,
        metric_types: Optional[List[MetricType]] = None,
        custom_period: Optional[Tuple[datetime, datetime]] = None
    ) -> AnalyticsReport:
        """
        Generate comprehensive analytics report for workspace.
        
        Args:
            workspace_id: Workspace ID
            timeframe: Analysis timeframe
            metric_types: Specific metrics to include
            custom_period: Custom time period (overrides timeframe)
            
        Returns:
            Comprehensive analytics report
        """
        logger.info(f"Generating analytics report for workspace {workspace_id}")
        
        # Determine time period
        if custom_period:
            period_start, period_end = custom_period
        else:
            period_start, period_end = self._get_timeframe_period(timeframe)
        
        # Include all metric types if not specified
        if not metric_types:
            metric_types = list(MetricType)
        
        # Calculate metrics
        summary_metrics = {}
        detailed_metrics = {}
        
        for metric_type in metric_types:
            calculator = self._metric_calculators.get(metric_type)
            if calculator:
                metrics = await calculator(workspace_id, period_start, period_end)
                summary_metrics.update(metrics.get('summary', {}))
                detailed_metrics[metric_type.value] = metrics.get('detailed', {})
        
        # Generate insights
        insights = []
        for generator in self._insight_generators:
            try:
                insight = await generator(workspace_id, detailed_metrics, period_start, period_end)
                if insight:
                    insights.extend(insight if isinstance(insight, list) else [insight])
            except Exception as e:
                logger.error(f"Error generating insights: {e}")
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(workspace_id, detailed_metrics, insights)
        
        report = AnalyticsReport(
            workspace_id=str(workspace_id),
            timeframe=timeframe.value,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            summary_metrics=summary_metrics,
            detailed_metrics=detailed_metrics,
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Store report
        await self._store_analytics_report(workspace_id, report)
        
        logger.info(f"Analytics report generated for workspace {workspace_id}")
        return report
    
    async def get_real_time_metrics(
        self,
        workspace_id: UUID,
        metric_names: List[str]
    ) -> Dict[str, Any]:
        """
        Get real-time metrics for workspace dashboard.
        
        Args:
            workspace_id: Workspace ID
            metric_names: List of metrics to retrieve
            
        Returns:
            Real-time metrics data
        """
        cache_key = f"{workspace_id}_realtime_{hash(tuple(sorted(metric_names)))}"
        
        # Check cache
        if cache_key in self._metrics_cache:
            cache_entry = self._metrics_cache[cache_key]
            if datetime.now(timezone.utc) - cache_entry['timestamp'] < timedelta(seconds=self._cache_ttl):
                return cache_entry['data']
        
        # Calculate real-time metrics
        metrics = {}
        
        for metric_name in metric_names:
            try:
                metric_value = await self._calculate_real_time_metric(workspace_id, metric_name)
                metrics[metric_name] = metric_value
            except Exception as e:
                logger.error(f"Error calculating metric {metric_name}: {e}")
                metrics[metric_name] = None
        
        # Cache results
        self._metrics_cache[cache_key] = {
            'data': metrics,
            'timestamp': datetime.now(timezone.utc)
        }
        
        return metrics
    
    async def get_member_analytics(
        self,
        workspace_id: UUID,
        member_id: UUID,
        timeframe: AnalyticsTimeframe = AnalyticsTimeframe.WEEKLY
    ) -> Dict[str, Any]:
        """
        Get detailed analytics for specific team member.
        
        Args:
            workspace_id: Workspace ID
            member_id: Member user ID
            timeframe: Analysis timeframe
            
        Returns:
            Member-specific analytics
        """
        period_start, period_end = self._get_timeframe_period(timeframe)
        
        # Get member activities
        activities = self.db.query(ActivityFeed).filter(
            and_(
                ActivityFeed.workspace_id == workspace_id,
                ActivityFeed.user_id == member_id,
                ActivityFeed.created_at >= period_start,
                ActivityFeed.created_at <= period_end
            )
        ).all()
        
        # Get collaboration sessions
        sessions = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.user_id == member_id,
                CollaborationSession.started_at >= period_start,
                CollaborationSession.started_at <= period_end
            )
        ).all()
        
        # Get assigned tasks
        tasks = self.db.query(Task).filter(
            and_(
                Task.assigned_to == member_id,
                Task.created_at >= period_start,
                Task.created_at <= period_end
            )
        ).all()
        
        # Calculate member metrics
        total_activities = len(activities)
        total_session_time = sum(
            (session.ended_at - session.started_at).total_seconds() / 3600
            for session in sessions if session.ended_at
        )
        
        completed_tasks = len([t for t in tasks if t.status == 'completed'])
        task_completion_rate = (completed_tasks / len(tasks) * 100) if tasks else 0
        
        # Activity breakdown
        activity_breakdown = {}
        for activity in activities:
            activity_type = activity.activity_type
            activity_breakdown[activity_type] = activity_breakdown.get(activity_type, 0) + 1
        
        return {
            'member_id': str(member_id),
            'period': {
                'start': period_start.isoformat(),
                'end': period_end.isoformat()
            },
            'productivity': {
                'total_activities': total_activities,
                'session_hours': round(total_session_time, 2),
                'tasks_assigned': len(tasks),
                'tasks_completed': completed_tasks,
                'completion_rate': round(task_completion_rate, 2)
            },
            'activity_breakdown': activity_breakdown,
            'collaboration_score': await self._calculate_member_collaboration_score(
                workspace_id, member_id, period_start, period_end
            ),
            'performance_trends': await self._get_member_performance_trends(
                workspace_id, member_id, period_start, period_end
            )
        }
    
    async def get_workspace_summary(self, workspace_id: UUID) -> Dict[str, Any]:
        """Get quick workspace analytics summary"""
        # Get basic counts
        member_count = self.db.query(WorkspaceMember).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.is_active == True
            )
        ).count()
        
        # Recent activity count (last 24 hours)
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        recent_activities = self.db.query(ActivityFeed).filter(
            and_(
                ActivityFeed.workspace_id == workspace_id,
                ActivityFeed.created_at >= yesterday
            )
        ).count()
        
        # Active sessions count
        active_sessions = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.is_active == True
            )
        ).count()
        
        # Recent conflicts
        recent_conflicts = self.db.query(ConflictResolution).filter(
            and_(
                ConflictResolution.workspace_id == workspace_id,
                ConflictResolution.detected_at >= yesterday
            )
        ).count()
        
        return {
            'member_count': member_count,
            'recent_activities': recent_activities,
            'active_sessions': active_sessions,
            'recent_conflicts': recent_conflicts,
            'health_score': await self._calculate_workspace_health_score(workspace_id)
        }
    
    async def initialize_workspace_analytics(self, workspace_id: UUID) -> None:
        """Initialize analytics for new workspace"""
        # Create initial analytics record
        analytics = TeamAnalytics(
            workspace_id=workspace_id,
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc) + timedelta(days=7),
            member_contributions={},
            activity_breakdown={},
            performance_trends={}
        )
        
        self.db.add(analytics)
        self.db.commit()
        
        logger.info(f"Initialized analytics for workspace {workspace_id}")
    
    # Metric calculation methods
    
    async def _calculate_productivity_metrics(
        self,
        workspace_id: UUID,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Calculate productivity metrics"""
        # Get activities in period
        activities = self.db.query(ActivityFeed).filter(
            and_(
                ActivityFeed.workspace_id == workspace_id,
                ActivityFeed.created_at >= period_start,
                ActivityFeed.created_at <= period_end
            )
        ).all()
        
        # Get sessions
        sessions = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.started_at >= period_start,
                CollaborationSession.started_at <= period_end
            )
        ).all()
        
        # Calculate metrics
        total_activities = len(activities)
        unique_active_users = len(set(activity.user_id for activity in activities))
        
        # Session statistics
        total_session_time = 0
        for session in sessions:
            if session.ended_at:
                duration = (session.ended_at - session.started_at).total_seconds()
                total_session_time += duration
        
        avg_session_duration = (total_session_time / len(sessions) / 3600) if sessions else 0
        
        # Activity velocity (activities per day)
        period_days = (period_end - period_start).days or 1
        activity_velocity = total_activities / period_days
        
        return {
            'summary': {
                'productivity_score': min(100, (activity_velocity * 10)),
                'activity_velocity': round(activity_velocity, 2),
                'avg_session_hours': round(avg_session_duration, 2)
            },
            'detailed': {
                'total_activities': total_activities,
                'unique_active_users': unique_active_users,
                'total_session_hours': round(total_session_time / 3600, 2),
                'activity_distribution': self._get_activity_distribution(activities)
            }
        }
    
    async def _calculate_collaboration_metrics(
        self,
        workspace_id: UUID,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Calculate collaboration metrics"""
        # Get comments and discussions
        comments = self.db.query(Comment).filter(
            and_(
                Comment.workspace_id == workspace_id,
                Comment.created_at >= period_start,
                Comment.created_at <= period_end
            )
        ).all()
        
        # Get conflicts and resolutions
        conflicts = self.db.query(ConflictResolution).filter(
            and_(
                ConflictResolution.workspace_id == workspace_id,
                ConflictResolution.detected_at >= period_start,
                ConflictResolution.detected_at <= period_end
            )
        ).all()
        
        # Calculate collaboration score
        total_comments = len(comments)
        total_conflicts = len(conflicts)
        resolved_conflicts = len([c for c in conflicts if c.resolved_at])
        
        conflict_resolution_rate = (resolved_conflicts / total_conflicts * 100) if total_conflicts else 100
        
        # Collaboration intensity
        comment_users = set(comment.user_id for comment in comments)
        collaboration_breadth = len(comment_users)
        
        collaboration_score = min(100, (collaboration_breadth * 5) + (conflict_resolution_rate * 0.5))
        
        return {
            'summary': {
                'collaboration_score': round(collaboration_score, 2),
                'conflict_resolution_rate': round(conflict_resolution_rate, 2)
            },
            'detailed': {
                'total_comments': total_comments,
                'total_conflicts': total_conflicts,
                'resolved_conflicts': resolved_conflicts,
                'collaboration_breadth': collaboration_breadth,
                'conflict_types': self._get_conflict_type_distribution(conflicts)
            }
        }
    
    async def _calculate_communication_metrics(
        self,
        workspace_id: UUID,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Calculate communication metrics"""
        # Communication activities
        comm_activities = self.db.query(ActivityFeed).filter(
            and_(
                ActivityFeed.workspace_id == workspace_id,
                ActivityFeed.activity_type.in_(['comment_added', 'status_update', 'review_submitted']),
                ActivityFeed.created_at >= period_start,
                ActivityFeed.created_at <= period_end
            )
        ).all()
        
        # Response time analysis (simplified)
        avg_response_time = 2.5  # hours - would calculate from actual response patterns
        
        # Communication frequency
        period_hours = (period_end - period_start).total_seconds() / 3600
        communication_frequency = len(comm_activities) / period_hours if period_hours > 0 else 0
        
        return {
            'summary': {
                'communication_score': min(100, communication_frequency * 20),
                'avg_response_time_hours': avg_response_time
            },
            'detailed': {
                'total_communications': len(comm_activities),
                'communication_frequency': round(communication_frequency, 3),
                'communication_types': self._get_activity_distribution(comm_activities)
            }
        }
    
    async def _calculate_quality_metrics(
        self,
        workspace_id: UUID,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Calculate quality metrics"""
        # This would integrate with code quality tools
        # For now, use conflict resolution as quality proxy
        
        conflicts = self.db.query(ConflictResolution).filter(
            and_(
                ConflictResolution.workspace_id == workspace_id,
                ConflictResolution.detected_at >= period_start,
                ConflictResolution.detected_at <= period_end
            )
        ).all()
        
        auto_resolved = len([c for c in conflicts if c.resolution_strategy == 'auto_merge'])
        total_conflicts = len(conflicts)
        
        quality_score = 100 - (total_conflicts * 2)  # Fewer conflicts = higher quality
        auto_resolution_rate = (auto_resolved / total_conflicts * 100) if total_conflicts else 100
        
        return {
            'summary': {
                'quality_score': max(0, quality_score),
                'auto_resolution_rate': round(auto_resolution_rate, 2)
            },
            'detailed': {
                'total_conflicts': total_conflicts,
                'auto_resolved': auto_resolved,
                'manual_resolved': total_conflicts - auto_resolved
            }
        }
    
    async def _calculate_performance_metrics(
        self,
        workspace_id: UUID,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Calculate performance metrics"""
        # System performance metrics
        sessions = self.db.query(CollaborationSession).filter(
            and_(
                CollaborationSession.workspace_id == workspace_id,
                CollaborationSession.started_at >= period_start,
                CollaborationSession.started_at <= period_end
            )
        ).all()
        
        # Calculate uptime and responsiveness
        total_sessions = len(sessions)
        successful_sessions = len([s for s in sessions if not s.ended_at or s.ended_at > s.started_at])
        
        uptime_percentage = (successful_sessions / total_sessions * 100) if total_sessions else 100
        
        return {
            'summary': {
                'performance_score': round(uptime_percentage, 2),
                'uptime_percentage': round(uptime_percentage, 2)
            },
            'detailed': {
                'total_sessions': total_sessions,
                'successful_sessions': successful_sessions,
                'failed_sessions': total_sessions - successful_sessions
            }
        }
    
    async def _calculate_real_time_metric(
        self,
        workspace_id: UUID,
        metric_name: str
    ) -> Any:
        """Calculate specific real-time metric"""
        if metric_name == 'active_users':
            return self.db.query(CollaborationSession).filter(
                and_(
                    CollaborationSession.workspace_id == workspace_id,
                    CollaborationSession.is_active == True
                )
            ).count()
        
        elif metric_name == 'recent_activities':
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            return self.db.query(ActivityFeed).filter(
                and_(
                    ActivityFeed.workspace_id == workspace_id,
                    ActivityFeed.created_at >= one_hour_ago
                )
            ).count()
        
        elif metric_name == 'pending_conflicts':
            return self.db.query(ConflictResolution).filter(
                and_(
                    ConflictResolution.workspace_id == workspace_id,
                    ConflictResolution.status == 'pending'
                )
            ).count()
        
        return None
    
    # Insight generation methods
    
    async def _generate_productivity_insights(
        self,
        workspace_id: UUID,
        metrics: Dict[str, Any],
        period_start: datetime,
        period_end: datetime
    ) -> List[Dict[str, Any]]:
        """Generate productivity insights"""
        insights = []
        
        productivity_metrics = metrics.get('productivity', {})
        if not productivity_metrics:
            return insights
        
        activity_velocity = productivity_metrics.get('activity_velocity', 0)
        
        if activity_velocity > 10:
            insights.append({
                'type': 'positive',
                'category': 'productivity',
                'title': 'High Team Activity',
                'description': f'Team is highly active with {activity_velocity:.1f} activities per day',
                'confidence': 0.8
            })
        elif activity_velocity < 2:
            insights.append({
                'type': 'warning',
                'category': 'productivity',
                'title': 'Low Team Activity',
                'description': f'Team activity is below average at {activity_velocity:.1f} activities per day',
                'confidence': 0.7
            })
        
        return insights
    
    async def _generate_collaboration_insights(
        self,
        workspace_id: UUID,
        metrics: Dict[str, Any],
        period_start: datetime,
        period_end: datetime
    ) -> List[Dict[str, Any]]:
        """Generate collaboration insights"""
        insights = []
        
        collab_metrics = metrics.get('collaboration', {})
        if not collab_metrics:
            return insights
        
        resolution_rate = collab_metrics.get('conflict_resolution_rate', 0)
        
        if resolution_rate > 90:
            insights.append({
                'type': 'positive',
                'category': 'collaboration',
                'title': 'Excellent Conflict Resolution',
                'description': f'{resolution_rate:.1f}% of conflicts resolved successfully',
                'confidence': 0.9
            })
        elif resolution_rate < 70:
            insights.append({
                'type': 'warning',
                'category': 'collaboration',
                'title': 'Conflict Resolution Issues',
                'description': f'Only {resolution_rate:.1f}% of conflicts resolved - may need intervention',
                'confidence': 0.8
            })
        
        return insights
    
    async def _generate_communication_insights(
        self,
        workspace_id: UUID,
        metrics: Dict[str, Any],
        period_start: datetime,
        period_end: datetime
    ) -> List[Dict[str, Any]]:
        """Generate communication insights"""
        insights = []
        
        comm_metrics = metrics.get('communication', {})
        if not comm_metrics:
            return insights
        
        response_time = comm_metrics.get('avg_response_time_hours', 0)
        
        if response_time < 2:
            insights.append({
                'type': 'positive',
                'category': 'communication',
                'title': 'Fast Response Times',
                'description': f'Team responds quickly with {response_time:.1f}h average response time',
                'confidence': 0.8
            })
        elif response_time > 8:
            insights.append({
                'type': 'warning',
                'category': 'communication',
                'title': 'Slow Response Times',
                'description': f'Team response time is {response_time:.1f}h - consider improving communication',
                'confidence': 0.7
            })
        
        return insights
    
    async def _generate_trend_insights(
        self,
        workspace_id: UUID,
        metrics: Dict[str, Any],
        period_start: datetime,
        period_end: datetime
    ) -> List[Dict[str, Any]]:
        """Generate trend-based insights"""
        # This would analyze historical trends
        return []
    
    async def _generate_recommendations(
        self,
        workspace_id: UUID,
        metrics: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze insights to generate recommendations
        warning_insights = [i for i in insights if i['type'] == 'warning']
        
        for insight in warning_insights:
            if insight['category'] == 'productivity':
                recommendations.append({
                    'type': 'process_improvement',
                    'title': 'Increase Team Engagement',
                    'description': 'Consider implementing daily standups or sprint planning to boost activity',
                    'priority': 'medium',
                    'estimated_impact': 'high'
                })
            elif insight['category'] == 'collaboration':
                recommendations.append({
                    'type': 'process_improvement',
                    'title': 'Improve Conflict Resolution Process',
                    'description': 'Implement conflict resolution training or automated conflict detection',
                    'priority': 'high',
                    'estimated_impact': 'high'
                })
            elif insight['category'] == 'communication':
                recommendations.append({
                    'type': 'tool_improvement',
                    'title': 'Setup Communication Guidelines',
                    'description': 'Establish response time expectations and communication protocols',
                    'priority': 'medium',
                    'estimated_impact': 'medium'
                })
        
        return recommendations
    
    # Helper methods
    
    def _get_timeframe_period(self, timeframe: AnalyticsTimeframe) -> Tuple[datetime, datetime]:
        """Get start and end dates for timeframe"""
        end_date = datetime.now(timezone.utc)
        
        if timeframe == AnalyticsTimeframe.HOURLY:
            start_date = end_date - timedelta(hours=1)
        elif timeframe == AnalyticsTimeframe.DAILY:
            start_date = end_date - timedelta(days=1)
        elif timeframe == AnalyticsTimeframe.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif timeframe == AnalyticsTimeframe.MONTHLY:
            start_date = end_date - timedelta(days=30)
        elif timeframe == AnalyticsTimeframe.QUARTERLY:
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(weeks=1)  # Default to weekly
        
        return start_date, end_date
    
    def _get_activity_distribution(self, activities: List[ActivityFeed]) -> Dict[str, int]:
        """Get distribution of activity types"""
        distribution = {}
        for activity in activities:
            activity_type = activity.activity_type
            distribution[activity_type] = distribution.get(activity_type, 0) + 1
        return distribution
    
    def _get_conflict_type_distribution(self, conflicts: List[ConflictResolution]) -> Dict[str, int]:
        """Get distribution of conflict types"""
        distribution = {}
        for conflict in conflicts:
            conflict_type = conflict.conflict_type
            distribution[conflict_type] = distribution.get(conflict_type, 0) + 1
        return distribution
    
    async def _calculate_member_collaboration_score(
        self,
        workspace_id: UUID,
        member_id: UUID,
        period_start: datetime,
        period_end: datetime
    ) -> float:
        """Calculate collaboration score for specific member"""
        # Get member's comments and interactions
        comments = self.db.query(Comment).filter(
            and_(
                Comment.workspace_id == workspace_id,
                Comment.user_id == member_id,
                Comment.created_at >= period_start,
                Comment.created_at <= period_end
            )
        ).count()
        
        # Get member's conflict resolutions
        resolved_conflicts = self.db.query(ConflictResolution).filter(
            and_(
                ConflictResolution.workspace_id == workspace_id,
                ConflictResolution.resolved_by == member_id,
                ConflictResolution.resolved_at >= period_start,
                ConflictResolution.resolved_at <= period_end
            )
        ).count()
        
        # Calculate score based on participation
        score = (comments * 2) + (resolved_conflicts * 5)
        return min(100, score)
    
    async def _get_member_performance_trends(
        self,
        workspace_id: UUID,
        member_id: UUID,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, List[float]]:
        """Get performance trends for member"""
        # This would analyze historical data to show trends
        # For now, return placeholder data
        return {
            'activity_trend': [8, 12, 15, 10, 18, 20, 16],
            'quality_trend': [85, 88, 90, 87, 92, 89, 91]
        }
    
    async def _calculate_workspace_health_score(self, workspace_id: UUID) -> float:
        """Calculate overall workspace health score"""
        # Get recent metrics
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        
        # Activity level
        recent_activities = self.db.query(ActivityFeed).filter(
            and_(
                ActivityFeed.workspace_id == workspace_id,
                ActivityFeed.created_at >= yesterday
            )
        ).count()
        
        # Conflict resolution
        recent_conflicts = self.db.query(ConflictResolution).filter(
            and_(
                ConflictResolution.workspace_id == workspace_id,
                ConflictResolution.detected_at >= yesterday
            )
        ).count()
        
        # Simple health score calculation
        activity_score = min(50, recent_activities * 5)
        conflict_penalty = recent_conflicts * 10
        
        health_score = max(0, min(100, activity_score - conflict_penalty + 50))
        return round(health_score, 1)
    
    async def _store_analytics_report(self, workspace_id: UUID, report: AnalyticsReport) -> None:
        """Store analytics report in database"""
        # Update or create team analytics record
        period_start = datetime.fromisoformat(report.period_start)
        period_end = datetime.fromisoformat(report.period_end)
        
        analytics = self.db.query(TeamAnalytics).filter(
            and_(
                TeamAnalytics.workspace_id == workspace_id,
                TeamAnalytics.period_start == period_start,
                TeamAnalytics.period_end == period_end
            )
        ).first()
        
        if not analytics:
            analytics = TeamAnalytics(
                workspace_id=workspace_id,
                period_start=period_start,
                period_end=period_end
            )
            self.db.add(analytics)
        
        # Update analytics data
        analytics.member_contributions = report.detailed_metrics
        analytics.activity_breakdown = report.summary_metrics
        analytics.performance_trends = {
            'insights': [i.__dict__ if hasattr(i, '__dict__') else i for i in report.insights],
            'recommendations': [r.__dict__ if hasattr(r, '__dict__') else r for r in report.recommendations]
        }
        
        # Update summary metrics
        summary_metrics = report.summary_metrics
        analytics.active_members = summary_metrics.get('unique_active_users', 0)
        analytics.total_activities = summary_metrics.get('total_activities', 0)
        analytics.collaboration_score = int(summary_metrics.get('collaboration_score', 0))
        
        self.db.commit()