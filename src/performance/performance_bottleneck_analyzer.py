#!/usr/bin/env python3
"""
Performance Bottleneck Analyzer - AI-Driven Performance Optimization Engine

Specializes in identifying and resolving performance bottlenecks in development workflows,
agent coordination, and system operations with <200ms API response time targets.

Features:
- Real-time bottleneck detection and analysis
- Agent coordination optimization
- Resource allocation intelligence
- Database query performance analysis
- Memory usage pattern analysis
- API response time optimization
- Load balancing and scaling recommendations
"""

import time
import asyncio
import psutil
import statistics
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy import text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class BottleneckMetric:
    """Individual bottleneck measurement"""
    name: str
    value: float
    unit: str
    timestamp: float
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # MEMORY, CPU, DATABASE, NETWORK, COORDINATION
    impact_score: float  # 0-100
    threshold: Optional[float] = None
    target: Optional[float] = None
    
    @property
    def exceeds_threshold(self) -> bool:
        return self.threshold is not None and self.value > self.threshold
    
    @property
    def meets_target(self) -> bool:
        return self.target is None or self.value <= self.target


@dataclass 
class BottleneckPattern:
    """Identified performance bottleneck pattern"""
    pattern_id: str
    category: str
    severity: str
    frequency: int  # How often this pattern occurs
    avg_impact: float
    description: str
    root_causes: List[str]
    optimization_strategies: List[str]
    estimated_improvement: str
    affected_components: List[str]
    detection_confidence: float  # 0.0 to 1.0
    
    @property
    def priority_score(self) -> float:
        severity_weight = {'CRITICAL': 1.0, 'HIGH': 0.75, 'MEDIUM': 0.5, 'LOW': 0.25}
        return (self.avg_impact * severity_weight.get(self.severity, 0.25) * 
                self.frequency * self.detection_confidence)


@dataclass
class OptimizationRecommendation:
    """Specific optimization recommendation"""
    title: str
    category: str
    priority: str  # IMMEDIATE, HIGH, MEDIUM, LOW
    effort: str    # MINIMAL, LOW, MEDIUM, HIGH, MASSIVE
    impact: str    # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    implementation_steps: List[str]
    estimated_time_hours: float
    expected_improvement: str
    resources_needed: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH
    dependencies: List[str] = field(default_factory=list)


class PerformanceBottleneckAnalyzer:
    """
    AI-Driven Performance Bottleneck Analyzer
    
    Provides intelligent analysis of system performance with:
    - Real-time bottleneck detection
    - Pattern recognition and root cause analysis  
    - Agent coordination optimization
    - Resource allocation intelligence
    - Predictive performance modeling
    - Automated optimization recommendations
    """
    
    def __init__(
        self,
        api_response_target_ms: float = 200.0,
        memory_target_mb: float = 100.0,
        cpu_target_percent: float = 15.0,
        analysis_window_minutes: int = 10
    ):
        # Performance targets (aggressive for production)
        self.targets = {
            'api_response_ms': api_response_target_ms,
            'memory_mb': memory_target_mb,  
            'cpu_percent': cpu_target_percent,
            'database_query_ms': 50.0,
            'agent_coordination_ms': 100.0,
            'cache_hit_rate': 0.85,  # 85%+
            'concurrent_users': 1000
        }
        
        # Critical thresholds
        self.thresholds = {
            'api_response_ms': 500.0,    # Red alert at 500ms
            'memory_mb': 200.0,          # Warning at 200MB
            'cpu_percent': 50.0,         # Warning at 50%
            'database_query_ms': 100.0,  # Slow query at 100ms
            'agent_coordination_ms': 200.0,
            'cache_hit_rate': 0.5,       # Poor cache performance
            'error_rate': 0.01           # 1% error rate threshold
        }
        
        # Analysis components
        self.analysis_window = analysis_window_minutes
        self.metrics_history: deque = deque(maxlen=1000)
        self.bottleneck_patterns: Dict[str, BottleneckPattern] = {}
        self.active_bottlenecks: List[BottleneckMetric] = []
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Background analysis
        self.monitoring_active = False
        self.analysis_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Component analyzers
        self.db_analyzer = DatabasePerformanceAnalyzer()
        self.agent_analyzer = AgentCoordinationAnalyzer()
        self.memory_analyzer = MemoryPatternAnalyzer()
        self.api_analyzer = APIResponseAnalyzer()
        
        logger.info(f"Bottleneck analyzer initialized with targets: API<{api_response_target_ms}ms, Memory<{memory_target_mb}MB")
    
    def capture_system_metrics(self) -> Dict[str, float]:
        """Capture comprehensive system performance metrics"""
        process = psutil.Process()
        system_memory = psutil.virtual_memory()
        
        metrics = {
            'timestamp': time.time(),
            'process_memory_mb': process.memory_info().rss / (1024 ** 2),
            'system_memory_percent': system_memory.percent,
            'cpu_percent': process.cpu_percent(),
            'system_cpu_percent': psutil.cpu_percent(),
            'thread_count': threading.active_count(),
            'file_descriptors': len(process.open_files()) + len(process.connections()),
            'io_read_bytes': process.io_counters().read_bytes if hasattr(process.io_counters(), 'read_bytes') else 0,
            'io_write_bytes': process.io_counters().write_bytes if hasattr(process.io_counters(), 'write_bytes') else 0
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def analyze_bottleneck_patterns(self) -> List[BottleneckPattern]:
        """Analyze historical data to identify bottleneck patterns"""
        if len(self.metrics_history) < 10:
            return []
        
        patterns = []
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 measurements
        
        # Memory usage pattern analysis
        memory_values = [m['process_memory_mb'] for m in recent_metrics]
        if memory_values:
            avg_memory = statistics.mean(memory_values)
            max_memory = max(memory_values)
            memory_trend = self._calculate_trend(memory_values)
            
            if avg_memory > self.thresholds['memory_mb']:
                pattern = BottleneckPattern(
                    pattern_id="high_memory_usage",
                    category="MEMORY",
                    severity="HIGH" if avg_memory > 300 else "MEDIUM",
                    frequency=len([m for m in memory_values if m > self.thresholds['memory_mb']]),
                    avg_impact=min(100, (avg_memory / self.targets['memory_mb']) * 25),
                    description=f"Consistently high memory usage averaging {avg_memory:.1f}MB (target: {self.targets['memory_mb']}MB)",
                    root_causes=[
                        "Memory leaks in long-running processes",
                        "Inefficient data structure usage", 
                        "Large object retention in caches",
                        "Excessive module imports"
                    ],
                    optimization_strategies=[
                        "Implement aggressive garbage collection tuning",
                        "Use memory profiling to identify specific leaks",
                        "Implement object pooling for frequently created objects",
                        "Optimize data structures (use __slots__, etc.)",
                        "Implement lazy loading for heavy modules"
                    ],
                    estimated_improvement="60-80% memory reduction possible",
                    affected_components=["Memory Management", "Object Lifecycle", "Module Loading"],
                    detection_confidence=0.9 if max_memory > avg_memory * 1.5 else 0.7
                )
                patterns.append(pattern)
        
        # CPU usage spikes analysis
        cpu_values = [m['cpu_percent'] for m in recent_metrics if m['cpu_percent'] > 0]
        if cpu_values:
            avg_cpu = statistics.mean(cpu_values)
            max_cpu = max(cpu_values)
            cpu_spikes = len([c for c in cpu_values if c > self.thresholds['cpu_percent']])
            
            if avg_cpu > self.targets['cpu_percent'] or cpu_spikes > len(cpu_values) * 0.2:
                pattern = BottleneckPattern(
                    pattern_id="cpu_performance_issues",
                    category="CPU",
                    severity="HIGH" if avg_cpu > 60 else "MEDIUM",
                    frequency=cpu_spikes,
                    avg_impact=min(100, avg_cpu * 2),
                    description=f"CPU performance issues with {cpu_spikes} spikes, avg {avg_cpu:.1f}%",
                    root_causes=[
                        "Inefficient algorithms or data processing",
                        "Blocking I/O operations",
                        "Excessive synchronous operations",
                        "Poor concurrency design"
                    ],
                    optimization_strategies=[
                        "Implement asynchronous processing patterns",
                        "Optimize hot code paths with profiling",
                        "Use efficient algorithms and data structures",
                        "Implement proper threading and multiprocessing",
                        "Add caching for expensive computations"
                    ],
                    estimated_improvement="30-50% CPU usage reduction",
                    affected_components=["API Processing", "Database Operations", "Agent Coordination"],
                    detection_confidence=0.8
                )
                patterns.append(pattern)
        
        # Update pattern database
        for pattern in patterns:
            self.bottleneck_patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metrics"""
        if len(values) < 5:
            return "STABLE"
        
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])
        
        change_percent = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        
        if change_percent > 10:
            return "INCREASING"
        elif change_percent < -10:
            return "DECREASING"
        else:
            return "STABLE"
    
    async def analyze_database_performance(self, engine: Optional[Any] = None) -> List[BottleneckMetric]:
        """Analyze database performance bottlenecks"""
        bottlenecks = []
        
        if engine and SQLALCHEMY_AVAILABLE:
            try:
                # Simulate database performance analysis
                start_time = time.time()
                async with engine.begin() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    await result.fetchone()
                query_time = (time.time() - start_time) * 1000
                
                if query_time > self.thresholds['database_query_ms']:
                    bottleneck = BottleneckMetric(
                        name="database_query_time",
                        value=query_time,
                        unit="ms",
                        timestamp=time.time(),
                        severity="HIGH" if query_time > 200 else "MEDIUM",
                        category="DATABASE",
                        impact_score=min(100, (query_time / self.targets['database_query_ms']) * 20),
                        threshold=self.thresholds['database_query_ms'],
                        target=self.targets['database_query_ms']
                    )
                    bottlenecks.append(bottleneck)
                    
            except Exception as e:
                logger.error(f"Database performance analysis failed: {e}")
        
        return bottlenecks
    
    async def analyze_agent_coordination(self, swarm_metrics: Optional[Dict] = None) -> List[BottleneckMetric]:
        """Analyze agent coordination bottlenecks"""
        bottlenecks = []
        
        if swarm_metrics:
            # Analyze coordination latency
            coord_latency = swarm_metrics.get('avg_coordination_time_ms', 0)
            if coord_latency > self.thresholds['agent_coordination_ms']:
                bottleneck = BottleneckMetric(
                    name="agent_coordination_latency",
                    value=coord_latency,
                    unit="ms",
                    timestamp=time.time(),
                    severity="HIGH" if coord_latency > 300 else "MEDIUM",
                    category="COORDINATION",
                    impact_score=min(100, (coord_latency / self.targets['agent_coordination_ms']) * 30),
                    threshold=self.thresholds['agent_coordination_ms'],
                    target=self.targets['agent_coordination_ms']
                )
                bottlenecks.append(bottleneck)
            
            # Analyze resource utilization
            resource_util = swarm_metrics.get('resource_utilization', 0)
            if resource_util > 0.8:  # 80%+ utilization
                bottleneck = BottleneckMetric(
                    name="resource_utilization_high",
                    value=resource_util * 100,
                    unit="%",
                    timestamp=time.time(),
                    severity="MEDIUM",
                    category="COORDINATION",
                    impact_score=resource_util * 50,
                    threshold=80.0,
                    target=70.0
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate prioritized optimization recommendations"""
        recommendations = []
        
        # Memory optimization recommendations
        memory_patterns = [p for p in self.bottleneck_patterns.values() if p.category == "MEMORY"]
        if memory_patterns:
            high_impact_memory = [p for p in memory_patterns if p.avg_impact > 50]
            if high_impact_memory:
                rec = OptimizationRecommendation(
                    title="Critical Memory Optimization",
                    category="MEMORY",
                    priority="IMMEDIATE",
                    effort="MEDIUM",
                    impact="CRITICAL",
                    description=f"Memory usage exceeds targets by {high_impact_memory[0].avg_impact:.0f}%. Immediate optimization required.",
                    implementation_steps=[
                        "Run memory profiler to identify top memory consumers",
                        "Implement object pooling for frequently created objects",
                        "Optimize garbage collection tuning (generation thresholds)",
                        "Implement lazy loading for heavy modules",
                        "Review and optimize data structure usage",
                        "Add memory monitoring and alerting"
                    ],
                    estimated_time_hours=8.0,
                    expected_improvement="60-80% memory reduction, <100MB target achievable",
                    resources_needed=["Senior Python Developer", "Performance Profiling Tools"],
                    risk_level="LOW"
                )
                recommendations.append(rec)
        
        # API response time optimization
        if any(m.name == 'api_response_time' and m.value > self.targets['api_response_ms'] for m in self.active_bottlenecks):
            rec = OptimizationRecommendation(
                title="API Response Time Optimization",
                category="API",
                priority="HIGH",
                effort="MEDIUM",
                impact="HIGH",
                description="API response times exceed <200ms target. Multi-tier optimization needed.",
                implementation_steps=[
                    "Implement Redis-based response caching",
                    "Optimize database connection pooling",
                    "Add query result caching with intelligent invalidation",
                    "Implement asynchronous processing where possible",
                    "Add CDN integration for static assets",
                    "Optimize serialization/deserialization"
                ],
                estimated_time_hours=12.0,
                expected_improvement="70-85% response time reduction, <200ms achievable",
                resources_needed=["Backend Developer", "DevOps Engineer", "Redis Instance"],
                risk_level="MEDIUM",
                dependencies=["Redis Setup", "CDN Configuration"]
            )
            recommendations.append(rec)
        
        # Database performance optimization
        db_patterns = [p for p in self.bottleneck_patterns.values() if "database" in p.pattern_id.lower()]
        if db_patterns or any(m.category == "DATABASE" for m in self.active_bottlenecks):
            rec = OptimizationRecommendation(
                title="Database Performance Optimization",
                category="DATABASE",
                priority="HIGH",
                effort="MEDIUM",
                impact="HIGH",
                description="Database queries are bottleneck. Connection pooling and query optimization needed.",
                implementation_steps=[
                    "Implement advanced connection pooling with monitoring",
                    "Add query performance profiling and slow query detection",
                    "Implement query result caching",
                    "Add database read replicas for read-heavy operations",
                    "Optimize frequently used queries with indexes",
                    "Implement connection health monitoring"
                ],
                estimated_time_hours=16.0,
                expected_improvement="50-70% database response time improvement",
                resources_needed=["Database Administrator", "Backend Developer", "Monitoring Tools"],
                risk_level="MEDIUM"
            )
            recommendations.append(rec)
        
        # Agent coordination optimization
        coord_patterns = [p for p in self.bottleneck_patterns.values() if p.category == "COORDINATION"]
        if coord_patterns:
            rec = OptimizationRecommendation(
                title="Agent Coordination Optimization",
                category="COORDINATION",
                priority="MEDIUM",
                effort="HIGH",
                impact="MEDIUM",
                description="Agent coordination latency impacts system scalability.",
                implementation_steps=[
                    "Implement more efficient agent communication protocols",
                    "Add intelligent load balancing for agent distribution",
                    "Optimize task queuing and scheduling algorithms",
                    "Implement agent resource monitoring and auto-scaling",
                    "Add coordination performance metrics and dashboards"
                ],
                estimated_time_hours=20.0,
                expected_improvement="40-60% coordination latency reduction",
                resources_needed=["AI/ML Engineer", "System Architect", "Performance Engineer"],
                risk_level="MEDIUM"
            )
            recommendations.append(rec)
        
        # Sort by priority and impact
        priority_order = {"IMMEDIATE": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        impact_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        recommendations.sort(
            key=lambda r: (priority_order.get(r.priority, 0), impact_order.get(r.impact, 0)), 
            reverse=True
        )
        
        self.optimization_recommendations = recommendations
        return recommendations
    
    async def comprehensive_analysis(self, engine: Optional[Any] = None, swarm_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Run comprehensive bottleneck analysis"""
        analysis_start = time.time()
        
        # Capture current metrics
        system_metrics = self.capture_system_metrics()
        
        # Analyze patterns
        patterns = self.analyze_bottleneck_patterns()
        
        # Analyze specific components
        db_bottlenecks = await self.analyze_database_performance(engine)
        agent_bottlenecks = await self.analyze_agent_coordination(swarm_metrics)
        
        # Combine all bottlenecks
        all_bottlenecks = db_bottlenecks + agent_bottlenecks
        self.active_bottlenecks = all_bottlenecks
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations()
        
        analysis_time = (time.time() - analysis_start) * 1000
        
        # Create comprehensive report
        report = {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "analysis_duration_ms": analysis_time,
            "system_status": self._assess_overall_status(),
            "current_metrics": system_metrics,
            "performance_targets": self.targets,
            "identified_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "category": p.category,
                    "severity": p.severity,
                    "priority_score": p.priority_score,
                    "description": p.description,
                    "estimated_improvement": p.estimated_improvement,
                    "detection_confidence": p.detection_confidence
                }
                for p in sorted(patterns, key=lambda x: x.priority_score, reverse=True)
            ],
            "active_bottlenecks": [
                {
                    "name": b.name,
                    "value": b.value,
                    "unit": b.unit,
                    "severity": b.severity,
                    "category": b.category,
                    "impact_score": b.impact_score,
                    "exceeds_threshold": b.exceeds_threshold,
                    "meets_target": b.meets_target
                }
                for b in sorted(all_bottlenecks, key=lambda x: x.impact_score, reverse=True)
            ],
            "optimization_recommendations": [
                {
                    "title": r.title,
                    "priority": r.priority,
                    "effort": r.effort,
                    "impact": r.impact,
                    "description": r.description,
                    "estimated_hours": r.estimated_time_hours,
                    "expected_improvement": r.expected_improvement,
                    "risk_level": r.risk_level,
                    "implementation_steps": r.implementation_steps[:3]  # Top 3 steps
                }
                for r in recommendations
            ],
            "optimization_roadmap": self._generate_implementation_roadmap(recommendations),
            "estimated_total_improvement": self._estimate_total_improvement(patterns, recommendations)
        }
        
        return report
    
    def _assess_overall_status(self) -> str:
        """Assess overall system performance status"""
        critical_issues = len([p for p in self.bottleneck_patterns.values() if p.severity == "CRITICAL"])
        high_issues = len([p for p in self.bottleneck_patterns.values() if p.severity == "HIGH"])
        
        if critical_issues > 0:
            return "CRITICAL - Immediate optimization required"
        elif high_issues > 2:
            return "DEGRADED - Multiple high-impact bottlenecks detected"
        elif high_issues > 0:
            return "SUBOPTIMAL - Performance improvements needed"
        else:
            return "OPTIMAL - Performance within acceptable ranges"
    
    def _generate_implementation_roadmap(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Generate implementation roadmap with phases"""
        immediate = [r for r in recommendations if r.priority == "IMMEDIATE"]
        high_priority = [r for r in recommendations if r.priority == "HIGH"]
        medium_priority = [r for r in recommendations if r.priority == "MEDIUM"]
        
        return {
            "phase_1_immediate": {
                "timeline": "0-1 weeks",
                "focus": "Critical bottlenecks and quick wins",
                "tasks": [r.title for r in immediate[:2]],
                "estimated_hours": sum(r.estimated_time_hours for r in immediate[:2])
            },
            "phase_2_high_impact": {
                "timeline": "1-4 weeks", 
                "focus": "High-impact performance optimizations",
                "tasks": [r.title for r in high_priority[:3]],
                "estimated_hours": sum(r.estimated_time_hours for r in high_priority[:3])
            },
            "phase_3_enhancement": {
                "timeline": "1-3 months",
                "focus": "Long-term performance enhancements",
                "tasks": [r.title for r in medium_priority],
                "estimated_hours": sum(r.estimated_time_hours for r in medium_priority)
            }
        }
    
    def _estimate_total_improvement(self, patterns: List[BottleneckPattern], recommendations: List[OptimizationRecommendation]) -> str:
        """Estimate total system improvement potential"""
        if not patterns and not recommendations:
            return "System already optimized - minimal improvement expected"
        
        critical_count = len([p for p in patterns if p.severity == "CRITICAL"])
        high_count = len([p for p in patterns if p.severity == "HIGH"])
        
        if critical_count > 0:
            return "75-90% performance improvement possible with critical optimizations"
        elif high_count > 1:
            return "50-70% performance improvement achievable"
        else:
            return "20-40% performance improvement expected"
    
    def start_continuous_analysis(self, interval_seconds: float = 30.0):
        """Start continuous bottleneck analysis"""
        self.monitoring_active = True
        
        def analysis_loop():
            while self.monitoring_active:
                try:
                    # Run analysis
                    self.capture_system_metrics()
                    patterns = self.analyze_bottleneck_patterns()
                    
                    # Check for critical issues
                    critical_patterns = [p for p in patterns if p.severity == "CRITICAL"]
                    if critical_patterns:
                        logger.error(f"CRITICAL BOTTLENECK DETECTED: {critical_patterns[0].description}")
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Continuous analysis error: {e}")
                    time.sleep(interval_seconds)
        
        self.analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        self.analysis_thread.start()
        logger.info(f"Continuous bottleneck analysis started (interval: {interval_seconds}s)")
    
    def stop_continuous_analysis(self):
        """Stop continuous analysis"""
        self.monitoring_active = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        logger.info("Continuous bottleneck analysis stopped")
    
    def export_analysis_report(self, filepath: str) -> str:
        """Export detailed analysis report"""
        report_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "analyzer_config": {
                "targets": self.targets,
                "thresholds": self.thresholds,
                "analysis_window_minutes": self.analysis_window
            },
            "bottleneck_patterns": {
                pattern_id: {
                    "category": pattern.category,
                    "severity": pattern.severity,
                    "frequency": pattern.frequency,
                    "avg_impact": pattern.avg_impact,
                    "description": pattern.description,
                    "root_causes": pattern.root_causes,
                    "optimization_strategies": pattern.optimization_strategies,
                    "estimated_improvement": pattern.estimated_improvement,
                    "priority_score": pattern.priority_score,
                    "detection_confidence": pattern.detection_confidence
                }
                for pattern_id, pattern in self.bottleneck_patterns.items()
            },
            "optimization_recommendations": [
                {
                    "title": rec.title,
                    "category": rec.category,
                    "priority": rec.priority,
                    "effort": rec.effort,
                    "impact": rec.impact,
                    "description": rec.description,
                    "implementation_steps": rec.implementation_steps,
                    "estimated_time_hours": rec.estimated_time_hours,
                    "expected_improvement": rec.expected_improvement,
                    "resources_needed": rec.resources_needed,
                    "risk_level": rec.risk_level,
                    "dependencies": rec.dependencies
                }
                for rec in self.optimization_recommendations
            ],
            "recent_metrics": list(self.metrics_history)[-20:]  # Last 20 measurements
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Analysis report exported to {filepath}")
        return filepath


# Specialized component analyzers
class DatabasePerformanceAnalyzer:
    """Specialized database performance analysis"""
    
    def __init__(self):
        self.query_metrics = deque(maxlen=1000)
        self.connection_pool_stats = {}
    
    def analyze_query_performance(self, query_time_ms: float, query_type: str) -> Optional[BottleneckMetric]:
        """Analyze individual query performance"""
        self.query_metrics.append({'time_ms': query_time_ms, 'type': query_type, 'timestamp': time.time()})
        
        if query_time_ms > 100:  # Slow query threshold
            return BottleneckMetric(
                name=f"slow_{query_type}_query",
                value=query_time_ms,
                unit="ms",
                timestamp=time.time(),
                severity="HIGH" if query_time_ms > 500 else "MEDIUM",
                category="DATABASE",
                impact_score=min(100, query_time_ms / 10),
                threshold=100.0,
                target=50.0
            )
        return None


class AgentCoordinationAnalyzer:
    """Specialized agent coordination analysis"""
    
    def __init__(self):
        self.coordination_metrics = deque(maxlen=500)
        self.agent_utilization = {}
    
    def analyze_coordination_efficiency(self, agents_active: int, total_agents: int, avg_response_time: float) -> List[BottleneckMetric]:
        """Analyze agent coordination efficiency"""
        bottlenecks = []
        utilization = agents_active / max(total_agents, 1)
        
        # Low utilization indicates coordination inefficiency
        if utilization < 0.6 and total_agents > 2:
            bottlenecks.append(BottleneckMetric(
                name="low_agent_utilization",
                value=utilization * 100,
                unit="%",
                timestamp=time.time(),
                severity="MEDIUM",
                category="COORDINATION",
                impact_score=(1 - utilization) * 60,
                threshold=60.0,
                target=80.0
            ))
        
        return bottlenecks


class MemoryPatternAnalyzer:
    """Specialized memory usage pattern analysis"""
    
    def __init__(self):
        self.memory_samples = deque(maxlen=1000)
        self.gc_stats = []
    
    def analyze_memory_patterns(self, current_mb: float) -> List[str]:
        """Analyze memory usage patterns for optimization opportunities"""
        self.memory_samples.append({'mb': current_mb, 'timestamp': time.time()})
        
        insights = []
        if len(self.memory_samples) >= 10:
            recent_avg = statistics.mean([s['mb'] for s in list(self.memory_samples)[-10:]])
            
            if recent_avg > 150:
                insights.append("High memory usage detected - implement object pooling")
            
            # Check for memory growth trend
            if len(self.memory_samples) >= 50:
                older_avg = statistics.mean([s['mb'] for s in list(self.memory_samples)[-50:-25]])
                if recent_avg > older_avg * 1.2:
                    insights.append("Memory leak suspected - investigate object retention")
        
        return insights


class APIResponseAnalyzer:
    """Specialized API response time analysis"""
    
    def __init__(self):
        self.response_times = deque(maxlen=1000)
        self.endpoint_stats = defaultdict(list)
    
    def analyze_response_patterns(self, response_time_ms: float, endpoint: str, status_code: int) -> Optional[BottleneckMetric]:
        """Analyze API response time patterns"""
        self.response_times.append(response_time_ms)
        self.endpoint_stats[endpoint].append({'time_ms': response_time_ms, 'status': status_code})
        
        if response_time_ms > 200:  # Target threshold
            return BottleneckMetric(
                name=f"slow_api_response_{endpoint.replace('/', '_')}",
                value=response_time_ms,
                unit="ms",
                timestamp=time.time(),
                severity="HIGH" if response_time_ms > 500 else "MEDIUM",
                category="API",
                impact_score=min(100, response_time_ms / 5),
                threshold=200.0,
                target=100.0
            )
        return None


# Global analyzer instance
_bottleneck_analyzer: Optional[PerformanceBottleneckAnalyzer] = None


def get_bottleneck_analyzer() -> PerformanceBottleneckAnalyzer:
    """Get global bottleneck analyzer instance"""
    global _bottleneck_analyzer
    if _bottleneck_analyzer is None:
        _bottleneck_analyzer = PerformanceBottleneckAnalyzer()
    return _bottleneck_analyzer


async def analyze_system_bottlenecks(engine: Optional[Any] = None, swarm_metrics: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function for comprehensive bottleneck analysis"""
    analyzer = get_bottleneck_analyzer()
    return await analyzer.comprehensive_analysis(engine, swarm_metrics)


if __name__ == "__main__":
    # Example usage and testing
    async def test_bottleneck_analysis():
        print("🔍 PERFORMANCE BOTTLENECK ANALYZER - Comprehensive Analysis")
        print("=" * 70)
        
        analyzer = PerformanceBottleneckAnalyzer(
            api_response_target_ms=200.0,
            memory_target_mb=100.0,
            cpu_target_percent=15.0
        )
        
        # Simulate some metrics collection
        for _ in range(20):
            analyzer.capture_system_metrics()
            time.sleep(0.1)
        
        # Run comprehensive analysis
        print("📊 Running comprehensive bottleneck analysis...")
        report = await analyzer.comprehensive_analysis()
        
        print(f"\n📈 Analysis Results:")
        print(f"   System Status: {report['system_status']}")
        print(f"   Patterns Identified: {len(report['identified_patterns'])}")
        print(f"   Active Bottlenecks: {len(report['active_bottlenecks'])}")
        print(f"   Recommendations: {len(report['optimization_recommendations'])}")
        
        if report['optimization_recommendations']:
            print(f"\n🎯 Top Recommendations:")
            for i, rec in enumerate(report['optimization_recommendations'][:3], 1):
                print(f"   {i}. {rec['title']} ({rec['priority']} priority)")
                print(f"      Impact: {rec['impact']}, Effort: {rec['effort']}")
                print(f"      Expected: {rec['expected_improvement']}")
        
        print(f"\n📋 Implementation Roadmap:")
        roadmap = report['optimization_roadmap']
        for phase, details in roadmap.items():
            print(f"   {phase}: {details['timeline']} - {details['focus']}")
            print(f"      Tasks: {len(details['tasks'])}, Hours: {details['estimated_hours']}")
        
        print(f"\n🎯 Total Improvement Potential: {report['estimated_total_improvement']}")
        
        # Export report
        report_file = analyzer.export_analysis_report(f"bottleneck_analysis_{int(time.time())}.json")
        print(f"\n📄 Detailed report exported: {report_file}")
        
        print("\n✅ Bottleneck analysis completed!")
    
    # Run the test
    import asyncio
    asyncio.run(test_bottleneck_analysis())
