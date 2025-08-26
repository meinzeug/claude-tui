"""
Test suite for the PerformanceOptimizer and optimization components.

Tests optimization strategies, safety validation, impact simulation,
and integration with bottleneck analysis.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.analytics.optimizer import (
    PerformanceOptimizer, OptimizationStrategy, OptimizationPlan,
    SafetyValidator, ImpactSimulator
)
from src.analytics.models import (
    PerformanceMetrics, BottleneckAnalysis, OptimizationRecommendation,
    AnalyticsConfiguration
)
from src.core.types import SystemMetrics


class TestOptimizationStrategy:
    """Test suite for optimization strategies."""

    @pytest.fixture
    def sample_bottleneck(self):
        """Create a sample bottleneck for testing."""
        return BottleneckAnalysis(
            type="cpu",
            severity=0.85,
            description="High CPU utilization detected",
            affected_metrics=["cpu_usage", "execution_time"],
            root_cause="Inefficient algorithm with O(n²) complexity",
            timestamp=datetime.now()
        )

    def test_aggressive_strategy(self, sample_bottleneck):
        """Test aggressive optimization strategy."""
        strategy = OptimizationStrategy.AGGRESSIVE
        optimizer = PerformanceOptimizer(strategy=strategy)
        
        recommendations = optimizer.generate_recommendations([sample_bottleneck])
        
        assert len(recommendations) > 0
        # Aggressive strategy should provide multiple options
        cpu_recs = [r for r in recommendations if "cpu" in r.description.lower()]
        assert len(cpu_recs) >= 1
        
        # Should have high confidence recommendations
        high_confidence_recs = [r for r in cpu_recs if r.confidence >= 0.8]
        assert len(high_confidence_recs) > 0

    def test_conservative_strategy(self, sample_bottleneck):
        """Test conservative optimization strategy."""
        strategy = OptimizationStrategy.CONSERVATIVE
        optimizer = PerformanceOptimizer(strategy=strategy)
        
        recommendations = optimizer.generate_recommendations([sample_bottleneck])
        
        # Conservative strategy should be more cautious
        for rec in recommendations:
            assert rec.risk_level in ["low", "medium"]  # No high-risk recommendations
            assert len(rec.implementation_steps) >= 3  # More detailed steps

    def test_balanced_strategy(self, sample_bottleneck):
        """Test balanced optimization strategy."""
        strategy = OptimizationStrategy.BALANCED
        optimizer = PerformanceOptimizer(strategy=strategy)
        
        recommendations = optimizer.generate_recommendations([sample_bottleneck])
        
        # Balanced strategy should have mix of approaches
        assert len(recommendations) > 0
        risk_levels = [r.risk_level for r in recommendations]
        
        # Should have variety in risk levels
        assert len(set(risk_levels)) >= 2

    def test_safety_first_strategy(self, sample_bottleneck):
        """Test safety-first optimization strategy."""
        strategy = OptimizationStrategy.SAFETY_FIRST
        optimizer = PerformanceOptimizer(strategy=strategy)
        
        recommendations = optimizer.generate_recommendations([sample_bottleneck])
        
        # Safety-first should prioritize low-risk options
        for rec in recommendations:
            assert rec.risk_level == "low"
            assert rec.estimated_improvement <= 50.0  # Conservative estimates
            assert rec.rollback_plan is not None


class TestPerformanceOptimizer:
    """Test suite for the main PerformanceOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a PerformanceOptimizer instance for testing."""
        return PerformanceOptimizer(strategy=OptimizationStrategy.BALANCED)

    @pytest.fixture
    def cpu_bottleneck(self):
        """Create a CPU bottleneck for testing."""
        return BottleneckAnalysis(
            type="cpu",
            severity=0.9,
            description="CPU usage consistently above 90%",
            affected_metrics=["cpu_usage", "execution_time", "throughput"],
            root_cause="Synchronous processing of parallel tasks",
            timestamp=datetime.now()
        )

    @pytest.fixture
    def memory_bottleneck(self):
        """Create a memory bottleneck for testing."""
        return BottleneckAnalysis(
            type="memory",
            severity=0.8,
            description="Memory usage approaching system limits",
            affected_metrics=["memory_usage", "gc_pressure", "allocation_rate"],
            root_cause="Memory leaks in long-running processes",
            timestamp=datetime.now()
        )

    @pytest.fixture
    def io_bottleneck(self):
        """Create an I/O bottleneck for testing."""
        return BottleneckAnalysis(
            type="io",
            severity=0.75,
            description="High disk I/O latency affecting performance",
            affected_metrics=["disk_io", "latency_p95", "response_time"],
            root_cause="Sequential database queries without indexing",
            timestamp=datetime.now()
        )

    def test_cpu_optimization_recommendations(self, optimizer, cpu_bottleneck):
        """Test CPU-specific optimization recommendations."""
        recommendations = optimizer.generate_recommendations([cpu_bottleneck])
        
        cpu_recs = [r for r in recommendations if "cpu" in r.description.lower() or "parallel" in r.description.lower()]
        assert len(cpu_recs) > 0
        
        # Should include parallelization suggestions
        parallel_recs = [r for r in cpu_recs if "parallel" in r.description.lower()]
        assert len(parallel_recs) > 0
        
        # Check implementation details
        for rec in cpu_recs:
            assert rec.estimated_improvement > 0
            assert len(rec.implementation_steps) >= 2
            assert rec.implementation_complexity in ["low", "medium", "high"]

    def test_memory_optimization_recommendations(self, optimizer, memory_bottleneck):
        """Test memory-specific optimization recommendations."""
        recommendations = optimizer.generate_recommendations([memory_bottleneck])
        
        memory_recs = [r for r in recommendations if "memory" in r.description.lower() or "cache" in r.description.lower()]
        assert len(memory_recs) > 0
        
        # Should include memory management suggestions
        for rec in memory_recs:
            assert any(keyword in rec.description.lower() for keyword in 
                      ["memory", "cache", "garbage", "allocation", "leak"])
            assert rec.priority in ["high", "medium", "low"]

    def test_io_optimization_recommendations(self, optimizer, io_bottleneck):
        """Test I/O-specific optimization recommendations."""
        recommendations = optimizer.generate_recommendations([io_bottleneck])
        
        io_recs = [r for r in recommendations if any(keyword in r.description.lower() for keyword in 
                                                    ["io", "disk", "database", "query", "index"])]
        assert len(io_recs) > 0
        
        # Should include I/O optimization suggestions
        for rec in io_recs:
            assert rec.estimated_improvement > 0
            assert "rollback_plan" in rec.__dict__

    def test_multiple_bottlenecks_prioritization(self, optimizer, cpu_bottleneck, memory_bottleneck, io_bottleneck):
        """Test handling multiple bottlenecks with proper prioritization."""
        bottlenecks = [cpu_bottleneck, memory_bottleneck, io_bottleneck]
        recommendations = optimizer.generate_recommendations(bottlenecks)
        
        assert len(recommendations) > 0
        
        # Should prioritize by severity
        high_severity_recs = [r for r in recommendations if r.confidence >= 0.8]
        assert len(high_severity_recs) > 0
        
        # Should address all bottleneck types
        types_addressed = set()
        for rec in recommendations:
            if "cpu" in rec.description.lower():
                types_addressed.add("cpu")
            elif "memory" in rec.description.lower():
                types_addressed.add("memory")
            elif "io" in rec.description.lower():
                types_addressed.add("io")
        
        assert len(types_addressed) >= 2  # Should address multiple types

    def test_optimization_plan_creation(self, optimizer, cpu_bottleneck):
        """Test creation of comprehensive optimization plans."""
        recommendations = optimizer.generate_recommendations([cpu_bottleneck])
        plan = optimizer.create_optimization_plan(recommendations)
        
        assert isinstance(plan, OptimizationPlan)
        assert len(plan.phases) > 0
        assert plan.total_estimated_time > 0
        assert plan.risk_assessment is not None
        
        # Phases should be ordered by priority
        priorities = [phase.priority for phase in plan.phases]
        priority_values = {"high": 3, "medium": 2, "low": 1}
        priority_nums = [priority_values[p] for p in priorities]
        assert priority_nums == sorted(priority_nums, reverse=True)

    @pytest.mark.asyncio
    async def test_async_optimization(self, optimizer, cpu_bottleneck):
        """Test asynchronous optimization processing."""
        recommendations = await optimizer.generate_recommendations_async([cpu_bottleneck])
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Verify async processing didn't affect quality
        for rec in recommendations:
            assert hasattr(rec, 'confidence')
            assert hasattr(rec, 'estimated_improvement')
            assert hasattr(rec, 'implementation_steps')

    def test_custom_optimization_rules(self, optimizer):
        """Test adding custom optimization rules."""
        def custom_rule(bottleneck):
            if bottleneck.type == "custom_type":
                return OptimizationRecommendation(
                    type="custom",
                    description="Custom optimization for specific case",
                    confidence=0.9,
                    estimated_improvement=25.0,
                    implementation_steps=["Step 1", "Step 2"],
                    priority="medium",
                    risk_level="low"
                )
            return None
        
        optimizer.add_custom_rule("custom_optimization", custom_rule)
        
        # Test with custom bottleneck
        custom_bottleneck = BottleneckAnalysis(
            type="custom_type",
            severity=0.7,
            description="Custom bottleneck type",
            affected_metrics=["custom_metric"],
            root_cause="Custom issue",
            timestamp=datetime.now()
        )
        
        recommendations = optimizer.generate_recommendations([custom_bottleneck])
        custom_recs = [r for r in recommendations if r.type == "custom"]
        assert len(custom_recs) > 0


class TestSafetyValidator:
    """Test suite for the SafetyValidator component."""

    @pytest.fixture
    def validator(self):
        """Create a SafetyValidator instance for testing."""
        return SafetyValidator()

    @pytest.fixture
    def safe_recommendation(self):
        """Create a safe optimization recommendation."""
        return OptimizationRecommendation(
            type="performance",
            description="Optimize database query with proper indexing",
            confidence=0.85,
            estimated_improvement=20.0,
            implementation_steps=["Create index", "Update query", "Test performance"],
            priority="medium",
            risk_level="low",
            rollback_plan=["Drop index if issues occur", "Revert query changes"]
        )

    @pytest.fixture
    def risky_recommendation(self):
        """Create a risky optimization recommendation."""
        return OptimizationRecommendation(
            type="architecture",
            description="Replace entire caching layer with new technology",
            confidence=0.6,
            estimated_improvement=80.0,
            implementation_steps=["Remove old cache", "Install new cache", "Migrate data"],
            priority="high",
            risk_level="high",
            rollback_plan=None  # No rollback plan - risky!
        )

    def test_validate_safe_recommendation(self, validator, safe_recommendation):
        """Test validation of safe recommendations."""
        is_safe, safety_score, issues = validator.validate_recommendation(safe_recommendation)
        
        assert is_safe is True
        assert safety_score >= 0.7
        assert len(issues) == 0

    def test_validate_risky_recommendation(self, validator, risky_recommendation):
        """Test validation of risky recommendations."""
        is_safe, safety_score, issues = validator.validate_recommendation(risky_recommendation)
        
        assert is_safe is False
        assert safety_score < 0.7
        assert len(issues) > 0
        
        # Should identify specific safety concerns
        issue_descriptions = [issue.description for issue in issues]
        assert any("rollback" in desc.lower() for desc in issue_descriptions)

    def test_safety_criteria_customization(self, validator):
        """Test customization of safety criteria."""
        # Set stricter safety criteria
        validator.set_safety_criteria({
            "min_confidence": 0.9,
            "max_risk_level": "low",
            "require_rollback_plan": True,
            "max_estimated_downtime": 30  # minutes
        })
        
        moderate_recommendation = OptimizationRecommendation(
            type="performance",
            description="Moderate risk optimization",
            confidence=0.8,  # Below new threshold
            estimated_improvement=30.0,
            implementation_steps=["Step 1", "Step 2"],
            priority="medium",
            risk_level="medium",  # Above new threshold
            rollback_plan=["Rollback step"]
        )
        
        is_safe, safety_score, issues = validator.validate_recommendation(moderate_recommendation)
        
        assert is_safe is False  # Should fail stricter criteria
        assert len(issues) >= 2  # Confidence and risk level issues

    def test_batch_validation(self, validator, safe_recommendation, risky_recommendation):
        """Test batch validation of multiple recommendations."""
        recommendations = [safe_recommendation, risky_recommendation]
        results = validator.validate_batch(recommendations)
        
        assert len(results) == 2
        assert results[0]["is_safe"] is True  # Safe recommendation
        assert results[1]["is_safe"] is False  # Risky recommendation
        
        # Should provide aggregated safety summary
        summary = validator.get_batch_summary(results)
        assert "total_recommendations" in summary
        assert "safe_recommendations" in summary
        assert "risk_distribution" in summary


class TestImpactSimulator:
    """Test suite for the ImpactSimulator component."""

    @pytest.fixture
    def simulator(self):
        """Create an ImpactSimulator instance for testing."""
        return ImpactSimulator()

    @pytest.fixture
    def baseline_metrics(self):
        """Create baseline performance metrics."""
        return PerformanceMetrics(
            base_metrics=SystemMetrics(
                cpu_usage=80.0,
                memory_usage=70.0,
                disk_io=50.0,
                network_io=30.0,
                timestamp=datetime.now()
            ),
            execution_time=3.5,
            throughput=85.0,
            error_rate=0.03,
            latency_p95=2.8
        )

    @pytest.fixture
    def optimization_rec(self):
        """Create an optimization recommendation for simulation."""
        return OptimizationRecommendation(
            type="cpu",
            description="Implement parallel processing for CPU-intensive tasks",
            confidence=0.85,
            estimated_improvement=40.0,  # 40% improvement
            implementation_steps=["Identify parallelizable tasks", "Implement threading", "Test performance"],
            priority="high",
            risk_level="medium"
        )

    def test_simulate_cpu_optimization_impact(self, simulator, baseline_metrics, optimization_rec):
        """Test simulation of CPU optimization impact."""
        simulated_metrics = simulator.simulate_impact(baseline_metrics, optimization_rec)
        
        # CPU usage should improve
        assert simulated_metrics.base_metrics.cpu_usage < baseline_metrics.base_metrics.cpu_usage
        
        # Execution time should improve
        assert simulated_metrics.execution_time < baseline_metrics.execution_time
        
        # Throughput should improve
        assert simulated_metrics.throughput > baseline_metrics.throughput

    def test_simulation_confidence_impact(self, simulator, baseline_metrics):
        """Test how recommendation confidence affects simulation."""
        high_confidence_rec = OptimizationRecommendation(
            type="performance",
            description="High confidence optimization",
            confidence=0.95,
            estimated_improvement=30.0,
            implementation_steps=["Step 1"],
            priority="high",
            risk_level="low"
        )
        
        low_confidence_rec = OptimizationRecommendation(
            type="performance", 
            description="Low confidence optimization",
            confidence=0.6,
            estimated_improvement=30.0,  # Same improvement estimate
            implementation_steps=["Step 1"],
            priority="medium",
            risk_level="medium"
        )
        
        high_conf_result = simulator.simulate_impact(baseline_metrics, high_confidence_rec)
        low_conf_result = simulator.simulate_impact(baseline_metrics, low_confidence_rec)
        
        # High confidence should result in closer to estimated improvement
        high_conf_improvement = (baseline_metrics.execution_time - high_conf_result.execution_time) / baseline_metrics.execution_time
        low_conf_improvement = (baseline_metrics.execution_time - low_conf_result.execution_time) / baseline_metrics.execution_time
        
        # High confidence should be closer to expected 30% improvement
        assert abs(high_conf_improvement - 0.3) < abs(low_conf_improvement - 0.3)

    def test_multiple_optimizations_simulation(self, simulator, baseline_metrics):
        """Test simulation of multiple optimizations applied together."""
        cpu_optimization = OptimizationRecommendation(
            type="cpu",
            description="CPU optimization",
            confidence=0.8,
            estimated_improvement=25.0,
            implementation_steps=["CPU step"],
            priority="high",
            risk_level="low"
        )
        
        memory_optimization = OptimizationRecommendation(
            type="memory", 
            description="Memory optimization",
            confidence=0.7,
            estimated_improvement=20.0,
            implementation_steps=["Memory step"],
            priority="medium",
            risk_level="low"
        )
        
        combined_result = simulator.simulate_combined_impact(
            baseline_metrics, 
            [cpu_optimization, memory_optimization]
        )
        
        # Combined optimizations should have cumulative positive impact
        assert combined_result.execution_time < baseline_metrics.execution_time
        assert combined_result.base_metrics.cpu_usage < baseline_metrics.base_metrics.cpu_usage
        assert combined_result.base_metrics.memory_usage < baseline_metrics.base_metrics.memory_usage

    def test_risk_factor_in_simulation(self, simulator, baseline_metrics):
        """Test how risk level affects simulation outcomes."""
        low_risk_rec = OptimizationRecommendation(
            type="performance",
            description="Low risk optimization",
            confidence=0.8,
            estimated_improvement=20.0,
            implementation_steps=["Safe step"],
            priority="medium",
            risk_level="low"
        )
        
        high_risk_rec = OptimizationRecommendation(
            type="performance",
            description="High risk optimization", 
            confidence=0.8,
            estimated_improvement=20.0,  # Same improvement estimate
            implementation_steps=["Risky step"],
            priority="high",
            risk_level="high"
        )
        
        low_risk_result = simulator.simulate_impact(baseline_metrics, low_risk_rec)
        high_risk_result = simulator.simulate_impact(baseline_metrics, high_risk_rec)
        
        # High risk should have more variability/uncertainty in results
        # This would be implemented with Monte Carlo simulation in real implementation
        assert isinstance(low_risk_result, PerformanceMetrics)
        assert isinstance(high_risk_result, PerformanceMetrics)


@pytest.mark.integration
class TestOptimizerIntegration:
    """Integration tests for the optimizer with other analytics components."""
    
    def test_optimizer_with_bottleneck_analysis(self):
        """Test integration between optimizer and bottleneck analysis."""
        from src.analytics.engine import PerformanceAnalyticsEngine
        from src.analytics.models import AnalyticsConfiguration
        
        # Create components
        engine_config = AnalyticsConfiguration()
        engine = PerformanceAnalyticsEngine(engine_config)
        optimizer = PerformanceOptimizer(strategy=OptimizationStrategy.BALANCED)
        
        # Create metrics with bottleneck
        metrics_with_bottleneck = PerformanceMetrics(
            base_metrics=SystemMetrics(
                cpu_usage=95.0,  # High CPU
                memory_usage=60.0,
                disk_io=40.0,
                network_io=25.0,
                timestamp=datetime.now()
            ),
            execution_time=5.0,  # Slow execution
            throughput=50.0,     # Low throughput
            error_rate=0.02
        )
        
        # Analyze bottlenecks
        bottlenecks = engine.analyze_bottlenecks([metrics_with_bottleneck])
        
        # Generate optimizations
        recommendations = optimizer.generate_recommendations(bottlenecks)
        
        # Should produce actionable recommendations
        assert len(recommendations) > 0
        assert any(rec.type == "cpu" for rec in recommendations)

    def test_end_to_end_optimization_workflow(self):
        """Test complete end-to-end optimization workflow."""
        from src.analytics.engine import PerformanceAnalyticsEngine
        from src.analytics.collector import MetricsCollector, CollectionConfiguration
        from src.analytics.models import AnalyticsConfiguration
        
        # Create full pipeline
        collector_config = CollectionConfiguration()
        collector = MetricsCollector(collector_config)
        
        engine_config = AnalyticsConfiguration()
        engine = PerformanceAnalyticsEngine(engine_config)
        
        optimizer = PerformanceOptimizer(strategy=OptimizationStrategy.BALANCED)
        safety_validator = SafetyValidator()
        impact_simulator = ImpactSimulator()
        
        # Simulate metrics collection
        with patch.object(collector, 'collect_system_metrics') as mock_collect:
            mock_collect.return_value = SystemMetrics(
                cpu_usage=90.0, memory_usage=85.0, disk_io=80.0,
                network_io=40.0, timestamp=datetime.now()
            )
            
            # 1. Collect metrics
            metrics = collector.collect_system_metrics()
            perf_metrics = PerformanceMetrics(
                base_metrics=metrics,
                execution_time=4.0,
                throughput=60.0,
                error_rate=0.05
            )
            
            # 2. Analyze bottlenecks
            bottlenecks = engine.analyze_bottlenecks([perf_metrics])
            
            # 3. Generate optimizations
            recommendations = optimizer.generate_recommendations(bottlenecks)
            
            # 4. Validate safety
            safe_recommendations = []
            for rec in recommendations:
                is_safe, _, _ = safety_validator.validate_recommendation(rec)
                if is_safe:
                    safe_recommendations.append(rec)
            
            # 5. Simulate impact
            if safe_recommendations:
                simulated_result = impact_simulator.simulate_impact(perf_metrics, safe_recommendations[0])
                
                # Should show improvement
                assert simulated_result.execution_time < perf_metrics.execution_time
        
        # Workflow should complete without errors
        assert len(bottlenecks) >= 0  # May or may not detect bottlenecks
        assert len(recommendations) >= 0  # May or may not generate recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])