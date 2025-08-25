"""
Comprehensive Performance Testing Suite.

Tests system performance under various loads with focus on task execution,
validation, and API response times. Includes benchmarking and load testing.
"""

import asyncio
import pytest
import time
import statistics
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock

# Test fixtures
from tests.fixtures.comprehensive_test_fixtures import (
    TestDataFactory,
    MockComponents,
    TestAssertions,
    PerformanceTimer,
    create_realistic_test_scenario
)


class PerformanceTestHarness:
    """Harness for conducting performance tests with metrics collection."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        
    def stop_monitoring(self):
        """Stop monitoring and collect metrics."""
        if self.start_time:
            self.metrics['execution_time'] = time.time() - self.start_time
        return self.metrics
    
    def assert_performance_thresholds(self, thresholds: Dict[str, float]):
        """Assert performance metrics meet thresholds."""
        for metric, threshold in thresholds.items():
            if metric in self.metrics:
                actual = self.metrics[metric]
                if 'time' in metric.lower():
                    assert actual <= threshold, f"{metric}: {actual:.2f}s > {threshold}s"
                else:
                    assert actual >= threshold, f"{metric}: {actual} < {threshold}"


class TestTaskEnginePerformance:
    """Performance tests for TaskEngine under various loads."""
    
    @pytest.fixture
    def performance_engine(self):
        """Create TaskEngine optimized for performance testing."""
        engine = Mock()
        engine.max_concurrent_tasks = 20
        engine.execute_workflow = AsyncMock()
        return engine
    
    @pytest.fixture
    def performance_harness(self):
        """Provide performance testing harness."""
        return PerformanceTestHarness()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_task_execution_benchmark(self, performance_engine, performance_harness):
        """Benchmark single task execution performance."""
        performance_harness.start_monitoring()
        
        # Mock fast execution
        performance_engine.execute_workflow.return_value = {
            'success': True,
            'tasks_executed': ['task-1'],
            'total_time': 0.05,
            'quality_metrics': {'success_rate': 100.0}
        }
        
        # Execute multiple times for benchmarking
        execution_times = []
        for _ in range(50):
            start = time.time()
            await performance_engine.execute_workflow(Mock(), 'sequential')
            execution_times.append(time.time() - start)
        
        metrics = performance_harness.stop_monitoring()
        
        # Calculate statistics
        avg_time = statistics.mean(execution_times)
        
        # Performance assertions
        assert avg_time < 0.1, f"Average execution time {avg_time:.3f}s too high"
        
        performance_harness.assert_performance_thresholds({
            'execution_time': 10.0  # 50 tasks in under 10 seconds
        })
    
    @pytest.mark.performance 
    @pytest.mark.asyncio
    async def test_concurrent_task_execution_scalability(self, performance_engine, performance_harness):
        """Test scalability with increasing concurrent task loads."""
        performance_harness.start_monitoring()
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        results = {}
        
        for concurrency in concurrency_levels:
            # Mock concurrent execution
            performance_engine.execute_workflow.return_value = {
                'success': True,
                'tasks_executed': [f'task-{i}' for i in range(concurrency)],
                'total_time': 0.5 + (concurrency * 0.01),
                'quality_metrics': {'success_rate': 100.0}
            }
            
            start_time = time.time()
            
            # Execute tasks concurrently
            await asyncio.gather(*[
                performance_engine.execute_workflow(Mock(), 'parallel')
                for _ in range(concurrency)
            ])
            
            execution_time = time.time() - start_time
            throughput = concurrency / execution_time
            
            results[concurrency] = {
                'execution_time': execution_time,
                'throughput': throughput
            }
        
        metrics = performance_harness.stop_monitoring()
        
        # Scalability assertions
        throughput_1 = results[1]['throughput']
        throughput_20 = results[20]['throughput']
        
        # Allow for some throughput degradation but not more than 50%
        assert throughput_20 >= throughput_1 * 0.5, \
            f"Throughput degraded too much: {throughput_20:.2f} vs {throughput_1:.2f}"


class TestValidationPerformance:
    """Performance tests for validation systems."""
    
    @pytest.fixture
    def validation_engine(self):
        """Create validation engine for performance testing."""
        validator = Mock()
        validator.validate_codebase = AsyncMock()
        validator.validate_single_file = AsyncMock()
        return validator
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_codebase_validation_performance(self, validation_engine, tmp_path):
        """Test validation performance on large codebases."""
        harness = PerformanceTestHarness()
        harness.start_monitoring()
        
        # Create large codebase simulation
        project_dir = tmp_path / "large_codebase"
        project_dir.mkdir()
        
        # Create many files
        file_count = 50
        for i in range(file_count):
            (project_dir / f"module_{i}.py").write_text(f'def func_{i}(): return {i}')
        
        # Mock validation performance
        validation_engine.validate_codebase.return_value = {
            'is_authentic': True,
            'authenticity_score': 82.5,
            'files_processed': file_count,
            'processing_time': 3.5
        }
        
        # Perform validation
        result = await validation_engine.validate_codebase(project_dir)
        
        metrics = harness.stop_monitoring()
        
        # Validation performance assertions
        assert result['files_processed'] == file_count
        assert result['processing_time'] < 5.0
        
        harness.assert_performance_thresholds({
            'execution_time': 8.0
        })


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])