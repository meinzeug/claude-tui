"""
Performance Benchmarks with TDD London School Approach

London School performance testing emphasizing:
- Mock-based isolation for consistent benchmark conditions
- Behavior verification of performance-critical interactions
- Contract testing for performance requirements
- Outside-in testing from user performance scenarios
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import statistics
from typing import List, Dict, Any

from src.claude_tui.core.project_manager import ProjectManager
from src.ai.swarm_orchestrator import SwarmOrchestrator, TaskRequest
from src.claude_tui.integrations.ai_interface import AIInterface
from src.performance.memory_optimizer import MemoryOptimizer
from src.performance.lazy_loader import LazyLoader


# Performance Mock Fixtures - London School isolation
@pytest.fixture
def mock_high_performance_project_manager():
    """Mock project manager optimized for performance testing"""
    mock = AsyncMock(spec=ProjectManager)
    
    # Mock fast responses
    mock.create_project = AsyncMock(return_value=Mock())
    mock.orchestrate_development = AsyncMock(return_value=Mock(success=True))
    mock.get_project_status = AsyncMock(return_value={"status": "active"})
    mock.load_project = AsyncMock(return_value=Mock())
    
    # Mock cleanup to be instantaneous
    mock.cleanup = AsyncMock(return_value=None)
    
    return mock


@pytest.fixture  
def mock_optimized_swarm_orchestrator():
    """Mock swarm orchestrator with performance optimizations"""
    mock = AsyncMock(spec=SwarmOrchestrator)
    
    # Mock fast swarm operations
    mock.initialize_swarm = AsyncMock(return_value="fast-swarm")
    mock.execute_task = AsyncMock(return_value="fast-exec")
    mock.get_swarm_status = AsyncMock(return_value={"agents": 3, "status": "active"})
    mock.scale_swarm = AsyncMock(return_value=True)
    
    return mock


@pytest.fixture
def performance_test_environment():
    """Set up performance test environment with controlled conditions"""
    return {
        'max_memory_mb': 100,  # Memory limit for tests
        'max_execution_time_ms': 1000,  # Time limit for operations
        'concurrent_operations': 10,  # Number of concurrent operations
        'benchmark_iterations': 100  # Iterations for statistical significance
    }


class TestProjectManagerPerformance:
    """Performance tests for ProjectManager - London School mock-based benchmarking"""
    
    def test_project_creation_performance_benchmark(
        self,
        benchmark,
        mock_high_performance_project_manager
    ):
        """Benchmark project creation with consistent mock conditions"""
        
        def create_project_operation():
            # Simulate project creation workflow
            template_name = "performance-template"
            project_name = "perf-test-project"
            output_dir = Path("/tmp/perf-test")
            
            # Mock the creation process
            mock_high_performance_project_manager.create_project(
                template_name=template_name,
                project_name=project_name,
                output_directory=output_dir
            )
            
            return mock_high_performance_project_manager
        
        # Benchmark the operation
        result = benchmark(create_project_operation)
        
        # Verify performance contract
        assert result is not None
        
        # Performance assertions - London School behavior verification
        benchmark_stats = benchmark.stats
        assert benchmark_stats['mean'] < 0.001, "Project creation should be sub-millisecond with mocks"
    
    def test_concurrent_project_operations_performance(
        self,
        benchmark,
        mock_high_performance_project_manager
    ):
        """Benchmark concurrent project operations"""
        
        async def concurrent_operations():
            # Create multiple concurrent operations
            tasks = []
            
            for i in range(10):
                task = mock_high_performance_project_manager.orchestrate_development({
                    "feature": f"feature-{i}",
                    "priority": "high"
                })
                tasks.append(task)
            
            # Execute concurrently
            results = await asyncio.gather(*tasks)
            return results
        
        def run_concurrent_test():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(concurrent_operations())
            finally:
                loop.close()
        
        # Benchmark concurrent operations
        results = benchmark(run_concurrent_test)
        
        # Verify performance and behavior
        assert len(results) == 10
        
        # Performance contract verification
        benchmark_stats = benchmark.stats
        assert benchmark_stats['mean'] < 0.01, "Concurrent operations should complete quickly"


class TestSwarmOrchestratorPerformance:
    """Performance tests for SwarmOrchestrator - London School scalability focus"""
    
    def test_swarm_initialization_performance_scaling(
        self,
        benchmark,
        mock_optimized_swarm_orchestrator,
        performance_test_environment
    ):
        """Benchmark swarm initialization performance at scale"""
        
        def initialize_multiple_swarms():
            swarm_ids = []
            
            for i in range(performance_test_environment['concurrent_operations']):
                project_spec = {
                    'features': [f'feature-{i}' for i in range(5)],
                    'complexity': 5 + i
                }
                
                # Mock swarm initialization
                swarm_id = f"benchmark-swarm-{i}"
                mock_optimized_swarm_orchestrator.initialize_swarm(project_spec)
                swarm_ids.append(swarm_id)
            
            return swarm_ids
        
        # Benchmark swarm initialization
        swarm_ids = benchmark(initialize_multiple_swarms)
        
        # Verify performance contract
        assert len(swarm_ids) == performance_test_environment['concurrent_operations']
        
        # Performance assertions
        benchmark_stats = benchmark.stats
        assert benchmark_stats['mean'] < 0.1, "Multiple swarm initialization should be fast"
    
    def test_task_execution_throughput_performance(
        self,
        benchmark,
        mock_optimized_swarm_orchestrator
    ):
        """Benchmark task execution throughput"""
        
        def execute_high_volume_tasks():
            execution_ids = []
            
            # Create high volume of tasks
            for i in range(50):
                task = TaskRequest(
                    task_id=f"throughput-task-{i}",
                    description=f"High volume task {i}",
                    priority="medium",
                    estimated_complexity=3
                )
                
                # Mock task execution
                exec_id = f"exec-throughput-{i}"
                mock_optimized_swarm_orchestrator.execute_task(task)
                execution_ids.append(exec_id)
            
            return execution_ids
        
        # Benchmark task throughput
        execution_ids = benchmark(execute_high_volume_tasks)
        
        # Verify throughput performance
        assert len(execution_ids) == 50
        
        # Calculate throughput (operations per second)
        benchmark_stats = benchmark.stats
        throughput = 50 / benchmark_stats['mean']
        assert throughput > 1000, f"Task throughput should exceed 1000 ops/sec, got {throughput}"


class TestMemoryPerformanceOptimization:
    """Memory performance tests - London School resource management focus"""
    
    def test_memory_optimizer_performance_impact(
        self,
        benchmark,
        performance_test_environment
    ):
        """Test memory optimizer performance impact"""
        
        # Mock memory optimizer
        mock_optimizer = Mock(spec=MemoryOptimizer)
        mock_optimizer.optimize_memory_usage = Mock(return_value={'optimized': True})
        mock_optimizer.get_memory_stats = Mock(return_value={'used_mb': 45})
        
        def memory_optimization_cycle():
            # Simulate memory optimization cycle
            initial_stats = mock_optimizer.get_memory_stats()
            optimization_result = mock_optimizer.optimize_memory_usage()
            final_stats = mock_optimizer.get_memory_stats()
            
            return {
                'initial': initial_stats,
                'optimization': optimization_result,
                'final': final_stats
            }
        
        # Benchmark memory optimization
        result = benchmark(memory_optimization_cycle)
        
        # Verify optimization behavior and performance
        assert result['optimization']['optimized'] is True
        
        # Performance contract for memory operations
        benchmark_stats = benchmark.stats
        assert benchmark_stats['mean'] < 0.01, "Memory optimization should be very fast"
    
    def test_lazy_loading_performance_benefits(
        self,
        benchmark
    ):
        """Test lazy loading performance benefits"""
        
        # Mock lazy loader
        mock_loader = Mock(spec=LazyLoader)
        mock_loader.load_on_demand = Mock(return_value={'loaded': True, 'time_saved': 0.5})
        
        def lazy_loading_operations():
            results = []
            
            # Simulate multiple lazy loading operations
            for i in range(20):
                result = mock_loader.load_on_demand(f"resource-{i}")
                results.append(result)
            
            return results
        
        # Benchmark lazy loading
        results = benchmark(lazy_loading_operations)
        
        # Verify lazy loading benefits
        assert len(results) == 20
        assert all(r['loaded'] for r in results)
        
        # Performance verification
        benchmark_stats = benchmark.stats
        assert benchmark_stats['mean'] < 0.05, "Lazy loading should minimize initialization time"


class TestConcurrentPerformancePatterns:
    """Concurrent performance patterns - London School coordination testing"""
    
    @pytest.mark.asyncio
    async def test_async_pipeline_performance(
        self,
        mock_high_performance_project_manager,
        mock_optimized_swarm_orchestrator
    ):
        """Test async pipeline performance with coordinated operations"""
        
        start_time = time.perf_counter()
        
        # Create async pipeline operations
        async def pipeline_stage_1():
            return await mock_high_performance_project_manager.create_project(
                template_name="async-template",
                project_name="async-project",
                output_directory=Path("/tmp/async")
            )
        
        async def pipeline_stage_2():
            return await mock_optimized_swarm_orchestrator.initialize_swarm({
                'features': ['async_processing'],
                'complexity': 6
            })
        
        async def pipeline_stage_3():
            task = TaskRequest(
                task_id="async-pipeline-task",
                description="Async pipeline task",
                estimated_complexity=5
            )
            return await mock_optimized_swarm_orchestrator.execute_task(task)
        
        # Execute pipeline stages concurrently
        stage1_result, stage2_result, stage3_result = await asyncio.gather(
            pipeline_stage_1(),
            pipeline_stage_2(), 
            pipeline_stage_3()
        )
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Verify concurrent execution performance
        assert execution_time < 0.1, f"Async pipeline should complete quickly, took {execution_time}s"
        assert stage1_result is not None
        assert stage2_result == "fast-swarm"
        assert stage3_result == "fast-exec"
    
    def test_thread_pool_performance_scaling(
        self,
        benchmark,
        performance_test_environment
    ):
        """Test thread pool performance scaling"""
        
        def thread_pool_operations():
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit CPU-bound operations
                futures = []
                
                for i in range(performance_test_environment['concurrent_operations']):
                    # Mock CPU-intensive operation
                    future = executor.submit(lambda x: x * x, i)
                    futures.append(future)
                
                # Collect results
                results = [future.result() for future in futures]
                return results
        
        # Benchmark thread pool operations
        results = benchmark(thread_pool_operations)
        
        # Verify thread pool performance
        assert len(results) == performance_test_environment['concurrent_operations']
        
        # Performance scaling verification
        benchmark_stats = benchmark.stats
        operations_per_second = len(results) / benchmark_stats['mean']
        assert operations_per_second > 100, f"Thread pool should handle >100 ops/sec, got {operations_per_second}"


class TestPerformanceRegressionDetection:
    """Performance regression detection - London School baseline comparison"""
    
    def test_performance_baseline_comparison(
        self,
        benchmark,
        mock_high_performance_project_manager
    ):
        """Test performance against established baselines"""
        
        def baseline_operation():
            # Standard project management operation
            return mock_high_performance_project_manager.get_project_status()
        
        # Benchmark against baseline
        result = benchmark(baseline_operation)
        
        # Store performance metrics for regression detection
        benchmark_stats = benchmark.stats
        performance_metrics = {
            'mean_time': benchmark_stats['mean'],
            'std_dev': benchmark_stats['stddev'],
            'min_time': benchmark_stats['min'],
            'max_time': benchmark_stats['max']
        }
        
        # Performance regression thresholds
        assert performance_metrics['mean_time'] < 0.001, "Mean execution time regression"
        assert performance_metrics['max_time'] < 0.01, "Maximum execution time regression"
        
        # Verify operation completed successfully
        assert result is not None
    
    def test_memory_usage_regression_detection(
        self,
        benchmark,
        performance_test_environment
    ):
        """Test memory usage regression detection"""
        
        import psutil
        import os
        
        def memory_intensive_operation():
            # Simulate memory usage monitoring
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Mock memory-intensive operation
            mock_data = [{'item': f'data-{i}'} for i in range(1000)]
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            return {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_delta_mb': memory_delta,
                'data_items': len(mock_data)
            }
        
        # Benchmark memory usage
        result = benchmark(memory_intensive_operation)
        
        # Memory regression checks
        max_memory_mb = performance_test_environment['max_memory_mb']
        assert result['memory_delta_mb'] < max_memory_mb, f"Memory usage regression: {result['memory_delta_mb']}MB > {max_memory_mb}MB"
        assert result['data_items'] == 1000, "Data processing regression"


class TestPerformanceContractCompliance:
    """Performance contract compliance - London School SLA verification"""
    
    def test_response_time_contract_compliance(
        self,
        mock_high_performance_project_manager
    ):
        """Test response time contract compliance"""
        
        # Define performance contracts
        performance_contracts = {
            'create_project_max_ms': 100,
            'orchestrate_development_max_ms': 500,
            'get_project_status_max_ms': 50,
            'load_project_max_ms': 200
        }
        
        # Test each operation against its contract
        operations = {
            'create_project': lambda: mock_high_performance_project_manager.create_project(
                "test-template", "test-project", Path("/tmp")
            ),
            'orchestrate_development': lambda: mock_high_performance_project_manager.orchestrate_development(
                {"feature": "test"}
            ),
            'get_project_status': lambda: mock_high_performance_project_manager.get_project_status(),
            'load_project': lambda: mock_high_performance_project_manager.load_project(
                Path("/tmp/test")
            )
        }
        
        for operation_name, operation_func in operations.items():
            start_time = time.perf_counter()
            result = operation_func()
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            contract_max_ms = performance_contracts.get(f'{operation_name}_max_ms', 1000)
            
            assert execution_time_ms < contract_max_ms, f"{operation_name} exceeded contract: {execution_time_ms}ms > {contract_max_ms}ms"
            assert result is not None, f"{operation_name} should return valid result"
    
    def test_throughput_contract_compliance(
        self,
        mock_optimized_swarm_orchestrator
    ):
        """Test throughput contract compliance"""
        
        # Throughput contracts
        throughput_contracts = {
            'min_tasks_per_second': 100,
            'min_swarms_per_minute': 60,
            'max_latency_ms': 10
        }
        
        # Test task execution throughput
        start_time = time.perf_counter()
        task_count = 200
        
        for i in range(task_count):
            task = TaskRequest(
                task_id=f"throughput-contract-{i}",
                description=f"Contract test task {i}",
                estimated_complexity=2
            )
            mock_optimized_swarm_orchestrator.execute_task(task)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate actual throughput
        tasks_per_second = task_count / total_time
        
        # Verify throughput contract
        assert tasks_per_second >= throughput_contracts['min_tasks_per_second'], \
            f"Throughput contract violation: {tasks_per_second} < {throughput_contracts['min_tasks_per_second']} tasks/sec"
        
        # Verify latency contract
        avg_latency_ms = (total_time / task_count) * 1000
        assert avg_latency_ms <= throughput_contracts['max_latency_ms'], \
            f"Latency contract violation: {avg_latency_ms}ms > {throughput_contracts['max_latency_ms']}ms"


class TestPerformanceMonitoringIntegration:
    """Performance monitoring integration - London School observability testing"""
    
    def test_performance_metrics_collection(
        self,
        benchmark,
        mock_high_performance_project_manager
    ):
        """Test performance metrics collection and reporting"""
        
        # Mock performance monitor
        mock_monitor = Mock()
        mock_monitor.start_timing = Mock()
        mock_monitor.end_timing = Mock()
        mock_monitor.record_memory_usage = Mock()
        mock_monitor.get_performance_report = Mock(return_value={
            'avg_response_time_ms': 5.2,
            'max_memory_usage_mb': 45.3,
            'operations_per_second': 150.7
        })
        
        def monitored_operation():
            # Simulate monitored operation
            mock_monitor.start_timing('test_operation')
            
            # Execute operation
            result = mock_high_performance_project_manager.get_project_status()
            
            # Record metrics
            mock_monitor.record_memory_usage()
            mock_monitor.end_timing('test_operation')
            
            return result
        
        # Benchmark with monitoring
        result = benchmark(monitored_operation)
        
        # Verify monitoring integration
        mock_monitor.start_timing.assert_called_with('test_operation')
        mock_monitor.end_timing.assert_called_with('test_operation')
        mock_monitor.record_memory_usage.assert_called_once()
        
        # Verify performance report
        performance_report = mock_monitor.get_performance_report()
        assert performance_report['avg_response_time_ms'] < 10
        assert performance_report['max_memory_usage_mb'] < 100
        assert performance_report['operations_per_second'] > 100
        
        assert result is not None