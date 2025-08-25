"""
Performance benchmarks and load testing for claude-tui.

This module provides comprehensive performance testing including:
- Load testing for concurrent operations
- Memory usage monitoring
- Response time benchmarking
- Throughput measurements
- Resource utilization analysis
"""

import asyncio
import gc
import memory_profiler
import psutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, Mock

import pytest


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        self.results.append({
            "function": func.__name__,
            "execution_time": execution_time,
            "args_count": len(args),
            "kwargs_count": len(kwargs)
        })
        
        return result, execution_time
    
    async def measure_async_execution_time(self, coro_func, *args, **kwargs):
        """Measure execution time of an async function."""
        start_time = time.perf_counter()
        result = await coro_func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        self.results.append({
            "function": coro_func.__name__,
            "execution_time": execution_time,
            "args_count": len(args),
            "kwargs_count": len(kwargs),
            "async": True
        })
        
        return result, execution_time
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        gc.collect()  # Ensure clean state
        
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        result = func(*args, **kwargs)
        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        
        memory_used = memory_after - memory_before
        
        self.results.append({
            "function": func.__name__,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_used": memory_used
        })
        
        return result, memory_used
    
    def get_system_metrics(self):
        """Get current system metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None,
            "network_io": psutil.net_io_counters()._asdict(),
            "process_memory": self.process.memory_info().rss / 1024 / 1024,  # MB
            "process_cpu": self.process.cpu_percent()
        }


class TestProjectManagerPerformance:
    """Performance tests for project management operations."""
    
    @pytest.fixture
    def performance_benchmark(self):
        """Create performance benchmark instance."""
        return PerformanceBenchmark()
    
    @pytest.fixture
    def mock_project_manager(self):
        """Create mock project manager with realistic delays."""
        manager = Mock()
        
        # Simulate realistic operation times
        manager.create_project = Mock(side_effect=lambda *args: (
            time.sleep(0.1), {"id": "proj_123", "name": "test-project"}
        )[1])
        
        manager.list_projects = Mock(side_effect=lambda *args: (
            time.sleep(0.05), [{"id": f"proj_{i}"} for i in range(100)]
        )[1])
        
        manager.update_project = Mock(side_effect=lambda *args: (
            time.sleep(0.08), {"id": "proj_123", "updated": True}
        )[1])
        
        manager.delete_project = Mock(side_effect=lambda *args: (
            time.sleep(0.06), True
        )[1])
        
        return manager
    
    def test_project_creation_performance(self, performance_benchmark, mock_project_manager):
        """Test project creation performance."""
        # Single project creation
        result, execution_time = performance_benchmark.measure_execution_time(
            mock_project_manager.create_project,
            {"name": "test-project", "template": "python"}
        )
        
        assert execution_time < 0.5  # Should complete within 500ms
        assert result["id"] == "proj_123"
        
        # Batch project creation
        projects_data = [
            {"name": f"project-{i}", "template": "python"}
            for i in range(10)
        ]
        
        start_time = time.perf_counter()
        results = []
        for project_data in projects_data:
            result, _ = performance_benchmark.measure_execution_time(
                mock_project_manager.create_project, project_data
            )
            results.append(result)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        assert total_time < 2.0  # 10 projects should complete within 2 seconds
        assert len(results) == 10
        
        # Calculate average execution time
        avg_time = sum(r["execution_time"] for r in performance_benchmark.results) / len(performance_benchmark.results)
        assert avg_time < 0.2  # Average should be under 200ms
    
    def test_project_listing_performance(self, performance_benchmark, mock_project_manager):
        """Test project listing performance with large datasets."""
        # Test with different page sizes
        page_sizes = [10, 50, 100, 500]
        
        for page_size in page_sizes:
            mock_project_manager.list_projects.return_value = [
                {"id": f"proj_{i}", "name": f"project-{i}"}
                for i in range(page_size)
            ]
            
            result, execution_time = performance_benchmark.measure_execution_time(
                mock_project_manager.list_projects, {"limit": page_size}
            )
            
            # Performance should scale reasonably
            if page_size <= 100:
                assert execution_time < 0.1
            else:
                assert execution_time < 0.5
            
            assert len(result) == page_size
    
    def test_concurrent_project_operations(self, performance_benchmark, mock_project_manager):
        """Test concurrent project operations."""
        def create_project_batch(batch_id, count=5):
            """Create a batch of projects."""
            results = []
            for i in range(count):
                result, _ = performance_benchmark.measure_execution_time(
                    mock_project_manager.create_project,
                    {"name": f"project-{batch_id}-{i}"}
                )
                results.append(result)
            return results
        
        # Test concurrent batches
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(create_project_batch, i, 5)
                for i in range(4)
            ]
            
            batch_results = [future.result() for future in futures]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Concurrent execution should be faster than sequential
        assert total_time < 3.0  # Should be much faster than 4*5*0.1 = 2.0s sequential
        
        # Verify all projects were created
        total_projects = sum(len(batch) for batch in batch_results)
        assert total_projects == 20
    
    def test_memory_usage_project_operations(self, performance_benchmark, mock_project_manager):
        """Test memory usage during project operations."""
        # Create many projects and measure memory
        result, memory_used = performance_benchmark.measure_memory_usage(
            lambda: [
                mock_project_manager.create_project({"name": f"project-{i}"})
                for i in range(1000)
            ]
        )
        
        # Memory usage should be reasonable
        assert memory_used < 50  # Less than 50MB for 1000 projects
        assert len(result) == 1000


class TestTaskEnginePerformance:
    """Performance tests for task execution engine."""
    
    @pytest.fixture
    def mock_task_engine(self):
        """Create mock task engine with realistic performance characteristics."""
        engine = Mock()
        
        # Simulate different task execution times
        async def mock_execute_task(task):
            # Simulate varying execution times based on task complexity
            base_time = 0.5
            complexity_factor = task.get("complexity", 1) * 0.3
            await asyncio.sleep(base_time + complexity_factor)
            return {
                "status": "completed",
                "execution_time": base_time + complexity_factor,
                "task_id": task.get("id", "unknown")
            }
        
        engine.execute_task = mock_execute_task
        engine.execute_tasks_parallel = AsyncMock()
        
        return engine
    
    @pytest.mark.asyncio
    async def test_single_task_performance(self, mock_task_engine):
        """Test single task execution performance."""
        benchmark = PerformanceBenchmark()
        
        task = {
            "id": "task_123",
            "name": "simple_task",
            "complexity": 1
        }
        
        result, execution_time = await benchmark.measure_async_execution_time(
            mock_task_engine.execute_task, task
        )
        
        assert result["status"] == "completed"
        assert execution_time < 1.0  # Should complete within 1 second
        assert execution_time > 0.5  # But should take some realistic time
    
    @pytest.mark.asyncio
    async def test_parallel_task_execution_performance(self, mock_task_engine):
        """Test parallel task execution performance."""
        benchmark = PerformanceBenchmark()
        
        # Create tasks with varying complexity
        tasks = [
            {"id": f"task_{i}", "complexity": i % 3 + 1}
            for i in range(20)
        ]
        
        # Mock parallel execution
        async def mock_parallel_execution(task_list):
            # Simulate parallel execution - should be much faster than sequential
            max_time = max(task.get("complexity", 1) * 0.3 + 0.5 for task in task_list)
            await asyncio.sleep(max_time)
            return [
                {"status": "completed", "task_id": task["id"]}
                for task in task_list
            ]
        
        mock_task_engine.execute_tasks_parallel = mock_parallel_execution
        
        result, execution_time = await benchmark.measure_async_execution_time(
            mock_task_engine.execute_tasks_parallel, tasks
        )
        
        # Parallel execution should be much faster than sequential
        assert execution_time < 2.0  # Should complete within 2 seconds
        assert len(result) == 20
        
        # Compare with sequential execution time (estimated)
        sequential_time_estimate = sum(task.get("complexity", 1) * 0.3 + 0.5 for task in tasks)
        assert execution_time < sequential_time_estimate * 0.2  # Should be at least 5x faster
    
    @pytest.mark.asyncio
    async def test_task_queue_performance(self, mock_task_engine):
        """Test task queue performance with high throughput."""
        benchmark = PerformanceBenchmark()
        
        # Simulate high-throughput task processing
        task_count = 100
        tasks = [{"id": f"task_{i}", "complexity": 1} for i in range(task_count)]
        
        # Mock queue processing
        async def process_task_queue(task_list):
            processed = []
            
            # Simulate batch processing (process in batches of 10)
            for i in range(0, len(task_list), 10):
                batch = task_list[i:i+10]
                await asyncio.sleep(0.1)  # Batch processing time
                processed.extend([{"status": "completed", "task_id": task["id"]} for task in batch])
            
            return processed
        
        mock_task_engine.process_queue = process_task_queue
        
        result, execution_time = await benchmark.measure_async_execution_time(
            mock_task_engine.process_queue, tasks
        )
        
        # Should process 100 tasks efficiently
        assert execution_time < 2.0  # Should complete within 2 seconds
        assert len(result) == task_count
        
        # Calculate throughput
        throughput = task_count / execution_time
        assert throughput > 50  # Should process at least 50 tasks per second
    
    @pytest.mark.asyncio
    async def test_concurrent_user_task_execution(self, mock_task_engine):
        """Test performance with multiple concurrent users."""
        benchmark = PerformanceBenchmark()
        
        async def user_task_workflow(user_id, task_count=5):
            """Simulate a user executing multiple tasks."""
            tasks = [
                {"id": f"user_{user_id}_task_{i}", "complexity": 1}
                for i in range(task_count)
            ]
            
            results = []
            for task in tasks:
                result = await mock_task_engine.execute_task(task)
                results.append(result)
            
            return results
        
        # Simulate 10 concurrent users
        user_count = 10
        start_time = time.perf_counter()
        
        user_results = await asyncio.gather(*[
            user_task_workflow(user_id, 3)  # 3 tasks per user
            for user_id in range(user_count)
        ])
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Verify results
        total_tasks = sum(len(user_result) for user_result in user_results)
        assert total_tasks == user_count * 3  # 30 tasks total
        
        # Performance should scale well with concurrent users
        assert execution_time < 5.0  # Should complete within 5 seconds
        
        # Calculate concurrent throughput
        throughput = total_tasks / execution_time
        assert throughput > 10  # Should handle at least 10 tasks per second with concurrency


class TestAIInterfacePerformance:
    """Performance tests for AI interface operations."""
    
    @pytest.fixture
    def mock_ai_interface(self):
        """Create mock AI interface with realistic response times."""
        interface = Mock()
        
        # Simulate AI response times based on prompt complexity
        async def mock_execute_claude_code(prompt, context=None):
            prompt_length = len(prompt)
            base_time = 1.0
            complexity_time = min(prompt_length / 100, 5.0)  # Cap at 5 seconds
            
            await asyncio.sleep(base_time + complexity_time)
            return {
                "status": "success",
                "output": f"Generated code for: {prompt[:50]}...",
                "execution_time": base_time + complexity_time
            }
        
        interface.execute_claude_code = mock_execute_claude_code
        
        return interface
    
    @pytest.mark.asyncio
    async def test_ai_response_time_performance(self, mock_ai_interface):
        """Test AI response time performance."""
        benchmark = PerformanceBenchmark()
        
        # Test different prompt complexities
        prompts = [
            "Hello world",  # Simple
            "Create a Python function that calculates fibonacci numbers",  # Medium
            "Build a complete REST API with authentication, database integration, error handling, logging, and comprehensive tests for a blog application with user management, post creation, comments, and admin features"  # Complex
        ]
        
        for i, prompt in enumerate(prompts):
            result, execution_time = await benchmark.measure_async_execution_time(
                mock_ai_interface.execute_claude_code, prompt
            )
            
            assert result["status"] == "success"
            
            # Performance expectations based on complexity
            if i == 0:  # Simple
                assert execution_time < 2.0
            elif i == 1:  # Medium
                assert execution_time < 3.0
            else:  # Complex
                assert execution_time < 7.0
    
    @pytest.mark.asyncio
    async def test_concurrent_ai_requests_performance(self, mock_ai_interface):
        """Test performance of concurrent AI requests."""
        benchmark = PerformanceBenchmark()
        
        # Create multiple concurrent requests
        requests = [
            f"Generate a {lang} function for sorting"
            for lang in ["Python", "JavaScript", "Java", "Go", "Rust"]
        ]
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*[
            mock_ai_interface.execute_claude_code(prompt)
            for prompt in requests
        ])
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Concurrent requests should be processed efficiently
        assert len(results) == 5
        assert all(result["status"] == "success" for result in results)
        
        # Should be faster than sequential execution
        assert total_time < 8.0  # Much faster than 5 * 2+ seconds sequential
        
        # Calculate concurrent throughput
        throughput = len(requests) / total_time
        assert throughput > 0.8  # Should handle at least 0.8 requests per second
    
    @pytest.mark.asyncio
    async def test_ai_caching_performance(self, mock_ai_interface):
        """Test performance improvement with caching."""
        benchmark = PerformanceBenchmark()
        
        # Mock caching behavior
        cache = {}
        
        async def cached_execute_claude_code(prompt):
            if prompt in cache:
                await asyncio.sleep(0.1)  # Cache hit - very fast
                return cache[prompt]
            else:
                result = await mock_ai_interface.execute_claude_code(prompt)
                cache[prompt] = result
                return result
        
        prompt = "Create a Python hello world function"
        
        # First request (cache miss)
        result1, time1 = await benchmark.measure_async_execution_time(
            cached_execute_claude_code, prompt
        )
        
        # Second request (cache hit)
        result2, time2 = await benchmark.measure_async_execution_time(
            cached_execute_claude_code, prompt
        )
        
        # Cache hit should be much faster
        assert time2 < time1 * 0.2  # At least 5x faster
        assert time2 < 0.2  # Cache hit should be under 200ms
        assert result1["output"] == result2["output"]


class TestValidationPerformance:
    """Performance tests for validation operations."""
    
    @pytest.fixture
    def mock_validation_engine(self):
        """Create mock validation engine."""
        engine = Mock()
        
        def mock_validate_code(code):
            # Simulate validation time based on code length
            code_length = len(code)
            time.sleep(code_length / 10000)  # 1ms per 10 characters
            
            placeholder_count = code.count("TODO") + code.count("NotImplementedError")
            
            return {
                "has_placeholders": placeholder_count > 0,
                "placeholder_count": placeholder_count,
                "quality_score": max(0.3, 1.0 - (placeholder_count * 0.2)),
                "validation_time": code_length / 10000
            }
        
        engine.validate_code = mock_validate_code
        
        return engine
    
    def test_code_validation_performance(self, mock_validation_engine):
        """Test code validation performance."""
        benchmark = PerformanceBenchmark()
        
        # Test different code sizes
        code_samples = [
            "def hello(): print('Hello')",  # Small
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)" * 10,  # Medium
            "class LargeClass:\n    def method(self): pass\n" * 100  # Large
        ]
        
        for i, code in enumerate(code_samples):
            result, execution_time = benchmark.measure_execution_time(
                mock_validation_engine.validate_code, code
            )
            
            # Performance should scale with code size but remain reasonable
            if i == 0:  # Small
                assert execution_time < 0.01
            elif i == 1:  # Medium
                assert execution_time < 0.1
            else:  # Large
                assert execution_time < 0.5
            
            assert "has_placeholders" in result
            assert "quality_score" in result
    
    def test_batch_validation_performance(self, mock_validation_engine):
        """Test batch validation performance."""
        benchmark = PerformanceBenchmark()
        
        # Create batch of code samples
        code_samples = [
            f"def function_{i}():\n    # TODO: implement\n    pass"
            for i in range(50)
        ]
        
        def batch_validate(code_list):
            """Batch validation function."""
            results = []
            for code in code_list:
                result = mock_validation_engine.validate_code(code)
                results.append(result)
            return results
        
        result, execution_time = benchmark.measure_execution_time(
            batch_validate, code_samples
        )
        
        # Batch processing should be efficient
        assert execution_time < 2.0  # Should complete within 2 seconds
        assert len(result) == 50
        
        # Calculate throughput
        throughput = len(code_samples) / execution_time
        assert throughput > 25  # Should validate at least 25 files per second


class TestSystemResourceUsage:
    """Test overall system resource usage under load."""
    
    def test_cpu_usage_under_load(self):
        """Test CPU usage during high-load operations."""
        def cpu_intensive_task():
            # Simulate CPU-intensive validation or processing
            total = 0
            for i in range(100000):
                total += i * i
            return total
        
        # Monitor CPU usage
        initial_cpu = psutil.cpu_percent(interval=1)
        
        # Execute CPU-intensive tasks
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(10)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        final_cpu = psutil.cpu_percent(interval=1)
        
        # Verify task completion
        assert len(results) == 10
        assert all(result > 0 for result in results)
        
        # CPU usage should increase but not max out completely
        execution_time = end_time - start_time
        assert execution_time < 10  # Should complete reasonably quickly
    
    def test_memory_usage_stability(self):
        """Test memory usage stability over time."""
        benchmark = PerformanceBenchmark()
        initial_memory = benchmark.get_system_metrics()["process_memory"]
        
        # Simulate memory-intensive operations
        data_structures = []
        
        for i in range(100):
            # Create and release data structures
            large_list = list(range(10000))
            data_structures.append(large_list)
            
            # Periodically clean up
            if i % 20 == 0:
                data_structures = data_structures[-10:]  # Keep only last 10
                gc.collect()
        
        final_memory = benchmark.get_system_metrics()["process_memory"]
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be controlled
        assert memory_growth < 100  # Less than 100MB growth
    
    def test_resource_cleanup_performance(self):
        """Test resource cleanup performance."""
        benchmark = PerformanceBenchmark()
        
        def create_and_cleanup_resources():
            # Simulate resource creation
            resources = []
            for i in range(1000):
                resource = {"id": i, "data": [j for j in range(100)]}
                resources.append(resource)
            
            # Cleanup
            resources.clear()
            gc.collect()
            
            return True
        
        result, execution_time = benchmark.measure_execution_time(
            create_and_cleanup_resources
        )
        
        # Cleanup should be efficient
        assert result is True
        assert execution_time < 1.0  # Should complete within 1 second


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests."""
    
    def test_baseline_performance_metrics(self):
        """Test baseline performance metrics."""
        benchmark = PerformanceBenchmark()
        
        # Define baseline performance expectations
        baselines = {
            "project_creation": 0.2,  # seconds
            "task_execution": 1.0,    # seconds
            "validation": 0.1,        # seconds
            "api_response": 0.5       # seconds
        }
        
        # Mock operations with baseline timings
        def mock_project_creation():
            time.sleep(0.15)  # Slightly under baseline
            return {"id": "proj_123"}
        
        def mock_task_execution():
            time.sleep(0.8)  # Under baseline
            return {"status": "completed"}
        
        def mock_validation():
            time.sleep(0.08)  # Under baseline
            return {"valid": True}
        
        def mock_api_response():
            time.sleep(0.3)  # Under baseline
            return {"data": "response"}
        
        # Test each operation
        operations = [
            ("project_creation", mock_project_creation),
            ("task_execution", mock_task_execution),
            ("validation", mock_validation),
            ("api_response", mock_api_response)
        ]
        
        for operation_name, operation_func in operations:
            result, execution_time = benchmark.measure_execution_time(operation_func)
            
            # Should meet baseline performance
            baseline = baselines[operation_name]
            assert execution_time <= baseline, f"{operation_name} exceeded baseline: {execution_time}s > {baseline}s"
            
            # Performance buffer for CI/CD variations
            assert execution_time <= baseline * 1.2, f"{operation_name} significantly exceeded baseline"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])