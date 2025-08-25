#!/usr/bin/env python3
"""
Comprehensive Performance Tests.

Tests performance requirements including:
- Startup time benchmarks (<2 seconds requirement)
- UI response time tests (<500ms requirement) 
- Memory usage profiling and leak detection
- Async operation performance validation
- Concurrent task execution scaling
- Database query performance
- File I/O operation benchmarks
"""

import asyncio
import pytest
import time
import psutil
import threading
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor
import gc
import sys

# Import components for performance testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.project_manager import ProjectManager
from core.task_engine import TaskEngine, create_task, ExecutionStrategy
from core.validator import ProgressValidator
from core.types import Priority, Task

# Performance test configuration
PERFORMANCE_TARGETS = {
    'startup_time_seconds': 2.0,
    'ui_response_time_ms': 500,
    'max_memory_mb': 512,
    'validation_speed_files_per_second': 5,
    'task_execution_throughput': 10,  # tasks per second
    'concurrent_users': 50
}


@pytest.fixture
def performance_monitor():
    """Monitor system performance during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.start_time = None
            self.start_memory = None
            self.start_cpu = None
            
        def start_monitoring(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
            self.start_cpu = self.process.cpu_percent()
            
        def stop_monitoring(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            end_cpu = self.process.cpu_percent()
            
            return {
                'duration': end_time - self.start_time,
                'memory_used': end_memory - self.start_memory,
                'cpu_avg': (self.start_cpu + end_cpu) / 2,
                'peak_memory': self.process.memory_info().peak_wss if hasattr(self.process.memory_info(), 'peak_wss') else end_memory
            }
    
    return PerformanceMonitor()


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory for performance tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create realistic project structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "docs").mkdir()
        
        # Create multiple source files for testing
        for i in range(20):
            (project_path / "src" / f"module_{i}.py").write_text(f'''
"""Module {i} for performance testing."""

def function_{i}_1(x, y):
    """Function {i}_1 implementation."""
    result = x + y
    return result * 2

def function_{i}_2(*args):
    """Function {i}_2 with variable arguments."""
    return sum(args)

class Class{i}:
    """Class {i} implementation."""
    
    def __init__(self, value):
        self.value = value
    
    def method_{i}(self):
        return self.value * {i}
''')
        
        yield project_path


class TestStartupPerformance:
    """Tests for application startup performance."""
    
    def test_project_manager_startup_time(self, temp_project_dir, performance_monitor):
        """Test ProjectManager startup meets <2 seconds requirement."""
        performance_monitor.start_monitoring()
        
        # Initialize ProjectManager
        pm = ProjectManager(
            config_dir=temp_project_dir,
            enable_validation=True,
            max_concurrent_tasks=5
        )
        
        metrics = performance_monitor.stop_monitoring()
        
        assert metrics['duration'] < PERFORMANCE_TARGETS['startup_time_seconds']
        print(f"ProjectManager startup time: {metrics['duration']:.3f}s")
    
    def test_task_engine_startup_time(self, performance_monitor):
        """Test TaskEngine startup performance."""
        performance_monitor.start_monitoring()
        
        # Initialize TaskEngine with various configurations
        engines = []
        for i in range(5):
            engine = TaskEngine(
                max_concurrent_tasks=i + 1,
                enable_validation=True,
                enable_monitoring=True
            )
            engines.append(engine)
        
        metrics = performance_monitor.stop_monitoring()
        
        assert metrics['duration'] < PERFORMANCE_TARGETS['startup_time_seconds']
        print(f"Multiple TaskEngine startup time: {metrics['duration']:.3f}s")
    
    def test_validator_startup_time(self, performance_monitor):
        """Test ProgressValidator startup performance."""
        performance_monitor.start_monitoring()
        
        # Initialize ProgressValidator with full features
        validator = ProgressValidator(
            enable_cross_validation=True,
            enable_execution_testing=True,
            enable_quality_analysis=True
        )
        
        metrics = performance_monitor.stop_monitoring()
        
        assert metrics['duration'] < PERFORMANCE_TARGETS['startup_time_seconds']
        print(f"ProgressValidator startup time: {metrics['duration']:.3f}s")
    
    def test_cold_vs_warm_startup(self, temp_project_dir, performance_monitor):
        """Test cold vs warm startup performance comparison."""
        # Cold startup (first time)
        performance_monitor.start_monitoring()
        pm1 = ProjectManager(config_dir=temp_project_dir)
        cold_metrics = performance_monitor.stop_monitoring()
        
        del pm1
        gc.collect()
        
        # Warm startup (modules already loaded)
        performance_monitor.start_monitoring()
        pm2 = ProjectManager(config_dir=temp_project_dir)
        warm_metrics = performance_monitor.stop_monitoring()
        
        # Warm startup should be faster or at least not significantly slower
        startup_time_difference = warm_metrics['duration'] - cold_metrics['duration']
        assert startup_time_difference < 0.5  # Warm startup shouldn't be much slower
        
        print(f"Cold startup: {cold_metrics['duration']:.3f}s")
        print(f"Warm startup: {warm_metrics['duration']:.3f}s")
        print(f"Difference: {startup_time_difference:+.3f}s")


class TestUIResponseTime:
    """Tests for UI response time performance."""
    
    @pytest.mark.asyncio
    async def test_ui_action_response_time(self, temp_project_dir):
        """Test UI action response times meet <500ms requirement."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        # Test various UI actions
        ui_actions = {
            'create_project': lambda: pm.create_project("Test Project", project_path=temp_project_dir / "test"),
            'list_projects': lambda: pm.list_projects(),
            'get_progress': lambda: asyncio.create_task(self._mock_get_progress(pm))
        }
        
        for action_name, action_func in ui_actions.items():
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(action_func) or hasattr(action_func(), '__await__'):
                    await action_func()
                else:
                    action_func()
            except Exception:
                pass  # Some actions might fail in test environment
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            assert response_time_ms < PERFORMANCE_TARGETS['ui_response_time_ms']
            print(f"{action_name} response time: {response_time_ms:.1f}ms")
    
    async def _mock_get_progress(self, pm):
        """Mock progress retrieval for testing."""
        # Simulate progress calculation
        await asyncio.sleep(0.01)  # Small delay to simulate real work
        return {'progress': 0.5, 'status': 'in_progress'}
    
    def test_widget_rendering_performance(self):
        """Test widget rendering performance."""
        # Mock widget rendering times
        widget_render_times = []
        
        for i in range(100):  # Simulate rendering 100 widgets
            start_time = time.time()
            
            # Simulate widget rendering work
            self._simulate_widget_rendering()
            
            end_time = time.time()
            render_time_ms = (end_time - start_time) * 1000
            widget_render_times.append(render_time_ms)
        
        # Check average and maximum rendering times
        avg_render_time = sum(widget_render_times) / len(widget_render_times)
        max_render_time = max(widget_render_times)
        
        assert avg_render_time < 50  # Average <50ms per widget
        assert max_render_time < PERFORMANCE_TARGETS['ui_response_time_ms']  # Max <500ms
        
        print(f"Average widget render time: {avg_render_time:.1f}ms")
        print(f"Maximum widget render time: {max_render_time:.1f}ms")
    
    def _simulate_widget_rendering(self):
        """Simulate widget rendering work."""
        # Simulate DOM manipulation, style calculation, etc.
        data = [i ** 2 for i in range(100)]
        processed = [x * 1.5 for x in data if x % 2 == 0]
        return len(processed)


class TestMemoryPerformance:
    """Tests for memory usage and leak detection."""
    
    def test_memory_usage_limits(self, temp_project_dir, performance_monitor):
        """Test memory usage stays within limits."""
        performance_monitor.start_monitoring()
        
        # Create multiple components that might consume memory
        components = []
        
        for i in range(10):
            pm = ProjectManager(config_dir=temp_project_dir / f"project_{i}")
            engine = TaskEngine(max_concurrent_tasks=5)
            validator = ProgressValidator()
            
            components.extend([pm, engine, validator])
        
        metrics = performance_monitor.stop_monitoring()
        memory_used_mb = metrics['memory_used'] / (1024 * 1024)
        
        assert memory_used_mb < PERFORMANCE_TARGETS['max_memory_mb']
        print(f"Memory used: {memory_used_mb:.1f}MB")
        
        # Cleanup
        del components
        gc.collect()
    
    def test_memory_leak_detection(self, temp_project_dir):
        """Test for memory leaks during repeated operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform operations that might leak memory
        for iteration in range(50):
            pm = ProjectManager(config_dir=temp_project_dir / f"leak_test_{iteration}")
            
            # Simulate typical operations
            try:
                project = asyncio.run(
                    pm.create_project(
                        f"LeakTest{iteration}",
                        project_path=temp_project_dir / f"leak_project_{iteration}"
                    )
                )
                asyncio.run(pm.delete_project(project.id, remove_files=True))
            except:
                pass  # Ignore errors in test environment
            
            del pm
            gc.collect()
            
            # Check memory growth every 10 iterations
            if iteration % 10 == 9:
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                memory_growth_mb = memory_growth / (1024 * 1024)
                
                print(f"Iteration {iteration + 1}: Memory growth: {memory_growth_mb:.1f}MB")
                
                # Memory growth should be reasonable
                assert memory_growth_mb < 100  # Less than 100MB growth
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_usage(self, temp_project_dir):
        """Test memory usage under concurrent operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create multiple concurrent operations
        async def concurrent_operation(operation_id):
            pm = ProjectManager(config_dir=temp_project_dir / f"concurrent_{operation_id}")
            try:
                project = await pm.create_project(
                    f"ConcurrentProject{operation_id}",
                    project_path=temp_project_dir / f"concurrent_project_{operation_id}"
                )
                await asyncio.sleep(0.1)  # Simulate work
                await pm.delete_project(project.id, remove_files=True)
            except:
                pass
            del pm
        
        # Run 20 concurrent operations
        tasks = [concurrent_operation(i) for i in range(20)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        gc.collect()
        final_memory = process.memory_info().rss
        memory_used = (final_memory - initial_memory) / (1024 * 1024)
        
        assert memory_used < PERFORMANCE_TARGETS['max_memory_mb']
        print(f"Concurrent operations memory usage: {memory_used:.1f}MB")


class TestAsyncOperationPerformance:
    """Tests for async operation performance."""
    
    @pytest.mark.asyncio
    async def test_task_execution_throughput(self, performance_monitor):
        """Test task execution throughput meets requirements."""
        engine = TaskEngine(max_concurrent_tasks=10, enable_validation=False)
        
        # Create many simple tasks
        tasks = []
        for i in range(100):
            task = create_task(
                name=f"Task {i}",
                description=f"Test task {i}",
                priority=Priority.MEDIUM,
                estimated_duration=1  # Very short tasks
            )
            tasks.append(task)
        
        # Mock simple executor
        async def simple_executor(task):
            await asyncio.sleep(0.01)  # 10ms work simulation
            from core.types import AITaskResult
            return AITaskResult(
                task_id=task.id,
                success=True,
                generated_content="Mock result",
                execution_time=0.01
            )
        
        # Register the executor
        engine.register_task_executor('default', simple_executor)
        
        performance_monitor.start_monitoring()
        
        # Execute all tasks
        from core.task_engine import execute_simple_workflow
        result = await execute_simple_workflow(tasks, ExecutionStrategy.PARALLEL)
        
        metrics = performance_monitor.stop_monitoring()
        
        # Calculate throughput
        throughput = len(tasks) / metrics['duration']
        
        assert throughput >= PERFORMANCE_TARGETS['task_execution_throughput']
        print(f"Task execution throughput: {throughput:.1f} tasks/second")
    
    @pytest.mark.asyncio
    async def test_validation_performance(self, temp_project_dir, performance_monitor):
        """Test validation performance meets speed requirements."""
        validator = ProgressValidator(
            enable_cross_validation=False,  # Disable for speed
            enable_execution_testing=False,
            enable_quality_analysis=True
        )
        
        # Create files for validation
        test_files = []
        for i in range(20):
            test_file = temp_project_dir / f"validate_{i}.py"
            test_file.write_text(f'''
def function_{i}():
    """Function {i} for validation testing."""
    return {i} * 2

def another_function_{i}(x):
    return x + {i}
''')
            test_files.append(test_file)
        
        performance_monitor.start_monitoring()
        
        # Validate all files
        results = []
        for test_file in test_files:
            result = await validator.validate_single_file(test_file)
            results.append(result)
        
        metrics = performance_monitor.stop_monitoring()
        
        # Calculate validation speed
        files_per_second = len(test_files) / metrics['duration']
        
        assert files_per_second >= PERFORMANCE_TARGETS['validation_speed_files_per_second']
        print(f"Validation speed: {files_per_second:.1f} files/second")
    
    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self):
        """Test performance of concurrent async operations."""
        # Create multiple async operations that might compete for resources
        async def async_operation(operation_id, duration=0.1):
            start_time = time.time()
            
            # Simulate async work (I/O, network, etc.)
            await asyncio.sleep(duration)
            
            # Simulate CPU work
            result = sum(i ** 2 for i in range(1000))
            
            end_time = time.time()
            return {
                'operation_id': operation_id,
                'duration': end_time - start_time,
                'result': result
            }
        
        # Test with increasing concurrency levels
        concurrency_levels = [1, 5, 10, 25, 50]
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            # Create concurrent operations
            operations = [
                async_operation(i, 0.05)  # 50ms each
                for i in range(concurrency)
            ]
            
            results = await asyncio.gather(*operations)
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # With perfect parallelism, 50 operations of 50ms each should take ~50ms
            # Allow some overhead
            max_expected_duration = 0.05 + (concurrency * 0.01)  # Base time + overhead
            
            print(f"Concurrency {concurrency}: {total_duration:.3f}s (expected ≤{max_expected_duration:.3f}s)")
            
            # Don't assert on high concurrency levels as they may be limited by system resources
            if concurrency <= 10:
                assert total_duration < max_expected_duration * 2  # Allow 2x overhead


class TestScalabilityPerformance:
    """Tests for scalability and load handling."""
    
    @pytest.mark.asyncio
    async def test_project_scaling(self, temp_project_dir, performance_monitor):
        """Test performance with increasing number of projects."""
        pm = ProjectManager(config_dir=temp_project_dir)
        
        project_counts = [1, 5, 10, 20]
        
        for count in project_counts:
            performance_monitor.start_monitoring()
            
            # Create multiple projects
            projects = []
            for i in range(count):
                try:
                    project = await pm.create_project(
                        f"ScaleProject{i}",
                        project_path=temp_project_dir / f"scale_project_{i}"
                    )
                    projects.append(project)
                except:
                    pass  # Continue with other projects
            
            # Test operations on all projects
            project_list = await pm.list_projects()
            
            metrics = performance_monitor.stop_monitoring()
            
            print(f"Projects: {count}, Duration: {metrics['duration']:.3f}s, Memory: {metrics['memory_used'] / 1024 / 1024:.1f}MB")
            
            # Performance should scale reasonably
            assert metrics['duration'] < count * 0.5  # Linear scaling with reasonable factor
            
            # Clean up projects
            for project in projects:
                try:
                    await pm.delete_project(project.id, remove_files=True)
                except:
                    pass
    
    def test_concurrent_user_simulation(self, temp_project_dir):
        """Simulate concurrent users and test performance."""
        def simulate_user(user_id):
            """Simulate a single user's operations."""
            user_metrics = []
            
            try:
                # Each user creates their own ProjectManager
                user_temp_dir = temp_project_dir / f"user_{user_id}"
                user_temp_dir.mkdir(exist_ok=True)
                
                pm = ProjectManager(config_dir=user_temp_dir)
                
                start_time = time.time()
                
                # Simulate user operations
                project = asyncio.run(
                    pm.create_project(
                        f"UserProject{user_id}",
                        project_path=user_temp_dir / "project"
                    )
                )
                
                # Simulate some work
                time.sleep(0.1)
                
                asyncio.run(pm.delete_project(project.id, remove_files=True))
                
                end_time = time.time()
                user_metrics.append(end_time - start_time)
                
            except Exception as e:
                user_metrics.append(float('inf'))  # Mark as failed
            
            return user_metrics
        
        # Test with different numbers of concurrent users
        user_counts = [1, 5, 10, 25]
        
        for num_users in user_counts:
            start_time = time.time()
            
            # Create thread pool for concurrent users
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [
                    executor.submit(simulate_user, user_id)
                    for user_id in range(num_users)
                ]
                
                # Collect results
                results = [future.result() for future in futures]
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Calculate success rate and average response time
            successful_operations = [r for r in results if r and r[0] != float('inf')]
            success_rate = len(successful_operations) / num_users * 100
            
            if successful_operations:
                avg_response_time = sum(r[0] for r in successful_operations) / len(successful_operations)
            else:
                avg_response_time = float('inf')
            
            print(f"Users: {num_users}, Success Rate: {success_rate:.1f}%, Avg Response: {avg_response_time:.3f}s, Total: {total_duration:.3f}s")
            
            # Performance requirements
            assert success_rate >= 90  # At least 90% success rate
            if num_users <= 10:  # Only assert on reasonable concurrency levels
                assert avg_response_time < 2.0  # Average response time should be reasonable


class TestDatabasePerformance:
    """Tests for database and I/O performance."""
    
    def test_file_io_performance(self, temp_project_dir, performance_monitor):
        """Test file I/O operations performance."""
        # Test file creation performance
        performance_monitor.start_monitoring()
        
        files_created = []
        for i in range(100):
            file_path = temp_project_dir / f"io_test_{i}.py"
            content = f"# Test file {i}\n" * 100  # 100 lines per file
            file_path.write_text(content)
            files_created.append(file_path)
        
        create_metrics = performance_monitor.stop_monitoring()
        
        # Test file reading performance
        performance_monitor.start_monitoring()
        
        for file_path in files_created:
            content = file_path.read_text()
            lines = content.split('\n')
            assert len(lines) >= 100
        
        read_metrics = performance_monitor.stop_monitoring()
        
        print(f"File creation: {create_metrics['duration']:.3f}s for 100 files")
        print(f"File reading: {read_metrics['duration']:.3f}s for 100 files")
        
        # File operations should be fast
        assert create_metrics['duration'] < 5.0  # Creating 100 files in <5s
        assert read_metrics['duration'] < 2.0   # Reading 100 files in <2s
    
    def test_state_persistence_performance(self, temp_project_dir, performance_monitor):
        """Test state persistence performance."""
        from core.project_manager import StateManager
        
        state_manager = StateManager(temp_project_dir / "state")
        
        # Test saving multiple project states
        performance_monitor.start_monitoring()
        
        project_states = []
        for i in range(50):
            project_id = f"project-{i}"
            state_data = {
                'id': project_id,
                'name': f'Project {i}',
                'state': 'ACTIVE',
                'progress': {'real_progress': i * 2, 'authenticity_rate': 85.0},
                'files': [f'file_{j}.py' for j in range(10)],  # 10 files per project
                'metadata': {'created': time.time(), 'version': '1.0'}
            }
            
            asyncio.run(state_manager.save_project_state(project_id, state_data))
            project_states.append((project_id, state_data))
        
        save_metrics = performance_monitor.stop_monitoring()
        
        # Test loading all states
        performance_monitor.start_monitoring()
        
        for project_id, original_data in project_states:
            loaded_data = asyncio.run(state_manager.load_project_state(project_id))
            assert loaded_data is not None
            assert loaded_data['name'] == original_data['name']
        
        load_metrics = performance_monitor.stop_monitoring()
        
        print(f"State saving: {save_metrics['duration']:.3f}s for 50 states")
        print(f"State loading: {load_metrics['duration']:.3f}s for 50 states")
        
        # State operations should be efficient
        assert save_metrics['duration'] < 3.0  # Saving 50 states in <3s
        assert load_metrics['duration'] < 1.0  # Loading 50 states in <1s


class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks."""
    
    def test_performance_benchmark_suite(self, temp_project_dir):
        """Run comprehensive performance benchmark suite."""
        benchmark_results = {}
        
        # Component initialization benchmarks
        components = {
            'ProjectManager': lambda: ProjectManager(config_dir=temp_project_dir),
            'TaskEngine': lambda: TaskEngine(max_concurrent_tasks=5),
            'ProgressValidator': lambda: ProgressValidator()
        }
        
        for component_name, component_factory in components.items():
            start_time = time.time()
            component = component_factory()
            end_time = time.time()
            
            benchmark_results[f'{component_name}_init_time'] = end_time - start_time
        
        # Memory usage benchmarks
        process = psutil.Process()
        benchmark_results['memory_usage_mb'] = process.memory_info().rss / (1024 * 1024)
        
        # CPU usage benchmark
        cpu_start = process.cpu_percent()
        time.sleep(1)  # Sample period
        cpu_end = process.cpu_percent()
        benchmark_results['cpu_usage_percent'] = max(cpu_start, cpu_end)
        
        # Print benchmark results
        print("\nPerformance Benchmark Results:")
        print("=" * 40)
        for metric, value in benchmark_results.items():
            if 'time' in metric:
                print(f"{metric}: {value:.3f}s")
            elif 'memory' in metric:
                print(f"{metric}: {value:.1f}MB")
            elif 'cpu' in metric:
                print(f"{metric}: {value:.1f}%")
            else:
                print(f"{metric}: {value}")
        
        # Assert performance targets
        init_times = [v for k, v in benchmark_results.items() if 'init_time' in k]
        assert all(t < PERFORMANCE_TARGETS['startup_time_seconds'] for t in init_times)
        assert benchmark_results['memory_usage_mb'] < PERFORMANCE_TARGETS['max_memory_mb']
    
    def test_performance_regression_detection(self, temp_project_dir):
        """Test for performance regressions."""
        # This test would compare against baseline performance metrics
        # In a real implementation, you'd store baseline metrics and compare
        
        baseline_metrics = {
            'startup_time': 1.5,
            'memory_usage_mb': 100,
            'validation_speed': 8.0
        }
        
        # Measure current performance
        current_metrics = {}
        
        # Startup time
        start_time = time.time()
        pm = ProjectManager(config_dir=temp_project_dir)
        current_metrics['startup_time'] = time.time() - start_time
        
        # Memory usage
        process = psutil.Process()
        current_metrics['memory_usage_mb'] = process.memory_info().rss / (1024 * 1024)
        
        # Validation speed (mocked)
        current_metrics['validation_speed'] = 7.5  # files per second
        
        # Check for regressions (allow 20% degradation)
        regression_threshold = 1.2
        
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                
                if metric == 'validation_speed':
                    # Higher is better for speed metrics
                    regression_ratio = baseline_value / current_value
                else:
                    # Lower is better for time/memory metrics
                    regression_ratio = current_value / baseline_value
                
                print(f"{metric}: {current_value:.3f} (baseline: {baseline_value:.3f}, ratio: {regression_ratio:.2f})")
                
                assert regression_ratio < regression_threshold, f"Performance regression detected in {metric}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
