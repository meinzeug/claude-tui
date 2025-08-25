"""
Comprehensive load and performance testing for Claude-TIU.

This module tests system performance under various load conditions,
scalability limits, and resource utilization patterns.
"""

import pytest
import asyncio
import time
import threading
import multiprocessing
import psutil
import gc
from typing import List, Dict, Any, Callable
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import statistics
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    success_rate: float
    error_count: int
    concurrent_requests: int


@dataclass
class LoadTestResult:
    """Load test result container."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    max_response_time: float
    min_response_time: float
    percentile_95: float
    throughput: float
    errors: List[str]


class TestPerformanceBenchmarks:
    """Performance benchmarking tests with specific targets."""
    
    @pytest.fixture
    def performance_targets(self):
        """Define performance targets for various operations."""
        return {
            'project_creation': {
                'max_time': 2.0,  # seconds
                'max_memory': 50 * 1024 * 1024,  # 50MB
                'min_throughput': 10  # operations per second
            },
            'task_execution': {
                'max_time': 30.0,  # seconds
                'max_memory': 200 * 1024 * 1024,  # 200MB
                'min_throughput': 2  # tasks per second
            },
            'validation': {
                'max_time': 5.0,  # seconds
                'max_memory': 100 * 1024 * 1024,  # 100MB
                'min_throughput': 5  # validations per second
            },
            'api_response': {
                'max_time': 1.0,  # seconds
                'max_memory': 20 * 1024 * 1024,  # 20MB
                'min_throughput': 100  # requests per second
            }
        }
    
    @pytest.fixture
    def mock_system_components(self):
        """Mock system components for performance testing."""
        class MockComponents:
            def __init__(self):
                self.project_manager = Mock()
                self.task_engine = Mock()
                self.ai_interface = Mock()
                self.validator = Mock()
                
                # Configure mock behaviors with realistic delays
                self.project_manager.create_project = Mock(
                    side_effect=lambda *args: self._simulate_work(0.1, {"id": "proj123"})
                )
                self.task_engine.execute_task = AsyncMock(
                    side_effect=lambda *args: self._simulate_async_work(2.0, {"status": "completed"})
                )
                self.ai_interface.validate_code = AsyncMock(
                    side_effect=lambda *args: self._simulate_async_work(1.0, {"authentic": True})
                )
                self.validator.validate_progress = AsyncMock(
                    side_effect=lambda *args: self._simulate_async_work(0.5, {"real_progress": 80})
                )
            
            def _simulate_work(self, delay: float, result: Any):
                """Simulate synchronous work with delay."""
                time.sleep(delay)
                return result
            
            async def _simulate_async_work(self, delay: float, result: Any):
                """Simulate asynchronous work with delay."""
                await asyncio.sleep(delay)
                return result
        
        return MockComponents()
    
    def measure_performance(self, operation: Callable, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of an operation."""
        # Get initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_cpu_percent = process.cpu_percent()
        
        # Execute operation with timing
        start_time = time.time()
        try:
            result = operation(*args, **kwargs)
            success = True
            error_count = 0
        except Exception as e:
            result = None
            success = False
            error_count = 1
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Get final resource usage
        final_memory = process.memory_info().rss
        final_cpu_percent = process.cpu_percent()
        
        # Calculate metrics
        memory_usage = final_memory - initial_memory
        cpu_usage = max(final_cpu_percent - initial_cpu_percent, 0)
        throughput = 1.0 / execution_time if execution_time > 0 else 0
        success_rate = 1.0 if success else 0.0
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            success_rate=success_rate,
            error_count=error_count,
            concurrent_requests=1
        )
    
    @pytest.mark.performance
    def test_project_creation_performance(self, mock_system_components, performance_targets):
        """Test project creation performance against targets."""
        # Arrange
        project_data = {
            "name": "perf-test-project",
            "template": "python",
            "description": "Performance testing project"
        }
        target = performance_targets['project_creation']
        
        # Act
        metrics = self.measure_performance(
            mock_system_components.project_manager.create_project,
            project_data
        )
        
        # Assert
        assert metrics.execution_time < target['max_time'], f"Execution time {metrics.execution_time:.2f}s exceeds target {target['max_time']}s"
        assert metrics.memory_usage < target['max_memory'], f"Memory usage {metrics.memory_usage} exceeds target {target['max_memory']}"
        assert metrics.throughput >= target['min_throughput'], f"Throughput {metrics.throughput:.2f} below target {target['min_throughput']}"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_task_execution_performance(self, mock_system_components, performance_targets):
        """Test task execution performance against targets."""
        # Arrange
        task_data = {
            "name": "perf-test-task",
            "prompt": "Generate performance test code",
            "type": "code_generation"
        }
        target = performance_targets['task_execution']
        
        # Act
        start_time = time.time()
        result = await mock_system_components.task_engine.execute_task(task_data)
        execution_time = time.time() - start_time
        
        # Assert
        assert execution_time < target['max_time'], f"Task execution time {execution_time:.2f}s exceeds target {target['max_time']}s"
        assert result is not None
    
    @pytest.mark.performance
    def test_memory_usage_stability(self, mock_system_components):
        """Test memory usage stability over many operations."""
        # Arrange
        initial_memory = psutil.Process().memory_info().rss
        operations = 100
        
        # Act - Perform many operations
        for i in range(operations):
            mock_system_components.project_manager.create_project({
                "name": f"test-project-{i}",
                "template": "python"
            })
            
            # Force garbage collection every 10 operations
            if i % 10 == 0:
                gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Assert - Memory growth should be reasonable
        max_acceptable_growth = 100 * 1024 * 1024  # 100MB
        assert memory_growth < max_acceptable_growth, f"Memory growth {memory_growth / 1024 / 1024:.2f}MB exceeds limit"
    
    @pytest.mark.performance
    def test_cpu_usage_efficiency(self, mock_system_components):
        """Test CPU usage efficiency during operations."""
        # Arrange
        process = psutil.Process()
        cpu_samples = []
        
        def monitor_cpu():
            """Monitor CPU usage in background."""
            for _ in range(10):
                cpu_samples.append(process.cpu_percent(interval=0.1))
        
        # Act - Start CPU monitoring and perform operations
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform CPU-intensive operations
        for i in range(20):
            mock_system_components.project_manager.create_project({
                "name": f"cpu-test-{i}",
                "template": "python"
            })
        
        monitor_thread.join()
        
        # Assert
        if cpu_samples:
            avg_cpu_usage = statistics.mean(cpu_samples)
            max_cpu_usage = max(cpu_samples)
            
            # CPU usage should be reasonable
            assert avg_cpu_usage < 80.0, f"Average CPU usage {avg_cpu_usage:.2f}% too high"
            assert max_cpu_usage < 95.0, f"Peak CPU usage {max_cpu_usage:.2f}% too high"


class TestConcurrentLoad:
    """Test concurrent load handling and scalability."""
    
    @pytest.fixture
    def load_test_config(self):
        """Configuration for load tests."""
        return {
            'concurrent_users': [1, 5, 10, 25, 50],
            'requests_per_user': 10,
            'max_response_time': 5.0,
            'min_success_rate': 0.95
        }
    
    @pytest.fixture
    def mock_api_server(self):
        """Mock API server for load testing."""
        class MockAPIServer:
            def __init__(self):
                self.request_count = 0
                self.active_requests = 0
                self.max_concurrent = 0
                self.request_times = []
                self.errors = []
                self.lock = threading.Lock()
            
            async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
                """Handle API request with realistic processing."""
                with self.lock:
                    self.active_requests += 1
                    self.max_concurrent = max(self.max_concurrent, self.active_requests)
                    self.request_count += 1
                
                start_time = time.time()
                
                try:
                    # Simulate request processing
                    endpoint = request_data.get('endpoint', '/unknown')
                    
                    if endpoint == '/projects':
                        await asyncio.sleep(0.1)  # Simulate database query
                        result = {"status": "success", "data": {"id": "proj123"}}\n                    elif endpoint == '/tasks':
                        await asyncio.sleep(0.5)  # Simulate task processing
                        result = {"status": "success", "data": {"task_id": "task456"}}
                    elif endpoint == '/validate':
                        await asyncio.sleep(0.2)  # Simulate validation
                        result = {"status": "success", "data": {"authentic": True}}
                    else:
                        result = {"status": "success", "data": {}}
                    
                    # Simulate occasional errors under high load
                    if self.active_requests > 30:
                        if time.time() % 10 < 1:  # 10% error rate under high load
                            raise Exception("Service overloaded")
                
                except Exception as e:
                    with self.lock:
                        self.errors.append(str(e))
                    result = {"status": "error", "message": str(e)}
                
                finally:
                    end_time = time.time()
                    with self.lock:
                        self.active_requests -= 1
                        self.request_times.append(end_time - start_time)
                
                return result
            
            def get_statistics(self) -> Dict[str, Any]:
                """Get server statistics."""
                with self.lock:
                    if self.request_times:
                        avg_response_time = statistics.mean(self.request_times)
                        max_response_time = max(self.request_times)
                        min_response_time = min(self.request_times)
                        percentile_95 = np.percentile(self.request_times, 95) if len(self.request_times) > 1 else avg_response_time
                    else:
                        avg_response_time = max_response_time = min_response_time = percentile_95 = 0
                    
                    return {
                        'total_requests': self.request_count,
                        'successful_requests': self.request_count - len(self.errors),
                        'failed_requests': len(self.errors),
                        'average_response_time': avg_response_time,
                        'max_response_time': max_response_time,
                        'min_response_time': min_response_time,
                        'percentile_95': percentile_95,
                        'max_concurrent': self.max_concurrent,
                        'errors': self.errors.copy()
                    }
        
        return MockAPIServer()
    
    async def simulate_user_load(self, server: Any, user_id: int, requests: int, endpoints: List[str]) -> List[float]:
        """Simulate load from a single user."""
        response_times = []
        
        for i in range(requests):
            endpoint = endpoints[i % len(endpoints)]
            request_data = {
                'endpoint': endpoint,
                'user_id': user_id,
                'request_id': i
            }
            
            start_time = time.time()
            try:
                await server.handle_request(request_data)
                response_time = time.time() - start_time
                response_times.append(response_time)
            except Exception as e:
                # Log error but continue
                response_times.append(-1)  # Error indicator
            
            # Small delay between requests
            await asyncio.sleep(0.01)
        
        return response_times
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_user_load(self, mock_api_server, load_test_config):
        """Test system behavior under concurrent user load."""
        # Test different concurrency levels
        results = {}
        
        for concurrent_users in load_test_config['concurrent_users']:
            # Reset server statistics
            mock_api_server.request_count = 0
            mock_api_server.errors = []
            mock_api_server.request_times = []
            
            # Simulate concurrent users
            endpoints = ['/projects', '/tasks', '/validate']
            user_tasks = []
            
            start_time = time.time()
            
            for user_id in range(concurrent_users):
                task = self.simulate_user_load(
                    mock_api_server, 
                    user_id, 
                    load_test_config['requests_per_user'],
                    endpoints
                )
                user_tasks.append(task)
            
            # Wait for all users to complete
            user_results = await asyncio.gather(*user_tasks)
            total_time = time.time() - start_time
            
            # Collect statistics
            stats = mock_api_server.get_statistics()
            throughput = stats['total_requests'] / total_time if total_time > 0 else 0
            success_rate = stats['successful_requests'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            
            results[concurrent_users] = LoadTestResult(
                test_name=f"concurrent_users_{concurrent_users}",
                total_requests=stats['total_requests'],
                successful_requests=stats['successful_requests'],
                failed_requests=stats['failed_requests'],
                average_response_time=stats['average_response_time'],
                max_response_time=stats['max_response_time'],
                min_response_time=stats['min_response_time'],
                percentile_95=stats['percentile_95'],
                throughput=throughput,
                errors=stats['errors']
            )
        
        # Assert performance requirements
        for concurrent_users, result in results.items():
            # Success rate should remain high
            success_rate = result.successful_requests / result.total_requests
            assert success_rate >= load_test_config['min_success_rate'], f"Success rate {success_rate:.2%} below minimum for {concurrent_users} users"
            
            # Response times should be reasonable
            assert result.average_response_time < load_test_config['max_response_time'], f"Average response time {result.average_response_time:.2f}s too high for {concurrent_users} users"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_sustained_load(self, mock_api_server):
        """Test system behavior under sustained load."""
        # Arrange
        duration_minutes = 2  # Reduced for testing
        requests_per_second = 10
        total_requests = duration_minutes * 60 * requests_per_second
        
        # Act - Generate sustained load
        start_time = time.time()
        request_tasks = []
        
        for i in range(total_requests):
            request_data = {
                'endpoint': '/projects' if i % 2 == 0 else '/validate',
                'request_id': i
            }
            task = mock_api_server.handle_request(request_data)
            request_tasks.append(task)
            
            # Control request rate
            if i % requests_per_second == 0:
                await asyncio.sleep(1.0)
        
        # Wait for completion
        await asyncio.gather(*request_tasks)
        total_time = time.time() - start_time
        
        # Assert
        stats = mock_api_server.get_statistics()
        
        # Throughput should be maintained
        actual_throughput = stats['total_requests'] / total_time
        expected_throughput = requests_per_second * 0.8  # Allow 20% variance
        assert actual_throughput >= expected_throughput, f"Throughput {actual_throughput:.2f} rps below expected {expected_throughput:.2f} rps"
        
        # Error rate should be low
        error_rate = len(stats['errors']) / stats['total_requests']
        assert error_rate < 0.05, f"Error rate {error_rate:.2%} too high for sustained load"
    
    @pytest.mark.performance
    def test_resource_utilization_under_load(self):
        """Test resource utilization patterns under high load."""
        # Arrange
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        cpu_samples = []
        memory_samples = []
        
        def monitor_resources():
            """Monitor system resources."""
            for _ in range(30):  # Monitor for 30 seconds
                cpu_samples.append(process.cpu_percent(interval=1.0))
                memory_samples.append(process.memory_info().rss)
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Generate load
        def cpu_intensive_task():
            """CPU intensive task."""
            for i in range(1000000):
                _ = i ** 2
        
        # Use thread pool to generate concurrent load
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(20)]
            for future in futures:
                future.result()
        
        monitor_thread.join()
        
        # Assert resource usage patterns
        if cpu_samples and memory_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_memory = max(memory_samples)
            memory_growth = max_memory - initial_memory
            
            # CPU should be utilized but not constantly maxed out
            assert avg_cpu < 90.0, f"Average CPU usage {avg_cpu:.2f}% too high"
            
            # Memory growth should be controlled
            max_memory_growth = 500 * 1024 * 1024  # 500MB
            assert memory_growth < max_memory_growth, f"Memory growth {memory_growth / 1024 / 1024:.2f}MB excessive"


class TestScalabilityLimits:
    """Test system scalability limits and breaking points."""
    
    @pytest.fixture
    def scalability_test_config(self):
        """Configuration for scalability tests."""
        return {
            'max_concurrent_projects': 100,
            'max_concurrent_tasks': 50,
            'max_memory_usage': 1024 * 1024 * 1024,  # 1GB
            'degradation_threshold': 0.5  # 50% performance degradation
        }
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_maximum_concurrent_projects(self, scalability_test_config):
        """Test maximum number of concurrent projects."""
        # Arrange
        max_projects = scalability_test_config['max_concurrent_projects']
        projects = []
        creation_times = []
        
        # Mock project manager with resource tracking
        class ResourceTrackingProjectManager:
            def __init__(self):
                self.active_projects = {}
                self.creation_count = 0
            
            def create_project(self, project_data):
                start_time = time.time()
                
                # Simulate resource usage
                time.sleep(0.01 * len(self.active_projects))  # Slower with more projects
                
                project_id = f"proj_{self.creation_count}"
                self.creation_count += 1
                self.active_projects[project_id] = {
                    'data': project_data,
                    'created_at': time.time()
                }
                
                creation_time = time.time() - start_time
                return {'id': project_id, 'creation_time': creation_time}
        
        project_manager = ResourceTrackingProjectManager()
        
        # Act - Create projects and measure degradation
        for i in range(max_projects):
            project_data = {
                'name': f'scalability-test-{i}',
                'template': 'python'
            }
            
            start_time = time.time()
            project = project_manager.create_project(project_data)
            creation_time = time.time() - start_time
            
            projects.append(project)
            creation_times.append(creation_time)
            
            # Check for performance degradation
            if i > 10:  # Skip initial warmup
                baseline_time = statistics.mean(creation_times[:10])
                current_time = creation_time
                degradation = (current_time - baseline_time) / baseline_time
                
                if degradation > scalability_test_config['degradation_threshold']:
                    print(f"Performance degraded by {degradation:.1%} at {i} projects")
        
        # Assert
        assert len(projects) > 50, "Should handle at least 50 concurrent projects"
        
        # Performance should not degrade excessively
        if len(creation_times) > 20:
            early_avg = statistics.mean(creation_times[:10])
            late_avg = statistics.mean(creation_times[-10:])
            degradation = (late_avg - early_avg) / early_avg
            
            assert degradation < 2.0, f"Performance degraded by {degradation:.1%}, exceeding 200% limit"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, scalability_test_config):
        """Test system behavior under memory pressure."""
        # Arrange
        max_memory = scalability_test_config['max_memory_usage']
        memory_allocations = []
        
        # Mock memory-intensive operations
        class MemoryIntensiveProcessor:
            def __init__(self):
                self.data_cache = {}
                self.processed_items = []
            
            async def process_large_dataset(self, size_mb: int) -> Dict[str, Any]:
                """Process large dataset that requires significant memory."""
                # Allocate memory for processing
                data = bytearray(size_mb * 1024 * 1024)  # Allocate size_mb MB
                
                # Simulate processing
                await asyncio.sleep(0.1)
                
                # Store in cache (simulating memory retention)
                cache_key = f"dataset_{len(self.data_cache)}"
                self.data_cache[cache_key] = data
                
                # Monitor memory usage
                process = psutil.Process()
                current_memory = process.memory_info().rss
                
                return {
                    'processed': True,
                    'cache_key': cache_key,
                    'memory_usage': current_memory
                }
        
        processor = MemoryIntensiveProcessor()
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Act - Process increasingly large datasets
        try:
            for size_mb in [10, 25, 50, 100, 200]:
                result = await processor.process_large_dataset(size_mb)
                current_memory = result['memory_usage']
                memory_growth = current_memory - initial_memory
                
                memory_allocations.append({
                    'dataset_size_mb': size_mb,
                    'memory_usage': current_memory,
                    'memory_growth': memory_growth
                })
                
                # Stop if approaching memory limit
                if memory_growth > max_memory * 0.8:  # 80% of limit
                    break
        
        except MemoryError:
            # This is expected when hitting memory limits
            pass
        
        # Assert
        assert len(memory_allocations) > 0, "Should process at least one dataset"
        
        # Memory usage should grow predictably
        if len(memory_allocations) > 1:
            memory_growths = [alloc['memory_growth'] for alloc in memory_allocations]
            # Memory growth should not be completely erratic
            growth_variance = statistics.variance(memory_growths) if len(memory_growths) > 1 else 0
            growth_mean = statistics.mean(memory_growths)
            
            if growth_mean > 0:
                coefficient_of_variation = (growth_variance ** 0.5) / growth_mean
                assert coefficient_of_variation < 2.0, "Memory usage growth too erratic"
    
    @pytest.mark.performance
    def test_error_handling_under_stress(self):
        """Test error handling and recovery under stress conditions."""
        # Arrange
        error_conditions = [
            'network_timeout',
            'database_connection_lost',
            'api_rate_limit',
            'insufficient_memory',
            'cpu_overload'
        ]
        
        class StressTestSystem:
            def __init__(self):
                self.error_count = 0
                self.recovery_count = 0
                self.consecutive_errors = 0
                self.max_consecutive_errors = 0
            
            def process_request_with_errors(self, request_id: int) -> Dict[str, Any]:
                """Process request with simulated error conditions."""
                # Simulate different error conditions based on request_id
                error_condition = error_conditions[request_id % len(error_conditions)]
                
                # Simulate error probability increases with consecutive errors
                error_probability = min(0.8, 0.1 + (self.consecutive_errors * 0.1))
                
                if time.time() % 1 < error_probability:
                    # Simulate error
                    self.error_count += 1
                    self.consecutive_errors += 1
                    self.max_consecutive_errors = max(self.max_consecutive_errors, self.consecutive_errors)
                    
                    # Simulate recovery mechanism
                    time.sleep(0.05)  # Recovery delay
                    
                    if self.consecutive_errors > 3:
                        # Circuit breaker pattern
                        self.recovery_count += 1
                        self.consecutive_errors = 0
                        return {'status': 'recovered', 'request_id': request_id}
                    else:
                        raise Exception(f"Simulated {error_condition} error")
                else:
                    # Successful processing
                    self.consecutive_errors = 0
                    time.sleep(0.01)  # Normal processing time
                    return {'status': 'success', 'request_id': request_id}
        
        system = StressTestSystem()
        
        # Act - Process many requests under stress
        successful_requests = 0
        failed_requests = 0
        
        for i in range(100):
            try:
                result = system.process_request_with_errors(i)
                if result['status'] in ['success', 'recovered']:
                    successful_requests += 1
                else:
                    failed_requests += 1
            except Exception:
                failed_requests += 1
        
        # Assert
        success_rate = successful_requests / (successful_requests + failed_requests)
        
        # System should maintain reasonable success rate even under stress
        assert success_rate >= 0.6, f"Success rate {success_rate:.2%} too low under stress"
        
        # Recovery mechanisms should activate
        assert system.recovery_count > 0, "Recovery mechanisms should have activated"
        
        # Consecutive errors should be bounded
        assert system.max_consecutive_errors <= 10, f"Too many consecutive errors: {system.max_consecutive_errors}"
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_long_running_stability(self):
        """Test system stability over extended periods."""
        # Arrange
        duration_minutes = 5  # Reduced for testing
        check_interval = 30  # seconds
        
        class LongRunningSystem:
            def __init__(self):
                self.start_time = time.time()
                self.operation_count = 0
                self.memory_samples = []
                self.error_count = 0
            
            def perform_operation(self) -> bool:
                """Perform a typical system operation."""
                try:
                    # Simulate varying workload
                    workload_factor = (time.time() - self.start_time) % 60 / 60  # 0-1 over 60s cycle
                    processing_time = 0.01 + (workload_factor * 0.05)
                    
                    time.sleep(processing_time)
                    self.operation_count += 1
                    
                    # Collect memory sample
                    memory_usage = psutil.Process().memory_info().rss
                    self.memory_samples.append(memory_usage)
                    
                    return True
                    
                except Exception as e:
                    self.error_count += 1
                    return False
            
            def get_stability_metrics(self) -> Dict[str, Any]:
                """Get stability metrics."""
                runtime = time.time() - self.start_time
                
                return {
                    'runtime_seconds': runtime,
                    'total_operations': self.operation_count,
                    'operations_per_second': self.operation_count / runtime if runtime > 0 else 0,
                    'error_count': self.error_count,
                    'error_rate': self.error_count / self.operation_count if self.operation_count > 0 else 0,
                    'memory_samples': len(self.memory_samples),
                    'memory_trend': self._calculate_memory_trend()
                }
            
            def _calculate_memory_trend(self) -> str:
                """Calculate memory usage trend."""
                if len(self.memory_samples) < 10:
                    return 'insufficient_data'
                
                # Compare first and last quartile
                first_quartile = self.memory_samples[:len(self.memory_samples)//4]
                last_quartile = self.memory_samples[-len(self.memory_samples)//4:]
                
                avg_first = statistics.mean(first_quartile)
                avg_last = statistics.mean(last_quartile)
                
                growth_rate = (avg_last - avg_first) / avg_first if avg_first > 0 else 0
                
                if growth_rate > 0.1:
                    return 'increasing'
                elif growth_rate < -0.1:
                    return 'decreasing'
                else:
                    return 'stable'
        
        system = LongRunningSystem()
        
        # Act - Run system for extended period
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            system.perform_operation()
            
            # Brief pause between operations
            time.sleep(0.1)
        
        # Get final metrics
        metrics = system.get_stability_metrics()
        
        # Assert
        assert metrics['runtime_seconds'] >= duration_minutes * 60 * 0.9, "Should run for nearly full duration"
        assert metrics['operations_per_second'] > 1.0, "Should maintain reasonable throughput"
        assert metrics['error_rate'] < 0.1, f"Error rate {metrics['error_rate']:.2%} too high for long-running stability"
        assert metrics['memory_trend'] != 'increasing', "Memory usage should not continuously increase"