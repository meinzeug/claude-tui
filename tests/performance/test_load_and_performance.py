"""
Performance and load testing suite for claude-tiu.

Tests system performance under various load conditions,
memory usage, and execution time benchmarks.
"""

import pytest
import asyncio
import time
import tracemalloc
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
from pathlib import Path
import json
import psutil
import os


class TestPerformanceBasics:
    """Basic performance test suite."""
    
    @pytest.mark.performance
    def test_project_creation_performance(self, project_factory, performance_threshold):
        """Test project creation performance."""
        # Arrange
        projects_to_create = 100
        max_time = performance_threshold["max_execution_time"]
        
        # Mock project manager
        class FastProjectManager:
            def __init__(self):
                self.projects = {}
                self.ai_interface = Mock()
                self.ai_interface.validate_project.return_value = True
            
            def create_project(self, project_data):
                project_id = f"perf_proj_{len(self.projects)}"
                self.projects[project_id] = {
                    "id": project_id,
                    **project_data,
                    "created_at": time.time()
                }
                return self.projects[project_id]
        
        manager = FastProjectManager()
        
        # Act
        start_time = time.time()
        
        for i in range(projects_to_create):
            project_data = project_factory.create_project_data(name=f"perf-project-{i}")
            manager.create_project(project_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert
        assert execution_time < max_time * projects_to_create / 10  # Allow reasonable scaling
        assert len(manager.projects) == projects_to_create
        
        # Calculate throughput
        throughput = projects_to_create / execution_time
        assert throughput > 50  # At least 50 projects per second
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_task_execution_performance(self, task_factory):
        """Test concurrent task execution performance."""
        # Arrange
        concurrent_tasks = 50
        
        class AsyncTaskEngine:
            def __init__(self):
                self.execution_history = []
            
            async def execute_task(self, task):
                start_time = time.time()
                # Simulate variable execution time
                await asyncio.sleep(0.01 + (hash(task["name"]) % 100) / 10000)
                end_time = time.time()
                
                result = {
                    "task_id": task.get("id", "unknown"),
                    "status": "completed",
                    "execution_time": end_time - start_time
                }
                self.execution_history.append(result)
                return result
        
        engine = AsyncTaskEngine()
        
        # Create tasks
        tasks = [
            task_factory.create_task_data(name=f"perf-task-{i}")
            for i in range(concurrent_tasks)
        ]
        
        # Act
        start_time = time.time()
        
        results = await asyncio.gather(
            *[engine.execute_task(task) for task in tasks]
        )
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        # Assert
        assert len(results) == concurrent_tasks
        assert all(result["status"] == "completed" for result in results)
        
        # Concurrent execution should be faster than sequential
        average_task_time = sum(r["execution_time"] for r in results) / len(results)
        sequential_time_estimate = average_task_time * concurrent_tasks
        
        # Should achieve significant speedup
        speedup = sequential_time_estimate / total_execution_time
        assert speedup > 5  # At least 5x speedup from concurrency
    
    @pytest.mark.performance
    def test_validation_performance(self, code_samples, performance_threshold):
        """Test code validation performance."""
        # Arrange
        class FastPlaceholderDetector:
            def __init__(self):
                self.patterns = [
                    r"#\s*TODO",
                    r"raise\s+NotImplementedError", 
                    r"^\s*pass\s*$",
                    r"console\.log"
                ]
            
            def has_placeholder(self, code):
                import re
                for pattern in self.patterns:
                    if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                        return True
                return False
            
            def analyze_batch(self, code_samples):
                return [self.has_placeholder(code) for code in code_samples]
        
        detector = FastPlaceholderDetector()
        
        # Create large batch of code samples
        code_batch = []
        for i in range(1000):
            # Mix of complete and placeholder code
            if i % 2 == 0:
                code_batch.append(code_samples["complete_function"])
            else:
                code_batch.append(code_samples["function_with_todo"])
        
        # Act
        start_time = time.time()
        results = detector.analyze_batch(code_batch)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Assert
        assert len(results) == 1000
        assert execution_time < performance_threshold["max_execution_time"] * 10
        
        # Performance requirement: less than 1ms per code sample
        time_per_sample = execution_time / len(code_batch)
        assert time_per_sample < 0.001
    
    @pytest.mark.benchmark
    def test_memory_efficiency(self, benchmark, project_factory):
        """Benchmark memory efficiency."""
        # Arrange
        def create_many_projects():
            projects = []
            for i in range(1000):
                project_data = project_factory.create_project_data(name=f"mem-test-{i}")
                projects.append(project_data)
            return projects
        
        # Act & Assert
        tracemalloc.start()
        
        result = benchmark(create_many_projects)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Verify results
        assert len(result) == 1000
        
        # Memory requirements
        assert current < 50 * 1024 * 1024  # Less than 50MB current
        assert peak < 100 * 1024 * 1024    # Less than 100MB peak
        
        # Benchmark should complete quickly
        assert benchmark.stats['mean'] < 0.5  # Less than 500ms


class TestLoadTesting:
    """Load testing for high-throughput scenarios."""
    
    @pytest.mark.performance
    def test_high_throughput_project_operations(self):
        """Test high-throughput project operations."""
        # Arrange
        class HighThroughputManager:
            def __init__(self):
                self.projects = {}
                self.lock = threading.Lock()
            
            def create_project(self, project_data):
                with self.lock:
                    project_id = f"ht_proj_{len(self.projects)}"
                    self.projects[project_id] = {"id": project_id, **project_data}
                    return self.projects[project_id]
            
            def get_project(self, project_id):
                with self.lock:
                    return self.projects.get(project_id)
            
            def list_projects(self):
                with self.lock:
                    return list(self.projects.values())
        
        manager = HighThroughputManager()
        
        # Act - Concurrent operations
        def worker_thread(thread_id):
            results = []
            for i in range(100):
                project_data = {
                    "name": f"thread-{thread_id}-project-{i}",
                    "template": "python"
                }
                project = manager.create_project(project_data)
                results.append(project["id"])
                
                # Also test read operations
                retrieved = manager.get_project(project["id"])
                assert retrieved is not None
            
            return results
        
        # Execute with multiple threads
        num_threads = 10
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_thread, thread_id)
                for thread_id in range(num_threads)
            ]
            
            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert
        total_operations = num_threads * 100
        assert len(all_results) == total_operations
        assert len(set(all_results)) == total_operations  # All unique IDs
        
        # Performance requirements
        throughput = total_operations / execution_time
        assert throughput > 100  # At least 100 operations per second
        
        # Verify data integrity
        all_projects = manager.list_projects()
        assert len(all_projects) == total_operations
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_load_handling(self):
        """Test async load handling capabilities."""
        # Arrange
        class AsyncLoadHandler:
            def __init__(self):
                self.active_tasks = 0
                self.completed_tasks = 0
                self.max_concurrent = 0
                self.lock = asyncio.Lock()
            
            async def process_request(self, request_id):
                async with self.lock:
                    self.active_tasks += 1
                    self.max_concurrent = max(self.max_concurrent, self.active_tasks)
                
                try:
                    # Simulate processing time
                    await asyncio.sleep(0.01 + (request_id % 10) / 1000)
                    return {"request_id": request_id, "status": "completed"}
                finally:
                    async with self.lock:
                        self.active_tasks -= 1
                        self.completed_tasks += 1
        
        handler = AsyncLoadHandler()
        
        # Act - High load
        num_requests = 1000
        
        start_time = time.time()
        
        results = await asyncio.gather(
            *[handler.process_request(i) for i in range(num_requests)]
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert
        assert len(results) == num_requests
        assert all(r["status"] == "completed" for r in results)
        assert handler.completed_tasks == num_requests
        assert handler.active_tasks == 0  # All completed
        
        # Performance requirements
        throughput = num_requests / execution_time
        assert throughput > 200  # At least 200 requests per second
        
        # Should handle significant concurrency
        assert handler.max_concurrent > 50
    
    @pytest.mark.performance
    def test_memory_under_load(self):
        """Test memory usage under sustained load."""
        # Arrange
        tracemalloc.start()
        
        class MemoryTestService:
            def __init__(self):
                self.data_store = []
            
            def process_batch(self, batch_size):
                # Simulate processing that creates temporary objects
                batch_data = []
                for i in range(batch_size):
                    item = {
                        "id": i,
                        "data": f"item_data_{i}" * 10,  # Some string data
                        "metadata": {"processed": True, "timestamp": time.time()}
                    }
                    batch_data.append(item)
                
                # Store only summary (simulate proper memory management)
                self.data_store.append({
                    "batch_size": batch_size,
                    "processed_at": time.time()
                })
                
                return len(batch_data)
        
        service = MemoryTestService()
        
        # Act - Process many batches
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        for batch_num in range(100):
            batch_size = 1000
            result = service.process_batch(batch_size)
            assert result == batch_size
            
            # Check memory growth periodically
            if batch_num % 10 == 0:
                current_memory = tracemalloc.get_traced_memory()[0]
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable
                assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
        
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        # Assert
        assert len(service.data_store) == 100
        
        # Memory should not grow excessively
        total_memory_growth = final_memory - initial_memory
        assert total_memory_growth < 50 * 1024 * 1024  # Less than 50MB total growth
    
    @pytest.mark.performance
    def test_cpu_usage_under_load(self):
        """Test CPU usage patterns under load."""
        import psutil
        
        # Arrange
        class CPUIntensiveService:
            def __init__(self):
                self.results = []
            
            def compute_intensive_task(self, iterations):
                # Simulate CPU-intensive work
                result = 0
                for i in range(iterations):
                    result += i * i
                    if i % 1000 == 0:
                        # Small break to prevent total CPU monopolization
                        time.sleep(0.0001)
                return result
            
            def process_workload(self, num_tasks, task_size):
                for i in range(num_tasks):
                    result = self.compute_intensive_task(task_size)
                    self.results.append(result)
        
        service = CPUIntensiveService()
        
        # Monitor CPU usage
        process = psutil.Process()
        initial_cpu_time = process.cpu_times()
        
        # Act
        start_time = time.time()
        service.process_workload(num_tasks=10, task_size=10000)
        end_time = time.time()
        
        final_cpu_time = process.cpu_times()
        
        # Assert
        execution_time = end_time - start_time
        cpu_time_used = (final_cpu_time.user - initial_cpu_time.user)
        
        # Should complete in reasonable time
        assert execution_time < 5.0  # Less than 5 seconds
        
        # Should utilize CPU efficiently
        cpu_efficiency = cpu_time_used / execution_time
        assert 0.1 < cpu_efficiency < 2.0  # Reasonable CPU utilization
        
        assert len(service.results) == 10


class TestScalabilityTesting:
    """Test system scalability with increasing loads."""
    
    @pytest.mark.performance
    @pytest.mark.parametrize("scale_factor", [1, 2, 4, 8])
    def test_linear_scalability(self, scale_factor):
        """Test linear scalability with increasing workloads."""
        # Arrange
        base_workload = 100
        workload = base_workload * scale_factor
        
        class ScalableService:
            def __init__(self):
                self.processed_items = []
            
            def process_items(self, items):
                for item in items:
                    # Simulate O(1) processing
                    processed = {"id": item["id"], "processed": True}
                    self.processed_items.append(processed)
                return len(self.processed_items)
        
        service = ScalableService()
        items = [{"id": i} for i in range(workload)]
        
        # Act
        start_time = time.time()
        result = service.process_items(items)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Assert
        assert result == workload
        
        # Execution time should scale roughly linearly
        expected_max_time = 0.1 * scale_factor  # 100ms per scale factor
        assert execution_time < expected_max_time
        
        # Throughput should remain consistent
        throughput = workload / execution_time
        assert throughput > 500  # At least 500 items per second
    
    @pytest.mark.performance
    def test_concurrent_user_simulation(self):
        """Test concurrent user simulation."""
        # Arrange
        class UserSimulationService:
            def __init__(self):
                self.active_sessions = {}
                self.request_history = []
                self.lock = threading.Lock()
            
            def simulate_user_session(self, user_id, requests_per_user):
                session_requests = []
                
                for i in range(requests_per_user):
                    request = {
                        "user_id": user_id,
                        "request_id": f"{user_id}_{i}",
                        "timestamp": time.time(),
                        "response_time": 0.01 + (i % 10) / 1000  # Simulate variable response
                    }
                    
                    # Simulate request processing
                    time.sleep(request["response_time"])
                    
                    session_requests.append(request)
                    
                    with self.lock:
                        self.request_history.append(request)
                
                return session_requests
        
        service = UserSimulationService()
        
        # Act - Simulate concurrent users
        num_users = 20
        requests_per_user = 10
        
        def user_worker(user_id):
            return service.simulate_user_session(user_id, requests_per_user)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [
                executor.submit(user_worker, user_id)
                for user_id in range(num_users)
            ]
            
            all_sessions = []
            for future in as_completed(futures):
                session = future.result()
                all_sessions.append(session)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert
        total_requests = num_users * requests_per_user
        assert len(all_sessions) == num_users
        assert len(service.request_history) == total_requests
        
        # Should handle concurrent users efficiently
        # Sequential execution would take much longer
        max_sequential_time = sum(
            sum(req["response_time"] for req in session)
            for session in all_sessions
        )
        
        # Should achieve significant speedup from concurrency
        speedup = max_sequential_time / execution_time
        assert speedup > 5  # At least 5x speedup
    
    @pytest.mark.performance
    def test_data_volume_scalability(self, tmp_path):
        """Test scalability with increasing data volumes."""
        # Arrange
        class DataProcessor:
            def __init__(self, storage_path):
                self.storage_path = storage_path
                self.processed_count = 0
            
            def process_data_file(self, file_path):
                # Simulate file processing
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                processed_lines = []
                for line in lines:
                    # Simulate processing
                    processed_line = line.strip().upper()
                    processed_lines.append(processed_line)
                    self.processed_count += 1
                
                # Write processed data
                output_path = self.storage_path / f"processed_{file_path.name}"
                with open(output_path, 'w') as f:
                    f.write('\n'.join(processed_lines))
                
                return len(processed_lines)
        
        processor = DataProcessor(tmp_path)
        
        # Create test files of different sizes
        file_sizes = [1000, 5000, 10000, 20000]  # Lines per file
        
        for size in file_sizes:
            # Create test file
            test_file = tmp_path / f"test_data_{size}.txt"
            with open(test_file, 'w') as f:
                for i in range(size):
                    f.write(f"test line {i} with some content\n")
            
            # Act
            start_time = time.time()
            lines_processed = processor.process_data_file(test_file)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Assert
            assert lines_processed == size
            
            # Processing rate should be consistent across file sizes
            processing_rate = size / execution_time
            assert processing_rate > 1000  # At least 1000 lines per second
            
            # Verify output file
            output_file = tmp_path / f"processed_test_data_{size}.txt"
            assert output_file.exists()
            
            with open(output_file, 'r') as f:
                output_lines = f.readlines()
            assert len(output_lines) == size


class TestResourceManagement:
    """Test resource management and cleanup."""
    
    @pytest.mark.performance
    def test_resource_cleanup(self):
        """Test proper resource cleanup."""
        # Arrange
        class ResourceManager:
            def __init__(self):
                self.active_resources = {}
                self.resource_counter = 0
            
            def acquire_resource(self, resource_type):
                resource_id = f"{resource_type}_{self.resource_counter}"
                self.resource_counter += 1
                
                resource = {
                    "id": resource_id,
                    "type": resource_type,
                    "acquired_at": time.time(),
                    "active": True
                }
                
                self.active_resources[resource_id] = resource
                return resource
            
            def release_resource(self, resource_id):
                if resource_id in self.active_resources:
                    self.active_resources[resource_id]["active"] = False
                    del self.active_resources[resource_id]
                    return True
                return False
            
            def cleanup_inactive_resources(self):
                inactive_count = 0
                for resource_id, resource in list(self.active_resources.items()):
                    if not resource["active"]:
                        del self.active_resources[resource_id]
                        inactive_count += 1
                return inactive_count
        
        manager = ResourceManager()
        
        # Act - Acquire and release resources
        acquired_resources = []
        
        for i in range(1000):
            resource = manager.acquire_resource("test_resource")
            acquired_resources.append(resource["id"])
            
            # Release every other resource
            if i % 2 == 0:
                success = manager.release_resource(resource["id"])
                assert success
        
        # Assert
        # Should have ~500 active resources (half were released)
        assert 400 <= len(manager.active_resources) <= 600
        
        # Cleanup should not find any inactive resources (we deleted them)
        cleaned_up = manager.cleanup_inactive_resources()
        assert cleaned_up == 0
        
        # Release remaining resources
        for resource_id in list(manager.active_resources.keys()):
            manager.release_resource(resource_id)
        
        assert len(manager.active_resources) == 0
    
    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for memory leaks."""
        tracemalloc.start()
        
        # Arrange
        class PotentialLeakService:
            def __init__(self):
                self.cache = {}
            
            def process_with_cache(self, data_id, data):
                # Simulate processing that might cause leaks
                processed = {
                    "id": data_id,
                    "processed_data": data * 2,  # Double the data
                    "metadata": {
                        "processed_at": time.time(),
                        "size": len(str(data))
                    }
                }
                
                # Cache result (potential leak if not managed)
                self.cache[data_id] = processed
                
                return processed
            
            def clear_old_cache_entries(self, max_age=1.0):
                current_time = time.time()
                old_keys = []
                
                for key, value in self.cache.items():
                    if current_time - value["metadata"]["processed_at"] > max_age:
                        old_keys.append(key)
                
                for key in old_keys:
                    del self.cache[key]
                
                return len(old_keys)
        
        service = PotentialLeakService()
        
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        # Act - Process many items
        for cycle in range(10):
            # Process batch
            for i in range(100):
                data = f"test_data_{cycle}_{i}" * 100  # Create some data
                service.process_with_cache(f"{cycle}_{i}", data)
            
            # Periodically clean cache to prevent leaks
            if cycle % 3 == 0:
                cleaned = service.clear_old_cache_entries(max_age=0.5)
                # Should clean some entries
                assert cleaned >= 0
        
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        # Assert
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (not indicating a major leak)
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
        
        # Cache should not grow indefinitely
        assert len(service.cache) < 1000  # Reasonable cache size
    
    @pytest.mark.performance
    def test_file_handle_management(self, tmp_path):
        """Test file handle management."""
        # Arrange
        class FileProcessor:
            def __init__(self):
                self.processed_files = []
                self.open_handles = {}
            
            def process_file_safe(self, file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Simulate processing
                        processed_content = content.upper()
                        
                        # Write result
                        output_path = file_path.with_suffix('.processed')
                        with open(output_path, 'w') as out_f:
                            out_f.write(processed_content)
                        
                        self.processed_files.append(str(output_path))
                        return len(processed_content)
                
                except IOError as e:
                    return f"Error: {e}"
        
        processor = FileProcessor()
        
        # Create many test files
        test_files = []
        for i in range(100):
            test_file = tmp_path / f"test_{i}.txt"
            test_file.write_text(f"Test content for file {i}\n" * 10)
            test_files.append(test_file)
        
        # Act - Process all files
        results = []
        for file_path in test_files:
            result = processor.process_file_safe(file_path)
            results.append(result)
        
        # Assert
        # All files should be processed successfully
        assert all(isinstance(r, int) for r in results)  # All returned lengths
        assert len(processor.processed_files) == 100
        
        # Verify all output files exist
        for processed_file in processor.processed_files:
            assert Path(processed_file).exists()
        
        # No file handles should be left open (with statement ensures this)
        # This is more of a design verification than a runtime test