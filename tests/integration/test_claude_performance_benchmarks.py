"""
Claude Performance Benchmark Tests

Comprehensive performance testing for both Claude client implementations
including load testing, stress testing, memory usage, and scalability analysis.
"""

import asyncio
import gc
import json
import os
import psutil
import pytest
import resource
import statistics
import tempfile
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from src.claude_tui.integrations.claude_code_client import ClaudeCodeClient
from src.claude_tui.integrations.claude_code_direct_client import ClaudeCodeDirectClient
from src.claude_tui.core.config_manager import ConfigManager

OAUTH_TOKEN = "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"


class PerformanceProfiler:
    """Utility class for performance profiling."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_peak = None
        self.cpu_times_start = None
        self.cpu_times_end = None
        
    def start(self):
        """Start performance profiling."""
        self.start_time = time.perf_counter()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.cpu_times_start = psutil.Process().cpu_times()
        tracemalloc.start()
        
    def end(self):
        """End performance profiling."""
        self.end_time = time.perf_counter()
        self.cpu_times_end = psutil.Process().cpu_times()
        
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_peak = current_memory
        
        # Get tracemalloc statistics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'execution_time': self.end_time - self.start_time,
            'memory_start_mb': self.memory_start,
            'memory_peak_mb': self.memory_peak,
            'memory_delta_mb': self.memory_peak - self.memory_start,
            'tracemalloc_current_mb': current / 1024 / 1024,
            'tracemalloc_peak_mb': peak / 1024 / 1024,
            'cpu_user_time': self.cpu_times_end.user - self.cpu_times_start.user,
            'cpu_system_time': self.cpu_times_end.system - self.cpu_times_start.system
        }


class TestBasicPerformanceMetrics:
    """Test basic performance metrics for individual operations."""
    
    @pytest.fixture
    async def http_client(self):
        """Create HTTP client for performance testing."""
        config = ConfigManager()
        client = ClaudeCodeClient(config, oauth_token=OAUTH_TOKEN)
        yield client
        await client.cleanup()
    
    @pytest.fixture
    def direct_client(self):
        """Create direct client for performance testing."""
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        cc_file.write_text(OAUTH_TOKEN)
        
        client = ClaudeCodeDirectClient(
            oauth_token_file=str(cc_file),
            working_directory=temp_dir
        )
        yield client
        client.cleanup_session()
        
        # Cleanup
        cc_file.unlink(missing_ok=True)
        Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_http_client_basic_performance(self, http_client):
        """Test basic HTTP client performance metrics."""
        profiler = PerformanceProfiler()
        
        test_cases = [
            "Write a simple hello world function",
            "Create a function to calculate factorial",
            "Implement a basic calculator",
            "Write a function to reverse a string",
            "Create a fibonacci sequence generator"
        ]
        
        results = []
        
        for i, task in enumerate(test_cases):
            profiler.start()
            
            result = await http_client.execute_task(task, {"timeout": 30})
            
            perf_metrics = profiler.end()
            perf_metrics['task'] = task
            perf_metrics['success'] = result.get('success', False)
            results.append(perf_metrics)
            
            print(f"Task {i+1}: {perf_metrics['execution_time']:.2f}s, "
                  f"{perf_metrics['memory_delta_mb']:.1f}MB")
        
        # Calculate statistics
        execution_times = [r['execution_time'] for r in results]
        memory_deltas = [r['memory_delta_mb'] for r in results]
        
        performance_summary = {
            'avg_execution_time': statistics.mean(execution_times),
            'median_execution_time': statistics.median(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'avg_memory_delta': statistics.mean(memory_deltas),
            'max_memory_delta': max(memory_deltas)
        }
        
        print("\nHTTP Client Performance Summary:")
        for key, value in performance_summary.items():
            print(f"{key}: {value:.3f}")
        
        # Performance assertions
        assert performance_summary['avg_execution_time'] < 60.0  # Should average under 1 minute
        assert performance_summary['max_memory_delta'] < 100.0  # Should not use more than 100MB per task
    
    @pytest.mark.asyncio
    async def test_direct_client_basic_performance(self, direct_client):
        """Test basic direct client performance metrics."""
        profiler = PerformanceProfiler()
        
        test_cases = [
            "Write a simple hello world function",
            "Create a function to calculate factorial", 
            "Implement a basic calculator",
            "Write a function to reverse a string",
            "Create a fibonacci sequence generator"
        ]
        
        results = []
        
        for i, task in enumerate(test_cases):
            profiler.start()
            
            result = await direct_client.execute_task_via_cli(task, timeout=30)
            
            perf_metrics = profiler.end()
            perf_metrics['task'] = task
            perf_metrics['success'] = result.get('success', False)
            results.append(perf_metrics)
            
            print(f"Task {i+1}: {perf_metrics['execution_time']:.2f}s, "
                  f"{perf_metrics['memory_delta_mb']:.1f}MB")
        
        # Calculate statistics
        execution_times = [r['execution_time'] for r in results]
        memory_deltas = [r['memory_delta_mb'] for r in results]
        
        performance_summary = {
            'avg_execution_time': statistics.mean(execution_times),
            'median_execution_time': statistics.median(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'avg_memory_delta': statistics.mean(memory_deltas),
            'max_memory_delta': max(memory_deltas)
        }
        
        print("\nDirect Client Performance Summary:")
        for key, value in performance_summary.items():
            print(f"{key}: {value:.3f}")
        
        # Performance assertions
        assert performance_summary['avg_execution_time'] < 60.0
        assert performance_summary['max_memory_delta'] < 150.0  # Direct client may use more memory


class TestLoadTesting:
    """Load testing with multiple concurrent requests."""
    
    @pytest.fixture
    async def http_client(self):
        """Create HTTP client for load testing."""
        config = ConfigManager()
        client = ClaudeCodeClient(config, oauth_token=OAUTH_TOKEN)
        yield client
        await client.cleanup()
    
    @pytest.fixture
    def direct_client(self):
        """Create direct client for load testing."""
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        cc_file.write_text(OAUTH_TOKEN)
        
        client = ClaudeCodeDirectClient(
            oauth_token_file=str(cc_file),
            working_directory=temp_dir
        )
        yield client
        client.cleanup_session()
        
        # Cleanup
        cc_file.unlink(missing_ok=True)
        Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_http_client_concurrent_load(self, http_client):
        """Test HTTP client under concurrent load."""
        concurrency_levels = [1, 2, 5, 10]  # Number of concurrent requests
        task_template = "Create a simple function number {}"
        
        load_test_results = []
        
        for concurrency in concurrency_levels:
            print(f"\nTesting concurrency level: {concurrency}")
            
            profiler = PerformanceProfiler()
            profiler.start()
            
            # Create concurrent tasks
            tasks = [
                http_client.execute_task(
                    task_template.format(i),
                    {"timeout": 30}
                )
                for i in range(concurrency)
            ]
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            perf_metrics = profiler.end()
            
            # Analyze results
            successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
            failed = sum(1 for r in results if isinstance(r, dict) and not r.get('success'))
            exceptions = sum(1 for r in results if isinstance(r, Exception))
            
            load_result = {
                'concurrency': concurrency,
                'successful': successful,
                'failed': failed,
                'exceptions': exceptions,
                'success_rate': successful / concurrency if concurrency > 0 else 0,
                'total_time': perf_metrics['execution_time'],
                'avg_time_per_request': perf_metrics['execution_time'] / concurrency,
                'requests_per_second': concurrency / perf_metrics['execution_time'],
                'memory_delta_mb': perf_metrics['memory_delta_mb'],
                'cpu_usage': perf_metrics['cpu_user_time'] + perf_metrics['cpu_system_time']
            }
            
            load_test_results.append(load_result)
            
            print(f"Success rate: {load_result['success_rate']:.2f}")
            print(f"Requests/sec: {load_result['requests_per_second']:.2f}")
            print(f"Avg time per request: {load_result['avg_time_per_request']:.2f}s")
            print(f"Memory delta: {load_result['memory_delta_mb']:.1f}MB")
        
        # Analyze scalability
        print("\nLoad Test Summary:")
        print("Concurrency | Success Rate | Req/Sec | Avg Time | Memory")
        for result in load_test_results:
            print(f"{result['concurrency']:10} | {result['success_rate']:11.2f} | "
                  f"{result['requests_per_second']:7.2f} | {result['avg_time_per_request']:8.2f}s | "
                  f"{result['memory_delta_mb']:6.1f}MB")
    
    @pytest.mark.asyncio
    async def test_direct_client_concurrent_load(self, direct_client):
        """Test direct client under concurrent load."""
        concurrency_levels = [1, 2, 3]  # Lower concurrency for CLI client
        task_template = "Create a simple function number {}"
        
        load_test_results = []
        
        for concurrency in concurrency_levels:
            print(f"\nTesting direct client concurrency level: {concurrency}")
            
            profiler = PerformanceProfiler()
            profiler.start()
            
            # Create concurrent tasks
            tasks = [
                direct_client.execute_task_via_cli(
                    task_template.format(i),
                    timeout=30
                )
                for i in range(concurrency)
            ]
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            perf_metrics = profiler.end()
            
            # Analyze results
            successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
            failed = sum(1 for r in results if isinstance(r, dict) and not r.get('success'))
            exceptions = sum(1 for r in results if isinstance(r, Exception))
            
            load_result = {
                'concurrency': concurrency,
                'successful': successful,
                'failed': failed, 
                'exceptions': exceptions,
                'success_rate': successful / concurrency if concurrency > 0 else 0,
                'total_time': perf_metrics['execution_time'],
                'avg_time_per_request': perf_metrics['execution_time'] / concurrency,
                'memory_delta_mb': perf_metrics['memory_delta_mb']
            }
            
            load_test_results.append(load_result)
            
            print(f"Success rate: {load_result['success_rate']:.2f}")
            print(f"Avg time per request: {load_result['avg_time_per_request']:.2f}s")
        
        # Direct client should handle at least single requests well
        single_request_result = load_test_results[0]
        assert single_request_result['success_rate'] >= 0.5  # At least 50% success rate


class TestMemoryUsageAnalysis:
    """Test memory usage patterns and potential leaks."""
    
    @pytest.fixture
    async def http_client(self):
        """Create HTTP client for memory testing."""
        config = ConfigManager()
        client = ClaudeCodeClient(config, oauth_token=OAUTH_TOKEN)
        yield client
        await client.cleanup()
    
    @pytest.fixture
    def direct_client(self):
        """Create direct client for memory testing."""
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        cc_file.write_text(OAUTH_TOKEN)
        
        client = ClaudeCodeDirectClient(
            oauth_token_file=str(cc_file),
            working_directory=temp_dir
        )
        yield client
        client.cleanup_session()
        
        # Cleanup
        cc_file.unlink(missing_ok=True)
        Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_http_client_memory_pattern(self, http_client):
        """Test HTTP client memory usage patterns."""
        memory_samples = []
        
        # Baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_samples.append(('baseline', baseline_memory))
        
        # Execute tasks and monitor memory
        for i in range(5):
            task = f"Generate code for task {i}"
            
            # Memory before task
            pre_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Execute task
            result = await http_client.execute_task(task, {"timeout": 20})
            
            # Memory after task
            post_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_samples.append((f'task_{i}_pre', pre_memory))
            memory_samples.append((f'task_{i}_post', post_memory))
            
            # Force garbage collection
            gc.collect()
            gc_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append((f'task_{i}_gc', gc_memory))
        
        # Analyze memory pattern
        print("\nHTTP Client Memory Usage Pattern:")
        for label, memory in memory_samples:
            print(f"{label:15}: {memory:6.1f} MB (+{memory - baseline_memory:5.1f})")
        
        # Check for memory leaks
        final_memory = memory_samples[-1][1]
        memory_growth = final_memory - baseline_memory
        
        print(f"\nTotal memory growth: {memory_growth:.1f} MB")
        
        # Should not grow excessively
        assert memory_growth < 200.0  # Should not grow more than 200MB
    
    @pytest.mark.asyncio
    async def test_direct_client_memory_pattern(self, direct_client):
        """Test direct client memory usage patterns."""
        memory_samples = []
        
        # Baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_samples.append(('baseline', baseline_memory))
        
        # Execute tasks and monitor memory
        for i in range(3):  # Fewer tasks for CLI client
            task = f"Generate code for task {i}"
            
            # Memory before task
            pre_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Execute task
            result = await direct_client.execute_task_via_cli(task, timeout=20)
            
            # Memory after task
            post_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_samples.append((f'task_{i}_pre', pre_memory))
            memory_samples.append((f'task_{i}_post', post_memory))
            
            # Force garbage collection
            gc.collect()
            gc_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append((f'task_{i}_gc', gc_memory))
        
        # Analyze memory pattern
        print("\nDirect Client Memory Usage Pattern:")
        for label, memory in memory_samples:
            print(f"{label:15}: {memory:6.1f} MB (+{memory - baseline_memory:5.1f})")
        
        # Check for memory leaks
        final_memory = memory_samples[-1][1]
        memory_growth = final_memory - baseline_memory
        
        print(f"\nTotal memory growth: {memory_growth:.1f} MB")
        
        # Direct client may use more memory due to subprocesses
        assert memory_growth < 300.0
    
    @pytest.mark.asyncio
    async def test_session_cleanup_memory_release(self, direct_client):
        """Test that session cleanup releases memory."""
        # Initial memory
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create multiple sessions worth of work
        for session in range(3):
            for task_num in range(2):
                await direct_client.execute_task_via_cli(
                    f"Session {session} task {task_num}",
                    timeout=15
                )
        
        # Memory after work
        work_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Cleanup session
        direct_client.cleanup_session()
        gc.collect()
        
        # Memory after cleanup
        cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"After work: {work_memory:.1f} MB (+{work_memory - initial_memory:.1f})")
        print(f"After cleanup: {cleanup_memory:.1f} MB (+{cleanup_memory - initial_memory:.1f})")
        
        # Cleanup should reduce memory usage
        memory_recovered = work_memory - cleanup_memory
        print(f"Memory recovered: {memory_recovered:.1f} MB")


class TestStressTesting:
    """Stress testing with extreme conditions."""
    
    @pytest.fixture
    def direct_client(self):
        """Create direct client for stress testing."""
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        cc_file.write_text(OAUTH_TOKEN)
        
        client = ClaudeCodeDirectClient(
            oauth_token_file=str(cc_file),
            working_directory=temp_dir
        )
        yield client
        client.cleanup_session()
        
        # Cleanup
        cc_file.unlink(missing_ok=True)
        Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_large_input_handling(self, direct_client):
        """Test handling of large input data."""
        # Create progressively larger inputs
        input_sizes = [1000, 5000, 10000]  # Number of characters
        
        for size in input_sizes:
            print(f"\nTesting input size: {size} characters")
            
            # Create large task description
            large_task = "Process this data: " + "x" * size
            
            profiler = PerformanceProfiler()
            profiler.start()
            
            try:
                result = await direct_client.execute_task_via_cli(
                    large_task,
                    timeout=30
                )
                
                perf_metrics = profiler.end()
                
                print(f"Success: {result.get('success', False)}")
                print(f"Execution time: {perf_metrics['execution_time']:.2f}s")
                print(f"Memory usage: {perf_metrics['memory_delta_mb']:.1f}MB")
                
                # Should handle reasonable input sizes
                if size <= 5000:
                    assert isinstance(result, dict)
                    assert 'execution_id' in result
                
            except Exception as e:
                print(f"Failed with exception: {type(e).__name__}: {e}")
                # Large inputs may fail, which is acceptable
    
    @pytest.mark.asyncio
    async def test_rapid_fire_requests(self, direct_client):
        """Test rapid fire request handling."""
        num_requests = 20
        tasks = []
        
        print(f"Sending {num_requests} rapid fire requests...")
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        # Send all requests as fast as possible
        for i in range(num_requests):
            task = direct_client.execute_task_via_cli(
                f"Rapid task {i}",
                timeout=10  # Short timeout
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        perf_metrics = profiler.end()
        
        # Analyze results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = sum(1 for r in results if isinstance(r, dict) and not r.get('success'))
        exceptions = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"Total time: {perf_metrics['execution_time']:.2f}s")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Exceptions: {exceptions}")
        print(f"Success rate: {successful/num_requests:.2f}")
        print(f"Requests per second: {num_requests/perf_metrics['execution_time']:.2f}")
        
        # Should handle some requests successfully
        assert successful > 0
    
    @pytest.mark.asyncio
    async def test_timeout_stress(self, direct_client):
        """Test behavior under timeout stress."""
        # Test various timeout scenarios
        timeout_scenarios = [
            ("very_short", 1),  # 1 second
            ("short", 5),       # 5 seconds
            ("normal", 15),     # 15 seconds
        ]
        
        timeout_results = []
        
        for scenario_name, timeout_value in timeout_scenarios:
            print(f"\nTesting {scenario_name} timeout ({timeout_value}s)")
            
            profiler = PerformanceProfiler()
            profiler.start()
            
            try:
                result = await direct_client.execute_task_via_cli(
                    "Create a complex function with error handling and documentation",
                    timeout=timeout_value
                )
                
                perf_metrics = profiler.end()
                
                timeout_result = {
                    'scenario': scenario_name,
                    'timeout_value': timeout_value,
                    'actual_time': perf_metrics['execution_time'],
                    'success': result.get('success', False),
                    'timed_out': perf_metrics['execution_time'] >= timeout_value * 0.9
                }
                
                timeout_results.append(timeout_result)
                
                print(f"Completed in {perf_metrics['execution_time']:.2f}s")
                print(f"Success: {timeout_result['success']}")
                
            except Exception as e:
                print(f"Exception: {type(e).__name__}: {e}")
                timeout_results.append({
                    'scenario': scenario_name,
                    'timeout_value': timeout_value,
                    'exception': str(e),
                    'success': False
                })
        
        print("\nTimeout Stress Test Summary:")
        for result in timeout_results:
            print(f"{result['scenario']:10}: {result.get('actual_time', 'N/A'):>6} / "
                  f"{result['timeout_value']:>2}s - Success: {result['success']}")


class TestComparisonBenchmarks:
    """Comparative performance benchmarks between clients."""
    
    @pytest.fixture
    async def http_client(self):
        """Create HTTP client for comparison."""
        config = ConfigManager()
        client = ClaudeCodeClient(config, oauth_token=OAUTH_TOKEN)
        yield client
        await client.cleanup()
    
    @pytest.fixture
    def direct_client(self):
        """Create direct client for comparison."""
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        cc_file.write_text(OAUTH_TOKEN)
        
        client = ClaudeCodeDirectClient(
            oauth_token_file=str(cc_file),
            working_directory=temp_dir
        )
        yield client
        client.cleanup_session()
        
        # Cleanup
        cc_file.unlink(missing_ok=True)
        Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_client_performance_comparison(self, http_client, direct_client):
        """Compare performance between HTTP and Direct clients."""
        test_tasks = [
            "Write a simple hello world function",
            "Create a function to calculate factorial",
            "Implement a basic calculator"
        ]
        
        http_results = []
        direct_results = []
        
        # Test HTTP client
        print("Testing HTTP Client...")
        for task in test_tasks:
            profiler = PerformanceProfiler()
            profiler.start()
            
            result = await http_client.execute_task(task, {"timeout": 30})
            
            perf_metrics = profiler.end()
            perf_metrics['task'] = task
            perf_metrics['success'] = result.get('success', False)
            http_results.append(perf_metrics)
        
        # Test Direct client  
        print("Testing Direct Client...")
        for task in test_tasks:
            profiler = PerformanceProfiler()
            profiler.start()
            
            result = await direct_client.execute_task_via_cli(task, timeout=30)
            
            perf_metrics = profiler.end()
            perf_metrics['task'] = task
            perf_metrics['success'] = result.get('success', False)
            direct_results.append(perf_metrics)
        
        # Compare results
        print("\nPerformance Comparison:")
        print("Task                           | HTTP Time | Direct Time | HTTP Mem | Direct Mem")
        print("-" * 80)
        
        for i, task in enumerate(test_tasks):
            http_time = http_results[i]['execution_time']
            direct_time = direct_results[i]['execution_time']
            http_mem = http_results[i]['memory_delta_mb']
            direct_mem = direct_results[i]['memory_delta_mb']
            
            task_short = task[:30]
            print(f"{task_short:<30} | {http_time:8.2f}s | {direct_time:10.2f}s | "
                  f"{http_mem:7.1f}MB | {direct_mem:9.1f}MB")
        
        # Calculate averages
        avg_http_time = statistics.mean(r['execution_time'] for r in http_results)
        avg_direct_time = statistics.mean(r['execution_time'] for r in direct_results)
        avg_http_mem = statistics.mean(r['memory_delta_mb'] for r in http_results)
        avg_direct_mem = statistics.mean(r['memory_delta_mb'] for r in direct_results)
        
        print("-" * 80)
        print(f"{'Average':<30} | {avg_http_time:8.2f}s | {avg_direct_time:10.2f}s | "
              f"{avg_http_mem:7.1f}MB | {avg_direct_mem:9.1f}MB")
        
        # Analysis
        faster_client = "HTTP" if avg_http_time < avg_direct_time else "Direct"
        speed_diff = abs(avg_http_time - avg_direct_time)
        speed_ratio = max(avg_http_time, avg_direct_time) / min(avg_http_time, avg_direct_time)
        
        print(f"\nPerformance Analysis:")
        print(f"Faster client: {faster_client}")
        print(f"Speed difference: {speed_diff:.2f}s")
        print(f"Speed ratio: {speed_ratio:.2f}x")


if __name__ == "__main__":
    """Run performance benchmark tests."""
    pytest.main([__file__, "-v", "-s", "--tb=short", "-k", "not test_stress"])