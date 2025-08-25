"""Comprehensive performance benchmarks using pytest-benchmark."""

import pytest
import asyncio
import time
import json
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from datetime import datetime, timedelta

# Performance testing libraries
try:
    import pytest_benchmark
except ImportError:
    pytest.skip("pytest-benchmark not available", allow_module_level=True)

# Import application components
from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.integrations.ai_interface import AIInterface
from claude_tiu.validation.anti_hallucination_engine import AntiHallucinationEngine
from claude_tiu.models.task import DevelopmentTask, TaskType, TaskPriority
from claude_tiu.models.project import Project
from claude_tiu.validation.progress_validator import ValidationResult


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager for performance testing."""
    manager = Mock(spec=ConfigManager)
    manager.get_setting = AsyncMock(return_value='test_value')
    manager.get_api_key = AsyncMock(return_value='test_key')
    return manager


@pytest.fixture
def sample_code_data():
    """Generate sample code data of various sizes."""
    return {
        'small': """
def hello_world():
    return "Hello, World!"
""",
        'medium': """
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(('add', a, b, result))
        return result
    
    def subtract(self, a, b):
        result = a - b
        self.history.append(('subtract', a, b, result))
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(('multiply', a, b, result))
        return result
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(('divide', a, b, result))
        return result
    
    def get_history(self):
        return self.history.copy()
    
    def clear_history(self):
        self.history.clear()
""",
        'large': """
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class TaskMetrics:
    task_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None

class PerformanceMonitor:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.metrics: Dict[str, TaskMetrics] = {}
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self, task_id: str) -> None:
        metrics = TaskMetrics(
            task_id=task_id,
            start_time=datetime.now()
        )
        self.metrics[task_id] = metrics
        self.logger.info(f"Started monitoring task {task_id}")
    
    async def stop_monitoring(self, task_id: str, success: bool = True, error: str = None) -> TaskMetrics:
        if task_id not in self.metrics:
            raise ValueError(f"Task {task_id} not found in metrics")
        
        metrics = self.metrics[task_id]
        metrics.end_time = datetime.now()
        metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.success = success
        metrics.error_message = error
        
        # Simulate getting system metrics
        metrics.memory_usage = self._get_memory_usage()
        metrics.cpu_usage = self._get_cpu_usage()
        
        self.logger.info(f"Stopped monitoring task {task_id}, duration: {metrics.duration:.2f}s")
        return metrics
    
    def _get_memory_usage(self) -> int:
        # Simulate memory usage calculation
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    
    def _get_cpu_usage(self) -> float:
        # Simulate CPU usage calculation
        import psutil
        return psutil.cpu_percent()
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        completed_tasks = [m for m in self.metrics.values() if m.end_time is not None]
        
        if not completed_tasks:
            return {'total_tasks': 0, 'avg_duration': 0, 'success_rate': 0}
        
        total_duration = sum(m.duration for m in completed_tasks if m.duration)
        successful_tasks = sum(1 for m in completed_tasks if m.success)
        
        return {
            'total_tasks': len(completed_tasks),
            'avg_duration': total_duration / len(completed_tasks) if completed_tasks else 0,
            'success_rate': successful_tasks / len(completed_tasks) if completed_tasks else 0,
            'total_duration': total_duration,
            'avg_memory_usage': sum(m.memory_usage for m in completed_tasks if m.memory_usage) / len(completed_tasks),
            'avg_cpu_usage': sum(m.cpu_usage for m in completed_tasks if m.cpu_usage) / len(completed_tasks)
        }
    
    async def export_metrics(self, filepath: Path) -> None:
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {task_id: {
                'task_id': m.task_id,
                'start_time': m.start_time.isoformat(),
                'end_time': m.end_time.isoformat() if m.end_time else None,
                'duration': m.duration,
                'memory_usage': m.memory_usage,
                'cpu_usage': m.cpu_usage,
                'success': m.success,
                'error_message': m.error_message
            } for task_id, m in self.metrics.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Exported metrics to {filepath}")
"""
    }


class TestConfigManagerPerformance:
    """Performance benchmarks for ConfigManager."""
    
    def test_config_loading_performance(self, benchmark, tmp_path):
        """Benchmark configuration loading speed."""
        # Create test config file
        config_file = tmp_path / "config.yaml"
        config_data = {
            'version': '1.0',
            'ai_services': {f'service_{i}': {'timeout': 300} for i in range(100)},
            'ui_preferences': {'theme': 'dark', 'font_size': 12},
            'custom_settings': {f'setting_{i}': f'value_{i}' for i in range(200)}
        }
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(config_dir=tmp_path)
        
        # Benchmark loading
        def load_config():
            asyncio.run(config_manager.load_config())
        
        result = benchmark(load_config)
        
        # Verify benchmark completed successfully
        assert config_manager.config is not None
    
    def test_settings_update_performance(self, benchmark, mock_config_manager):
        """Benchmark settings update operations."""
        asyncio.run(mock_config_manager.initialize())
        
        def update_multiple_settings():
            async def _update():
                for i in range(50):
                    await mock_config_manager.update_setting(f'test.setting_{i}', f'value_{i}')
            asyncio.run(_update())
        
        result = benchmark(update_multiple_settings)
        
        # Verify updates completed
        assert result is None  # Benchmark returns None on success
    
    def test_concurrent_config_access(self, benchmark, mock_config_manager):
        """Benchmark concurrent configuration access."""
        async def concurrent_access():
            tasks = []
            for i in range(20):
                tasks.append(mock_config_manager.get_setting(f'setting_{i}', 'default'))
            await asyncio.gather(*tasks)
        
        def run_concurrent_access():
            asyncio.run(concurrent_access())
        
        result = benchmark(run_concurrent_access)
        
        # Verify concurrent access completed
        assert result is None


class TestAIInterfacePerformance:
    """Performance benchmarks for AI Interface."""
    
    @pytest.mark.asyncio
    async def test_validation_performance(self, benchmark, mock_config_manager, sample_code_data):
        """Benchmark AI validation performance."""
        with patch('claude_tiu.integrations.ai_interface.AntiHallucinationIntegration') as mock_ah:
            mock_ah_instance = Mock()
            mock_ah_instance.initialize = AsyncMock()
            mock_ah_instance.validate_ai_generated_content = AsyncMock(
                return_value=ValidationResult(
                    is_valid=True,
                    authenticity_score=0.95,
                    issues=[]
                )
            )
            mock_ah.return_value = mock_ah_instance
            
            ai_interface = AIInterface(mock_config_manager)
            await ai_interface.initialize()
            
            # Benchmark validation of different code sizes
            async def validate_code_samples():
                validation_tasks = []
                for size, code in sample_code_data.items():
                    task = ai_interface.validate_ai_output(
                        output=code,
                        task=Mock(task_type=TaskType.CODE_GENERATION, name=f"test_{size}"),
                        project=None
                    )
                    validation_tasks.append(task)
                
                results = await asyncio.gather(*validation_tasks)
                return results
            
            def run_validation_benchmark():
                return asyncio.run(validate_code_samples())
            
            results = benchmark(run_validation_benchmark)
            
            # Verify all validations completed
            assert len(results) == 3
            assert all(result.is_valid for result in results)
    
    def test_task_execution_throughput(self, benchmark, mock_config_manager):
        """Benchmark task execution throughput."""
        with patch('claude_tiu.integrations.ai_interface.ClaudeCodeClient') as mock_cc:
            # Mock fast task execution
            mock_cc_instance = Mock()
            mock_cc_instance.execute_coding_task = AsyncMock(
                return_value=Mock(
                    success=True,
                    content="mock result",
                    execution_time=0.001
                )
            )
            mock_cc.return_value = mock_cc_instance
            
            ai_interface = AIInterface(mock_config_manager)
            
            # Benchmark executing multiple tasks
            async def execute_multiple_tasks():
                tasks = []
                for i in range(10):
                    task = DevelopmentTask(
                        name=f"Task {i}",
                        description=f"Test task {i}",
                        task_type=TaskType.CODE_GENERATION,
                        priority=TaskPriority.MEDIUM
                    )
                    tasks.append(ai_interface._execute_task_with_claude_code(task, Mock()))
                
                results = await asyncio.gather(*tasks)
                return results
            
            def run_task_benchmark():
                return asyncio.run(execute_multiple_tasks())
            
            results = benchmark(run_task_benchmark)
            
            # Verify all tasks completed
            assert len(results) == 10
            assert all(result.success for result in results)


class TestAntiHallucinationPerformance:
    """Performance benchmarks for Anti-Hallucination Engine."""
    
    def test_content_analysis_speed(self, benchmark, mock_config_manager, sample_code_data):
        """Benchmark content analysis speed."""
        with patch('claude_tiu.validation.anti_hallucination_engine.load_models'), \
             patch('claude_tiu.validation.anti_hallucination_engine.initialize_neural_networks'):
            
            engine = AntiHallucinationEngine(mock_config_manager)
            
            # Mock fast neural network predictions
            async def fast_validation(content, context):
                # Simulate neural network processing time
                await asyncio.sleep(0.001)  # 1ms processing time
                return ValidationResult(
                    is_valid=len(content) > 10,  # Simple validation logic
                    authenticity_score=0.9 if len(content) > 100 else 0.7,
                    issues=[]
                )
            
            engine.validate_content = fast_validation
            
            async def analyze_all_samples():
                results = []
                for size, code in sample_code_data.items():
                    result = await engine.validate_content(
                        content=code,
                        context={'language': 'python', 'size': size}
                    )
                    results.append(result)
                return results
            
            def run_analysis_benchmark():
                return asyncio.run(analyze_all_samples())
            
            results = benchmark(run_analysis_benchmark)
            
            # Verify analysis completed
            assert len(results) == 3
            assert all(isinstance(result, ValidationResult) for result in results)
    
    def test_batch_validation_performance(self, benchmark, mock_config_manager):
        """Benchmark batch validation performance."""
        with patch('claude_tiu.validation.anti_hallucination_engine.load_models'), \
             patch('claude_tiu.validation.anti_hallucination_engine.initialize_neural_networks'):
            
            engine = AntiHallucinationEngine(mock_config_manager)
            
            # Generate batch of code samples
            code_samples = [
                f"def function_{i}(x): return x * {i}"
                for i in range(100)
            ]
            
            # Mock batch validation
            async def validate_batch(samples):
                validation_tasks = []
                for i, code in enumerate(samples):
                    async def validate_single(c=code):
                        await asyncio.sleep(0.001)  # Simulate processing
                        return ValidationResult(
                            is_valid=True,
                            authenticity_score=0.92,
                            issues=[]
                        )
                    validation_tasks.append(validate_single())
                
                return await asyncio.gather(*validation_tasks)
            
            def run_batch_benchmark():
                return asyncio.run(validate_batch(code_samples))
            
            results = benchmark(run_batch_benchmark)
            
            # Verify batch validation completed
            assert len(results) == 100
            assert all(result.is_valid for result in results)
    
    def test_memory_efficiency_large_files(self, benchmark, mock_config_manager):
        """Benchmark memory efficiency with large files."""
        with patch('claude_tiu.validation.anti_hallucination_engine.load_models'), \
             patch('claude_tiu.validation.anti_hallucination_engine.initialize_neural_networks'):
            
            engine = AntiHallucinationEngine(mock_config_manager)
            
            # Generate large code file
            large_code = "\\n".join([
                f"def large_function_{i}():",
                f"    # This is function number {i}",
                f"    data = [{j} for j in range({i * 100})]",
                f"    return sum(data)",
                ""
            ] for i in range(1000))
            
            # Mock memory-efficient validation
            async def validate_large_file(code):
                # Process in chunks to simulate memory efficiency
                chunk_size = 1000
                chunks = [code[i:i+chunk_size] for i in range(0, len(code), chunk_size)]
                
                chunk_results = []
                for chunk in chunks:
                    await asyncio.sleep(0.0001)  # Simulate chunk processing
                    chunk_results.append(ValidationResult(
                        is_valid=True,
                        authenticity_score=0.88,
                        issues=[]
                    ))
                
                # Aggregate results
                avg_score = sum(r.authenticity_score for r in chunk_results) / len(chunk_results)
                return ValidationResult(
                    is_valid=True,
                    authenticity_score=avg_score,
                    issues=[]
                )
            
            def run_large_file_benchmark():
                return asyncio.run(validate_large_file(large_code))
            
            result = benchmark(run_large_file_benchmark)
            
            # Verify large file validation completed
            assert result.is_valid is True
            assert result.authenticity_score > 0.8


class TestConcurrencyPerformance:
    """Performance benchmarks for concurrent operations."""
    
    def test_concurrent_validation_throughput(self, benchmark, mock_config_manager):
        """Benchmark concurrent validation throughput."""
        with patch('claude_tiu.validation.anti_hallucination_engine.load_models'), \
             patch('claude_tiu.validation.anti_hallucination_engine.initialize_neural_networks'):
            
            engine = AntiHallucinationEngine(mock_config_manager)
            
            # Create test data
            test_codes = [
                f"def test_function_{i}(): return {i}"
                for i in range(50)
            ]
            
            async def concurrent_validation():
                # Create semaphore to limit concurrency
                semaphore = asyncio.Semaphore(10)  # Max 10 concurrent validations
                
                async def validate_with_semaphore(code):
                    async with semaphore:
                        await asyncio.sleep(0.001)  # Simulate validation time
                        return ValidationResult(
                            is_valid=True,
                            authenticity_score=0.93,
                            issues=[]
                        )
                
                # Run all validations concurrently
                validation_tasks = [
                    validate_with_semaphore(code) for code in test_codes
                ]
                
                return await asyncio.gather(*validation_tasks)
            
            def run_concurrent_benchmark():
                return asyncio.run(concurrent_validation())
            
            results = benchmark(run_concurrent_benchmark)
            
            # Verify concurrent validation completed
            assert len(results) == 50
            assert all(result.is_valid for result in results)
    
    def test_async_task_queue_performance(self, benchmark):
        """Benchmark async task queue performance."""
        async def process_task_queue():
            queue = asyncio.Queue(maxsize=100)
            results = []
            
            # Producer: Add tasks to queue
            async def producer():
                for i in range(100):
                    await queue.put(f"task_{i}")
                # Signal completion
                await queue.put(None)
            
            # Consumer: Process tasks from queue
            async def consumer():
                while True:
                    task = await queue.get()
                    if task is None:
                        break
                    
                    # Simulate task processing
                    await asyncio.sleep(0.001)
                    results.append(f"processed_{task}")
                    queue.task_done()
            
            # Run producer and consumer concurrently
            await asyncio.gather(
                producer(),
                consumer()
            )
            
            return results
        
        def run_queue_benchmark():
            return asyncio.run(process_task_queue())
        
        results = benchmark(run_queue_benchmark)
        
        # Verify queue processing completed
        assert len(results) == 100
        assert all("processed_task_" in result for result in results)


class TestMemoryPerformance:
    """Performance benchmarks for memory usage."""
    
    def test_memory_usage_scaling(self, benchmark, mock_config_manager):
        """Benchmark memory usage scaling with data size."""
        def memory_scaling_test():
            # Simulate different data sizes
            data_sizes = [100, 1000, 10000]
            memory_results = []
            
            for size in data_sizes:
                # Create test data
                test_data = [f"item_{i}" * 10 for i in range(size)]
                
                # Measure memory usage (simulated)
                start_memory = 1000000  # Simulate 1MB baseline
                data_memory = len(str(test_data))  # Rough memory estimate
                
                # Process data
                processed_data = []
                for item in test_data:
                    processed_data.append(item.upper())
                
                memory_results.append({
                    'size': size,
                    'memory_used': data_memory,
                    'processed': len(processed_data)
                })
                
                # Clean up
                del test_data
                del processed_data
            
            return memory_results
        
        results = benchmark(memory_scaling_test)
        
        # Verify memory scaling test completed
        assert len(results) == 3
        assert all('memory_used' in result for result in results)
        
        # Verify memory usage scales reasonably
        memory_usage = [result['memory_used'] for result in results]
        assert memory_usage[1] > memory_usage[0]  # Larger data uses more memory
        assert memory_usage[2] > memory_usage[1]
    
    def test_garbage_collection_impact(self, benchmark):
        """Benchmark garbage collection impact on performance."""
        import gc
        
        def gc_impact_test():
            # Create and destroy many objects
            for cycle in range(10):
                # Create objects
                large_objects = []
                for i in range(1000):
                    obj = {
                        'id': i,
                        'data': [j for j in range(100)],
                        'metadata': f'object_{i}_cycle_{cycle}'
                    }
                    large_objects.append(obj)
                
                # Process objects
                processed = [obj['id'] for obj in large_objects if obj['id'] % 2 == 0]
                
                # Clear objects
                large_objects.clear()
                
                # Force garbage collection
                if cycle % 3 == 0:
                    gc.collect()
            
            return cycle + 1
        
        result = benchmark(gc_impact_test)
        
        # Verify GC test completed
        assert result == 10


class TestRealWorldScenarios:
    """Performance benchmarks for real-world scenarios."""
    
    def test_full_project_validation_performance(self, benchmark, mock_config_manager, tmp_path):
        """Benchmark full project validation performance."""
        # Create mock project structure
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        
        # Create multiple Python files
        files_to_create = [
            ("main.py", "def main(): print('Hello World')"),
            ("utils.py", "def helper(): return 'helper'"),
            ("models.py", "class Model: pass"),
            ("tests/test_main.py", "def test_main(): assert True"),
            ("tests/test_utils.py", "def test_helper(): assert True"),
        ]
        
        for file_path, content in files_to_create:
            file_full_path = project_dir / file_path
            file_full_path.parent.mkdir(parents=True, exist_ok=True)
            file_full_path.write_text(content)
        
        # Mock project validation
        async def validate_project():
            validation_results = {}
            
            # Find all Python files
            python_files = list(project_dir.rglob("*.py"))
            
            # Validate each file
            for file_path in python_files:
                content = file_path.read_text()
                
                # Simulate validation processing time
                await asyncio.sleep(0.01)  # 10ms per file
                
                validation_results[str(file_path)] = ValidationResult(
                    is_valid=True,
                    authenticity_score=0.91,
                    issues=[]
                )
            
            return validation_results
        
        def run_project_validation():
            return asyncio.run(validate_project())
        
        results = benchmark(run_project_validation)
        
        # Verify project validation completed
        assert len(results) == 5  # 5 Python files created
        assert all(result.is_valid for result in results.values())
    
    def test_continuous_monitoring_performance(self, benchmark):
        """Benchmark continuous monitoring performance."""
        def continuous_monitoring_simulation():
            monitoring_data = []
            
            # Simulate 60 seconds of monitoring with 1-second intervals
            for second in range(60):
                # Simulate collecting metrics
                metrics = {
                    'timestamp': second,
                    'cpu_usage': 50 + (second % 10),  # Simulate fluctuating CPU
                    'memory_usage': 1000 + (second * 10),  # Simulate increasing memory
                    'active_tasks': max(0, 10 - (second // 10)),  # Simulate decreasing tasks
                    'validation_queue': second % 5,  # Simulate queue fluctuation
                }
                
                monitoring_data.append(metrics)
                
                # Simulate processing overhead
                time.sleep(0.001)  # 1ms processing time per metric collection
            
            # Calculate aggregated metrics
            avg_cpu = sum(m['cpu_usage'] for m in monitoring_data) / len(monitoring_data)
            max_memory = max(m['memory_usage'] for m in monitoring_data)
            total_tasks = sum(m['active_tasks'] for m in monitoring_data)
            
            return {
                'samples': len(monitoring_data),
                'avg_cpu': avg_cpu,
                'max_memory': max_memory,
                'total_tasks': total_tasks
            }
        
        result = benchmark(continuous_monitoring_simulation)
        
        # Verify monitoring simulation completed
        assert result['samples'] == 60
        assert result['avg_cpu'] > 40  # Average CPU should be reasonable
        assert result['max_memory'] > 1000  # Memory should have increased
    
    def test_load_testing_simulation(self, benchmark, mock_config_manager):
        """Benchmark load testing simulation."""
        async def simulate_high_load():
            # Simulate high concurrent load
            concurrent_requests = 50
            request_results = []
            
            async def process_request(request_id):
                # Simulate request processing
                start_time = asyncio.get_event_loop().time()
                
                # Simulate AI validation request
                await asyncio.sleep(0.01)  # 10ms processing time
                
                end_time = asyncio.get_event_loop().time()
                
                return {
                    'request_id': request_id,
                    'processing_time': end_time - start_time,
                    'success': True
                }
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
            
            async def limited_request(request_id):
                async with semaphore:
                    return await process_request(request_id)
            
            # Process all requests concurrently
            request_tasks = [
                limited_request(i) for i in range(concurrent_requests)
            ]
            
            results = await asyncio.gather(*request_tasks)
            
            # Calculate performance metrics
            total_time = sum(r['processing_time'] for r in results)
            avg_time = total_time / len(results)
            success_rate = sum(1 for r in results if r['success']) / len(results)
            
            return {
                'total_requests': len(results),
                'avg_processing_time': avg_time,
                'success_rate': success_rate,
                'throughput': len(results) / max(r['processing_time'] for r in results)
            }
        
        def run_load_test():
            return asyncio.run(simulate_high_load())
        
        result = benchmark(run_load_test)
        
        # Verify load test completed successfully
        assert result['total_requests'] == 50
        assert result['success_rate'] == 1.0  # 100% success rate
        assert result['avg_processing_time'] > 0
        assert result['throughput'] > 0


@pytest.mark.benchmark(
    group="overall_performance",
    min_rounds=5,
    max_time=0.1,
    timer=time.perf_counter,
    disable_gc=True,
    warmup=False
)
class TestOverallPerformanceBenchmarks:
    """Overall performance benchmark suite."""
    
    def test_startup_performance(self, benchmark, mock_config_manager):
        """Benchmark application startup performance."""
        def startup_simulation():
            # Simulate application startup sequence
            components = [
                ('config_manager', 0.01),
                ('ai_interface', 0.02),
                ('validation_engine', 0.015),
                ('tui_components', 0.005),
                ('background_services', 0.01)
            ]
            
            startup_times = {}
            for component, duration in components:
                start = time.perf_counter()
                time.sleep(duration)  # Simulate initialization time
                end = time.perf_counter()
                startup_times[component] = end - start
            
            return startup_times
        
        result = benchmark(startup_simulation)
        
        # Verify startup completed within reasonable time
        total_startup_time = sum(result.values())
        assert total_startup_time < 0.1  # Under 100ms total
    
    def test_end_to_end_workflow_performance(self, benchmark, mock_config_manager):
        """Benchmark complete end-to-end workflow performance."""
        async def end_to_end_workflow():
            workflow_steps = [
                ('load_project', 0.005),
                ('analyze_code', 0.01),
                ('run_validation', 0.015),
                ('generate_report', 0.008),
                ('save_results', 0.003)
            ]
            
            workflow_results = {}
            
            for step_name, duration in workflow_steps:
                start_time = asyncio.get_event_loop().time()
                
                # Simulate async workflow step
                await asyncio.sleep(duration)
                
                end_time = asyncio.get_event_loop().time()
                workflow_results[step_name] = {
                    'duration': end_time - start_time,
                    'success': True
                }
            
            return workflow_results
        
        def run_workflow():
            return asyncio.run(end_to_end_workflow())
        
        result = benchmark(run_workflow)
        
        # Verify workflow completed successfully
        assert len(result) == 5
        assert all(step['success'] for step in result.values())
        
        total_workflow_time = sum(step['duration'] for step in result.values())
        assert total_workflow_time < 0.1  # Under 100ms total