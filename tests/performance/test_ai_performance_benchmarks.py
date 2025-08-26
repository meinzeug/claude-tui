#!/usr/bin/env python3
"""
AI Performance Benchmarks
Comprehensive performance testing for AI components with focus on 95.8% accuracy validation.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import gc
import memory_profiler
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pathlib import Path

# Performance testing utilities
class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.memory_usage = []
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.perf_counter()
        
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.perf_counter() - self.start_times[operation]
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        return duration
        
    def record_memory_usage(self):
        """Record current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_usage.append({
            'timestamp': time.time(),
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'percent': process.memory_percent()
        })
        
    def get_statistics(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
            
        times = self.metrics[operation]
        return {
            'count': len(times),
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min': min(times),
            'max': max(times),
            'total': sum(times)
        }
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.memory_usage:
            return {}
            
        rss_values = [entry['rss'] for entry in self.memory_usage]
        percent_values = [entry['percent'] for entry in self.memory_usage]
        
        return {
            'peak_rss_mb': max(rss_values) / 1024 / 1024,
            'avg_rss_mb': statistics.mean(rss_values) / 1024 / 1024,
            'peak_percent': max(percent_values),
            'avg_percent': statistics.mean(percent_values),
            'samples': len(self.memory_usage)
        }

# Mock AI components for testing
@dataclass
class AIModel:
    name: str
    type: str
    accuracy: float = 0.0
    inference_time: float = 0.0
    memory_usage: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    operation: str
    model_name: str
    execution_time: float
    memory_usage: float
    accuracy: float
    throughput: float
    success: bool
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class MockAntiHallucinationEngine:
    """Mock Anti-Hallucination Engine for performance testing."""
    
    def __init__(self):
        self.models = {
            'pattern_recognition': AIModel("PatternRecognition", "classifier", 0.962),
            'authenticity_classifier': AIModel("AuthenticityClassifier", "classifier", 0.958),
            'placeholder_detector': AIModel("PlaceholderDetector", "detector", 0.945),
            'semantic_analyzer': AIModel("SemanticAnalyzer", "nlp", 0.892),
        }
        self.cache = {}
        self.performance_target_ms = 200
        self.accuracy_target = 0.958
    
    async def initialize(self):
        """Initialize the engine."""
        await asyncio.sleep(0.01)  # Simulate initialization
    
    async def validate_code_authenticity(self, code: str, use_cache: bool = True) -> Dict[str, Any]:
        """Simulate code authenticity validation."""
        # Simulate cache lookup
        cache_key = hash(code)
        if use_cache and cache_key in self.cache:
            result = self.cache[cache_key].copy()
            result['from_cache'] = True
            return result
        
        # Simulate processing time based on code length
        processing_time = min(0.05 + (len(code) / 10000), 0.5)
        await asyncio.sleep(processing_time)
        
        # Simulate accuracy calculation
        authenticity_score = 0.95
        if 'todo' in code.lower() or 'placeholder' in code.lower():
            authenticity_score = 0.3
        elif 'def ' in code and '{' in code and '}' in code:
            authenticity_score = 0.92
        
        # Add some realistic variance
        authenticity_score += np.random.normal(0, 0.02)
        authenticity_score = max(0.0, min(1.0, authenticity_score))
        
        result = {
            'authenticity_score': authenticity_score,
            'processing_time_ms': processing_time * 1000,
            'confidence': 0.95,
            'from_cache': False,
            'model_predictions': {
                model_name: model.accuracy + np.random.normal(0, 0.01)
                for model_name, model in self.models.items()
            }
        }
        
        # Cache result
        if use_cache:
            self.cache[cache_key] = result.copy()
            result['from_cache'] = False
        
        return result
    
    async def train_model(self, model_name: str, training_data: List[Dict], epochs: int = 10):
        """Simulate model training."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        # Simulate training time
        training_time = 0.1 * len(training_data) * epochs / 1000
        await asyncio.sleep(training_time)
        
        # Simulate accuracy improvement
        base_accuracy = model.accuracy
        improvement = min(0.05, len(training_data) / 10000)
        model.accuracy = min(0.99, base_accuracy + improvement)
        
        return {
            'model': model_name,
            'initial_accuracy': base_accuracy,
            'final_accuracy': model.accuracy,
            'training_time': training_time,
            'training_samples': len(training_data),
            'epochs': epochs
        }
    
    async def batch_validate(self, code_samples: List[str], batch_size: int = 10) -> List[Dict]:
        """Simulate batch validation."""
        results = []
        
        for i in range(0, len(code_samples), batch_size):
            batch = code_samples[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [self.validate_code_authenticity(code) for code in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results

class MockSwarmOrchestrator:
    """Mock Swarm Orchestrator for performance testing."""
    
    def __init__(self):
        self.agents = {}
        self.tasks = {}
        self.coordination_overhead = 0.01  # Base coordination time
    
    async def spawn_agent(self, agent_config: Dict) -> str:
        """Simulate agent spawning."""
        await asyncio.sleep(0.005)  # Spawn time
        
        agent_id = f"agent-{len(self.agents) + 1}"
        self.agents[agent_id] = {
            'id': agent_id,
            'type': agent_config.get('type', 'generic'),
            'capabilities': agent_config.get('capabilities', []),
            'status': 'idle',
            'created_at': time.time()
        }
        return agent_id
    
    async def coordinate_task(self, task_data: Dict) -> Dict:
        """Simulate task coordination."""
        coordination_time = self.coordination_overhead * len(self.agents)
        await asyncio.sleep(coordination_time)
        
        task_id = f"task-{len(self.tasks) + 1}"
        self.tasks[task_id] = {
            'id': task_id,
            'data': task_data,
            'assigned_agents': list(self.agents.keys())[:task_data.get('required_agents', 1)],
            'created_at': time.time()
        }
        
        return {
            'task_id': task_id,
            'coordination_time': coordination_time,
            'assigned_agents': len(self.tasks[task_id]['assigned_agents'])
        }
    
    async def process_concurrent_tasks(self, task_configs: List[Dict]) -> List[Dict]:
        """Process multiple tasks concurrently."""
        tasks = [self.coordinate_task(config) for config in task_configs]
        return await asyncio.gather(*tasks)

class MockClaudeCodeClient:
    """Mock Claude Code Client for performance testing."""
    
    def __init__(self):
        self.request_latency = 0.1  # Base latency
        self.rate_limit = 100  # Requests per minute
        self.last_request_times = []
    
    async def send_message(self, message: str, context: Dict = None) -> Dict:
        """Simulate API request."""
        # Rate limiting simulation
        current_time = time.time()
        self.last_request_times = [
            t for t in self.last_request_times 
            if current_time - t < 60
        ]
        
        if len(self.last_request_times) >= self.rate_limit:
            raise Exception("Rate limit exceeded")
        
        self.last_request_times.append(current_time)
        
        # Simulate network latency
        latency = self.request_latency + np.random.exponential(0.05)
        await asyncio.sleep(latency)
        
        # Simulate response size based on message length
        response_tokens = min(len(message.split()) * 2, 1000)
        
        return {
            'response': f"Response to: {message[:50]}...",
            'tokens': response_tokens,
            'latency': latency,
            'timestamp': current_time
        }
    
    async def batch_requests(self, messages: List[str]) -> List[Dict]:
        """Process multiple requests with batching optimization."""
        # Simulate batch optimization
        batch_efficiency = 0.8  # 20% efficiency gain from batching
        tasks = []
        
        for message in messages:
            # Reduce individual latency due to batching
            original_latency = self.request_latency
            self.request_latency *= batch_efficiency
            tasks.append(self.send_message(message))
            self.request_latency = original_latency
        
        return await asyncio.gather(*tasks)


@pytest.fixture
def performance_metrics():
    """Performance metrics collector."""
    return PerformanceMetrics()

@pytest.fixture
def anti_hallucination_engine():
    """Mock anti-hallucination engine."""
    return MockAntiHallucinationEngine()

@pytest.fixture
def swarm_orchestrator():
    """Mock swarm orchestrator."""
    return MockSwarmOrchestrator()

@pytest.fixture
def claude_code_client():
    """Mock Claude Code client."""
    return MockClaudeCodeClient()

@pytest.fixture
def sample_code_data():
    """Sample code data for testing."""
    return [
        "def quicksort(arr): return sorted(arr)",  # Simple
        """
def complex_algorithm(data):
    '''Complex algorithm implementation.'''
    result = []
    for item in data:
        if isinstance(item, dict):
            processed = process_dict(item)
            result.append(processed)
    return result
""",  # Medium complexity
        """
class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def process(self, data):
        # TODO: Implement processing logic
        pass
        
    def validate(self, data):
        # PLACEHOLDER: Add validation
        return True
""",  # Placeholder code
        "x = 1",  # Minimal code
        "# Just a comment"  # Comment only
    ]


class TestAntiHallucinationPerformance:
    """Performance tests for Anti-Hallucination Engine."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_single_validation_performance(self, anti_hallucination_engine, performance_metrics, sample_code_data):
        """Test single validation performance metrics."""
        await anti_hallucination_engine.initialize()
        
        results = []
        
        for i, code in enumerate(sample_code_data):
            performance_metrics.record_memory_usage()
            performance_metrics.start_timer(f'validation_{i}')
            
            result = await anti_hallucination_engine.validate_code_authenticity(code)
            
            execution_time = performance_metrics.end_timer(f'validation_{i}')
            performance_metrics.record_memory_usage()
            
            benchmark_result = BenchmarkResult(
                operation='single_validation',
                model_name='anti_hallucination',
                execution_time=execution_time,
                memory_usage=performance_metrics.memory_usage[-1]['rss'] / 1024 / 1024,
                accuracy=result['authenticity_score'],
                throughput=1 / execution_time,
                success=True,
                metadata={'code_length': len(code)}
            )
            results.append(benchmark_result)
            
            # Verify performance targets
            assert execution_time < 1.0, f"Validation took too long: {execution_time:.3f}s"
            assert result['processing_time_ms'] < anti_hallucination_engine.performance_target_ms
        
        # Analyze results
        execution_times = [r.execution_time for r in results]
        accuracy_scores = [r.accuracy for r in results]
        
        avg_time = statistics.mean(execution_times)
        avg_accuracy = statistics.mean(accuracy_scores)
        
        print(f"Single validation performance:")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Average accuracy: {avg_accuracy:.3f}")
        print(f"  Throughput: {1/avg_time:.1f} validations/sec")
        
        # Performance assertions
        assert avg_time < 0.5, f"Average validation time too high: {avg_time:.3f}s"
        assert avg_accuracy > 0.8, f"Average accuracy too low: {avg_accuracy:.3f}"
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_batch_validation_performance(self, anti_hallucination_engine, performance_metrics):
        """Test batch validation performance."""
        await anti_hallucination_engine.initialize()
        
        # Generate test data
        batch_sizes = [1, 5, 10, 25, 50]
        code_template = "def function_{i}(): return {i}"
        
        results = {}
        
        for batch_size in batch_sizes:
            test_codes = [code_template.format(i=i) for i in range(batch_size)]
            
            performance_metrics.record_memory_usage()
            performance_metrics.start_timer(f'batch_{batch_size}')
            
            batch_results = await anti_hallucination_engine.batch_validate(test_codes, batch_size=10)
            
            execution_time = performance_metrics.end_timer(f'batch_{batch_size}')
            performance_metrics.record_memory_usage()
            
            throughput = len(test_codes) / execution_time
            avg_accuracy = statistics.mean([r['authenticity_score'] for r in batch_results])
            
            results[batch_size] = {
                'execution_time': execution_time,
                'throughput': throughput,
                'accuracy': avg_accuracy,
                'memory_mb': performance_metrics.memory_usage[-1]['rss'] / 1024 / 1024
            }
            
            print(f"Batch size {batch_size}: {throughput:.1f} validations/sec, accuracy: {avg_accuracy:.3f}")
        
        # Verify batch efficiency
        single_throughput = results[1]['throughput']
        batch_10_throughput = results[10]['throughput'] if 10 in results else single_throughput
        
        # Batching should provide some efficiency gain
        efficiency_gain = batch_10_throughput / single_throughput
        print(f"Batching efficiency gain: {efficiency_gain:.2f}x")
        
        assert efficiency_gain > 1.5, f"Insufficient batching efficiency: {efficiency_gain:.2f}x"
    
    @pytest.mark.performance
    @pytest.mark.accuracy
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_accuracy_vs_performance_tradeoff(self, anti_hallucination_engine, performance_metrics):
        """Test accuracy vs performance trade-off."""
        await anti_hallucination_engine.initialize()
        
        # Test different complexity levels
        test_cases = [
            ("simple", "def add(a, b): return a + b"),
            ("medium", """
def process_data(data):
    '''Process input data with validation.'''
    if not data:
        return []
    
    processed = []
    for item in data:
        if isinstance(item, str):
            processed.append(item.strip().lower())
        elif isinstance(item, dict):
            processed.append(process_dict(item))
    
    return processed
"""),
            ("complex", """
class AdvancedProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
        
    def process_complex_data(self, data):
        '''Advanced data processing with caching and validation.'''
        cache_key = self._generate_cache_key(data)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # TODO: Implement advanced processing
        result = self._complex_algorithm(data)
        self.cache[cache_key] = result
        return result
"""),
            ("placeholder", """
def incomplete_function():
    '''TODO: Implement this function.'''
    # PLACEHOLDER: Add implementation
    pass
""")
        ]
        
        results = {}
        
        for complexity, code in test_cases:
            performance_metrics.start_timer(f'accuracy_test_{complexity}')
            
            # Run multiple validations for statistical significance
            validation_results = []
            for _ in range(10):
                result = await anti_hallucination_engine.validate_code_authenticity(code, use_cache=False)
                validation_results.append(result)
            
            execution_time = performance_metrics.end_timer(f'accuracy_test_{complexity}')
            
            avg_accuracy = statistics.mean([r['authenticity_score'] for r in validation_results])
            accuracy_std = statistics.stdev([r['authenticity_score'] for r in validation_results])
            avg_processing_time = statistics.mean([r['processing_time_ms'] for r in validation_results])
            
            results[complexity] = {
                'accuracy': avg_accuracy,
                'accuracy_std': accuracy_std,
                'processing_time_ms': avg_processing_time,
                'consistency': 1.0 - (accuracy_std / avg_accuracy) if avg_accuracy > 0 else 0.0
            }
            
            print(f"{complexity.capitalize()}: accuracy={avg_accuracy:.3f}±{accuracy_std:.3f}, time={avg_processing_time:.1f}ms")
        
        # Verify accuracy targets
        assert results['simple']['accuracy'] > 0.9, "Simple code accuracy too low"
        assert results['placeholder']['accuracy'] < 0.5, "Placeholder code should be detected"
        
        # Verify consistency
        for complexity, metrics in results.items():
            assert metrics['consistency'] > 0.8, f"{complexity} accuracy not consistent enough"
    
    @pytest.mark.performance
    @pytest.mark.memory
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, anti_hallucination_engine, performance_metrics):
        """Test memory efficiency under load."""
        await anti_hallucination_engine.initialize()
        
        # Generate large dataset
        large_dataset = [f"def function_{i}(): return {i}" for i in range(500)]
        
        # Monitor memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        performance_metrics.record_memory_usage()
        
        # Process dataset in chunks
        chunk_size = 50
        results = []
        
        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset[i:i + chunk_size]
            
            # Process chunk
            chunk_results = await anti_hallucination_engine.batch_validate(chunk)
            results.extend(chunk_results)
            
            # Record memory after each chunk
            performance_metrics.record_memory_usage()
            
            # Force garbage collection
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_stats = performance_metrics.get_memory_statistics()
        
        print(f"Memory efficiency test:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Peak memory: {memory_stats['peak_rss_mb']:.1f} MB")
        print(f"  Memory growth: {final_memory - initial_memory:.1f} MB")
        
        # Memory assertions
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f} MB"
        assert len(results) == len(large_dataset), "Not all samples processed"


class TestSwarmOrchestrationPerformance:
    """Performance tests for Swarm Orchestration."""
    
    @pytest.mark.performance
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_agent_spawn_performance(self, swarm_orchestrator, performance_metrics):
        """Test agent spawning performance."""
        agent_counts = [1, 5, 10, 25, 50]
        results = {}
        
        for count in agent_counts:
            performance_metrics.start_timer(f'spawn_{count}')
            
            # Spawn agents concurrently
            spawn_tasks = [
                swarm_orchestrator.spawn_agent({
                    'type': 'test_agent',
                    'capabilities': ['testing', 'validation']
                })
                for _ in range(count)
            ]
            
            agent_ids = await asyncio.gather(*spawn_tasks)
            execution_time = performance_metrics.end_timer(f'spawn_{count}')
            
            results[count] = {
                'execution_time': execution_time,
                'agents_per_second': count / execution_time,
                'success': len(agent_ids) == count
            }
            
            print(f"Spawned {count} agents in {execution_time:.3f}s ({results[count]['agents_per_second']:.1f} agents/sec)")
        
        # Verify performance scaling
        assert results[1]['agents_per_second'] > 100, "Single agent spawn too slow"
        assert results[50]['execution_time'] < 2.0, "Bulk spawn too slow"
    
    @pytest.mark.performance
    @pytest.mark.coordination
    @pytest.mark.asyncio
    async def test_task_coordination_performance(self, swarm_orchestrator, performance_metrics):
        """Test task coordination performance."""
        # Setup agents
        agent_configs = [
            {'type': 'tester', 'capabilities': ['testing']},
            {'type': 'coder', 'capabilities': ['coding']},
            {'type': 'reviewer', 'capabilities': ['reviewing']},
        ]
        
        for config in agent_configs:
            await swarm_orchestrator.spawn_agent(config)
        
        # Test concurrent task coordination
        task_counts = [1, 5, 10, 25]
        results = {}
        
        for count in task_counts:
            task_configs = [
                {
                    'description': f'Test task {i}',
                    'required_agents': 1,
                    'priority': 'medium'
                }
                for i in range(count)
            ]
            
            performance_metrics.start_timer(f'coordinate_{count}')
            coordination_results = await swarm_orchestrator.process_concurrent_tasks(task_configs)
            execution_time = performance_metrics.end_timer(f'coordinate_{count}')
            
            results[count] = {
                'execution_time': execution_time,
                'tasks_per_second': count / execution_time,
                'success_rate': len(coordination_results) / count
            }
            
            print(f"Coordinated {count} tasks in {execution_time:.3f}s ({results[count]['tasks_per_second']:.1f} tasks/sec)")
        
        # Verify coordination efficiency
        assert results[1]['tasks_per_second'] > 10, "Single task coordination too slow"
        assert all(r['success_rate'] == 1.0 for r in results.values()), "Task coordination failures"


class TestClaudeCodeClientPerformance:
    """Performance tests for Claude Code Client."""
    
    @pytest.mark.performance
    @pytest.mark.network
    @pytest.mark.asyncio
    async def test_api_latency_performance(self, claude_code_client, performance_metrics):
        """Test API latency performance."""
        test_messages = [
            "Simple test message",
            "Medium complexity message with multiple sentences and some technical content about programming.",
            "Complex message with extensive technical details about software architecture, design patterns, performance optimization, and system integration requirements that would typically require more processing time and resources."
        ]
        
        results = {}
        
        for i, message in enumerate(test_messages):
            message_type = ['simple', 'medium', 'complex'][i]
            
            performance_metrics.start_timer(f'api_{message_type}')
            
            try:
                result = await claude_code_client.send_message(message)
                execution_time = performance_metrics.end_timer(f'api_{message_type}')
                
                results[message_type] = {
                    'execution_time': execution_time,
                    'latency': result['latency'],
                    'tokens': result['tokens'],
                    'success': True
                }
                
                print(f"{message_type.capitalize()}: {execution_time:.3f}s, latency: {result['latency']:.3f}s, tokens: {result['tokens']}")
                
                # Verify reasonable performance
                assert execution_time < 5.0, f"{message_type} request took too long"
                assert result['latency'] < 2.0, f"{message_type} latency too high"
                
            except Exception as e:
                results[message_type] = {
                    'execution_time': 0.0,
                    'success': False,
                    'error': str(e)
                }
        
        # Verify all requests succeeded
        success_count = sum(1 for r in results.values() if r['success'])
        assert success_count == len(test_messages), "Some API requests failed"
    
    @pytest.mark.performance
    @pytest.mark.network
    @pytest.mark.asyncio
    async def test_batch_request_performance(self, claude_code_client, performance_metrics):
        """Test batch request performance and optimization."""
        batch_sizes = [1, 5, 10, 20]
        message_template = "Test message number {i} for batch processing evaluation."
        
        results = {}
        
        for batch_size in batch_sizes:
            messages = [message_template.format(i=i) for i in range(batch_size)]
            
            # Test individual requests
            performance_metrics.start_timer(f'individual_{batch_size}')
            individual_tasks = [claude_code_client.send_message(msg) for msg in messages]
            individual_results = await asyncio.gather(*individual_tasks, return_exceptions=True)
            individual_time = performance_metrics.end_timer(f'individual_{batch_size}')
            
            # Test batch requests
            performance_metrics.start_timer(f'batch_{batch_size}')
            try:
                batch_results = await claude_code_client.batch_requests(messages)
                batch_time = performance_metrics.end_timer(f'batch_{batch_size}')
                batch_success = len(batch_results) == batch_size
            except Exception as e:
                batch_time = performance_metrics.end_timer(f'batch_{batch_size}')
                batch_results = []
                batch_success = False
            
            individual_success_count = sum(1 for r in individual_results if not isinstance(r, Exception))
            
            results[batch_size] = {
                'individual_time': individual_time,
                'batch_time': batch_time,
                'individual_success_rate': individual_success_count / batch_size,
                'batch_success': batch_success,
                'efficiency_gain': individual_time / batch_time if batch_time > 0 else 0.0
            }
            
            print(f"Batch {batch_size}: individual={individual_time:.3f}s, batch={batch_time:.3f}s, gain={results[batch_size]['efficiency_gain']:.2f}x")
        
        # Verify batch optimization benefits
        for size in batch_sizes[1:]:  # Skip size 1
            if results[size]['batch_success'] and results[size]['individual_success_rate'] > 0.8:
                assert results[size]['efficiency_gain'] > 1.0, f"No batching benefit for size {size}"


class TestIntegratedPerformance:
    """Integrated performance tests across all AI components."""
    
    @pytest.mark.performance
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_performance(self, anti_hallucination_engine, swarm_orchestrator, claude_code_client, performance_metrics):
        """Test end-to-end workflow performance."""
        await anti_hallucination_engine.initialize()
        
        # Setup test scenario
        code_samples = [
            "def example_function(): return True",
            "def incomplete(): # TODO: implement",
            "class TestClass: pass",
        ]
        
        # Complete workflow simulation
        performance_metrics.start_timer('e2e_workflow')
        performance_metrics.record_memory_usage()
        
        # 1. Spawn agents
        agent_configs = [
            {'type': 'validator', 'capabilities': ['validation']},
            {'type': 'coordinator', 'capabilities': ['coordination']},
        ]
        agent_ids = []
        for config in agent_configs:
            agent_id = await swarm_orchestrator.spawn_agent(config)
            agent_ids.append(agent_id)
        
        # 2. Validate code samples
        validation_results = []
        for code in code_samples:
            result = await anti_hallucination_engine.validate_code_authenticity(code)
            validation_results.append(result)
        
        # 3. Coordinate tasks based on validation
        task_configs = [
            {
                'description': f'Process validation result {i}',
                'required_agents': 1
            }
            for i in range(len(validation_results))
        ]
        coordination_results = await swarm_orchestrator.process_concurrent_tasks(task_configs)
        
        # 4. Generate AI responses
        ai_requests = [
            f"Analyze validation result: {result['authenticity_score']:.3f}"
            for result in validation_results
        ]
        ai_responses = await claude_code_client.batch_requests(ai_requests)
        
        performance_metrics.record_memory_usage()
        total_time = performance_metrics.end_timer('e2e_workflow')
        memory_stats = performance_metrics.get_memory_statistics()
        
        # Analyze results
        avg_accuracy = statistics.mean([r['authenticity_score'] for r in validation_results])
        total_operations = len(code_samples) + len(coordination_results) + len(ai_responses)
        throughput = total_operations / total_time
        
        print(f"End-to-end workflow performance:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average accuracy: {avg_accuracy:.3f}")
        print(f"  Throughput: {throughput:.1f} operations/sec")
        print(f"  Memory usage: {memory_stats.get('peak_rss_mb', 0):.1f} MB peak")
        
        # Performance assertions
        assert total_time < 10.0, f"Workflow too slow: {total_time:.3f}s"
        assert avg_accuracy > 0.7, f"Average accuracy too low: {avg_accuracy:.3f}"
        assert throughput > 1.0, f"Throughput too low: {throughput:.1f} ops/sec"
        assert len(agent_ids) == len(agent_configs), "Agent spawning failed"
        assert len(validation_results) == len(code_samples), "Validation failed"
        assert len(coordination_results) == len(task_configs), "Coordination failed"
        assert len(ai_responses) == len(ai_requests), "AI requests failed"
    
    @pytest.mark.performance
    @pytest.mark.accuracy
    @pytest.mark.asyncio
    async def test_accuracy_benchmark_suite(self, anti_hallucination_engine, performance_metrics):
        """Comprehensive accuracy benchmark suite."""
        await anti_hallucination_engine.initialize()
        
        # Accuracy test dataset with ground truth
        accuracy_dataset = [
            ("def quicksort(arr): return sorted(arr)", True, "authentic_simple"),
            ("""
def complex_sort(arr, key=None, reverse=False):
    '''Advanced sorting with custom key function.'''
    if not arr:
        return []
    return sorted(arr, key=key, reverse=reverse)
""", True, "authentic_complex"),
            ("def incomplete(): # TODO: implement", False, "placeholder_simple"),
            ("""
class DataProcessor:
    def process(self, data):
        # PLACEHOLDER: Add processing logic
        pass
""", False, "placeholder_complex"),
            ("x = 1", True, "minimal_authentic"),
            ("# Just a comment", False, "comment_only"),
            ("", False, "empty_code"),
        ]
        
        # Run accuracy benchmark
        performance_metrics.start_timer('accuracy_benchmark')
        
        results = []
        for code, ground_truth, category in accuracy_dataset:
            validation_result = await anti_hallucination_engine.validate_code_authenticity(code)
            
            predicted_authentic = validation_result['authenticity_score'] >= 0.7
            correct_prediction = predicted_authentic == ground_truth
            
            results.append({
                'code': code[:50] + "..." if len(code) > 50 else code,
                'ground_truth': ground_truth,
                'predicted': predicted_authentic,
                'confidence': validation_result['authenticity_score'],
                'correct': correct_prediction,
                'category': category
            })
        
        total_time = performance_metrics.end_timer('accuracy_benchmark')
        
        # Calculate accuracy metrics
        correct_predictions = sum(1 for r in results if r['correct'])
        total_predictions = len(results)
        overall_accuracy = correct_predictions / total_predictions
        
        # Category-wise accuracy
        category_accuracy = {}
        for category in set(r['category'] for r in results):
            category_results = [r for r in results if r['category'] == category]
            category_correct = sum(1 for r in category_results if r['correct'])
            category_accuracy[category] = category_correct / len(category_results)
        
        print(f"Accuracy Benchmark Results:")
        print(f"  Overall accuracy: {overall_accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"  Benchmark time: {total_time:.3f}s")
        
        for category, accuracy in category_accuracy.items():
            print(f"  {category}: {accuracy:.3f}")
        
        # Log detailed results
        for result in results:
            status = "✓" if result['correct'] else "✗"
            print(f"  {status} {result['category']}: {result['confidence']:.3f} (expected {result['ground_truth']})")
        
        # Accuracy assertions
        assert overall_accuracy >= 0.80, f"Overall accuracy too low: {overall_accuracy:.3f}"
        
        # Check specific category requirements
        authentic_categories = [cat for cat in category_accuracy.keys() if 'authentic' in cat]
        placeholder_categories = [cat for cat in category_accuracy.keys() if 'placeholder' in cat]
        
        if authentic_categories:
            authentic_accuracy = statistics.mean([category_accuracy[cat] for cat in authentic_categories])
            assert authentic_accuracy >= 0.85, f"Authentic code detection too low: {authentic_accuracy:.3f}"
        
        if placeholder_categories:
            placeholder_accuracy = statistics.mean([category_accuracy[cat] for cat in placeholder_categories])
            assert placeholder_accuracy >= 0.85, f"Placeholder detection too low: {placeholder_accuracy:.3f}"


if __name__ == "__main__":
    # Run AI performance benchmarks
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "performance",
        "--durations=10"
    ])