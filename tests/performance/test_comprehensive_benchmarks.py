"""
Comprehensive Performance Benchmark Tests

Tests for the performance benchmarking system including load testing,
quantum intelligence performance, and regression detection.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from performance.benchmarking.comprehensive_benchmarker import (
    ComprehensivePerformanceBenchmarker,
    LoadTestEngine,
    LoadTestScenario,
    QuantumIntelligencePerformanceTester,
    DatabasePerformanceOptimizer,
    PerformanceRegressionDetector,
    PerformanceMetricsCollector
)

class TestPerformanceMetricsCollector:
    """Test performance metrics collection"""
    
    @pytest.fixture
    def metrics_collector(self):
        return PerformanceMetricsCollector()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, metrics_collector):
        """Test basic metrics collection"""
        # Start monitoring briefly
        monitoring_task = asyncio.create_task(
            metrics_collector.start_monitoring(interval=0.1)
        )
        
        # Let it collect a few samples
        await asyncio.sleep(0.5)
        
        # Stop monitoring
        metrics_collector.stop_monitoring()
        monitoring_task.cancel()
        
        # Verify metrics were collected
        assert len(metrics_collector.metrics_history) > 0
        
        # Check metric structure
        sample = metrics_collector.metrics_history[0]
        assert 'timestamp' in sample
        assert 'cpu' in sample
        assert 'memory' in sample
        assert 'disk' in sample
        assert 'network' in sample
    
    def test_metrics_summary(self, metrics_collector):
        """Test metrics summary generation"""
        # Add some test data
        test_metrics = [
            {
                'cpu': {'percent': 50.0},
                'memory': {'percent': 60.0}
            },
            {
                'cpu': {'percent': 70.0},
                'memory': {'percent': 65.0}
            },
            {
                'cpu': {'percent': 60.0},
                'memory': {'percent': 70.0}
            }
        ]
        
        metrics_collector.measurements = test_metrics
        summary = metrics_collector.get_metrics_summary()
        
        assert 'cpu' in summary
        assert 'memory' in summary
        assert summary['cpu']['avg'] == 60.0
        assert summary['memory']['max'] == 70.0

class TestLoadTestEngine:
    """Test load testing functionality"""
    
    @pytest.fixture
    def load_test_engine(self):
        return LoadTestEngine(base_url="http://localhost:8000")
    
    @pytest.fixture
    def test_scenario(self):
        return LoadTestScenario(
            name="Test Scenario",
            duration=10,  # Short duration for testing
            concurrent_users=5,
            ramp_up_time=2,
            think_time=0.1,
            endpoints=["/health", "/status"],
            request_patterns={}
        )
    
    @pytest.mark.asyncio
    async def test_load_test_scenario_structure(self, test_scenario):
        """Test load test scenario structure"""
        assert test_scenario.name == "Test Scenario"
        assert test_scenario.duration == 10
        assert test_scenario.concurrent_users == 5
        assert len(test_scenario.endpoints) == 2
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_load_test_execution(self, mock_session_class, load_test_engine, test_scenario):
        """Test load test execution with mocked HTTP client"""
        # Mock HTTP session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session
        
        # Mock context manager behavior
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        # Run load test
        result = await load_test_engine.run_load_test(test_scenario)
        
        # Verify results
        assert 'scenario' in result
        assert 'duration' in result
        assert 'total_requests' in result
        assert 'throughput' in result
        assert 'latency' in result
        assert result['scenario'] == "Test Scenario"
    
    def test_analyze_throughput_measurements(self, load_test_engine):
        """Test throughput measurement analysis"""
        # Create test measurements
        measurements = [
            {
                'actualThroughput': 100,
                'successRate': 0.95,
                'averageLatency': 0.1,
                'p95Latency': 0.2,
                'p99Latency': 0.3
            },
            {
                'actualThroughput': 120,
                'successRate': 0.98,
                'averageLatency': 0.12,
                'p95Latency': 0.22,
                'p99Latency': 0.32
            },
            {
                'actualThroughput': 80,
                'successRate': 0.90,
                'averageLatency': 0.15,
                'p95Latency': 0.25,
                'p99Latency': 0.35
            }
        ]
        
        analysis = load_test_engine._analyze_throughput_measurements(measurements)
        
        assert analysis['averageThroughput'] == 100.0  # (100 + 120 + 80) / 3
        assert analysis['maxThroughput'] == 120
        assert analysis['averageSuccessRate'] == pytest.approx(0.943, abs=0.001)

class TestQuantumIntelligencePerformanceTester:
    """Test quantum intelligence performance testing"""
    
    @pytest.fixture
    def quantum_tester(self):
        return QuantumIntelligencePerformanceTester()
    
    def test_quantum_modules_list(self, quantum_tester):
        """Test quantum modules are properly defined"""
        expected_modules = [
            'quantum_consciousness',
            'quantum_memory',
            'quantum_reasoning',
            'quantum_optimization'
        ]
        
        assert quantum_tester.quantum_modules == expected_modules
    
    @pytest.mark.asyncio
    async def test_benchmark_single_module(self, quantum_tester):
        """Test benchmarking of a single quantum module"""
        module_result = await quantum_tester._benchmark_module('quantum_consciousness')
        
        assert 'module' in module_result
        assert 'total_duration' in module_result
        assert 'load_test_results' in module_result
        assert 'scalability_score' in module_result
        assert module_result['module'] == 'quantum_consciousness'
        
        # Check load test results structure
        load_results = module_result['load_test_results']
        assert len(load_results) > 0
        
        for result in load_results:
            assert 'load_level' in result
            assert 'duration' in result
            assert 'operations_per_second' in result
            assert 'error_rate' in result
    
    @pytest.mark.asyncio
    async def test_quantum_operation_simulation(self, quantum_tester):
        """Test quantum operation simulation"""
        operation_result = await quantum_tester._quantum_operation('quantum_memory')
        
        assert 'module' in operation_result
        assert 'duration' in operation_result
        assert 'timestamp' in operation_result
        assert operation_result['module'] == 'quantum_memory'
        assert operation_result['duration'] > 0
    
    def test_scalability_score_calculation(self, quantum_tester):
        """Test scalability score calculation"""
        test_results = [
            {
                'load_level': 1,
                'operations_per_second': 100
            },
            {
                'load_level': 10,
                'operations_per_second': 80  # 800 total ops/s
            }
        ]
        
        scalability_score = quantum_tester._calculate_scalability_score(test_results)
        
        # Ideal would be 1000 ops/s (100 * 10), actual is 800
        # Score should be 800/1000 = 0.8
        assert scalability_score == 0.8
    
    @pytest.mark.asyncio
    async def test_benchmark_all_modules(self, quantum_tester):
        """Test benchmarking all quantum modules"""
        results = await quantum_tester.benchmark_quantum_modules()
        
        # Should have results for all modules
        assert len(results) == len(quantum_tester.quantum_modules)
        
        for module in quantum_tester.quantum_modules:
            assert module in results
            assert 'scalability_score' in results[module]

class TestDatabasePerformanceOptimizer:
    """Test database performance optimization"""
    
    @pytest.fixture
    def db_optimizer(self):
        return DatabasePerformanceOptimizer({})
    
    @pytest.mark.asyncio
    async def test_query_benchmarking(self, db_optimizer):
        """Test database query benchmarking"""
        query_results = await db_optimizer._benchmark_queries()
        
        assert 'total_queries' in query_results
        assert 'avg_execution_time' in query_results
        assert 'max_execution_time' in query_results
        assert 'query_details' in query_results
        
        # Should have tested multiple queries
        assert query_results['total_queries'] > 0
        assert len(query_results['query_details']) == query_results['total_queries']
    
    @pytest.mark.asyncio
    async def test_connection_pool_benchmarking(self, db_optimizer):
        """Test connection pool performance benchmarking"""
        pool_results = await db_optimizer._benchmark_connection_pool()
        
        assert 'pool_benchmarks' in pool_results
        assert 'optimal_pool_size' in pool_results
        
        # Should have tested multiple pool sizes
        benchmarks = pool_results['pool_benchmarks']
        assert len(benchmarks) > 0
        
        for benchmark in benchmarks:
            assert 'pool_size' in benchmark
            assert 'ops_per_second' in benchmark
    
    @pytest.mark.asyncio
    async def test_cache_benchmarking(self, db_optimizer):
        """Test cache performance benchmarking"""
        cache_results = await db_optimizer._benchmark_cache()
        
        assert 'total_operations' in cache_results
        assert 'cache_hits' in cache_results
        assert 'cache_misses' in cache_results
        assert 'hit_rate' in cache_results
        assert 'performance_improvement' in cache_results
        
        # Hit rate should be between 0 and 1
        assert 0 <= cache_results['hit_rate'] <= 1
        
        # Total operations should equal hits + misses
        total_ops = cache_results['total_operations']
        hits = cache_results['cache_hits']
        misses = cache_results['cache_misses']
        assert hits + misses == total_ops
    
    @pytest.mark.asyncio
    async def test_index_analysis(self, db_optimizer):
        """Test database index analysis"""
        index_results = await db_optimizer._analyze_indexes()
        
        assert 'total_indexes' in index_results
        assert 'unused_indexes' in index_results
        assert 'duplicate_indexes' in index_results
        assert 'missing_indexes' in index_results
        assert 'index_recommendations' in index_results
        assert 'estimated_performance_gain' in index_results

class TestPerformanceRegressionDetector:
    """Test performance regression detection"""
    
    @pytest.fixture
    def regression_detector(self):
        return PerformanceRegressionDetector(baseline_file="test_baseline.json")
    
    @pytest.fixture
    def sample_baseline(self):
        return {
            'throughput': 1000,
            'latency': {
                'p95': 0.5
            },
            'error_rate': 0.01,
            'resource_usage': {
                'cpu': {
                    'avg': 50.0
                },
                'memory': {
                    'avg': 60.0
                }
            }
        }
    
    @pytest.fixture
    def sample_current_data(self):
        return {
            'throughput': 800,  # 20% degradation
            'latency': {
                'p95': 0.6  # 20% degradation
            },
            'error_rate': 0.015,  # 50% increase
            'resource_usage': {
                'cpu': {
                    'avg': 60.0  # 20% increase
                },
                'memory': {
                    'avg': 58.0  # 3.3% improvement
                }
            }
        }
    
    def test_get_nested_value(self, regression_detector):
        """Test nested value extraction"""
        test_data = {
            'level1': {
                'level2': {
                    'value': 42
                }
            }
        }
        
        # Test successful extraction
        value = regression_detector._get_nested_value(test_data, 'level1.level2.value')
        assert value == 42
        
        # Test missing path
        value = regression_detector._get_nested_value(test_data, 'missing.path')
        assert value is None
    
    def test_regression_detection(self, regression_detector, sample_baseline, sample_current_data):
        """Test regression detection logic"""
        # Set baseline data
        regression_detector.baseline_data = sample_baseline
        
        # Detect regressions
        result = regression_detector.detect_regressions(sample_current_data)
        
        assert 'regressions' in result
        assert 'improvements' in result
        assert 'status' in result
        
        # Should detect regressions
        regressions = result['regressions']
        assert len(regressions) > 0
        
        # Check specific regression
        throughput_regression = next(
            (r for r in regressions if r['metric'] == 'throughput'), 
            None
        )
        assert throughput_regression is not None
        assert throughput_regression['change_percent'] == -20.0
        assert throughput_regression['severity'] == 'HIGH'  # >20% degradation
    
    def test_no_baseline_scenario(self, regression_detector, sample_current_data):
        """Test behavior when no baseline exists"""
        # Ensure no baseline
        regression_detector.baseline_data = {}
        
        result = regression_detector.detect_regressions(sample_current_data)
        
        assert result['status'] == 'no_baseline'
        assert len(result['regressions']) == 0
        assert len(result['improvements']) == 0
    
    def test_save_baseline(self, regression_detector, sample_baseline, tmp_path):
        """Test baseline saving"""
        # Use temporary file
        temp_baseline = tmp_path / "test_baseline.json"
        regression_detector.baseline_file = str(temp_baseline)
        
        # Save baseline
        regression_detector.save_baseline(sample_baseline)
        
        # Verify file was created and contains correct data
        assert temp_baseline.exists()
        
        with open(temp_baseline, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data == sample_baseline

class TestComprehensivePerformanceBenchmarker:
    """Test the main benchmarking orchestrator"""
    
    @pytest.fixture
    def benchmarker(self):
        return ComprehensivePerformanceBenchmarker()
    
    @pytest.mark.asyncio
    async def test_memory_analysis(self, benchmarker):
        """Test memory pattern analysis"""
        memory_analysis = await benchmarker._analyze_memory_patterns()
        
        assert 'memory_summary' in memory_analysis
        assert 'memory_growth' in memory_analysis
        assert 'leak_detected' in memory_analysis
        assert 'operations_analyzed' in memory_analysis
        assert 'recommendations' in memory_analysis
        
        # Should analyze some operations
        assert memory_analysis['operations_analyzed'] > 0
        
        # Should have recommendations
        assert len(memory_analysis['recommendations']) > 0
    
    def test_memory_recommendations(self, benchmarker):
        """Test memory recommendation generation"""
        # Test high memory growth
        high_growth_recommendations = benchmarker._generate_memory_recommendations(15.0)
        assert len(high_growth_recommendations) >= 3
        assert any('memory leak' in rec.lower() for rec in high_growth_recommendations)
        
        # Test moderate memory growth
        moderate_growth_recommendations = benchmarker._generate_memory_recommendations(7.0)
        assert len(moderate_growth_recommendations) >= 2
        
        # Test low memory growth
        low_growth_recommendations = benchmarker._generate_memory_recommendations(1.0)
        assert len(low_growth_recommendations) >= 2  # Always has basic recommendations
    
    def test_performance_summary_generation(self, benchmarker):
        """Test performance summary generation"""
        sample_results = {
            'benchmark_id': 'test_123',
            'total_duration': 120.5,
            'load_tests': {
                'Light Load': {
                    'error_rate': 0.02,  # Should trigger warning
                    'latency': {
                        'p95': 1.5  # Should be OK
                    }
                },
                'Heavy Load': {
                    'error_rate': 0.01,  # Should be OK
                    'latency': {
                        'p95': 3.0  # Should trigger warning
                    }
                }
            },
            'quantum_performance': {
                'quantum_consciousness': {
                    'scalability_score': 0.7  # Should trigger warning
                },
                'quantum_memory': {
                    'scalability_score': 0.9  # Should be OK
                }
            },
            'memory_analysis': {
                'leak_detected': False
            },
            'regression_analysis': {
                'regression_count': 1  # Should trigger critical
            }
        }
        
        summary = benchmarker._generate_performance_summary(sample_results)
        
        assert summary['benchmark_id'] == 'test_123'
        assert summary['total_duration'] == 120.5
        assert summary['overall_status'] == 'CRITICAL'  # Due to regression
        assert len(summary['issues_found']) > 0
        assert len(summary['recommendations']) > 0
    
    @pytest.mark.asyncio
    @patch('builtins.open', new_callable=Mock)
    @patch('json.dump')
    async def test_report_generation(self, mock_json_dump, mock_open, benchmarker):
        """Test performance report generation"""
        sample_results = {
            'benchmark_id': 'test_123',
            'total_duration': 60.0
        }
        
        await benchmarker._generate_performance_report(sample_results)
        
        # Should have called open twice (main report + summary)
        assert mock_open.call_count == 2
        
        # Should have called json.dump twice
        assert mock_json_dump.call_count == 2

@pytest.mark.integration
class TestPerformanceBenchmarkIntegration:
    """Integration tests for performance benchmarking"""
    
    @pytest.mark.asyncio
    async def test_full_benchmark_execution(self):
        """Test complete benchmark execution (short version)"""
        benchmarker = ComprehensivePerformanceBenchmarker()
        
        # Mock external dependencies to speed up test
        with patch.object(benchmarker.load_test_engine, 'run_load_test') as mock_load_test:
            mock_load_test.return_value = {
                'scenario': 'Test',
                'throughput': 500,
                'latency': {'p95': 1.0},
                'error_rate': 0.01
            }
            
            with patch.object(benchmarker.quantum_tester, 'benchmark_quantum_modules') as mock_quantum:
                mock_quantum.return_value = {
                    'quantum_consciousness': {'scalability_score': 0.8}
                }
                
                with patch.object(benchmarker.db_optimizer, 'benchmark_database_performance') as mock_db:
                    mock_db.return_value = {'query_performance': {'avg_execution_time': 0.1}}
                    
                    # Run benchmark
                    results = await benchmarker.run_comprehensive_benchmark()
                    
                    # Verify structure
                    assert 'benchmark_id' in results
                    assert 'load_tests' in results
                    assert 'quantum_performance' in results
                    assert 'database_performance' in results
                    assert 'memory_analysis' in results
                    assert 'regression_analysis' in results
                    assert 'total_duration' in results

if __name__ == '__main__':
    pytest.main([__file__, '-v'])