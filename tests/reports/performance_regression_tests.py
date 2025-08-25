#!/usr/bin/env python3
"""
Performance Regression Test Framework
Automated tests to detect performance degradations
"""

import os
import sys
import time
import json
import psutil
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import tracemalloc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    test_name: str
    duration_ms: float
    memory_peak_mb: float
    memory_final_mb: float
    cpu_usage_percent: float
    throughput_ops_per_sec: Optional[float] = None
    error_rate_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RegressionTestResult:
    """Regression test execution result."""
    test_name: str
    current_benchmark: PerformanceBenchmark
    baseline_benchmark: Optional[PerformanceBenchmark] = None
    regression_detected: bool = False
    performance_change_percent: Dict[str, float] = field(default_factory=dict)
    status: str = "UNKNOWN"  # PASS, REGRESSION, IMPROVEMENT, UNKNOWN


class PerformanceRegressionTester:
    """Automated performance regression testing."""
    
    # Performance thresholds (% change that triggers regression alert)
    DURATION_REGRESSION_THRESHOLD = 20.0  # 20% slower
    MEMORY_REGRESSION_THRESHOLD = 15.0     # 15% more memory
    THROUGHPUT_REGRESSION_THRESHOLD = 10.0 # 10% less throughput
    
    def __init__(self, baseline_file: Optional[Path] = None):
        self.baseline_file = baseline_file or Path(__file__).parent / "performance_baseline.json"
        self.baseline_data: Dict[str, PerformanceBenchmark] = {}
        self.load_baseline()
        
    def load_baseline(self):
        """Load baseline performance data."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                    
                for test_name, benchmark_data in data.items():
                    self.baseline_data[test_name] = PerformanceBenchmark(
                        test_name=benchmark_data['test_name'],
                        duration_ms=benchmark_data['duration_ms'],
                        memory_peak_mb=benchmark_data['memory_peak_mb'],
                        memory_final_mb=benchmark_data['memory_final_mb'],
                        cpu_usage_percent=benchmark_data['cpu_usage_percent'],
                        throughput_ops_per_sec=benchmark_data.get('throughput_ops_per_sec'),
                        error_rate_percent=benchmark_data.get('error_rate_percent', 0.0),
                        metadata=benchmark_data.get('metadata', {})
                    )
            except Exception as e:
                print(f"⚠️ Could not load baseline data: {e}")
                
    def save_baseline(self, benchmarks: Dict[str, PerformanceBenchmark]):
        """Save new baseline performance data."""
        baseline_data = {}
        for test_name, benchmark in benchmarks.items():
            baseline_data[test_name] = {
                'test_name': benchmark.test_name,
                'duration_ms': benchmark.duration_ms,
                'memory_peak_mb': benchmark.memory_peak_mb,
                'memory_final_mb': benchmark.memory_final_mb,
                'cpu_usage_percent': benchmark.cpu_usage_percent,
                'throughput_ops_per_sec': benchmark.throughput_ops_per_sec,
                'error_rate_percent': benchmark.error_rate_percent,
                'metadata': benchmark.metadata
            }
            
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
            
    def benchmark_function(self, func, *args, **kwargs) -> PerformanceBenchmark:
        """Benchmark a function execution."""
        # Start memory tracing
        tracemalloc.start()
        
        # Get initial metrics
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Start CPU monitoring in background
        cpu_usage = []
        stop_cpu_monitor = threading.Event()
        
        def monitor_cpu():
            while not stop_cpu_monitor.is_set():
                cpu_usage.append(psutil.cpu_percent())
                time.sleep(0.1)
                
        cpu_thread = threading.Thread(target=monitor_cpu)
        cpu_thread.start()
        
        # Execute function
        start_time = time.time()
        errors = 0
        operations = 0
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            operations = 1
        except Exception as e:
            errors = 1
            result = None
            print(f"Error in benchmark: {e}")
            
        end_time = time.time()
        
        # Stop CPU monitoring
        stop_cpu_monitor.set()
        cpu_thread.join()
        
        # Get final metrics
        final_memory = process.memory_info().rss / 1024 / 1024
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        duration_ms = (end_time - start_time) * 1000
        avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
        throughput = operations / (end_time - start_time) if (end_time - start_time) > 0 else 0
        error_rate = (errors / max(operations, 1)) * 100
        
        return PerformanceBenchmark(
            test_name=func.__name__,
            duration_ms=duration_ms,
            memory_peak_mb=peak_memory / 1024 / 1024,
            memory_final_mb=final_memory,
            cpu_usage_percent=avg_cpu,
            throughput_ops_per_sec=throughput,
            error_rate_percent=error_rate,
            metadata={
                'initial_memory_mb': initial_memory,
                'operations': operations,
                'errors': errors
            }
        )
        
    def test_memory_allocation_performance(self) -> PerformanceBenchmark:
        """Test memory allocation and deallocation performance."""
        def memory_test():
            # Allocate large amounts of memory
            data_structures = []
            
            for i in range(1000):
                # Create various data structures
                data_structures.append({
                    'list': [j for j in range(100)],
                    'dict': {f'key_{j}': f'value_{j}' for j in range(50)},
                    'string': f'test_string_{i}' * 10,
                    'nested': {'level1': {'level2': [i, i+1, i+2]}}
                })
                
            # Process data  
            processed = []
            for item in data_structures:
                processed.append(len(item['list']) + len(item['dict']))
                
            # Clean up
            data_structures.clear()
            processed.clear()
            
            # Force garbage collection
            gc.collect()
            
            return len(data_structures)
            
        return self.benchmark_function(memory_test)
        
    def test_api_simulation_performance(self) -> PerformanceBenchmark:
        """Test simulated API performance."""
        async def api_test():
            results = []
            
            # Simulate 100 API calls
            async def simulate_api_call(call_id: int):
                await asyncio.sleep(0.001)  # 1ms simulated processing
                return {'call_id': call_id, 'result': f'success_{call_id}'}
                
            # Process calls in batches to simulate real load
            batch_size = 10
            for batch_start in range(0, 100, batch_size):
                batch_tasks = []
                for i in range(batch_start, min(batch_start + batch_size, 100)):
                    batch_tasks.append(simulate_api_call(i))
                    
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
                
            return len(results)
            
        benchmark = self.benchmark_function(api_test)
        benchmark.test_name = "api_simulation"
        return benchmark
        
    def test_file_processing_performance(self) -> PerformanceBenchmark:
        """Test file processing scalability."""
        def file_processing_test():
            # Simulate processing many files
            file_data = []
            
            for i in range(5000):  # 5000 simulated files
                file_content = f"File {i} content: " + "data " * 50
                
                # Simulate file processing
                processed_content = file_content.upper().replace("DATA", "processed")
                
                # Store metadata instead of full content to save memory
                file_data.append({
                    'file_id': i,
                    'size': len(processed_content),
                    'checksum': hash(processed_content) % 10000
                })
                
                # Periodic cleanup to avoid memory buildup
                if i % 1000 == 0:
                    # Keep only recent files in memory
                    if len(file_data) > 500:
                        file_data = file_data[-500:]
                        
            return len(file_data)
            
        benchmark = self.benchmark_function(file_processing_test)
        benchmark.test_name = "file_processing"
        return benchmark
        
    def test_concurrent_processing_performance(self) -> PerformanceBenchmark:
        """Test concurrent processing performance."""
        def concurrent_test():
            results = []
            
            def worker_task(task_id: int):
                # Simulate CPU-intensive work
                total = 0
                for i in range(1000):
                    total += i * task_id
                return total
                
            # Use ThreadPoolExecutor for CPU-bound tasks
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker_task, i) for i in range(50)]
                
                for future in as_completed(futures):
                    results.append(future.result())
                    
            return len(results)
            
        benchmark = self.benchmark_function(concurrent_test)
        benchmark.test_name = "concurrent_processing" 
        return benchmark
        
    def test_memory_leak_detection(self) -> PerformanceBenchmark:
        """Test for memory leaks."""
        def memory_leak_test():
            # Track object creation and cleanup
            test_objects = []
            
            for iteration in range(10):
                # Create objects
                iteration_objects = []
                for i in range(1000):
                    obj = {
                        'id': i + (iteration * 1000),
                        'data': [j for j in range(10)],
                        'metadata': f'iteration_{iteration}_object_{i}'
                    }
                    iteration_objects.append(obj)
                    
                # Add to global list
                test_objects.extend(iteration_objects)
                
                # Clean up older objects
                if len(test_objects) > 5000:
                    test_objects = test_objects[-2000:]  # Keep only recent objects
                    
                # Force garbage collection
                if iteration % 3 == 0:
                    gc.collect()
                    
            # Final cleanup
            final_count = len(test_objects)
            test_objects.clear()
            gc.collect()
            
            return final_count
            
        benchmark = self.benchmark_function(memory_leak_test)
        benchmark.test_name = "memory_leak_detection"
        return benchmark
        
    def compare_with_baseline(self, current_benchmark: PerformanceBenchmark) -> RegressionTestResult:
        """Compare current benchmark with baseline."""
        baseline = self.baseline_data.get(current_benchmark.test_name)
        
        if not baseline:
            return RegressionTestResult(
                test_name=current_benchmark.test_name,
                current_benchmark=current_benchmark,
                status="UNKNOWN"
            )
            
        # Calculate performance changes
        performance_changes = {}
        
        # Duration change
        duration_change = ((current_benchmark.duration_ms - baseline.duration_ms) / baseline.duration_ms) * 100
        performance_changes['duration'] = duration_change
        
        # Memory change
        memory_change = ((current_benchmark.memory_peak_mb - baseline.memory_peak_mb) / baseline.memory_peak_mb) * 100
        performance_changes['memory'] = memory_change
        
        # Throughput change
        if baseline.throughput_ops_per_sec and current_benchmark.throughput_ops_per_sec:
            throughput_change = ((current_benchmark.throughput_ops_per_sec - baseline.throughput_ops_per_sec) / baseline.throughput_ops_per_sec) * 100
            performance_changes['throughput'] = throughput_change
        
        # Determine if regression occurred
        regression_detected = (
            duration_change > self.DURATION_REGRESSION_THRESHOLD or
            memory_change > self.MEMORY_REGRESSION_THRESHOLD or
            (performance_changes.get('throughput', 0) < -self.THROUGHPUT_REGRESSION_THRESHOLD)
        )
        
        # Determine status
        if regression_detected:
            status = "REGRESSION"
        elif (duration_change < -5 or memory_change < -5 or performance_changes.get('throughput', 0) > 5):
            status = "IMPROVEMENT"
        else:
            status = "PASS"
            
        return RegressionTestResult(
            test_name=current_benchmark.test_name,
            current_benchmark=current_benchmark,
            baseline_benchmark=baseline,
            regression_detected=regression_detected,
            performance_change_percent=performance_changes,
            status=status
        )
        
    def run_all_regression_tests(self) -> List[RegressionTestResult]:
        """Run all regression tests and compare with baseline."""
        print("🚀 Running Performance Regression Tests...")
        
        # Define all test functions
        test_functions = [
            self.test_memory_allocation_performance,
            self.test_api_simulation_performance,
            self.test_file_processing_performance,
            self.test_concurrent_processing_performance,
            self.test_memory_leak_detection
        ]
        
        results = []
        current_benchmarks = {}
        
        for test_func in test_functions:
            print(f"  Running {test_func.__name__}...")
            benchmark = test_func()
            current_benchmarks[benchmark.test_name] = benchmark
            
            # Compare with baseline
            regression_result = self.compare_with_baseline(benchmark)
            results.append(regression_result)
            
            # Print immediate result
            status_icon = "✅" if regression_result.status == "PASS" else "⚠️" if regression_result.status == "IMPROVEMENT" else "❌"
            print(f"    {status_icon} {regression_result.test_name}: {regression_result.status}")
            
        # Update baseline if no regressions
        regressions = [r for r in results if r.status == "REGRESSION"]
        if not regressions:
            print("  📊 Updating baseline with current benchmarks...")
            self.save_baseline(current_benchmarks)
            
        return results
        
    def generate_regression_report(self, results: List[RegressionTestResult]) -> Dict[str, Any]:
        """Generate comprehensive regression test report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': len(results),
                'regressions': len([r for r in results if r.status == "REGRESSION"]),
                'improvements': len([r for r in results if r.status == "IMPROVEMENT"]),
                'passes': len([r for r in results if r.status == "PASS"]),
                'unknowns': len([r for r in results if r.status == "UNKNOWN"])
            },
            'results': []
        }
        
        for result in results:
            result_data = {
                'test_name': result.test_name,
                'status': result.status,
                'regression_detected': result.regression_detected,
                'current_metrics': {
                    'duration_ms': result.current_benchmark.duration_ms,
                    'memory_peak_mb': result.current_benchmark.memory_peak_mb,
                    'memory_final_mb': result.current_benchmark.memory_final_mb,
                    'cpu_usage_percent': result.current_benchmark.cpu_usage_percent,
                    'throughput_ops_per_sec': result.current_benchmark.throughput_ops_per_sec,
                    'error_rate_percent': result.current_benchmark.error_rate_percent
                },
                'performance_changes': result.performance_change_percent
            }
            
            if result.baseline_benchmark:
                result_data['baseline_metrics'] = {
                    'duration_ms': result.baseline_benchmark.duration_ms,
                    'memory_peak_mb': result.baseline_benchmark.memory_peak_mb,
                    'memory_final_mb': result.baseline_benchmark.memory_final_mb,
                    'cpu_usage_percent': result.baseline_benchmark.cpu_usage_percent,
                    'throughput_ops_per_sec': result.baseline_benchmark.throughput_ops_per_sec,
                    'error_rate_percent': result.baseline_benchmark.error_rate_percent
                }
                
            report['results'].append(result_data)
            
        return report


def main():
    """Main execution function."""
    tester = PerformanceRegressionTester()
    results = tester.run_all_regression_tests()
    report = tester.generate_regression_report(results)
    
    # Save report
    report_file = Path(__file__).parent / f"regression_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    # Print summary
    print(f"\n{'='*60}")
    print("🔍 PERFORMANCE REGRESSION TEST REPORT")
    print(f"{'='*60}")
    print(f"Tests Run: {report['summary']['total_tests']}")
    print(f"Regressions: {report['summary']['regressions']} ❌")
    print(f"Improvements: {report['summary']['improvements']} ⚠️")
    print(f"Passes: {report['summary']['passes']} ✅")
    print(f"Unknown: {report['summary']['unknowns']} ❓")
    
    # Detailed results
    for result in results:
        if result.status == "REGRESSION":
            print(f"\n❌ REGRESSION DETECTED: {result.test_name}")
            for metric, change in result.performance_change_percent.items():
                print(f"  {metric}: {change:+.1f}%")
        elif result.status == "IMPROVEMENT":
            print(f"\n⚠️ IMPROVEMENT: {result.test_name}")
            for metric, change in result.performance_change_percent.items():
                print(f"  {metric}: {change:+.1f}%")
                
    print(f"\n📁 Full report saved to: {report_file}")
    print(f"{'='*60}")
    
    return len([r for r in results if r.status == "REGRESSION"]) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)