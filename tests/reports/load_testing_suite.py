#!/usr/bin/env python3
"""
Comprehensive Load Testing Suite
High-performance load testing for Claude-TIU system validation
"""

import asyncio
import time
import json
import psutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import aiohttp
import queue
import statistics
import gc


@dataclass
class LoadTestConfig:
    """Load test configuration."""
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int = 0
    target_rps: Optional[int] = None  # Requests per second
    max_response_time_ms: int = 5000
    success_threshold_percent: float = 95.0
    memory_limit_mb: Optional[int] = None


@dataclass
class LoadTestMetrics:
    """Load test execution metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    requests_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    response_times: List[float] = field(default_factory=list)


@dataclass
class LoadTestResult:
    """Complete load test result."""
    config: LoadTestConfig
    metrics: LoadTestMetrics
    success: bool
    start_time: datetime
    end_time: datetime
    system_resources: Dict[str, Any] = field(default_factory=dict)


class SystemResourceMonitor:
    """Monitor system resources during load testing."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start monitoring system resources."""
        self.monitoring = True
        self.metrics = []
        
        def monitor():
            while self.monitoring:
                try:
                    cpu = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    process = psutil.Process()
                    
                    self.metrics.append({
                        'timestamp': time.time(),
                        'cpu_percent': cpu,
                        'memory_percent': memory.percent,
                        'memory_used_mb': memory.used / 1024 / 1024,
                        'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                        'process_cpu_percent': process.cpu_percent()
                    })
                    
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        if not self.metrics:
            return {}
            
        return {
            'peak_cpu_percent': max(m['cpu_percent'] for m in self.metrics),
            'avg_cpu_percent': statistics.mean(m['cpu_percent'] for m in self.metrics),
            'peak_memory_mb': max(m['process_memory_mb'] for m in self.metrics),
            'avg_memory_mb': statistics.mean(m['process_memory_mb'] for m in self.metrics),
            'memory_growth_mb': self.metrics[-1]['process_memory_mb'] - self.metrics[0]['process_memory_mb'],
            'duration_seconds': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp']
        }


class LoadTestRunner:
    """High-performance load test execution engine."""
    
    def __init__(self):
        self.resource_monitor = SystemResourceMonitor()
        
    async def simulate_api_request(self, session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
        """Simulate an API request."""
        start_time = time.time()
        
        try:
            # Simulate API call processing
            await asyncio.sleep(0.001)  # 1ms minimum processing time
            
            # Add some variability
            if hash(endpoint) % 10 == 0:  # 10% of requests are slower
                await asyncio.sleep(0.01)  # 10ms additional delay
                
            duration = time.time() - start_time
            
            return {
                'success': True,
                'duration': duration,
                'status_code': 200,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'status_code': 500,
                'error': str(e)
            }
            
    async def simulate_validation_request(self, content: str, complexity: int = 1) -> Dict[str, Any]:
        """Simulate validation processing."""
        start_time = time.time()
        
        try:
            # Simulate validation processing time based on content complexity
            base_time = 0.002  # 2ms base processing
            processing_time = base_time * complexity * (1 + len(content) / 1000)
            
            await asyncio.sleep(processing_time)
            
            # Simulate validation result
            is_valid = len(content) > 10 and not any(word in content.lower() for word in ['error', 'failed'])
            
            duration = time.time() - start_time
            
            return {
                'success': True,
                'duration': duration,
                'is_valid': is_valid,
                'authenticity_score': 0.9 if is_valid else 0.6,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'is_valid': False,
                'error': str(e)
            }
            
    def simulate_file_processing(self, file_count: int, file_size_kb: int = 10) -> Dict[str, Any]:
        """Simulate file processing operation."""
        start_time = time.time()
        
        try:
            processed_files = 0
            
            for i in range(file_count):
                # Simulate file content
                file_content = f"File {i} content: " + "x" * (file_size_kb * 1024)
                
                # Simulate processing
                processed_content = file_content.upper()
                checksum = hash(processed_content) % 10000
                
                # Simulate memory management
                if i % 100 == 0:
                    gc.collect()
                    
                processed_files += 1
                
            duration = time.time() - start_time
            
            return {
                'success': True,
                'duration': duration,
                'processed_files': processed_files,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'processed_files': 0,
                'error': str(e)
            }
            
    async def run_concurrent_api_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Run concurrent API load test."""
        metrics = LoadTestMetrics()
        start_time = datetime.now()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        async with aiohttp.ClientSession() as session:
            # Calculate request distribution
            total_requests = config.target_rps * config.duration_seconds if config.target_rps else config.concurrent_users * 10
            
            async def user_session(user_id: int):
                """Simulate single user session."""
                session_metrics = {
                    'requests': 0,
                    'successes': 0,
                    'failures': 0,
                    'response_times': []
                }
                
                session_duration = config.duration_seconds
                start_session = time.time()
                
                while (time.time() - start_session) < session_duration:
                    # Make request
                    result = await self.simulate_api_request(session, f"/api/test/{user_id}")
                    
                    session_metrics['requests'] += 1
                    session_metrics['response_times'].append(result['duration'])
                    
                    if result['success']:
                        session_metrics['successes'] += 1
                    else:
                        session_metrics['failures'] += 1
                        
                    # Rate limiting
                    if config.target_rps:
                        await asyncio.sleep(1.0 / (config.target_rps / config.concurrent_users))
                    else:
                        await asyncio.sleep(0.01)  # 100 RPS per user default
                        
                return session_metrics
                
            # Execute concurrent user sessions
            tasks = [user_session(i) for i in range(config.concurrent_users)]
            session_results = await asyncio.gather(*tasks)
            
        # Aggregate metrics
        for session_result in session_results:
            metrics.total_requests += session_result['requests']
            metrics.successful_requests += session_result['successes']
            metrics.failed_requests += session_result['failures']
            metrics.response_times.extend(session_result['response_times'])
            
        end_time = datetime.now()
        metrics.total_duration = (end_time - start_time).total_seconds()
        
        # Calculate response time statistics
        if metrics.response_times:
            metrics.min_response_time = min(metrics.response_times)
            metrics.max_response_time = max(metrics.response_times)
            metrics.avg_response_time = statistics.mean(metrics.response_times)
            metrics.p95_response_time = statistics.quantiles(metrics.response_times, n=20)[18]  # 95th percentile
            metrics.p99_response_time = statistics.quantiles(metrics.response_times, n=100)[98]  # 99th percentile
            
        metrics.requests_per_second = metrics.total_requests / metrics.total_duration if metrics.total_duration > 0 else 0
        
        # Get resource metrics
        resource_metrics = self.resource_monitor.stop_monitoring()
        metrics.peak_memory_mb = resource_metrics.get('peak_memory_mb', 0)
        metrics.peak_cpu_percent = resource_metrics.get('peak_cpu_percent', 0)
        
        # Determine success
        success_rate = (metrics.successful_requests / metrics.total_requests) * 100 if metrics.total_requests > 0 else 0
        response_time_ok = metrics.avg_response_time * 1000 < config.max_response_time_ms
        success = success_rate >= config.success_threshold_percent and response_time_ok
        
        return LoadTestResult(
            config=config,
            metrics=metrics,
            success=success,
            start_time=start_time,
            end_time=end_time,
            system_resources=resource_metrics
        )
        
    async def run_validation_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Run validation-focused load test."""
        metrics = LoadTestMetrics()
        start_time = datetime.now()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Generate test content
        test_contents = [
            f"def function_{i}():\n    return {i} * 2\n"
            for i in range(100)
        ]
        
        async def validation_worker(worker_id: int):
            """Worker for validation tasks."""
            worker_metrics = {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'response_times': []
            }
            
            session_duration = config.duration_seconds
            start_session = time.time()
            
            while (time.time() - start_session) < session_duration:
                # Select test content
                content = test_contents[worker_id % len(test_contents)]
                complexity = (worker_id % 5) + 1  # Varying complexity
                
                # Run validation
                result = await self.simulate_validation_request(content, complexity)
                
                worker_metrics['requests'] += 1
                worker_metrics['response_times'].append(result['duration'])
                
                if result['success']:
                    worker_metrics['successes'] += 1
                else:
                    worker_metrics['failures'] += 1
                    
                # Control rate
                await asyncio.sleep(0.01)  # 100 validations per second per worker
                
            return worker_metrics
            
        # Execute validation workers
        tasks = [validation_worker(i) for i in range(config.concurrent_users)]
        worker_results = await asyncio.gather(*tasks)
        
        # Aggregate metrics
        for worker_result in worker_results:
            metrics.total_requests += worker_result['requests']
            metrics.successful_requests += worker_result['successes']
            metrics.failed_requests += worker_result['failures']
            metrics.response_times.extend(worker_result['response_times'])
            
        end_time = datetime.now()
        metrics.total_duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        if metrics.response_times:
            metrics.min_response_time = min(metrics.response_times)
            metrics.max_response_time = max(metrics.response_times)
            metrics.avg_response_time = statistics.mean(metrics.response_times)
            
        metrics.requests_per_second = metrics.total_requests / metrics.total_duration if metrics.total_duration > 0 else 0
        
        # Get resource metrics
        resource_metrics = self.resource_monitor.stop_monitoring()
        metrics.peak_memory_mb = resource_metrics.get('peak_memory_mb', 0)
        metrics.peak_cpu_percent = resource_metrics.get('peak_cpu_percent', 0)
        
        # Determine success
        success_rate = (metrics.successful_requests / metrics.total_requests) * 100 if metrics.total_requests > 0 else 0
        success = success_rate >= config.success_threshold_percent
        
        return LoadTestResult(
            config=config,
            metrics=metrics,
            success=success,
            start_time=start_time,
            end_time=end_time,
            system_resources=resource_metrics
        )
        
    def run_file_processing_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Run file processing load test."""
        metrics = LoadTestMetrics()
        start_time = datetime.now()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        def file_worker(worker_id: int):
            """Worker for file processing tasks."""
            worker_metrics = {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'response_times': []
            }
            
            session_duration = config.duration_seconds
            start_session = time.time()
            
            while (time.time() - start_session) < session_duration:
                # Process batch of files
                file_count = 50 + (worker_id % 50)  # 50-100 files per batch
                
                result = self.simulate_file_processing(file_count, file_size_kb=5)
                
                worker_metrics['requests'] += 1
                worker_metrics['response_times'].append(result['duration'])
                
                if result['success']:
                    worker_metrics['successes'] += 1
                else:
                    worker_metrics['failures'] += 1
                    
                # Brief pause between batches
                time.sleep(0.1)
                
            return worker_metrics
            
        # Execute file processing workers
        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = [executor.submit(file_worker, i) for i in range(config.concurrent_users)]
            worker_results = [future.result() for future in as_completed(futures)]
            
        # Aggregate metrics
        for worker_result in worker_results:
            metrics.total_requests += worker_result['requests']
            metrics.successful_requests += worker_result['successes']
            metrics.failed_requests += worker_result['failures']
            metrics.response_times.extend(worker_result['response_times'])
            
        end_time = datetime.now()
        metrics.total_duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        if metrics.response_times:
            metrics.min_response_time = min(metrics.response_times)
            metrics.max_response_time = max(metrics.response_times)
            metrics.avg_response_time = statistics.mean(metrics.response_times)
            
        metrics.requests_per_second = metrics.total_requests / metrics.total_duration if metrics.total_duration > 0 else 0
        
        # Get resource metrics
        resource_metrics = self.resource_monitor.stop_monitoring()
        metrics.peak_memory_mb = resource_metrics.get('peak_memory_mb', 0)
        metrics.peak_cpu_percent = resource_metrics.get('peak_cpu_percent', 0)
        
        # Determine success
        success_rate = (metrics.successful_requests / metrics.total_requests) * 100 if metrics.total_requests > 0 else 0
        memory_ok = not config.memory_limit_mb or metrics.peak_memory_mb <= config.memory_limit_mb
        success = success_rate >= config.success_threshold_percent and memory_ok
        
        return LoadTestResult(
            config=config,
            metrics=metrics,
            success=success,
            start_time=start_time,
            end_time=end_time,
            system_resources=resource_metrics
        )


class LoadTestSuite:
    """Comprehensive load testing suite."""
    
    def __init__(self):
        self.runner = LoadTestRunner()
        
    def get_test_configurations(self) -> List[LoadTestConfig]:
        """Get predefined test configurations."""
        return [
            # Light load test
            LoadTestConfig(
                name="light_api_load",
                concurrent_users=10,
                duration_seconds=30,
                target_rps=50,
                max_response_time_ms=200
            ),
            
            # Medium load test
            LoadTestConfig(
                name="medium_api_load",
                concurrent_users=25,
                duration_seconds=60,
                target_rps=100,
                max_response_time_ms=300
            ),
            
            # Heavy load test
            LoadTestConfig(
                name="heavy_api_load",
                concurrent_users=50,
                duration_seconds=120,
                target_rps=200,
                max_response_time_ms=500
            ),
            
            # Validation-focused test
            LoadTestConfig(
                name="validation_load",
                concurrent_users=20,
                duration_seconds=60,
                max_response_time_ms=1000,
                success_threshold_percent=98.0
            ),
            
            # File processing test
            LoadTestConfig(
                name="file_processing_load",
                concurrent_users=8,
                duration_seconds=90,
                memory_limit_mb=300,
                success_threshold_percent=95.0
            )
        ]
        
    async def run_all_tests(self) -> List[LoadTestResult]:
        """Run all load tests."""
        configs = self.get_test_configurations()
        results = []
        
        print("🚀 Starting Comprehensive Load Testing Suite...")
        
        for config in configs:
            print(f"  Running {config.name}...")
            
            if "api" in config.name:
                result = await self.runner.run_concurrent_api_load_test(config)
            elif "validation" in config.name:
                result = await self.runner.run_validation_load_test(config)
            elif "file" in config.name:
                result = self.runner.run_file_processing_load_test(config)
            else:
                continue
                
            results.append(result)
            
            # Print immediate result
            status_icon = "✅" if result.success else "❌"
            print(f"    {status_icon} {config.name}: {'PASS' if result.success else 'FAIL'}")
            print(f"      RPS: {result.metrics.requests_per_second:.1f}")
            print(f"      Avg Response: {result.metrics.avg_response_time*1000:.1f}ms")
            print(f"      Success Rate: {(result.metrics.successful_requests/result.metrics.total_requests*100):.1f}%")
            print(f"      Peak Memory: {result.metrics.peak_memory_mb:.1f}MB")
            
            # Brief pause between tests
            await asyncio.sleep(5)
            
        return results
        
    def generate_load_test_report(self, results: List[LoadTestResult]) -> Dict[str, Any]:
        """Generate comprehensive load test report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': len(results),
                'passed_tests': len([r for r in results if r.success]),
                'failed_tests': len([r for r in results if not r.success]),
                'total_requests': sum(r.metrics.total_requests for r in results),
                'total_duration': sum(r.metrics.total_duration for r in results),
                'avg_rps': statistics.mean([r.metrics.requests_per_second for r in results if r.metrics.requests_per_second > 0]),
                'peak_memory_mb': max(r.metrics.peak_memory_mb for r in results),
                'peak_cpu_percent': max(r.metrics.peak_cpu_percent for r in results)
            },
            'test_results': []
        }
        
        for result in results:
            test_data = {
                'config': {
                    'name': result.config.name,
                    'concurrent_users': result.config.concurrent_users,
                    'duration_seconds': result.config.duration_seconds,
                    'target_rps': result.config.target_rps,
                    'max_response_time_ms': result.config.max_response_time_ms
                },
                'metrics': {
                    'total_requests': result.metrics.total_requests,
                    'successful_requests': result.metrics.successful_requests,
                    'failed_requests': result.metrics.failed_requests,
                    'success_rate_percent': (result.metrics.successful_requests / result.metrics.total_requests * 100) if result.metrics.total_requests > 0 else 0,
                    'requests_per_second': result.metrics.requests_per_second,
                    'avg_response_time_ms': result.metrics.avg_response_time * 1000,
                    'min_response_time_ms': result.metrics.min_response_time * 1000,
                    'max_response_time_ms': result.metrics.max_response_time * 1000,
                    'p95_response_time_ms': result.metrics.p95_response_time * 1000 if result.metrics.p95_response_time else 0,
                    'p99_response_time_ms': result.metrics.p99_response_time * 1000 if result.metrics.p99_response_time else 0,
                    'peak_memory_mb': result.metrics.peak_memory_mb,
                    'peak_cpu_percent': result.metrics.peak_cpu_percent
                },
                'success': result.success,
                'duration': (result.end_time - result.start_time).total_seconds(),
                'system_resources': result.system_resources
            }
            
            report['test_results'].append(test_data)
            
        return report


async def main():
    """Main execution function."""
    suite = LoadTestSuite()
    results = await suite.run_all_tests()
    report = suite.generate_load_test_report(results)
    
    # Save report
    report_file = Path(__file__).parent / f"load_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    # Print summary
    print(f"\n{'='*60}")
    print("🔥 LOAD TESTING REPORT")
    print(f"{'='*60}")
    print(f"Tests Run: {report['summary']['total_tests']}")
    print(f"Tests Passed: {report['summary']['passed_tests']} ✅")
    print(f"Tests Failed: {report['summary']['failed_tests']} ❌")
    print(f"Total Requests: {report['summary']['total_requests']:,}")
    print(f"Average RPS: {report['summary']['avg_rps']:.1f}")
    print(f"Peak Memory: {report['summary']['peak_memory_mb']:.1f}MB")
    print(f"Peak CPU: {report['summary']['peak_cpu_percent']:.1f}%")
    
    # Detailed results
    for result in results:
        status_icon = "✅" if result.success else "❌"
        print(f"\n{status_icon} {result.config.name.upper()}:")
        print(f"  Users: {result.config.concurrent_users}")
        print(f"  Duration: {result.config.duration_seconds}s")
        print(f"  Requests: {result.metrics.total_requests:,}")
        print(f"  RPS: {result.metrics.requests_per_second:.1f}")
        print(f"  Success Rate: {(result.metrics.successful_requests/result.metrics.total_requests*100):.1f}%")
        print(f"  Avg Response: {result.metrics.avg_response_time*1000:.1f}ms")
        print(f"  Peak Memory: {result.metrics.peak_memory_mb:.1f}MB")
        
    print(f"\n📁 Full report saved to: {report_file}")
    print(f"{'='*60}")
    
    return len([r for r in results if not r.success]) == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)