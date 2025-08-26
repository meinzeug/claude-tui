#!/usr/bin/env python3
"""
Performance Benchmark Suite - Comprehensive Performance Testing

Provides comprehensive benchmarking capabilities:
- API endpoint response time benchmarks
- Database query performance testing
- Memory usage profiling
- CPU utilization monitoring
- Throughput and concurrent user simulation
- Load testing with realistic scenarios
- Performance regression detection
"""

import asyncio
import time
import statistics
import gc
import psutil
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import aiohttp
import random
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark test result"""
    test_name: str
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    test_suite: str
    start_time: datetime
    end_time: datetime
    total_tests: int
    successful_tests: int
    failed_tests: int
    results: List[BenchmarkResult] = field(default_factory=list)
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        return self.successful_tests / self.total_tests if self.total_tests > 0 else 0.0
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


class SystemMetricsCollector:
    """Collects system performance metrics during benchmarks"""
    
    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.collecting = False
        self.metrics_history = []
        self.collection_thread = None
        
    def start_collection(self):
        """Start collecting system metrics"""
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.start()
        
    def stop_collection(self):
        """Stop collecting system metrics"""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join()
    
    def _collect_metrics(self):
        """Collect system metrics in background thread"""
        process = psutil.Process(os.getpid())
        
        while self.collecting:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_count = psutil.cpu_count()
                
                # Memory metrics
                memory_info = process.memory_info()
                system_memory = psutil.virtual_memory()
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                
                # Network I/O (if available)
                try:
                    network_io = psutil.net_io_counters()
                except:
                    network_io = None
                
                metrics = {
                    'timestamp': time.time(),
                    'cpu': {
                        'percent': cpu_percent,
                        'count': cpu_count
                    },
                    'memory': {
                        'rss_mb': memory_info.rss / 1024 / 1024,
                        'vms_mb': memory_info.vms / 1024 / 1024,
                        'system_percent': system_memory.percent,
                        'system_available_mb': system_memory.available / 1024 / 1024
                    },
                    'disk_io': {
                        'read_bytes': disk_io.read_bytes if disk_io else 0,
                        'write_bytes': disk_io.write_bytes if disk_io else 0
                    } if disk_io else {},
                    'network_io': {
                        'bytes_sent': network_io.bytes_sent,
                        'bytes_recv': network_io.bytes_recv
                    } if network_io else {},
                    'gc_objects': len(gc.get_objects())
                }
                
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                    
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(self.interval)
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        if not self.metrics_history:
            return {}
        
        # Extract time series data
        cpu_values = [m['cpu']['percent'] for m in self.metrics_history]
        memory_values = [m['memory']['rss_mb'] for m in self.metrics_history]
        
        return {
            'duration_seconds': len(self.metrics_history) * self.interval,
            'samples_collected': len(self.metrics_history),
            'cpu': {
                'avg_percent': statistics.mean(cpu_values),
                'max_percent': max(cpu_values),
                'min_percent': min(cpu_values)
            },
            'memory': {
                'avg_rss_mb': statistics.mean(memory_values),
                'max_rss_mb': max(memory_values),
                'min_rss_mb': min(memory_values)
            },
            'first_sample': self.metrics_history[0],
            'last_sample': self.metrics_history[-1]
        }


class APIBenchmarkSuite:
    """Benchmark suite for API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.results = []
        
    @asynccontextmanager
    async def get_session(self):
        """Get aiohttp session for benchmarks"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=100)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        
        try:
            yield self.session
        finally:
            # Keep session open for reuse
            pass
    
    async def benchmark_endpoint(
        self, 
        endpoint: str, 
        method: str = 'GET',
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        expected_status: int = 200
    ) -> BenchmarkResult:
        """Benchmark a single API endpoint"""
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                async with session.request(method, url, json=data, headers=headers) as response:
                    response_data = await response.text()
                    duration_ms = (time.time() - start_time) * 1000
                    
                    success = response.status == expected_status
                    error_message = None if success else f"Expected {expected_status}, got {response.status}"
                    
                    result = BenchmarkResult(
                        test_name=f"{method} {endpoint}",
                        duration_ms=duration_ms,
                        success=success,
                        error_message=error_message,
                        metrics={
                            'status_code': response.status,
                            'response_size_bytes': len(response_data),
                            'endpoint': endpoint,
                            'method': method
                        }
                    )
                    
                    self.results.append(result)
                    return result
                    
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = BenchmarkResult(
                test_name=f"{method} {endpoint}",
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metrics={'endpoint': endpoint, 'method': method}
            )
            self.results.append(result)
            return result
    
    async def load_test_endpoint(
        self,
        endpoint: str,
        concurrent_users: int = 10,
        requests_per_user: int = 100,
        ramp_up_seconds: float = 10.0
    ) -> List[BenchmarkResult]:
        """Load test an endpoint with concurrent users"""
        
        async def user_session(user_id: int, delay: float):
            """Simulate a single user session"""
            await asyncio.sleep(delay)  # Ramp-up delay
            
            user_results = []
            for request_num in range(requests_per_user):
                result = await self.benchmark_endpoint(endpoint)
                result.metrics['user_id'] = user_id
                result.metrics['request_num'] = request_num
                user_results.append(result)
                
                # Small delay between requests
                await asyncio.sleep(random.uniform(0.1, 0.5))
            
            return user_results
        
        # Calculate ramp-up delays
        ramp_up_delay = ramp_up_seconds / concurrent_users if concurrent_users > 1 else 0
        
        # Start all user sessions
        tasks = []
        for user_id in range(concurrent_users):
            delay = user_id * ramp_up_delay
            task = asyncio.create_task(user_session(user_id, delay))
            tasks.append(task)
        
        # Wait for all sessions to complete
        all_results = []
        for task in as_completed(tasks):
            user_results = await task
            all_results.extend(user_results)
        
        return all_results
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()


class DatabaseBenchmarkSuite:
    """Benchmark suite for database operations"""
    
    def __init__(self):
        self.results = []
    
    async def benchmark_query(
        self, 
        name: str, 
        query_func: Callable,
        iterations: int = 100
    ) -> List[BenchmarkResult]:
        """Benchmark a database query function"""
        
        results = []
        for i in range(iterations):
            start_time = time.time()
            
            try:
                result = await query_func() if asyncio.iscoroutinefunction(query_func) else query_func()
                duration_ms = (time.time() - start_time) * 1000
                
                benchmark_result = BenchmarkResult(
                    test_name=f"{name}_iteration_{i}",
                    duration_ms=duration_ms,
                    success=True,
                    metrics={
                        'iteration': i,
                        'result_size': len(result) if hasattr(result, '__len__') else 1
                    }
                )
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                benchmark_result = BenchmarkResult(
                    test_name=f"{name}_iteration_{i}",
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e),
                    metrics={'iteration': i}
                )
            
            results.append(benchmark_result)
            self.results.append(benchmark_result)
        
        return results
    
    async def benchmark_crud_operations(
        self,
        create_func: Callable,
        read_func: Callable,
        update_func: Callable,
        delete_func: Callable,
        iterations: int = 50
    ) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark CRUD operations"""
        
        crud_results = {
            'create': [],
            'read': [],
            'update': [],
            'delete': []
        }
        
        # Test CRUD operations
        for i in range(iterations):
            # Create
            create_result = await self._benchmark_single_operation(
                f"create_iteration_{i}", create_func, {'iteration': i}
            )
            crud_results['create'].append(create_result)
            
            if create_result.success:
                # Read
                read_result = await self._benchmark_single_operation(
                    f"read_iteration_{i}", read_func, {'iteration': i}
                )
                crud_results['read'].append(read_result)
                
                # Update
                update_result = await self._benchmark_single_operation(
                    f"update_iteration_{i}", update_func, {'iteration': i}
                )
                crud_results['update'].append(update_result)
                
                # Delete
                delete_result = await self._benchmark_single_operation(
                    f"delete_iteration_{i}", delete_func, {'iteration': i}
                )
                crud_results['delete'].append(delete_result)
        
        return crud_results
    
    async def _benchmark_single_operation(
        self, 
        name: str, 
        func: Callable, 
        metrics: Dict[str, Any]
    ) -> BenchmarkResult:
        """Benchmark a single database operation"""
        start_time = time.time()
        
        try:
            result = await func() if asyncio.iscoroutinefunction(func) else func()
            duration_ms = (time.time() - start_time) * 1000
            
            return BenchmarkResult(
                test_name=name,
                duration_ms=duration_ms,
                success=True,
                metrics=metrics
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return BenchmarkResult(
                test_name=name,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metrics=metrics
            )


class PerformanceBenchmarkRunner:
    """Main benchmark runner orchestrating all tests"""
    
    def __init__(self):
        self.system_metrics = SystemMetricsCollector()
        self.api_benchmarks = APIBenchmarkSuite()
        self.db_benchmarks = DatabaseBenchmarkSuite()
        self.results = []
        
    async def run_comprehensive_benchmark(
        self,
        api_endpoints: List[Tuple[str, str]] = None,  # (endpoint, method)
        database_queries: Dict[str, Callable] = None,
        load_test_config: Dict[str, Any] = None
    ) -> PerformanceReport:
        """Run comprehensive performance benchmark"""
        
        start_time = datetime.utcnow()
        
        # Start system metrics collection
        self.system_metrics.start_collection()
        
        try:
            all_results = []
            
            # API Endpoint Benchmarks
            if api_endpoints:
                logger.info("Running API endpoint benchmarks...")
                for endpoint, method in api_endpoints:
                    result = await self.api_benchmarks.benchmark_endpoint(endpoint, method)
                    all_results.append(result)
                    logger.info(f"  {method} {endpoint}: {result.duration_ms:.1f}ms")
            
            # Load Testing
            if load_test_config:
                logger.info("Running load tests...")
                load_results = await self.api_benchmarks.load_test_endpoint(
                    load_test_config.get('endpoint', '/'),
                    load_test_config.get('concurrent_users', 10),
                    load_test_config.get('requests_per_user', 100)
                )
                all_results.extend(load_results)
                logger.info(f"  Load test completed: {len(load_results)} requests")
            
            # Database Benchmarks
            if database_queries:
                logger.info("Running database benchmarks...")
                for name, query_func in database_queries.items():
                    db_results = await self.db_benchmarks.benchmark_query(name, query_func, 50)
                    all_results.extend(db_results)
                    avg_time = statistics.mean(r.duration_ms for r in db_results)
                    logger.info(f"  {name}: {avg_time:.1f}ms average")
            
        finally:
            # Stop system metrics collection
            self.system_metrics.stop_collection()
            await self.api_benchmarks.close()
        
        end_time = datetime.utcnow()
        
        # Create comprehensive report
        successful_tests = sum(1 for r in all_results if r.success)
        failed_tests = len(all_results) - successful_tests
        
        report = PerformanceReport(
            test_suite="Comprehensive Performance Benchmark",
            start_time=start_time,
            end_time=end_time,
            total_tests=len(all_results),
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            results=all_results,
            system_metrics=self.system_metrics.get_summary_metrics()
        )
        
        return report
    
    def analyze_performance_regression(
        self, 
        current_report: PerformanceReport,
        baseline_report: PerformanceReport,
        regression_threshold: float = 0.2  # 20% slowdown threshold
    ) -> Dict[str, Any]:
        """Analyze performance regression between two reports"""
        
        regressions = []
        improvements = []
        
        # Group results by test name
        current_results = {r.test_name: r for r in current_report.results}
        baseline_results = {r.test_name: r for r in baseline_report.results}
        
        for test_name in current_results:
            if test_name in baseline_results:
                current_time = current_results[test_name].duration_ms
                baseline_time = baseline_results[test_name].duration_ms
                
                if baseline_time > 0:
                    change_ratio = (current_time - baseline_time) / baseline_time
                    
                    if change_ratio > regression_threshold:
                        regressions.append({
                            'test_name': test_name,
                            'current_ms': current_time,
                            'baseline_ms': baseline_time,
                            'regression_percent': change_ratio * 100,
                            'severity': 'critical' if change_ratio > 0.5 else 'warning'
                        })
                    elif change_ratio < -0.1:  # 10% improvement
                        improvements.append({
                            'test_name': test_name,
                            'current_ms': current_time,
                            'baseline_ms': baseline_time,
                            'improvement_percent': abs(change_ratio) * 100
                        })
        
        return {
            'regressions': regressions,
            'improvements': improvements,
            'regression_count': len(regressions),
            'improvement_count': len(improvements),
            'overall_status': 'REGRESSION_DETECTED' if regressions else 'PERFORMANCE_STABLE',
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def generate_performance_report_html(self, report: PerformanceReport) -> str:
        """Generate HTML performance report"""
        
        # Calculate statistics
        if report.results:
            response_times = [r.duration_ms for r in report.results if r.success]
            if response_times:
                avg_response_time = statistics.mean(response_times)
                p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
                max_response_time = max(response_times)
            else:
                avg_response_time = p95_response_time = max_response_time = 0
        else:
            avg_response_time = p95_response_time = max_response_time = 0
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Benchmark Report</h1>
        <p><strong>Test Suite:</strong> {report.test_suite}</p>
        <p><strong>Duration:</strong> {report.duration_seconds:.2f} seconds</p>
        <p><strong>Test Period:</strong> {report.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {report.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Summary Metrics</h2>
    <div class="metric">
        <strong>Total Tests:</strong> {report.total_tests}
    </div>
    <div class="metric success">
        <strong>Successful:</strong> {report.successful_tests}
    </div>
    <div class="metric error">
        <strong>Failed:</strong> {report.failed_tests}
    </div>
    <div class="metric">
        <strong>Success Rate:</strong> {report.success_rate:.2%}
    </div>
    <div class="metric">
        <strong>Avg Response Time:</strong> {avg_response_time:.1f}ms
    </div>
    <div class="metric">
        <strong>P95 Response Time:</strong> {p95_response_time:.1f}ms
    </div>
    <div class="metric">
        <strong>Max Response Time:</strong> {max_response_time:.1f}ms
    </div>
    
    <h2>System Metrics During Test</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Average</th>
            <th>Maximum</th>
            <th>Minimum</th>
        </tr>
        <tr>
            <td>CPU Usage (%)</td>
            <td>{report.system_metrics.get('cpu', {}).get('avg_percent', 0):.1f}</td>
            <td>{report.system_metrics.get('cpu', {}).get('max_percent', 0):.1f}</td>
            <td>{report.system_metrics.get('cpu', {}).get('min_percent', 0):.1f}</td>
        </tr>
        <tr>
            <td>Memory Usage (MB)</td>
            <td>{report.system_metrics.get('memory', {}).get('avg_rss_mb', 0):.1f}</td>
            <td>{report.system_metrics.get('memory', {}).get('max_rss_mb', 0):.1f}</td>
            <td>{report.system_metrics.get('memory', {}).get('min_rss_mb', 0):.1f}</td>
        </tr>
    </table>
    
    <h2>Individual Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Duration (ms)</th>
            <th>Status</th>
            <th>Error Message</th>
        </tr>
        """
        
        for result in report.results[:100]:  # Show first 100 results
            status_class = "success" if result.success else "error"
            error_msg = result.error_message or ""
            html += f"""
        <tr>
            <td>{result.test_name}</td>
            <td>{result.duration_ms:.1f}</td>
            <td class="{status_class}">{'✓ Success' if result.success else '✗ Failed'}</td>
            <td>{error_msg}</td>
        </tr>
            """
        
        html += """
    </table>
    
    <h2>Performance Recommendations</h2>
    <ul>
        """
        
        # Add recommendations based on results
        if avg_response_time > 1000:
            html += "<li>Average response time is over 1 second. Consider optimizing slow endpoints.</li>"
        if p95_response_time > 2000:
            html += "<li>95th percentile response time is over 2 seconds. Some requests are very slow.</li>"
        if report.success_rate < 0.95:
            html += "<li>Success rate is below 95%. Investigate failing requests.</li>"
        
        html += """
    </ul>
    
    <footer>
        <p><em>Report generated on {timestamp}</em></p>
    </footer>
</body>
</html>
        """.format(timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        
        return html


if __name__ == "__main__":
    # Example usage and testing
    async def test_benchmark_suite():
        print("🚀 PERFORMANCE BENCHMARK SUITE - Testing")
        print("=" * 60)
        
        runner = PerformanceBenchmarkRunner()
        
        # Define test endpoints
        api_endpoints = [
            ('/', 'GET'),
            ('/health', 'GET'),
            ('/api/v1/status', 'GET')
        ]
        
        # Define database test queries
        def sample_query():
            time.sleep(0.01)  # Simulate DB query
            return [{"id": i, "data": f"value_{i}"} for i in range(10)]
        
        database_queries = {
            'sample_select': sample_query
        }
        
        # Load test configuration
        load_test_config = {
            'endpoint': '/',
            'concurrent_users': 5,
            'requests_per_user': 20
        }
        
        print("📊 Running comprehensive benchmark...")
        report = await runner.run_comprehensive_benchmark(
            api_endpoints=api_endpoints,
            database_queries=database_queries,
            load_test_config=load_test_config
        )
        
        print(f"\n📈 Benchmark Results:")
        print(f"   Total Tests: {report.total_tests}")
        print(f"   Success Rate: {report.success_rate:.2%}")
        print(f"   Duration: {report.duration_seconds:.2f} seconds")
        print(f"   System CPU Avg: {report.system_metrics.get('cpu', {}).get('avg_percent', 0):.1f}%")
        print(f"   System Memory Avg: {report.system_metrics.get('memory', {}).get('avg_rss_mb', 0):.1f}MB")
        
        # Generate HTML report
        html_report = runner.generate_performance_report_html(report)
        
        # Save report to file
        with open('performance_benchmark_report.html', 'w') as f:
            f.write(html_report)
        
        print(f"\n✅ HTML report saved to: performance_benchmark_report.html")
        print("✅ Performance benchmark suite test completed!")
    
    # Run test
    asyncio.run(test_benchmark_suite())