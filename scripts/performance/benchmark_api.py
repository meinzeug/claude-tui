#!/usr/bin/env python3
"""
High-Performance API Benchmarking Suite.

Comprehensive benchmarking tool to measure and optimize API performance.
Targets <500ms response times with load testing capabilities.
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    endpoint: str
    method: str
    response_time: float
    status_code: int
    success: bool
    error: Optional[str] = None
    response_size: int = 0


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    total_duration: float
    results: List[BenchmarkResult] = field(default_factory=list)


class APIBenchmark:
    """
    High-performance API benchmarking tool.
    
    Features:
    - Concurrent request execution
    - Detailed performance metrics
    - Load testing capabilities
    - Response time analysis
    - Throughput measurement
    """
    
    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        timeout: int = 30,
        max_concurrent: int = 50
    ):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        
        # Headers
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if auth_token:
            self.headers['Authorization'] = f'Bearer {auth_token}'
    
    async def benchmark_endpoint(
        self,
        endpoint: str,
        method: str = 'GET',
        payload: Optional[Dict[str, Any]] = None,
        num_requests: int = 100,
        concurrent_requests: int = 10
    ) -> BenchmarkSuite:
        """
        Benchmark a single endpoint.
        
        Args:
            endpoint: API endpoint to benchmark
            method: HTTP method
            payload: Request payload for POST/PUT requests
            num_requests: Total number of requests
            concurrent_requests: Number of concurrent requests
            
        Returns:
            BenchmarkSuite with detailed results
        """
        logger.info(f"Benchmarking {method} {endpoint} - {num_requests} requests, {concurrent_requests} concurrent")
        
        results = []
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request(session: aiohttp.ClientSession, request_id: int) -> BenchmarkResult:
            """Make a single request."""
            async with semaphore:
                start_time = time.time()
                
                try:
                    url = f"{self.base_url}{endpoint}"
                    
                    if method.upper() == 'GET':
                        async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                            response_text = await response.text()
                            response_time = time.time() - start_time
                            
                            return BenchmarkResult(
                                endpoint=endpoint,
                                method=method,
                                response_time=response_time,
                                status_code=response.status,
                                success=response.status < 400,
                                response_size=len(response_text)
                            )
                    
                    elif method.upper() == 'POST':
                        async with session.post(
                            url, 
                            json=payload,
                            headers=self.headers,
                            timeout=self.timeout
                        ) as response:
                            response_text = await response.text()
                            response_time = time.time() - start_time
                            
                            return BenchmarkResult(
                                endpoint=endpoint,
                                method=method,
                                response_time=response_time,
                                status_code=response.status,
                                success=response.status < 400,
                                response_size=len(response_text)
                            )
                
                except asyncio.TimeoutError:
                    response_time = time.time() - start_time
                    return BenchmarkResult(
                        endpoint=endpoint,
                        method=method,
                        response_time=response_time,
                        status_code=408,
                        success=False,
                        error="Request timeout"
                    )
                
                except Exception as e:
                    response_time = time.time() - start_time
                    return BenchmarkResult(
                        endpoint=endpoint,
                        method=method,
                        response_time=response_time,
                        status_code=500,
                        success=False,
                        error=str(e)
                    )
        
        # Execute benchmark
        total_start_time = time.time()
        
        # Configure aiohttp session with performance optimizations
        connector = aiohttp.TCPConnector(
            limit=concurrent_requests * 2,
            limit_per_host=concurrent_requests,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout_config = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config
        ) as session:
            # Create tasks
            tasks = [
                make_request(session, i)
                for i in range(num_requests)
            ]
            
            # Execute with progress tracking
            completed = 0
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                completed += 1
                
                if completed % (num_requests // 10) == 0:
                    progress = (completed / num_requests) * 100
                    logger.info(f"Progress: {completed}/{num_requests} ({progress:.1f}%)")
        
        total_duration = time.time() - total_start_time
        
        # Calculate metrics
        successful_results = [r for r in results if r.success]
        successful_count = len(successful_results)
        failed_count = len(results) - successful_count
        
        if successful_results:
            response_times = [r.response_time for r in successful_results]
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0
            min_response_time = max_response_time = 0
        
        requests_per_second = num_requests / total_duration
        
        return BenchmarkSuite(
            total_requests=num_requests,
            successful_requests=successful_count,
            failed_requests=failed_count,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            total_duration=total_duration,
            results=results
        )
    
    async def run_comprehensive_benchmark(self) -> Dict[str, BenchmarkSuite]:
        """
        Run comprehensive benchmark suite covering all critical endpoints.
        """
        logger.info("Starting comprehensive API benchmark suite")
        
        # Define test scenarios
        test_scenarios = [
            {
                'name': 'health_check',
                'endpoint': '/health',
                'method': 'GET',
                'num_requests': 1000,
                'concurrent': 20
            },
            {
                'name': 'ai_code_generation',
                'endpoint': '/api/v1/ai/code/generate',
                'method': 'POST',
                'payload': {
                    'prompt': 'def fibonacci(n): pass',
                    'language': 'python',
                    'validate_response': True
                },
                'num_requests': 50,
                'concurrent': 5
            },
            {
                'name': 'ai_validation',
                'endpoint': '/api/v1/ai/validate',
                'method': 'POST',
                'payload': {
                    'response': {'code': 'print("hello world")'},
                    'validation_type': 'code'
                },
                'num_requests': 100,
                'concurrent': 10
            },
            {
                'name': 'performance_metrics',
                'endpoint': '/api/v1/ai/performance',
                'method': 'GET',
                'num_requests': 200,
                'concurrent': 15
            },
            {
                'name': 'batch_processing',
                'endpoint': '/api/v1/performance/batch/code/generate',
                'method': 'POST',
                'payload': {
                    'requests': [
                        {'prompt': 'def add(a, b): pass', 'language': 'python'},
                        {'prompt': 'def multiply(a, b): pass', 'language': 'python'}
                    ],
                    'parallel_execution': True
                },
                'num_requests': 20,
                'concurrent': 3
            }
        ]
        
        # Run benchmarks
        results = {}
        
        for scenario in test_scenarios:
            logger.info(f"Running scenario: {scenario['name']}")
            
            try:
                result = await self.benchmark_endpoint(
                    endpoint=scenario['endpoint'],
                    method=scenario['method'],
                    payload=scenario.get('payload'),
                    num_requests=scenario['num_requests'],
                    concurrent_requests=scenario['concurrent']
                )
                
                results[scenario['name']] = result
                
                # Log results
                logger.info(
                    f"Scenario {scenario['name']} completed: "
                    f"Avg: {result.avg_response_time*1000:.1f}ms, "
                    f"P95: {result.p95_response_time*1000:.1f}ms, "
                    f"Success: {result.successful_requests}/{result.total_requests} "
                    f"({result.successful_requests/result.total_requests*100:.1f}%)"
                )
                
            except Exception as e:
                logger.error(f"Scenario {scenario['name']} failed: {e}")
                continue
            
            # Brief pause between scenarios
            await asyncio.sleep(1)
        
        return results
    
    def generate_report(self, results: Dict[str, BenchmarkSuite]) -> str:
        """
        Generate comprehensive benchmark report.
        """
        report = []
        report.append("=" * 80)
        report.append("API PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Target: {self.base_url}")
        report.append("")
        
        # Summary table
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"{'Endpoint':<30} {'Avg (ms)':<10} {'P95 (ms)':<10} {'RPS':<8} {'Success%':<8}")
        report.append("-" * 40)
        
        for name, result in results.items():
            success_rate = (result.successful_requests / result.total_requests) * 100
            report.append(
                f"{name:<30} {result.avg_response_time*1000:<10.1f} "
                f"{result.p95_response_time*1000:<10.1f} {result.requests_per_second:<8.1f} "
                f"{success_rate:<8.1f}"
            )
        
        report.append("")
        
        # Detailed results
        for name, result in results.items():
            report.append(f"DETAILED RESULTS: {name.upper()}")
            report.append("-" * 40)
            report.append(f"Total Requests: {result.total_requests}")
            report.append(f"Successful: {result.successful_requests}")
            report.append(f"Failed: {result.failed_requests}")
            report.append(f"Success Rate: {(result.successful_requests/result.total_requests)*100:.2f}%")
            report.append(f"Total Duration: {result.total_duration:.2f}s")
            report.append(f"Requests/Second: {result.requests_per_second:.2f}")
            report.append("")
            report.append("Response Times (ms):")
            report.append(f"  Average: {result.avg_response_time*1000:.2f}")
            report.append(f"  Median (P50): {result.p50_response_time*1000:.2f}")
            report.append(f"  P95: {result.p95_response_time*1000:.2f}")
            report.append(f"  P99: {result.p99_response_time*1000:.2f}")
            report.append(f"  Min: {result.min_response_time*1000:.2f}")
            report.append(f"  Max: {result.max_response_time*1000:.2f}")
            report.append("")
            
            # Performance assessment
            if result.p95_response_time * 1000 < 500:
                report.append("✅ PERFORMANCE: EXCELLENT (P95 < 500ms)")
            elif result.p95_response_time * 1000 < 1000:
                report.append("⚠️  PERFORMANCE: GOOD (P95 < 1000ms)")
            elif result.p95_response_time * 1000 < 2000:
                report.append("⚠️  PERFORMANCE: ACCEPTABLE (P95 < 2000ms)")
            else:
                report.append("❌ PERFORMANCE: NEEDS OPTIMIZATION (P95 > 2000ms)")
            
            report.append("")
        
        # System information
        report.append("SYSTEM INFORMATION")
        report.append("-" * 40)
        report.append(f"CPU Cores: {psutil.cpu_count()}")
        report.append(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        report.append(f"Max Concurrent: {self.max_concurrent}")
        report.append("")
        
        return "\n".join(report)


async def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="API Performance Benchmark Tool")
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--token', help='Auth token')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout')
    parser.add_argument('--concurrent', type=int, default=50, help='Max concurrent requests')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = APIBenchmark(
        base_url=args.url,
        auth_token=args.token,
        timeout=args.timeout,
        max_concurrent=args.concurrent
    )
    
    # Run comprehensive benchmark
    logger.info("Starting comprehensive API benchmark...")
    start_time = time.time()
    
    try:
        results = await benchmark.run_comprehensive_benchmark()
        total_time = time.time() - start_time
        
        # Generate report
        report = benchmark.generate_report(results)
        
        # Output report
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {args.output}")
        else:
            print(report)
        
        logger.info(f"Benchmark completed in {total_time:.2f}s")
        
        # Check if performance targets are met
        failed_endpoints = []
        for name, result in results.items():
            if result.p95_response_time * 1000 > 1000:  # P95 > 1s
                failed_endpoints.append(name)
        
        if failed_endpoints:
            logger.warning(f"Performance targets not met for: {', '.join(failed_endpoints)}")
            exit(1)
        else:
            logger.info("✅ All endpoints meet performance targets!")
            exit(0)
    
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())