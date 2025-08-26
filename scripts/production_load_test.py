#!/usr/bin/env python3
"""
Production Load Testing Suite
Comprehensive load testing for Claude TUI production deployment
"""

import asyncio
import aiohttp
import time
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@dataclass
class LoadTestResult:
    """Results from a single load test request"""
    response_time: float
    status_code: int
    success: bool
    error: str = ""

@dataclass
class LoadTestSummary:
    """Summary of load test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    duration: float

class ProductionLoadTester:
    """
    Production-grade load testing suite for Claude TUI
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", max_workers: int = 100):
        self.base_url = base_url
        self.max_workers = max_workers
        self.session = None
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    async def _make_request(self, endpoint: str, method: str = "GET", 
                           payload: Dict = None) -> LoadTestResult:
        """Make a single HTTP request and measure performance"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        await response.read()  # Ensure full response is received
                        response_time = time.time() - start_time
                        return LoadTestResult(
                            response_time=response_time,
                            status_code=response.status,
                            success=response.status < 400
                        )
                elif method.upper() == "POST":
                    async with session.post(f"{self.base_url}{endpoint}", 
                                          json=payload) as response:
                        await response.read()
                        response_time = time.time() - start_time
                        return LoadTestResult(
                            response_time=response_time,
                            status_code=response.status,
                            success=response.status < 400
                        )
        except Exception as e:
            response_time = time.time() - start_time
            return LoadTestResult(
                response_time=response_time,
                status_code=0,
                success=False,
                error=str(e)
            )
    
    async def run_concurrent_requests(self, endpoint: str, num_requests: int, 
                                    method: str = "GET", payload: Dict = None) -> List[LoadTestResult]:
        """Run multiple concurrent requests"""
        self.logger.info(f"Starting {num_requests} concurrent requests to {endpoint}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def limited_request():
            async with semaphore:
                return await self._make_request(endpoint, method, payload)
        
        # Execute all requests concurrently
        tasks = [limited_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def analyze_results(self, results: List[LoadTestResult], 
                       duration: float) -> LoadTestSummary:
        """Analyze load test results and generate summary"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        response_times = [r.response_time for r in successful_results]
        
        if not response_times:
            response_times = [0]
        
        return LoadTestSummary(
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            avg_response_time=statistics.mean(response_times),
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0],
            p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) > 1 else response_times[0],
            requests_per_second=len(successful_results) / duration if duration > 0 else 0,
            error_rate=(len(failed_results) / len(results)) * 100 if results else 0,
            duration=duration
        )
    
    async def health_check_test(self) -> LoadTestSummary:
        """Test health check endpoint under load"""
        self.logger.info("🏥 Running health check load test...")
        
        start_time = time.time()
        results = await self.run_concurrent_requests("/health", 100)
        duration = time.time() - start_time
        
        return self.analyze_results(results, duration)
    
    async def api_endpoints_test(self) -> Dict[str, LoadTestSummary]:
        """Test various API endpoints under load"""
        self.logger.info("🔗 Running API endpoints load test...")
        
        endpoints = [
            ("/api/v1/projects", "GET"),
            ("/api/v1/tasks", "GET"),
            ("/api/v1/auth/status", "GET"),
        ]
        
        results = {}
        
        for endpoint, method in endpoints:
            self.logger.info(f"Testing {method} {endpoint}")
            start_time = time.time()
            test_results = await self.run_concurrent_requests(endpoint, 50, method)
            duration = time.time() - start_time
            
            results[f"{method} {endpoint}"] = self.analyze_results(test_results, duration)
        
        return results
    
    async def stress_test(self, duration_seconds: int = 60) -> LoadTestSummary:
        """Run sustained stress test"""
        self.logger.info(f"💪 Running {duration_seconds}-second stress test...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        all_results = []
        
        while time.time() < end_time:
            batch_results = await self.run_concurrent_requests("/health", 20)
            all_results.extend(batch_results)
            await asyncio.sleep(0.1)  # Brief pause between batches
        
        total_duration = time.time() - start_time
        return self.analyze_results(all_results, total_duration)
    
    async def database_load_test(self) -> LoadTestSummary:
        """Test database operations under load"""
        self.logger.info("🗄️ Running database load test...")
        
        # Test data for project creation
        test_project = {
            "name": f"Load Test Project {int(time.time())}",
            "description": "Automated load test project",
            "language": "python"
        }
        
        start_time = time.time()
        results = await self.run_concurrent_requests(
            "/api/v1/projects", 25, "POST", test_project
        )
        duration = time.time() - start_time
        
        return self.analyze_results(results, duration)
    
    def print_summary(self, name: str, summary: LoadTestSummary):
        """Print formatted test summary"""
        print(f"\n{'='*60}")
        print(f"📊 {name}")
        print(f"{'='*60}")
        print(f"Total Requests: {summary.total_requests}")
        print(f"Successful: {summary.successful_requests}")
        print(f"Failed: {summary.failed_requests}")
        print(f"Error Rate: {summary.error_rate:.2f}%")
        print(f"Duration: {summary.duration:.2f}s")
        print(f"Requests/Second: {summary.requests_per_second:.2f}")
        print(f"\nResponse Times:")
        print(f"  Average: {summary.avg_response_time*1000:.2f}ms")
        print(f"  Min: {summary.min_response_time*1000:.2f}ms")
        print(f"  Max: {summary.max_response_time*1000:.2f}ms")
        print(f"  95th percentile: {summary.p95_response_time*1000:.2f}ms")
        print(f"  99th percentile: {summary.p99_response_time*1000:.2f}ms")
        
        # Performance evaluation
        if summary.avg_response_time < 0.2:
            print("✅ Performance: EXCELLENT")
        elif summary.avg_response_time < 0.5:
            print("✅ Performance: GOOD")
        elif summary.avg_response_time < 1.0:
            print("⚠️ Performance: ACCEPTABLE")
        else:
            print("❌ Performance: NEEDS IMPROVEMENT")
        
        if summary.error_rate == 0:
            print("✅ Reliability: EXCELLENT")
        elif summary.error_rate < 1:
            print("✅ Reliability: GOOD")
        elif summary.error_rate < 5:
            print("⚠️ Reliability: ACCEPTABLE")
        else:
            print("❌ Reliability: NEEDS IMPROVEMENT")
    
    async def run_comprehensive_load_test(self) -> Dict[str, Any]:
        """Run comprehensive load testing suite"""
        self.logger.info("🚀 Starting comprehensive production load test...")
        
        results = {}
        
        # 1. Health check test
        health_summary = await self.health_check_test()
        self.print_summary("Health Check Load Test", health_summary)
        results['health_check'] = health_summary
        
        # 2. API endpoints test
        api_results = await self.api_endpoints_test()
        for endpoint, summary in api_results.items():
            self.print_summary(f"API Endpoint: {endpoint}", summary)
        results['api_endpoints'] = api_results
        
        # 3. Database load test
        try:
            db_summary = await self.database_load_test()
            self.print_summary("Database Load Test", db_summary)
            results['database'] = db_summary
        except Exception as e:
            self.logger.error(f"Database load test failed: {e}")
            results['database'] = f"Failed: {e}"
        
        # 4. Stress test
        stress_summary = await self.stress_test(30)  # 30-second stress test
        self.print_summary("30-Second Stress Test", stress_summary)
        results['stress_test'] = stress_summary
        
        # Overall assessment
        self._print_overall_assessment(results)
        
        return results
    
    def _print_overall_assessment(self, results: Dict[str, Any]):
        """Print overall production readiness assessment"""
        print(f"\n{'='*60}")
        print("🎯 OVERALL PRODUCTION READINESS ASSESSMENT")
        print(f"{'='*60}")
        
        # Calculate overall score
        scores = []
        
        # Health check score
        health = results.get('health_check')
        if health and health.error_rate == 0 and health.avg_response_time < 0.2:
            scores.append(100)
            print("✅ Health Check: PRODUCTION READY")
        elif health and health.error_rate < 5 and health.avg_response_time < 0.5:
            scores.append(75)
            print("⚠️ Health Check: NEEDS MINOR IMPROVEMENTS")
        else:
            scores.append(50)
            print("❌ Health Check: NEEDS MAJOR IMPROVEMENTS")
        
        # API endpoints score
        api_scores = []
        api_results = results.get('api_endpoints', {})
        for endpoint, summary in api_results.items():
            if isinstance(summary, LoadTestSummary):
                if summary.error_rate == 0 and summary.avg_response_time < 0.3:
                    api_scores.append(100)
                elif summary.error_rate < 5 and summary.avg_response_time < 1.0:
                    api_scores.append(75)
                else:
                    api_scores.append(50)
        
        if api_scores:
            avg_api_score = statistics.mean(api_scores)
            scores.append(avg_api_score)
            if avg_api_score >= 90:
                print("✅ API Endpoints: PRODUCTION READY")
            elif avg_api_score >= 75:
                print("⚠️ API Endpoints: NEEDS MINOR IMPROVEMENTS")
            else:
                print("❌ API Endpoints: NEEDS MAJOR IMPROVEMENTS")
        
        # Stress test score
        stress = results.get('stress_test')
        if stress and stress.error_rate < 1 and stress.requests_per_second > 50:
            scores.append(100)
            print("✅ Stress Test: PRODUCTION READY")
        elif stress and stress.error_rate < 10 and stress.requests_per_second > 20:
            scores.append(75)
            print("⚠️ Stress Test: NEEDS MINOR IMPROVEMENTS")
        else:
            scores.append(50)
            print("❌ Stress Test: NEEDS MAJOR IMPROVEMENTS")
        
        # Overall score
        if scores:
            overall_score = statistics.mean(scores)
            print(f"\n📊 OVERALL SCORE: {overall_score:.1f}/100")
            
            if overall_score >= 90:
                print("🚀 PRODUCTION DEPLOYMENT: APPROVED")
                print("System is ready for production deployment")
            elif overall_score >= 75:
                print("⚠️ PRODUCTION DEPLOYMENT: CONDITIONAL APPROVAL")
                print("Address minor issues before production deployment")
            else:
                print("❌ PRODUCTION DEPLOYMENT: NOT APPROVED")
                print("Critical issues must be resolved before production deployment")

async def main():
    """Main function to run load tests"""
    tester = ProductionLoadTester()
    
    try:
        results = await tester.run_comprehensive_load_test()
        
        # Save results to file
        timestamp = int(time.time())
        results_file = f"load_test_results_{timestamp}.json"
        
        # Convert LoadTestSummary objects to dicts for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, LoadTestSummary):
                json_results[key] = value.__dict__
            elif isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, LoadTestSummary):
                        json_results[key][k] = v.__dict__
                    else:
                        json_results[key][k] = str(v)
            else:
                json_results[key] = str(value)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Load testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)