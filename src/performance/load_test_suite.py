#!/usr/bin/env python3
"""
Load Test Suite for Performance Validation

Comprehensive load testing to validate <200ms API response times with 1000+ concurrent users.
Includes realistic scenarios, performance benchmarking, and automated optimization validation.
"""

import asyncio
import aiohttp
import time
import json
import statistics
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import psutil
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 1000
    test_duration_seconds: int = 300  # 5 minutes
    ramp_up_time_seconds: int = 60    # 1 minute
    target_response_time_ms: float = 200.0
    max_acceptable_response_time_ms: float = 500.0
    max_error_rate_percent: float = 1.0
    scenarios: List[str] = field(default_factory=lambda: [
        "api_health_check",
        "ai_code_generation", 
        "swarm_status_check",
        "performance_metrics",
        "project_operations"
    ])


@dataclass 
class TestResult:
    """Individual test result"""
    scenario: str
    endpoint: str
    response_time_ms: float
    status_code: int
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    user_id: int = 0


@dataclass
class LoadTestReport:
    """Comprehensive load test report"""
    config: LoadTestConfig
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success_rate_percent(self) -> float:
        return (self.successful_requests / max(self.total_requests, 1)) * 100
    
    @property
    def error_rate_percent(self) -> float:
        return (self.failed_requests / max(self.total_requests, 1)) * 100
    
    @property
    def requests_per_second(self) -> float:
        return self.total_requests / max(self.duration_seconds, 1)


class LoadTestExecutor:
    """Executes comprehensive load tests"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.test_scenarios = {
            "api_health_check": self._test_health_check,
            "ai_code_generation": self._test_ai_code_generation,
            "swarm_status_check": self._test_swarm_status,
            "performance_metrics": self._test_performance_metrics,
            "project_operations": self._test_project_operations
        }
        
        # Performance monitoring during tests
        self.system_metrics: List[Dict[str, Any]] = []
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def run_load_test(self) -> LoadTestReport:
        """Execute comprehensive load test"""
        print(f"🚀 LOAD TEST STARTING")
        print(f"   Target: {self.config.concurrent_users} concurrent users")
        print(f"   Duration: {self.config.test_duration_seconds}s")
        print(f"   Response Time Target: <{self.config.target_response_time_ms}ms")
        print(f"   Base URL: {self.config.base_url}")
        
        start_time = time.time()
        
        # Initialize HTTP session with optimized settings
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_read=5)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "LoadTest/1.0"}
        )
        
        try:
            # Start system monitoring
            self.monitoring_task = asyncio.create_task(self._monitor_system_performance())
            
            # Execute load test phases
            await self._execute_ramp_up_phase()
            await self._execute_sustained_load_phase()
            
            end_time = time.time()
            
            # Generate comprehensive report
            report = LoadTestReport(
                config=self.config,
                start_time=start_time,
                end_time=end_time,
                total_requests=len(self.results),
                successful_requests=len([r for r in self.results if r.success]),
                failed_requests=len([r for r in self.results if not r.success]),
                results=self.results
            )
            
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            return report
            
        finally:
            await self.session.close()
    
    async def _execute_ramp_up_phase(self):
        """Execute gradual ramp-up phase"""
        print(f"📈 RAMP-UP PHASE: Gradually increasing to {self.config.concurrent_users} users")
        
        ramp_step_duration = self.config.ramp_up_time_seconds / 10  # 10 steps
        users_per_step = max(1, self.config.concurrent_users // 10)
        
        current_users = 0
        
        for step in range(10):
            current_users = min(current_users + users_per_step, self.config.concurrent_users)
            
            print(f"   Step {step + 1}/10: {current_users} concurrent users")
            
            # Launch tasks for this step
            tasks = []
            for user_id in range(current_users):
                task = asyncio.create_task(
                    self._simulate_user_session(user_id, duration=ramp_step_duration)
                )
                tasks.append(task)
            
            # Wait for step completion
            await asyncio.sleep(ramp_step_duration)
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
    
    async def _execute_sustained_load_phase(self):
        """Execute sustained load phase"""
        print(f"⚡ SUSTAINED LOAD PHASE: {self.config.concurrent_users} users for {self.config.test_duration_seconds}s")
        
        # Calculate requests per user
        remaining_duration = self.config.test_duration_seconds - self.config.ramp_up_time_seconds
        
        # Launch all concurrent user sessions
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task = asyncio.create_task(
                self._simulate_user_session(user_id, duration=remaining_duration)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        completed_tasks = 0
        for task in asyncio.as_completed(tasks):
            try:
                await task
                completed_tasks += 1
                if completed_tasks % 100 == 0:
                    print(f"   Completed: {completed_tasks}/{len(tasks)} user sessions")
            except Exception as e:
                logger.error(f"User session error: {e}")
    
    async def _simulate_user_session(self, user_id: int, duration: float):
        """Simulate realistic user session"""
        session_start = time.time()
        requests_made = 0
        
        while (time.time() - session_start) < duration:
            try:
                # Select random scenario with realistic weights
                scenario = random.choices(
                    self.config.scenarios,
                    weights=[5, 2, 3, 1, 2],  # Health checks most frequent
                    k=1
                )[0]
                
                # Execute scenario
                if scenario in self.test_scenarios:
                    result = await self.test_scenarios[scenario](user_id)
                    self.results.append(result)
                    requests_made += 1
                
                # Realistic user think time
                think_time = random.uniform(0.5, 2.0)  # 0.5-2 seconds between requests
                await asyncio.sleep(think_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"User {user_id} session error: {e}")
                
                # Record error
                error_result = TestResult(
                    scenario="session_error",
                    endpoint="unknown",
                    response_time_ms=0,
                    status_code=0,
                    success=False,
                    error_message=str(e),
                    user_id=user_id
                )
                self.results.append(error_result)
                
                # Brief pause before retry
                await asyncio.sleep(1)
    
    async def _test_health_check(self, user_id: int) -> TestResult:
        """Test API health check endpoint"""
        return await self._make_request(
            "api_health_check",
            "GET",
            "/health",
            user_id
        )
    
    async def _test_ai_code_generation(self, user_id: int) -> TestResult:
        """Test AI code generation endpoint"""
        payload = {
            "prompt": "Create a simple Python function to calculate factorial",
            "language": "python",
            "complexity": random.randint(1, 5)
        }
        
        return await self._make_request(
            "ai_code_generation",
            "POST", 
            "/api/v1/ai/code/generate",
            user_id,
            json_payload=payload
        )
    
    async def _test_swarm_status(self, user_id: int) -> TestResult:
        """Test swarm status endpoint"""
        return await self._make_request(
            "swarm_status_check",
            "GET",
            "/api/v1/swarm/status",
            user_id
        )
    
    async def _test_performance_metrics(self, user_id: int) -> TestResult:
        """Test performance metrics endpoint"""
        return await self._make_request(
            "performance_metrics",
            "GET",
            "/api/v1/performance/metrics",
            user_id
        )
    
    async def _test_project_operations(self, user_id: int) -> TestResult:
        """Test project operations"""
        # Simulate different project operations
        operations = [
            ("GET", "/api/v1/projects"),
            ("GET", f"/api/v1/projects/{random.randint(1, 100)}"),
            ("POST", "/api/v1/projects", {"name": f"test-project-{user_id}"}),
        ]
        
        method, endpoint, *payload = random.choice(operations)
        json_payload = payload[0] if payload else None
        
        return await self._make_request(
            "project_operations",
            method,
            endpoint,
            user_id,
            json_payload=json_payload
        )
    
    async def _make_request(
        self, 
        scenario: str, 
        method: str, 
        endpoint: str, 
        user_id: int,
        json_payload: Optional[Dict] = None
    ) -> TestResult:
        """Make HTTP request and measure performance"""
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            async with self.session.request(
                method, 
                url, 
                json=json_payload,
                headers={"X-User-ID": str(user_id)}
            ) as response:
                # Read response body to ensure complete request processing
                await response.read()
                
                response_time_ms = (time.time() - start_time) * 1000
                
                return TestResult(
                    scenario=scenario,
                    endpoint=endpoint,
                    response_time_ms=response_time_ms,
                    status_code=response.status,
                    success=200 <= response.status < 400,
                    user_id=user_id
                )
                
        except asyncio.TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            return TestResult(
                scenario=scenario,
                endpoint=endpoint,
                response_time_ms=response_time_ms,
                status_code=0,
                success=False,
                error_message="Request timeout",
                user_id=user_id
            )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return TestResult(
                scenario=scenario,
                endpoint=endpoint,
                response_time_ms=response_time_ms,
                status_code=0,
                success=False,
                error_message=str(e),
                user_id=user_id
            )
    
    async def _monitor_system_performance(self):
        """Monitor system performance during load test"""
        while True:
            try:
                process = psutil.Process()
                system_memory = psutil.virtual_memory()
                
                metrics = {
                    "timestamp": time.time(),
                    "process_memory_mb": process.memory_info().rss / (1024 ** 2),
                    "system_memory_percent": system_memory.percent,
                    "cpu_percent": process.cpu_percent(),
                    "system_cpu_percent": psutil.cpu_percent(),
                    "active_connections": len(process.connections()),
                    "open_files": len(process.open_files())
                }
                
                self.system_metrics.append(metrics)
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(5)


class LoadTestAnalyzer:
    """Analyzes load test results and generates insights"""
    
    @staticmethod
    def analyze_results(report: LoadTestReport) -> Dict[str, Any]:
        """Analyze load test results comprehensively"""
        
        if not report.results:
            return {"error": "No test results to analyze"}
        
        # Basic statistics
        successful_results = [r for r in report.results if r.success]
        failed_results = [r for r in report.results if not r.success]
        
        response_times = [r.response_time_ms for r in successful_results]
        
        analysis = {
            "summary": {
                "total_requests": report.total_requests,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate_percent": report.success_rate_percent,
                "error_rate_percent": report.error_rate_percent,
                "duration_seconds": report.duration_seconds,
                "requests_per_second": report.requests_per_second
            },
            "performance_metrics": {},
            "scenario_breakdown": {},
            "performance_validation": {},
            "optimization_recommendations": []
        }
        
        # Performance metrics analysis
        if response_times:
            analysis["performance_metrics"] = {
                "response_time_ms": {
                    "min": min(response_times),
                    "max": max(response_times),
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "p95": response_times[int(len(response_times) * 0.95)] if len(response_times) > 20 else max(response_times),
                    "p99": response_times[int(len(response_times) * 0.99)] if len(response_times) > 100 else max(response_times),
                    "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
                },
                "target_response_time_ms": report.config.target_response_time_ms,
                "meets_target": statistics.mean(response_times) <= report.config.target_response_time_ms
            }
        
        # Scenario breakdown
        scenario_stats = {}
        for scenario in report.config.scenarios:
            scenario_results = [r for r in report.results if r.scenario == scenario]
            
            if scenario_results:
                scenario_response_times = [r.response_time_ms for r in scenario_results if r.success]
                scenario_stats[scenario] = {
                    "total_requests": len(scenario_results),
                    "successful_requests": len([r for r in scenario_results if r.success]),
                    "success_rate_percent": (len([r for r in scenario_results if r.success]) / len(scenario_results)) * 100,
                    "avg_response_time_ms": statistics.mean(scenario_response_times) if scenario_response_times else 0,
                    "p95_response_time_ms": scenario_response_times[int(len(scenario_response_times) * 0.95)] if len(scenario_response_times) > 20 else (max(scenario_response_times) if scenario_response_times else 0)
                }
        
        analysis["scenario_breakdown"] = scenario_stats
        
        # Performance validation against targets
        validation = {
            "response_time_target_met": False,
            "error_rate_target_met": False,
            "concurrent_users_handled": report.config.concurrent_users,
            "overall_performance_grade": "F"
        }
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = response_times[int(len(response_times) * 0.95)] if len(response_times) > 20 else max(response_times)
            
            validation["response_time_target_met"] = avg_response_time <= report.config.target_response_time_ms
            validation["error_rate_target_met"] = report.error_rate_percent <= report.config.max_error_rate_percent
            
            # Performance grading
            if (validation["response_time_target_met"] and 
                validation["error_rate_target_met"] and 
                p95_response_time <= report.config.max_acceptable_response_time_ms):
                validation["overall_performance_grade"] = "A"
            elif (avg_response_time <= report.config.target_response_time_ms * 1.5 and 
                  report.error_rate_percent <= report.config.max_error_rate_percent * 2):
                validation["overall_performance_grade"] = "B" 
            elif (avg_response_time <= report.config.target_response_time_ms * 2 and 
                  report.error_rate_percent <= report.config.max_error_rate_percent * 5):
                validation["overall_performance_grade"] = "C"
            else:
                validation["overall_performance_grade"] = "D"
        
        analysis["performance_validation"] = validation
        
        # Generate optimization recommendations
        recommendations = LoadTestAnalyzer._generate_recommendations(report, analysis)
        analysis["optimization_recommendations"] = recommendations
        
        return analysis
    
    @staticmethod
    def _generate_recommendations(report: LoadTestReport, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on test results"""
        recommendations = []
        
        perf_metrics = analysis.get("performance_metrics", {})
        response_time_data = perf_metrics.get("response_time_ms", {})
        
        # Response time recommendations
        if response_time_data:
            avg_response_time = response_time_data.get("mean", 0)
            p95_response_time = response_time_data.get("p95", 0)
            
            if avg_response_time > report.config.target_response_time_ms:
                severity = "HIGH" if avg_response_time > report.config.target_response_time_ms * 2 else "MEDIUM"
                recommendations.append({
                    "category": "response_time",
                    "severity": severity,
                    "title": "Optimize API Response Times",
                    "description": f"Average response time ({avg_response_time:.0f}ms) exceeds target ({report.config.target_response_time_ms:.0f}ms)",
                    "actions": [
                        "Implement aggressive caching strategies",
                        "Optimize database queries and connection pooling", 
                        "Add CDN for static content",
                        "Consider horizontal scaling",
                        "Profile and optimize hot code paths"
                    ]
                })
            
            if p95_response_time > report.config.max_acceptable_response_time_ms:
                recommendations.append({
                    "category": "response_time_outliers",
                    "severity": "MEDIUM",
                    "title": "Address Response Time Outliers", 
                    "description": f"P95 response time ({p95_response_time:.0f}ms) indicates performance outliers",
                    "actions": [
                        "Investigate slow queries and operations",
                        "Implement request timeout handling",
                        "Add performance monitoring and alerting",
                        "Consider async processing for heavy operations"
                    ]
                })
        
        # Error rate recommendations
        if report.error_rate_percent > report.config.max_error_rate_percent:
            severity = "HIGH" if report.error_rate_percent > 5 else "MEDIUM"
            recommendations.append({
                "category": "error_rate",
                "severity": severity,
                "title": "Reduce Error Rate",
                "description": f"Error rate ({report.error_rate_percent:.1f}%) exceeds acceptable threshold ({report.config.max_error_rate_percent:.1f}%)",
                "actions": [
                    "Implement better error handling and retry logic",
                    "Add circuit breaker patterns",
                    "Increase resource limits (memory, CPU)",
                    "Improve connection pool configuration",
                    "Add comprehensive logging and monitoring"
                ]
            })
        
        # Scenario-specific recommendations
        scenario_breakdown = analysis.get("scenario_breakdown", {})
        for scenario, stats in scenario_breakdown.items():
            if stats["success_rate_percent"] < 95:  # Less than 95% success rate
                recommendations.append({
                    "category": "scenario_reliability",
                    "severity": "MEDIUM",
                    "title": f"Improve {scenario.replace('_', ' ').title()} Reliability",
                    "description": f"Scenario success rate ({stats['success_rate_percent']:.1f}%) below target",
                    "actions": [
                        f"Focus optimization efforts on {scenario} endpoint",
                        "Add specific monitoring for this scenario",
                        "Implement scenario-specific caching",
                        "Consider breaking down complex operations"
                    ]
                })
        
        # Capacity recommendations
        if report.config.concurrent_users >= 1000 and report.success_rate_percent < 99:
            recommendations.append({
                "category": "capacity",
                "severity": "HIGH",
                "title": "Scale for High Concurrency",
                "description": f"System struggles with {report.config.concurrent_users} concurrent users",
                "actions": [
                    "Implement horizontal auto-scaling",
                    "Add load balancing across multiple instances",
                    "Optimize resource allocation",
                    "Consider microservices architecture",
                    "Implement queuing for resource-intensive operations"
                ]
            })
        
        return recommendations
    
    @staticmethod
    def export_results(report: LoadTestReport, analysis: Dict[str, Any], filepath: str):
        """Export comprehensive test results"""
        export_data = {
            "test_configuration": {
                "base_url": report.config.base_url,
                "concurrent_users": report.config.concurrent_users,
                "test_duration_seconds": report.config.test_duration_seconds,
                "target_response_time_ms": report.config.target_response_time_ms,
                "scenarios": report.config.scenarios
            },
            "test_execution": {
                "start_time": datetime.fromtimestamp(report.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(report.end_time).isoformat(),
                "duration_seconds": report.duration_seconds,
                "total_requests": report.total_requests,
                "requests_per_second": report.requests_per_second
            },
            "performance_results": analysis,
            "raw_results_sample": [
                {
                    "scenario": r.scenario,
                    "endpoint": r.endpoint,
                    "response_time_ms": r.response_time_ms,
                    "status_code": r.status_code,
                    "success": r.success,
                    "timestamp": r.timestamp
                }
                for r in report.results[:100]  # Sample of first 100 results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📊 Load test results exported to: {filepath}")


async def run_performance_validation_test(concurrent_users: int = 1000, duration_minutes: int = 5) -> Dict[str, Any]:
    """Run comprehensive performance validation test"""
    
    print(f"🎯 PERFORMANCE VALIDATION TEST")
    print(f"Target: Validate <200ms response times with {concurrent_users} concurrent users")
    print("=" * 80)
    
    # Configure load test
    config = LoadTestConfig(
        base_url="http://localhost:8000",  # Update with actual API URL
        concurrent_users=concurrent_users,
        test_duration_seconds=duration_minutes * 60,
        ramp_up_time_seconds=min(60, duration_minutes * 10),  # 10% of test time or 60s max
        target_response_time_ms=200.0,
        max_acceptable_response_time_ms=500.0,
        max_error_rate_percent=1.0
    )
    
    # Execute load test
    executor = LoadTestExecutor(config)
    report = await executor.run_load_test()
    
    # Analyze results  
    analysis = LoadTestAnalyzer.analyze_results(report)
    
    # Export results
    timestamp = int(time.time())
    results_file = f"load_test_results_{timestamp}.json"
    LoadTestAnalyzer.export_results(report, analysis, results_file)
    
    # Print summary
    print(f"\n📋 LOAD TEST SUMMARY")
    print(f"{'='*50}")
    
    summary = analysis["summary"]
    print(f"Total Requests: {summary['total_requests']:,}")
    print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
    print(f"Error Rate: {summary['error_rate_percent']:.1f}%")
    print(f"Requests/sec: {summary['requests_per_second']:.1f}")
    
    perf_metrics = analysis.get("performance_metrics", {})
    if perf_metrics:
        rt = perf_metrics["response_time_ms"]
        print(f"\\nResponse Time Metrics:")
        print(f"  Average: {rt['mean']:.1f}ms")
        print(f"  Median: {rt['median']:.1f}ms")
        print(f"  P95: {rt['p95']:.1f}ms")
        print(f"  P99: {rt['p99']:.1f}ms")
    
    validation = analysis["performance_validation"]
    print(f"\\n🎯 PERFORMANCE VALIDATION")
    print(f"Response Time Target Met: {'✅' if validation['response_time_target_met'] else '❌'}")
    print(f"Error Rate Target Met: {'✅' if validation['error_rate_target_met'] else '❌'}")
    print(f"Overall Grade: {validation['overall_performance_grade']}")
    
    # Show top recommendations
    recommendations = analysis.get("optimization_recommendations", [])
    if recommendations:
        print(f"\\n🔧 TOP OPTIMIZATION RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec['title']} ({rec['severity']} priority)")
            print(f"     {rec['description']}")
    
    print(f"\\n📁 Detailed results saved to: {results_file}")
    
    return {
        "report": report,
        "analysis": analysis,
        "results_file": results_file,
        "performance_grade": validation["overall_performance_grade"],
        "meets_targets": validation["response_time_target_met"] and validation["error_rate_target_met"]
    }


if __name__ == "__main__":
    # Run performance validation test
    async def main():
        try:
            results = await run_performance_validation_test(
                concurrent_users=100,  # Start with smaller load for testing
                duration_minutes=2     # Shorter duration for testing
            )
            
            if results["meets_targets"]:
                print("\\n🎉 PERFORMANCE TARGETS ACHIEVED!")
            else:
                print("\\n⚠️ PERFORMANCE TARGETS NOT MET - Optimization needed")
                
        except Exception as e:
            print(f"\\n❌ Load test failed: {e}")
            logger.exception("Load test execution failed")
    
    # Run the test
    asyncio.run(main())