"""
Production Smoke Testing Suite
Automated post-deployment validation and production smoke tests
"""

import asyncio
import time
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import httpx
import psutil
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.api.v1.health import HealthChecker
from src.monitoring.sla_monitor import sla_monitor
from src.validation.production_validator import production_validator
from src.database.disaster_recovery import disaster_recovery

logger = logging.getLogger(__name__)


class SmokeTestStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


class TestPriority(str, Enum):
    CRITICAL = "critical"    # Must pass for production deployment
    HIGH = "high"           # Important for user experience
    MEDIUM = "medium"       # Good to have working
    LOW = "low"             # Nice to have


@dataclass
class SmokeTestResult:
    """Result of a single smoke test"""
    test_name: str
    status: SmokeTestStatus
    priority: TestPriority
    duration_seconds: float
    timestamp: datetime
    message: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    recommendations: Optional[List[str]] = None


@dataclass
class SmokeTestSuite:
    """Complete smoke test suite results"""
    suite_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    overall_status: SmokeTestStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    skipped_tests: int
    critical_failures: int
    results: List[SmokeTestResult]
    environment_info: Dict[str, Any]
    deployment_ready: bool


class ProductionSmokeTestRunner:
    """Automated production smoke testing"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.health_checker = HealthChecker()
        self.suite_id = f"smoke_test_{int(time.time())}"
        
    async def test_basic_connectivity(self) -> SmokeTestResult:
        """Test basic application connectivity"""
        start_time = time.time()
        test_name = "Basic Connectivity"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/", timeout=10.0)
                
                if response.status_code == 200:
                    return SmokeTestResult(
                        test_name=test_name,
                        status=SmokeTestStatus.PASS,
                        priority=TestPriority.CRITICAL,
                        duration_seconds=time.time() - start_time,
                        timestamp=datetime.now(timezone.utc),
                        message="Application is responding to HTTP requests",
                        details={"status_code": response.status_code, "response_time_ms": (time.time() - start_time) * 1000}
                    )
                else:
                    return SmokeTestResult(
                        test_name=test_name,
                        status=SmokeTestStatus.FAIL,
                        priority=TestPriority.CRITICAL,
                        duration_seconds=time.time() - start_time,
                        timestamp=datetime.now(timezone.utc),
                        message=f"Application returned status code {response.status_code}",
                        details={"status_code": response.status_code},
                        recommendations=["Check application logs", "Verify deployment status"]
                    )
        except Exception as e:
            return SmokeTestResult(
                test_name=test_name,
                status=SmokeTestStatus.FAIL,
                priority=TestPriority.CRITICAL,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message="Failed to connect to application",
                error=str(e),
                recommendations=["Check if application is running", "Verify network connectivity", "Check firewall rules"]
            )
    
    async def test_health_endpoints(self) -> SmokeTestResult:
        """Test all health check endpoints"""
        start_time = time.time()
        test_name = "Health Endpoints"
        
        try:
            health_endpoints = [
                "/health",
                "/health/liveness", 
                "/health/readiness",
                "/health/startup"
            ]
            
            endpoint_results = {}
            
            async with httpx.AsyncClient() as client:
                for endpoint in health_endpoints:
                    try:
                        response = await client.get(f"{self.base_url}{endpoint}", timeout=5.0)
                        endpoint_results[endpoint] = {
                            "status_code": response.status_code,
                            "healthy": response.status_code == 200,
                            "response_time_ms": response.elapsed.total_seconds() * 1000 if response.elapsed else 0
                        }
                    except Exception as e:
                        endpoint_results[endpoint] = {
                            "status_code": None,
                            "healthy": False,
                            "error": str(e)
                        }
            
            # Determine overall health status
            healthy_endpoints = sum(1 for result in endpoint_results.values() if result.get("healthy", False))
            total_endpoints = len(health_endpoints)
            
            if healthy_endpoints == total_endpoints:
                status = SmokeTestStatus.PASS
                message = "All health endpoints are responding correctly"
            elif healthy_endpoints >= total_endpoints * 0.75:
                status = SmokeTestStatus.WARNING
                message = f"{healthy_endpoints}/{total_endpoints} health endpoints are healthy"
            else:
                status = SmokeTestStatus.FAIL
                message = f"Critical: Only {healthy_endpoints}/{total_endpoints} health endpoints are healthy"
            
            return SmokeTestResult(
                test_name=test_name,
                status=status,
                priority=TestPriority.CRITICAL,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message=message,
                details={"endpoints": endpoint_results},
                recommendations=["Check application health status", "Review logs for errors"] if status != SmokeTestStatus.PASS else []
            )
            
        except Exception as e:
            return SmokeTestResult(
                test_name=test_name,
                status=SmokeTestStatus.FAIL,
                priority=TestPriority.CRITICAL,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message="Health endpoint testing failed",
                error=str(e),
                recommendations=["Check application deployment", "Verify health check implementation"]
            )
    
    async def test_database_connectivity(self) -> SmokeTestResult:
        """Test database connectivity and basic operations"""
        start_time = time.time()
        test_name = "Database Connectivity"
        
        try:
            db = next(get_db())
            
            # Test basic query
            result = db.execute("SELECT 1 as test_value")
            row = result.fetchone()
            
            if row and row[0] == 1:
                # Test table existence (if any)
                try:
                    # Try to query a likely table
                    db.execute("SELECT COUNT(*) FROM users LIMIT 1")
                    table_test = True
                except Exception:
                    table_test = False
                
                return SmokeTestResult(
                    test_name=test_name,
                    status=SmokeTestStatus.PASS,
                    priority=TestPriority.CRITICAL,
                    duration_seconds=time.time() - start_time,
                    timestamp=datetime.now(timezone.utc),
                    message="Database is accessible and responding",
                    details={"basic_query": True, "table_access": table_test}
                )
            else:
                return SmokeTestResult(
                    test_name=test_name,
                    status=SmokeTestStatus.FAIL,
                    priority=TestPriority.CRITICAL,
                    duration_seconds=time.time() - start_time,
                    timestamp=datetime.now(timezone.utc),
                    message="Database query returned unexpected result",
                    recommendations=["Check database configuration", "Verify database schema"]
                )
                
        except Exception as e:
            return SmokeTestResult(
                test_name=test_name,
                status=SmokeTestStatus.FAIL,
                priority=TestPriority.CRITICAL,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message="Database connection failed",
                error=str(e),
                recommendations=["Check database server status", "Verify connection string", "Check network connectivity"]
            )
        finally:
            if 'db' in locals():
                db.close()
    
    async def test_api_endpoints(self) -> SmokeTestResult:
        """Test critical API endpoints"""
        start_time = time.time()
        test_name = "API Endpoints"
        
        try:
            critical_endpoints = [
                {"path": "/docs", "method": "GET", "expected_status": 200},
                {"path": "/openapi.json", "method": "GET", "expected_status": 200},
                {"path": "/api/v1/health", "method": "GET", "expected_status": 200}
            ]
            
            endpoint_results = {}
            
            async with httpx.AsyncClient() as client:
                for endpoint in critical_endpoints:
                    try:
                        if endpoint["method"] == "GET":
                            response = await client.get(f"{self.base_url}{endpoint['path']}", timeout=10.0)
                        else:
                            response = await client.request(
                                endpoint["method"],
                                f"{self.base_url}{endpoint['path']}",
                                timeout=10.0
                            )
                        
                        endpoint_results[endpoint["path"]] = {
                            "status_code": response.status_code,
                            "expected_status": endpoint["expected_status"],
                            "success": response.status_code == endpoint["expected_status"],
                            "response_time_ms": response.elapsed.total_seconds() * 1000 if response.elapsed else 0
                        }
                        
                    except Exception as e:
                        endpoint_results[endpoint["path"]] = {
                            "status_code": None,
                            "expected_status": endpoint["expected_status"],
                            "success": False,
                            "error": str(e)
                        }
            
            # Determine overall status
            successful_endpoints = sum(1 for result in endpoint_results.values() if result.get("success", False))
            total_endpoints = len(critical_endpoints)
            
            if successful_endpoints == total_endpoints:
                status = SmokeTestStatus.PASS
                message = "All critical API endpoints are responding correctly"
            elif successful_endpoints >= total_endpoints * 0.75:
                status = SmokeTestStatus.WARNING
                message = f"{successful_endpoints}/{total_endpoints} API endpoints are working"
            else:
                status = SmokeTestStatus.FAIL
                message = f"Critical: Only {successful_endpoints}/{total_endpoints} API endpoints are working"
            
            return SmokeTestResult(
                test_name=test_name,
                status=status,
                priority=TestPriority.HIGH,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message=message,
                details={"endpoints": endpoint_results},
                recommendations=["Check API routing configuration", "Review application logs"] if status != SmokeTestStatus.PASS else []
            )
            
        except Exception as e:
            return SmokeTestResult(
                test_name=test_name,
                status=SmokeTestStatus.FAIL,
                priority=TestPriority.HIGH,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message="API endpoint testing failed",
                error=str(e),
                recommendations=["Check application routing", "Verify API configuration"]
            )
    
    async def test_quantum_intelligence_modules(self) -> SmokeTestResult:
        """Test quantum intelligence modules operational status"""
        start_time = time.time()
        test_name = "Quantum Intelligence Modules"
        
        try:
            # Test quantum modules through health endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=15.0)
                
                if response.status_code == 200:
                    health_data = response.json()
                    quantum_component = health_data.get("components", {}).get("quantum_intelligence", {})
                    
                    if quantum_component.get("status") == "healthy":
                        quantum_details = quantum_component.get("details", {})
                        quantum_modules = quantum_details.get("quantum_modules", {})
                        
                        operational_modules = sum(1 for module in quantum_modules.values() if module.get("status") == "operational")
                        total_modules = len(quantum_modules)
                        
                        if operational_modules == 4 and total_modules == 4:
                            status = SmokeTestStatus.PASS
                            message = "All 4 quantum intelligence modules are operational"
                        elif operational_modules >= 3:
                            status = SmokeTestStatus.WARNING
                            message = f"{operational_modules}/4 quantum intelligence modules operational"
                        else:
                            status = SmokeTestStatus.FAIL
                            message = f"Critical: Only {operational_modules}/4 quantum intelligence modules operational"
                        
                        return SmokeTestResult(
                            test_name=test_name,
                            status=status,
                            priority=TestPriority.HIGH,
                            duration_seconds=time.time() - start_time,
                            timestamp=datetime.now(timezone.utc),
                            message=message,
                            details={"quantum_modules": quantum_modules, "operational_count": operational_modules},
                            recommendations=[
                                "Check quantum module initialization",
                                "Review quantum intelligence logs",
                                "Verify quantum processing dependencies"
                            ] if status != SmokeTestStatus.PASS else []
                        )
                    else:
                        return SmokeTestResult(
                            test_name=test_name,
                            status=SmokeTestStatus.FAIL,
                            priority=TestPriority.HIGH,
                            duration_seconds=time.time() - start_time,
                            timestamp=datetime.now(timezone.utc),
                            message="Quantum intelligence component is not healthy",
                            details={"component_status": quantum_component.get("status")},
                            recommendations=["Check quantum intelligence system initialization"]
                        )
                else:
                    return SmokeTestResult(
                        test_name=test_name,
                        status=SmokeTestStatus.FAIL,
                        priority=TestPriority.HIGH,
                        duration_seconds=time.time() - start_time,
                        timestamp=datetime.now(timezone.utc),
                        message="Could not retrieve quantum intelligence status",
                        details={"health_endpoint_status": response.status_code},
                        recommendations=["Check health endpoint functionality"]
                    )
                
        except Exception as e:
            return SmokeTestResult(
                test_name=test_name,
                status=SmokeTestStatus.FAIL,
                priority=TestPriority.HIGH,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message="Quantum intelligence module testing failed",
                error=str(e),
                recommendations=["Check quantum intelligence system deployment", "Verify module dependencies"]
            )
    
    async def test_performance_baseline(self) -> SmokeTestResult:
        """Test basic performance baseline"""
        start_time = time.time()
        test_name = "Performance Baseline"
        
        try:
            # Run a small load test
            response_times = []
            error_count = 0
            test_requests = 20
            
            async with httpx.AsyncClient() as client:
                for _ in range(test_requests):
                    request_start = time.time()
                    try:
                        response = await client.get(f"{self.base_url}/health/liveness", timeout=5.0)
                        response_time = (time.time() - request_start) * 1000  # Convert to ms
                        response_times.append(response_time)
                        
                        if response.status_code != 200:
                            error_count += 1
                            
                    except Exception:
                        error_count += 1
                        response_times.append(5000)  # 5s timeout
            
            # Calculate metrics
            avg_response_time = sum(response_times) / len(response_times) if response_times else 5000
            error_rate = (error_count / test_requests) * 100
            
            # Performance thresholds
            if avg_response_time <= 100 and error_rate <= 0.1:
                status = SmokeTestStatus.PASS
                message = f"Performance baseline met: {avg_response_time:.1f}ms avg, {error_rate:.1f}% errors"
            elif avg_response_time <= 500 and error_rate <= 5:
                status = SmokeTestStatus.WARNING
                message = f"Performance acceptable: {avg_response_time:.1f}ms avg, {error_rate:.1f}% errors"
            else:
                status = SmokeTestStatus.FAIL
                message = f"Performance below baseline: {avg_response_time:.1f}ms avg, {error_rate:.1f}% errors"
            
            return SmokeTestResult(
                test_name=test_name,
                status=status,
                priority=TestPriority.MEDIUM,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message=message,
                details={
                    "average_response_time_ms": avg_response_time,
                    "error_rate_percentage": error_rate,
                    "total_requests": test_requests,
                    "failed_requests": error_count
                },
                recommendations=[
                    "Check system resource utilization",
                    "Review application performance",
                    "Consider scaling if needed"
                ] if status != SmokeTestStatus.PASS else []
            )
            
        except Exception as e:
            return SmokeTestResult(
                test_name=test_name,
                status=SmokeTestStatus.FAIL,
                priority=TestPriority.MEDIUM,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message="Performance baseline testing failed",
                error=str(e),
                recommendations=["Check application responsiveness", "Verify system health"]
            )
    
    async def test_disaster_recovery(self) -> SmokeTestResult:
        """Test disaster recovery capabilities"""
        start_time = time.time()
        test_name = "Disaster Recovery"
        
        try:
            # Run disaster recovery test
            dr_results = await disaster_recovery.test_disaster_recovery()
            
            if dr_results["summary"]["overall_status"] == "pass":
                status = SmokeTestStatus.PASS
                message = "Disaster recovery system is functional"
            else:
                status = SmokeTestStatus.WARNING
                message = f"Disaster recovery test: {dr_results['summary']['passed_tests']}/{dr_results['summary']['total_tests']} tests passed"
            
            return SmokeTestResult(
                test_name=test_name,
                status=status,
                priority=TestPriority.MEDIUM,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message=message,
                details=dr_results,
                recommendations=[
                    "Review disaster recovery procedures",
                    "Check backup system functionality",
                    "Verify restoration capabilities"
                ] if status != SmokeTestStatus.PASS else []
            )
            
        except Exception as e:
            return SmokeTestResult(
                test_name=test_name,
                status=SmokeTestStatus.WARNING,
                priority=TestPriority.MEDIUM,
                duration_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                message="Disaster recovery testing failed",
                error=str(e),
                recommendations=["Check disaster recovery system", "Verify backup procedures"]
            )
    
    async def run_comprehensive_smoke_tests(self) -> SmokeTestSuite:
        """Run complete smoke test suite"""
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting production smoke tests: {self.suite_id}")
        
        # Define all smoke tests
        test_methods = [
            self.test_basic_connectivity,
            self.test_health_endpoints,
            self.test_database_connectivity,
            self.test_api_endpoints,
            self.test_quantum_intelligence_modules,
            self.test_performance_baseline,
            self.test_disaster_recovery
        ]
        
        # Run all tests concurrently
        results = await asyncio.gather(*[test() for test in test_methods], return_exceptions=True)
        
        # Process results
        test_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_results.append(SmokeTestResult(
                    test_name=f"Test {i+1}",
                    status=SmokeTestStatus.FAIL,
                    priority=TestPriority.HIGH,
                    duration_seconds=0.0,
                    timestamp=datetime.now(timezone.utc),
                    message="Test execution failed",
                    error=str(result)
                ))
            else:
                test_results.append(result)
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        # Calculate summary statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == SmokeTestStatus.PASS)
        failed_tests = sum(1 for r in test_results if r.status == SmokeTestStatus.FAIL)
        warning_tests = sum(1 for r in test_results if r.status == SmokeTestStatus.WARNING)
        skipped_tests = sum(1 for r in test_results if r.status == SmokeTestStatus.SKIP)
        
        critical_failures = sum(1 for r in test_results if r.status == SmokeTestStatus.FAIL and r.priority == TestPriority.CRITICAL)
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = SmokeTestStatus.FAIL
            deployment_ready = False
        elif failed_tests > 0:
            overall_status = SmokeTestStatus.FAIL
            deployment_ready = False
        elif warning_tests > 0:
            overall_status = SmokeTestStatus.WARNING
            deployment_ready = True  # Warnings don't block deployment
        else:
            overall_status = SmokeTestStatus.PASS
            deployment_ready = True
        
        # Collect environment information
        environment_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hostname": os.getenv("HOSTNAME", "unknown"),
            "environment": os.getenv("ENVIRONMENT", "production"),
            "version": os.getenv("APP_VERSION", "unknown"),
            "python_version": f"{psutil.Process().exe}",
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3)
        }
        
        suite = SmokeTestSuite(
            suite_id=self.suite_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            overall_status=overall_status,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            skipped_tests=skipped_tests,
            critical_failures=critical_failures,
            results=test_results,
            environment_info=environment_info,
            deployment_ready=deployment_ready
        )
        
        # Save test results
        await self.save_test_results(suite)
        
        logger.info(f"Smoke tests completed: {self.suite_id} - Status: {overall_status}, Deployment Ready: {deployment_ready}")
        
        return suite
    
    async def save_test_results(self, suite: SmokeTestSuite) -> None:
        """Save smoke test results to file"""
        try:
            results_dir = Path("smoke_test_results")
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / f"{suite.suite_id}.json"
            
            # Convert to serializable format
            suite_data = asdict(suite)
            suite_data["start_time"] = suite.start_time.isoformat()
            suite_data["end_time"] = suite.end_time.isoformat()
            
            for result in suite_data["results"]:
                result["timestamp"] = result["timestamp"].isoformat() if isinstance(result["timestamp"], datetime) else result["timestamp"]
            
            with open(results_file, 'w') as f:
                json.dump(suite_data, f, indent=2)
                
            logger.info(f"Smoke test results saved: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save smoke test results: {e}")


# Global smoke test runner
smoke_test_runner = ProductionSmokeTestRunner()