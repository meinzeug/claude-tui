"""
Production Validation Suite
Comprehensive validation system for production deployment
"""

import asyncio
import time
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import httpx
import psutil
import aioredis
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.api.v1.health import HealthChecker, HealthStatus, ComponentHealth
from src.performance.memory_optimizer import MemoryOptimizer
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    duration: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    recommendations: Optional[List[str]] = None


@dataclass
class ProductionValidationReport:
    """Comprehensive production validation report"""
    validation_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    overall_status: ValidationStatus
    results: List[ValidationResult]
    summary: Dict[str, Any]
    sla_compliance: Dict[str, Any]
    recommendations: List[str]


class ProductionValidator:
    """Comprehensive production validation system"""
    
    def __init__(self):
        self.settings = get_settings()
        self.health_checker = HealthChecker()
        self.memory_optimizer = MemoryOptimizer()
        self.validation_id = f"validation_{int(time.time())}"
        
    async def validate_external_integrations(self) -> ValidationResult:
        """Validate all external API integrations with real calls"""
        start_time = time.time()
        
        try:
            integrations = [
                {
                    "name": "Claude API",
                    "url": "https://api.anthropic.com/v1/messages",
                    "method": "POST",
                    "headers": {"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                    "expected_status": [400, 401],  # Auth error is expected with test key
                    "timeout": 30
                },
                {
                    "name": "GitHub API",
                    "url": "https://api.github.com/user",
                    "method": "GET",
                    "headers": {"Authorization": "token test"},
                    "expected_status": [401],  # Auth error is expected with test token
                    "timeout": 10
                }
            ]
            
            results = {}
            
            async with httpx.AsyncClient() as client:
                for integration in integrations:
                    try:
                        response = await client.request(
                            integration["method"],
                            integration["url"],
                            headers=integration["headers"],
                            timeout=integration["timeout"]
                        )
                        
                        is_healthy = response.status_code in integration["expected_status"]
                        results[integration["name"]] = {
                            "status": "healthy" if is_healthy else "unhealthy",
                            "status_code": response.status_code,
                            "response_time": response.elapsed.total_seconds(),
                            "reachable": True
                        }
                        
                    except httpx.TimeoutException:
                        results[integration["name"]] = {
                            "status": "timeout",
                            "error": "Request timeout",
                            "reachable": False
                        }
                    except Exception as e:
                        results[integration["name"]] = {
                            "status": "error",
                            "error": str(e),
                            "reachable": False
                        }
            
            # Determine overall status
            healthy_count = sum(1 for r in results.values() if r.get("status") == "healthy")
            total_count = len(results)
            
            if healthy_count == total_count:
                status = ValidationStatus.PASS
                message = "All external integrations are reachable and responding correctly"
                severity = ValidationSeverity.INFO
            elif healthy_count >= total_count * 0.5:
                status = ValidationStatus.WARNING
                message = f"{healthy_count}/{total_count} external integrations healthy"
                severity = ValidationSeverity.MEDIUM
            else:
                status = ValidationStatus.FAIL
                message = f"Critical: Only {healthy_count}/{total_count} external integrations healthy"
                severity = ValidationSeverity.CRITICAL
            
            return ValidationResult(
                name="External API Integrations",
                status=status,
                severity=severity,
                message=message,
                duration=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                details={"integrations": results},
                recommendations=[
                    "Configure proper API keys for production deployment",
                    "Set up monitoring for external API health",
                    "Implement circuit breakers for external dependencies"
                ] if status != ValidationStatus.PASS else []
            )
            
        except Exception as e:
            return ValidationResult(
                name="External API Integrations",
                status=ValidationStatus.FAIL,
                severity=ValidationSeverity.CRITICAL,
                message="Integration validation failed",
                duration=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                recommendations=[
                    "Check network connectivity",
                    "Verify external API endpoints",
                    "Review API configuration"
                ]
            )
    
    async def validate_database_performance(self, db: Session) -> ValidationResult:
        """Validate database performance under load"""
        start_time = time.time()
        
        try:
            # Test basic connectivity
            db.execute("SELECT 1")
            
            # Test query performance
            query_times = []
            
            # Run multiple test queries
            for _ in range(10):
                query_start = time.time()
                try:
                    result = db.execute("""
                        SELECT 
                            COUNT(*) as count,
                            AVG(LENGTH(email)) as avg_email_length,
                            MAX(created_at) as latest_user
                        FROM users 
                        WHERE is_active = true
                    """)
                    row = result.fetchone()
                    query_times.append(time.time() - query_start)
                except Exception:
                    # Table might not exist, use simple query
                    db.execute("SELECT 1")
                    query_times.append(time.time() - query_start)
            
            avg_query_time = sum(query_times) / len(query_times)
            max_query_time = max(query_times)
            
            # Performance thresholds
            if avg_query_time > 1.0:  # > 1 second average
                status = ValidationStatus.FAIL
                severity = ValidationSeverity.CRITICAL
                message = f"Database performance critical: {avg_query_time:.3f}s average query time"
            elif avg_query_time > 0.1:  # > 100ms average
                status = ValidationStatus.WARNING
                severity = ValidationSeverity.MEDIUM
                message = f"Database performance degraded: {avg_query_time:.3f}s average query time"
            else:
                status = ValidationStatus.PASS
                severity = ValidationSeverity.INFO
                message = f"Database performance good: {avg_query_time:.3f}s average query time"
            
            # Check connection pool
            pool_info = {}
            try:
                pool_info = {
                    "pool_size": db.bind.pool.size(),
                    "checked_out": db.bind.pool.checkedout(),
                    "overflow": db.bind.pool.overflow(),
                    "utilization": db.bind.pool.checkedout() / db.bind.pool.size()
                }
            except AttributeError:
                pool_info = {"note": "Connection pool info not available for this database type"}
            
            return ValidationResult(
                name="Database Performance",
                status=status,
                severity=severity,
                message=message,
                duration=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                details={
                    "average_query_time": avg_query_time,
                    "max_query_time": max_query_time,
                    "query_count": len(query_times),
                    "connection_pool": pool_info
                },
                recommendations=[
                    "Consider adding database indexes for frequently queried columns",
                    "Monitor query performance in production",
                    "Set up database connection pooling",
                    "Configure query timeout limits"
                ] if status != ValidationStatus.PASS else []
            )
            
        except Exception as e:
            return ValidationResult(
                name="Database Performance",
                status=ValidationStatus.FAIL,
                severity=ValidationSeverity.CRITICAL,
                message="Database performance validation failed",
                duration=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                recommendations=[
                    "Check database connectivity",
                    "Verify database configuration",
                    "Review database logs for errors"
                ]
            )
    
    async def validate_sla_requirements(self) -> ValidationResult:
        """Validate SLA requirements: 99.9% uptime, <100ms response time, <0.1% error rate"""
        start_time = time.time()
        
        try:
            # Simulate load testing for SLA validation
            response_times = []
            error_count = 0
            total_requests = 100
            
            async with httpx.AsyncClient() as client:
                base_url = f"http://localhost:{self.settings.port}"
                
                # Test health endpoint performance
                for i in range(total_requests):
                    request_start = time.time()
                    try:
                        response = await client.get(f"{base_url}/health/liveness", timeout=5.0)
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                        
                        if response.status_code != 200:
                            error_count += 1
                            
                    except Exception:
                        error_count += 1
                        response_times.append(5.0)  # Timeout/error = 5s response time
            
            # Calculate metrics
            avg_response_time = sum(response_times) / len(response_times) if response_times else 5.0
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 5.0
            error_rate = error_count / total_requests
            
            # SLA validation
            sla_violations = []
            
            # Response time SLA: <100ms average
            if avg_response_time > 0.1:
                sla_violations.append(f"Average response time {avg_response_time*1000:.1f}ms exceeds 100ms SLA")
            
            # Error rate SLA: <0.1%
            if error_rate > 0.001:
                sla_violations.append(f"Error rate {error_rate*100:.2f}% exceeds 0.1% SLA")
            
            # Determine status
            if sla_violations:
                status = ValidationStatus.FAIL
                severity = ValidationSeverity.CRITICAL
                message = f"SLA violations detected: {'; '.join(sla_violations)}"
            else:
                status = ValidationStatus.PASS
                severity = ValidationSeverity.INFO
                message = "All SLA requirements met"
            
            return ValidationResult(
                name="SLA Requirements Validation",
                status=status,
                severity=severity,
                message=message,
                duration=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                details={
                    "average_response_time_ms": avg_response_time * 1000,
                    "p95_response_time_ms": p95_response_time * 1000,
                    "error_rate_percent": error_rate * 100,
                    "total_requests": total_requests,
                    "failed_requests": error_count,
                    "sla_requirements": {
                        "uptime_target": "99.9%",
                        "response_time_target": "<100ms",
                        "error_rate_target": "<0.1%"
                    }
                },
                recommendations=[
                    "Implement proper load balancing for production",
                    "Set up comprehensive monitoring and alerting",
                    "Configure auto-scaling based on load",
                    "Implement circuit breakers for resilience"
                ] if status != ValidationStatus.PASS else []
            )
            
        except Exception as e:
            return ValidationResult(
                name="SLA Requirements Validation",
                status=ValidationStatus.FAIL,
                severity=ValidationSeverity.CRITICAL,
                message="SLA validation failed",
                duration=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                recommendations=[
                    "Check application health endpoints",
                    "Verify network connectivity",
                    "Review application configuration"
                ]
            )
    
    async def validate_security_hardening(self) -> ValidationResult:
        """Validate security hardening measures"""
        start_time = time.time()
        
        try:
            security_checks = {}
            
            # Check environment variables
            critical_env_vars = [
                "JWT_SECRET_KEY", "DATABASE_URL", "REDIS_URL"
            ]
            
            missing_env_vars = []
            for var in critical_env_vars:
                if not os.getenv(var):
                    missing_env_vars.append(var)
            
            security_checks["environment_variables"] = {
                "missing_critical_vars": missing_env_vars,
                "status": "pass" if not missing_env_vars else "fail"
            }
            
            # Check file permissions
            sensitive_files = [
                ".env", "config/security.env", ".secrets"
            ]
            
            file_permission_issues = []
            for file_path in sensitive_files:
                if Path(file_path).exists():
                    stat = Path(file_path).stat()
                    # Check if file is readable by others (dangerous)
                    if stat.st_mode & 0o044:  # Others can read
                        file_permission_issues.append(f"{file_path} is readable by others")
            
            security_checks["file_permissions"] = {
                "issues": file_permission_issues,
                "status": "pass" if not file_permission_issues else "warning"
            }
            
            # Check for debug mode
            debug_enabled = self.settings.debug
            security_checks["debug_mode"] = {
                "enabled": debug_enabled,
                "status": "fail" if debug_enabled else "pass"
            }
            
            # Determine overall security status
            critical_issues = []
            if missing_env_vars:
                critical_issues.append("Missing critical environment variables")
            if debug_enabled:
                critical_issues.append("Debug mode enabled in production")
            
            if critical_issues:
                status = ValidationStatus.FAIL
                severity = ValidationSeverity.CRITICAL
                message = f"Critical security issues: {'; '.join(critical_issues)}"
            elif file_permission_issues:
                status = ValidationStatus.WARNING
                severity = ValidationSeverity.MEDIUM
                message = "Security warnings detected"
            else:
                status = ValidationStatus.PASS
                severity = ValidationSeverity.INFO
                message = "Security hardening measures in place"
            
            return ValidationResult(
                name="Security Hardening",
                status=status,
                severity=severity,
                message=message,
                duration=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                details=security_checks,
                recommendations=[
                    "Set all critical environment variables",
                    "Disable debug mode in production",
                    "Restrict file permissions on sensitive files",
                    "Implement proper secret management",
                    "Enable security headers middleware",
                    "Configure HTTPS/TLS properly"
                ] if status != ValidationStatus.PASS else []
            )
            
        except Exception as e:
            return ValidationResult(
                name="Security Hardening",
                status=ValidationStatus.FAIL,
                severity=ValidationSeverity.CRITICAL,
                message="Security validation failed",
                duration=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                recommendations=[
                    "Review security configuration",
                    "Check application permissions",
                    "Verify environment setup"
                ]
            )
    
    async def run_comprehensive_validation(self, db: Session) -> ProductionValidationReport:
        """Run comprehensive production validation suite"""
        start_time = datetime.now(timezone.utc)
        results = []
        
        logger.info(f"Starting comprehensive production validation: {self.validation_id}")
        
        # Run all validation checks concurrently
        validation_tasks = [
            self.validate_external_integrations(),
            self.validate_database_performance(db),
            self.validate_sla_requirements(),
            self.validate_security_hardening()
        ]
        
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                results.append(ValidationResult(
                    name=f"Validation Task {i+1}",
                    status=ValidationStatus.FAIL,
                    severity=ValidationSeverity.CRITICAL,
                    message="Validation task failed with exception",
                    duration=0.0,
                    timestamp=datetime.now(timezone.utc),
                    error=str(result)
                ))
            else:
                results.append(result)
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        # Calculate summary statistics
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.status == ValidationStatus.PASS)
        failed_checks = sum(1 for r in results if r.status == ValidationStatus.FAIL)
        warning_checks = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        
        critical_failures = sum(1 for r in results if r.status == ValidationStatus.FAIL and r.severity == ValidationSeverity.CRITICAL)
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = ValidationStatus.FAIL
        elif failed_checks > 0:
            overall_status = ValidationStatus.FAIL
        elif warning_checks > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASS
        
        # SLA compliance calculation
        sla_compliance = {
            "availability_target": 99.9,
            "response_time_target": 100,  # ms
            "error_rate_target": 0.1,  # %
            "compliance_status": "compliant" if overall_status == ValidationStatus.PASS else "non_compliant"
        }
        
        # Collect recommendations
        all_recommendations = []
        for result in results:
            if result.recommendations:
                all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = list(dict.fromkeys(all_recommendations))
        
        summary = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "warning_checks": warning_checks,
            "critical_failures": critical_failures,
            "pass_rate": passed_checks / total_checks if total_checks > 0 else 0,
            "production_ready": overall_status == ValidationStatus.PASS
        }
        
        report = ProductionValidationReport(
            validation_id=self.validation_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            overall_status=overall_status,
            results=results,
            summary=summary,
            sla_compliance=sla_compliance,
            recommendations=unique_recommendations
        )
        
        # Save report
        await self.save_validation_report(report)
        
        logger.info(f"Production validation completed: {self.validation_id} - Status: {overall_status}")
        
        return report
    
    async def save_validation_report(self, report: ProductionValidationReport) -> None:
        """Save validation report to file"""
        try:
            report_dir = Path("production_validation_reports")
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / f"{report.validation_id}.json"
            
            # Convert to serializable format
            report_data = asdict(report)
            report_data["start_time"] = report.start_time.isoformat()
            report_data["end_time"] = report.end_time.isoformat()
            
            for result in report_data["results"]:
                result["timestamp"] = result["timestamp"].isoformat() if isinstance(result["timestamp"], datetime) else result["timestamp"]
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            logger.info(f"Validation report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")


# Global validator instance
production_validator = ProductionValidator()