"""
Production Validation API Endpoints
REST API endpoints for comprehensive production validation
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.validation.production_validator import production_validator, ProductionValidationReport
from src.monitoring.sla_monitor import sla_monitor, SLAMetrics
from src.testing.production_smoke_tests import smoke_test_runner, SmokeTestSuite
from src.database.disaster_recovery import disaster_recovery, BackupMetadata
from src.api.v1.health import health_checker

router = APIRouter(prefix="/production-validation", tags=["production-validation"])


class ValidationRequest(BaseModel):
    """Request model for production validation"""
    include_integration_tests: bool = True
    include_performance_tests: bool = True
    include_security_audit: bool = True
    include_backup_test: bool = True


class SLAMonitoringRequest(BaseModel):
    """Request model for SLA monitoring"""
    duration_hours: int = 24
    enable_alerts: bool = True


class BackupRequest(BaseModel):
    """Request model for backup operations"""
    backup_type: str = "full"
    include_config: bool = True


class ProductionDashboard(BaseModel):
    """Production readiness dashboard data"""
    timestamp: datetime
    overall_status: str
    deployment_ready: bool
    health_status: Dict[str, Any]
    sla_metrics: Optional[Dict[str, Any]]
    recent_validations: List[Dict[str, Any]]
    recent_smoke_tests: List[Dict[str, Any]]
    backup_status: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]


@router.get("/dashboard", response_model=ProductionDashboard)
async def get_production_dashboard(db: Session = Depends(get_db)):
    """Get comprehensive production readiness dashboard"""
    try:
        # Get health status
        health_status = await health_checker.health_check(db)
        health_data = health_status.dict() if hasattr(health_status, 'dict') else health_status.__dict__
        
        # Get SLA metrics summary
        sla_summary = await sla_monitor.get_metrics_summary(hours=24)
        
        # Get recent backup status
        recent_backups = await disaster_recovery.list_backups()
        backup_status = {
            "last_backup": recent_backups[0].timestamp.isoformat() if recent_backups else None,
            "backup_count": len(recent_backups),
            "latest_backup_size_mb": recent_backups[0].size_bytes / (1024*1024) if recent_backups else 0,
            "backup_health": "healthy" if recent_backups and recent_backups[0].status == "success" else "warning"
        }
        
        # Determine overall status
        health_healthy = health_data.get("status") == "healthy"
        sla_compliant = sla_summary.get("overall_sla_compliance", {}).get("compliant", False)
        backup_healthy = backup_status["backup_health"] == "healthy"
        
        if health_healthy and sla_compliant and backup_healthy:
            overall_status = "healthy"
            deployment_ready = True
        elif health_healthy and sla_compliant:
            overall_status = "warning"
            deployment_ready = True
        else:
            overall_status = "unhealthy"
            deployment_ready = False
        
        # Collect recommendations
        recommendations = []
        if not health_healthy:
            recommendations.append("Address health check issues")
        if not sla_compliant:
            recommendations.append("Improve SLA compliance metrics")
        if not backup_healthy:
            recommendations.append("Create fresh backup")
        
        return ProductionDashboard(
            timestamp=datetime.now(timezone.utc),
            overall_status=overall_status,
            deployment_ready=deployment_ready,
            health_status=health_data,
            sla_metrics=sla_summary,
            recent_validations=[],  # Would be populated from validation history
            recent_smoke_tests=[],  # Would be populated from smoke test history
            backup_status=backup_status,
            alerts=[],  # Would be populated from active alerts
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")


@router.post("/validate")
async def run_production_validation(
    request: ValidationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Run comprehensive production validation"""
    try:
        # Start validation in background
        validation_task = production_validator.run_comprehensive_validation(db)
        
        # Return immediate response with validation ID
        return JSONResponse(
            status_code=202,
            content={
                "message": "Production validation started",
                "validation_id": production_validator.validation_id,
                "status": "in_progress",
                "estimated_duration_minutes": 5
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation startup failed: {str(e)}")


@router.post("/smoke-tests")
async def run_smoke_tests(background_tasks: BackgroundTasks):
    """Run production smoke tests"""
    try:
        # Run smoke tests
        suite = await smoke_test_runner.run_comprehensive_smoke_tests()
        
        return JSONResponse(
            content={
                "message": "Smoke tests completed",
                "suite_id": suite.suite_id,
                "overall_status": suite.overall_status,
                "deployment_ready": suite.deployment_ready,
                "summary": {
                    "total_tests": suite.total_tests,
                    "passed_tests": suite.passed_tests,
                    "failed_tests": suite.failed_tests,
                    "warning_tests": suite.warning_tests,
                    "critical_failures": suite.critical_failures
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smoke test execution failed: {str(e)}")


@router.get("/sla-metrics")
async def get_sla_metrics(hours: int = 24):
    """Get SLA metrics summary"""
    try:
        summary = await sla_monitor.get_metrics_summary(hours=hours)
        return JSONResponse(content=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SLA metrics retrieval failed: {str(e)}")


@router.post("/sla-load-test")
async def run_sla_load_test(duration_seconds: int = 300, requests_per_second: int = 10):
    """Run SLA load testing"""
    try:
        # Validate parameters
        if duration_seconds > 3600:  # 1 hour max
            raise HTTPException(status_code=400, detail="Duration cannot exceed 3600 seconds")
        
        if requests_per_second > 100:  # 100 RPS max
            raise HTTPException(status_code=400, detail="Requests per second cannot exceed 100")
        
        # Run load test
        test_results = await sla_monitor.simulate_load_for_testing(
            duration_seconds=duration_seconds,
            requests_per_second=requests_per_second
        )
        
        return JSONResponse(content={
            "message": "SLA load test completed",
            "results": test_results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SLA load test failed: {str(e)}")


@router.post("/backup")
async def create_backup(request: BackupRequest):
    """Create system backup"""
    try:
        if request.backup_type == "full":
            backup_results = await disaster_recovery.create_full_system_backup()
            
            return JSONResponse(content={
                "message": "Full system backup completed",
                "backup_results": {
                    "database": {
                        "backup_id": backup_results["database"].backup_id if backup_results["database"] else None,
                        "status": backup_results["database"].status if backup_results["database"] else "failed",
                        "size_mb": backup_results["database"].size_bytes / (1024*1024) if backup_results["database"] else 0
                    },
                    "configuration": {
                        "backup_id": backup_results["configuration"].backup_id if backup_results["configuration"] else None,
                        "status": backup_results["configuration"].status if backup_results["configuration"] else "failed",
                        "size_mb": backup_results["configuration"].size_bytes / (1024*1024) if backup_results["configuration"] else 0
                    }
                }
            })
        else:
            # Database only backup
            backup_result = await disaster_recovery.backup_database()
            
            return JSONResponse(content={
                "message": "Database backup completed",
                "backup_id": backup_result.backup_id,
                "status": backup_result.status,
                "size_mb": backup_result.size_bytes / (1024*1024),
                "duration_seconds": backup_result.duration_seconds
            })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup creation failed: {str(e)}")


@router.get("/backups")
async def list_backups():
    """List all available backups"""
    try:
        backups = await disaster_recovery.list_backups()
        
        backup_list = []
        for backup in backups:
            backup_list.append({
                "backup_id": backup.backup_id,
                "backup_type": backup.backup_type,
                "timestamp": backup.timestamp.isoformat(),
                "status": backup.status,
                "size_mb": backup.size_bytes / (1024*1024),
                "duration_seconds": backup.duration_seconds,
                "tables_count": len(backup.tables_backed_up)
            })
        
        return JSONResponse(content={
            "backups": backup_list,
            "total_count": len(backup_list)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup listing failed: {str(e)}")


@router.post("/disaster-recovery-test")
async def test_disaster_recovery():
    """Test disaster recovery procedures"""
    try:
        test_results = await disaster_recovery.test_disaster_recovery()
        
        return JSONResponse(content={
            "message": "Disaster recovery test completed",
            "test_id": test_results["test_id"],
            "overall_status": test_results["summary"]["overall_status"],
            "results": test_results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disaster recovery test failed: {str(e)}")


@router.post("/restore/{backup_id}")
async def restore_from_backup(backup_id: str):
    """Restore system from backup"""
    try:
        # Note: This is a dangerous operation and should require additional authentication
        restore_result = await disaster_recovery.restore_database(backup_id)
        
        return JSONResponse(content={
            "message": "Database restore completed",
            "restore_id": restore_result.restore_id,
            "backup_id": restore_result.backup_id,
            "status": restore_result.status,
            "tables_restored": len(restore_result.tables_restored),
            "records_restored": restore_result.records_restored,
            "duration_seconds": restore_result.duration_seconds
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore operation failed: {str(e)}")


@router.get("/quantum-status")
async def get_quantum_intelligence_status(db: Session = Depends(get_db)):
    """Get quantum intelligence modules status"""
    try:
        # Get quantum status from health check
        quantum_check = await health_checker.check_quantum_intelligence_modules()
        
        return JSONResponse(content={
            "status": quantum_check.status,
            "message": quantum_check.message,
            "response_time_ms": quantum_check.response_time * 1000 if quantum_check.response_time else 0,
            "details": quantum_check.details
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantum status check failed: {str(e)}")


@router.get("/deployment-readiness")
async def check_deployment_readiness(db: Session = Depends(get_db)):
    """Check if system is ready for production deployment"""
    try:
        # Run all critical checks
        health_status = await health_checker.health_check(db)
        smoke_suite = await smoke_test_runner.run_comprehensive_smoke_tests()
        sla_summary = await sla_monitor.get_metrics_summary(hours=1)  # Last hour only
        
        # Determine deployment readiness
        health_ready = health_status.status == "healthy"
        smoke_ready = smoke_suite.deployment_ready
        sla_ready = sla_summary.get("overall_sla_compliance", {}).get("compliant", False)
        
        deployment_ready = health_ready and smoke_ready and sla_ready
        
        # Generate deployment checklist
        checklist = {
            "health_checks": {
                "status": "pass" if health_ready else "fail",
                "details": "All health checks passing" if health_ready else "Health check issues detected"
            },
            "smoke_tests": {
                "status": "pass" if smoke_ready else "fail",
                "details": f"{smoke_suite.passed_tests}/{smoke_suite.total_tests} tests passed"
            },
            "sla_compliance": {
                "status": "pass" if sla_ready else "fail",
                "details": "SLA requirements met" if sla_ready else "SLA requirements not met"
            }
        }
        
        return JSONResponse(content={
            "deployment_ready": deployment_ready,
            "overall_status": "ready" if deployment_ready else "not_ready",
            "checklist": checklist,
            "recommendations": [
                "All systems operational - ready for deployment"
            ] if deployment_ready else [
                "Address health check issues",
                "Fix failing smoke tests",
                "Improve SLA compliance"
            ]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment readiness check failed: {str(e)}")


@router.post("/cleanup-old-data")
async def cleanup_old_data(retention_days: int = 30):
    """Clean up old validation data and backups"""
    try:
        # Cleanup old backups
        cleaned_backups = await disaster_recovery.cleanup_old_backups(retention_days)
        
        # Note: Additional cleanup for validation reports, smoke test results, etc. would go here
        
        return JSONResponse(content={
            "message": "Cleanup completed",
            "cleaned_backups": cleaned_backups,
            "retention_days": retention_days
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup operation failed: {str(e)}")


@router.get("/validation-history")
async def get_validation_history(limit: int = 10):
    """Get recent validation history"""
    try:
        # This would typically query a database or read from files
        # For now, return a placeholder response
        return JSONResponse(content={
            "message": "Validation history endpoint - implementation needed",
            "limit": limit,
            "validations": []
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation history retrieval failed: {str(e)}")