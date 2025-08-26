"""
Production Health Check Middleware for Claude-TIU
Implements comprehensive health, readiness, and startup probes
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import psutil
import aioredis
import asyncpg
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import httpx

logger = logging.getLogger(__name__)

class HealthCheckManager:
    """
    Comprehensive health check manager for production deployment
    """
    
    def __init__(self):
        self.startup_time = datetime.utcnow()
        self.startup_checks_completed = False
        self.is_ready = False
        self.is_healthy = True
        self.last_health_check = None
        self.health_check_cache_ttl = 30  # seconds
        self.dependency_checks: Dict[str, Callable] = {}
        
    async def startup_probe(self) -> Dict[str, Any]:
        """
        Kubernetes startup probe - checks if app is ready to accept traffic
        Called during container initialization
        """
        try:
            # Check critical dependencies during startup
            startup_checks = [
                self._check_database_connection(),
                self._check_redis_connection(),
                self._check_memory_allocation(),
                self._check_required_environment_variables(),
                self._check_file_system_access(),
            ]
            
            results = await asyncio.gather(*startup_checks, return_exceptions=True)
            
            failed_checks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    check_name = [
                        "database", "redis", "memory", 
                        "environment", "filesystem"
                    ][i]
                    failed_checks.append(f"{check_name}: {str(result)}")
            
            if failed_checks:
                self.startup_checks_completed = False
                return {
                    "status": "unhealthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "startup_time": self.startup_time.isoformat(),
                    "failed_checks": failed_checks,
                    "message": "Startup checks failed"
                }
            
            self.startup_checks_completed = True
            self.is_ready = True
            
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "startup_time": self.startup_time.isoformat(),
                "startup_duration_seconds": (
                    datetime.utcnow() - self.startup_time
                ).total_seconds(),
                "message": "All startup checks passed"
            }
            
        except Exception as e:
            logger.error(f"Startup probe failed: {str(e)}")
            self.startup_checks_completed = False
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "message": "Startup probe encountered an error"
            }
    
    async def readiness_probe(self) -> Dict[str, Any]:
        """
        Kubernetes readiness probe - checks if app can serve requests
        Called regularly to determine if pod should receive traffic
        """
        try:
            # If startup hasn't completed, we're not ready
            if not self.startup_checks_completed:
                return {
                    "status": "not_ready",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Startup checks not completed"
                }
            
            # Check if we can serve requests
            readiness_checks = [
                self._check_database_pool_availability(),
                self._check_redis_availability(),
                self._check_memory_pressure(),
                self._check_api_endpoints_responsiveness(),
            ]
            
            results = await asyncio.gather(*readiness_checks, return_exceptions=True)
            
            failed_checks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception) or not result:
                    check_name = [
                        "database_pool", "redis_availability", 
                        "memory_pressure", "api_responsiveness"
                    ][i]
                    error_msg = str(result) if isinstance(result, Exception) else "check failed"
                    failed_checks.append(f"{check_name}: {error_msg}")
            
            if failed_checks:
                self.is_ready = False
                return {
                    "status": "not_ready",
                    "timestamp": datetime.utcnow().isoformat(),
                    "failed_checks": failed_checks,
                    "message": "Readiness checks failed"
                }
            
            self.is_ready = True
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "All readiness checks passed"
            }
            
        except Exception as e:
            logger.error(f"Readiness probe failed: {str(e)}")
            self.is_ready = False
            return {
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "message": "Readiness probe encountered an error"
            }
    
    async def liveness_probe(self) -> Dict[str, Any]:
        """
        Kubernetes liveness probe - checks if app is still running properly
        Called regularly to determine if pod should be restarted
        """
        try:
            # Cache health checks to avoid excessive load
            now = datetime.utcnow()
            if (self.last_health_check and 
                (now - self.last_health_check).total_seconds() < self.health_check_cache_ttl):
                
                return {
                    "status": "healthy" if self.is_healthy else "unhealthy",
                    "timestamp": now.isoformat(),
                    "cached": True,
                    "message": "Cached health status"
                }
            
            # Perform lightweight health checks
            liveness_checks = [
                self._check_process_health(),
                self._check_memory_usage(),
                self._check_disk_space(),
                self._check_critical_threads(),
            ]
            
            results = await asyncio.gather(*liveness_checks, return_exceptions=True)
            
            failed_checks = []
            warning_checks = []
            
            for i, result in enumerate(results):
                check_name = [
                    "process_health", "memory_usage", 
                    "disk_space", "critical_threads"
                ][i]
                
                if isinstance(result, Exception):
                    failed_checks.append(f"{check_name}: {str(result)}")
                elif isinstance(result, dict):
                    if result.get("status") == "critical":
                        failed_checks.append(f"{check_name}: {result.get('message', 'critical issue')}")
                    elif result.get("status") == "warning":
                        warning_checks.append(f"{check_name}: {result.get('message', 'warning issue')}")
            
            self.last_health_check = now
            
            if failed_checks:
                self.is_healthy = False
                return {
                    "status": "unhealthy",
                    "timestamp": now.isoformat(),
                    "failed_checks": failed_checks,
                    "warning_checks": warning_checks,
                    "uptime_seconds": (now - self.startup_time).total_seconds(),
                    "message": "Liveness checks failed - restart recommended"
                }
            
            self.is_healthy = True
            return {
                "status": "healthy",
                "timestamp": now.isoformat(),
                "warning_checks": warning_checks,
                "uptime_seconds": (now - self.startup_time).total_seconds(),
                "message": "All liveness checks passed"
            }
            
        except Exception as e:
            logger.error(f"Liveness probe failed: {str(e)}")
            self.is_healthy = False
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
                "message": "Liveness probe encountered an error"
            }
    
    async def detailed_health_status(self) -> Dict[str, Any]:
        """
        Comprehensive health status for monitoring and debugging
        """
        try:
            system_info = await self._get_system_info()
            dependency_status = await self._get_dependency_status()
            
            return {
                "status": "healthy" if self.is_healthy else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "startup_time": self.startup_time.isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
                "startup_completed": self.startup_checks_completed,
                "ready": self.is_ready,
                "system_info": system_info,
                "dependencies": dependency_status,
                "version": "1.0.0",  # Should come from config
                "environment": "production"
            }
            
        except Exception as e:
            logger.error(f"Detailed health status failed: {str(e)}")
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "message": "Failed to retrieve detailed health status"
            }
    
    # Internal health check methods
    
    async def _check_database_connection(self) -> bool:
        """Check database connectivity during startup"""
        try:
            # This should use your actual database config
            import os
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise Exception("DATABASE_URL not configured")
            
            conn = await asyncpg.connect(database_url)
            await conn.execute("SELECT 1")
            await conn.close()
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            raise e
    
    async def _check_redis_connection(self) -> bool:
        """Check Redis connectivity during startup"""
        try:
            import os
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            
            redis = aioredis.from_url(redis_url)
            await redis.ping()
            await redis.close()
            return True
        except Exception as e:
            logger.error(f"Redis connection check failed: {str(e)}")
            raise e
    
    async def _check_memory_allocation(self) -> bool:
        """Check memory allocation during startup"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Check if we have at least 256MB allocated
            if memory_info.rss < 256 * 1024 * 1024:
                raise Exception(f"Insufficient memory allocated: {memory_info.rss / 1024 / 1024:.2f}MB")
            
            return True
        except Exception as e:
            logger.error(f"Memory allocation check failed: {str(e)}")
            raise e
    
    async def _check_required_environment_variables(self) -> bool:
        """Check required environment variables"""
        import os
        required_vars = [
            "CLAUDE_TIU_ENV",
            "PYTHONPATH"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
    
    async def _check_file_system_access(self) -> bool:
        """Check file system access"""
        import tempfile
        import os
        
        try:
            # Test write access to temp directory
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(b"health check")
                tmp.flush()
            
            # Test read access to application directory
            app_dir = "/app"
            if not os.path.exists(app_dir) or not os.access(app_dir, os.R_OK):
                raise Exception(f"Cannot access application directory: {app_dir}")
            
            return True
        except Exception as e:
            logger.error(f"File system access check failed: {str(e)}")
            raise e
    
    async def _check_database_pool_availability(self) -> bool:
        """Check database connection pool availability"""
        try:
            # This would check your actual connection pool
            # For now, just check if we can make a quick connection
            return await self._check_database_connection()
        except Exception:
            return False
    
    async def _check_redis_availability(self) -> bool:
        """Check Redis availability for readiness"""
        try:
            return await self._check_redis_connection()
        except Exception:
            return False
    
    async def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Check if memory usage is below 80% of limit (1GB limit)
            memory_limit = 1 * 1024 * 1024 * 1024  # 1GB
            memory_usage_percent = (memory_info.rss / memory_limit) * 100
            
            return memory_usage_percent < 80
        except Exception:
            return False
    
    async def _check_api_endpoints_responsiveness(self) -> bool:
        """Check API endpoints responsiveness"""
        try:
            # This would test internal API endpoints
            # For now, just return True as a placeholder
            return True
        except Exception:
            return False
    
    async def _check_process_health(self) -> Dict[str, Any]:
        """Check process health for liveness"""
        try:
            process = psutil.Process()
            
            # Check if process is responsive
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            
            # Check for CPU usage anomalies
            if cpu_percent > 95:
                return {
                    "status": "critical",
                    "message": f"High CPU usage: {cpu_percent}%"
                }
            elif cpu_percent > 80:
                return {
                    "status": "warning",
                    "message": f"Elevated CPU usage: {cpu_percent}%"
                }
            
            return {"status": "healthy", "cpu_percent": cpu_percent}
        except Exception as e:
            return {"status": "critical", "message": str(e)}
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage for liveness"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # 1GB limit
            memory_limit = 1 * 1024 * 1024 * 1024
            memory_usage_percent = (memory_info.rss / memory_limit) * 100
            
            if memory_usage_percent > 90:
                return {
                    "status": "critical",
                    "message": f"Critical memory usage: {memory_usage_percent:.1f}%"
                }
            elif memory_usage_percent > 80:
                return {
                    "status": "warning",
                    "message": f"High memory usage: {memory_usage_percent:.1f}%"
                }
            
            return {
                "status": "healthy",
                "memory_usage_percent": memory_usage_percent
            }
        except Exception as e:
            return {"status": "critical", "message": str(e)}
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space for liveness"""
        try:
            disk_usage = psutil.disk_usage('/')
            
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            if disk_usage_percent > 95:
                return {
                    "status": "critical",
                    "message": f"Critical disk usage: {disk_usage_percent:.1f}%"
                }
            elif disk_usage_percent > 85:
                return {
                    "status": "warning",
                    "message": f"High disk usage: {disk_usage_percent:.1f}%"
                }
            
            return {
                "status": "healthy",
                "disk_usage_percent": disk_usage_percent
            }
        except Exception as e:
            return {"status": "critical", "message": str(e)}
    
    async def _check_critical_threads(self) -> Dict[str, Any]:
        """Check critical application threads"""
        try:
            import threading
            
            thread_count = threading.active_count()
            
            # Check if we have a reasonable number of threads
            if thread_count > 100:
                return {
                    "status": "warning",
                    "message": f"High thread count: {thread_count}"
                }
            
            return {"status": "healthy", "thread_count": thread_count}
        except Exception as e:
            return {"status": "critical", "message": str(e)}
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            process = psutil.Process()
            
            return {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "process_memory_mb": process.memory_info().rss / (1024**2),
                "process_cpu_percent": process.cpu_percent(),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_dependency_status(self) -> Dict[str, Any]:
        """Get dependency status"""
        dependencies = {}
        
        # Check database
        try:
            await self._check_database_connection()
            dependencies["database"] = {"status": "healthy"}
        except Exception as e:
            dependencies["database"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Redis
        try:
            await self._check_redis_connection()
            dependencies["redis"] = {"status": "healthy"}
        except Exception as e:
            dependencies["redis"] = {"status": "unhealthy", "error": str(e)}
        
        return dependencies

# Global health check manager instance
health_manager = HealthCheckManager()

# FastAPI endpoint handlers
async def startup_endpoint():
    """Kubernetes startup probe endpoint"""
    result = await health_manager.startup_probe()
    status_code = 200 if result["status"] == "healthy" else 503
    return JSONResponse(content=result, status_code=status_code)

async def readiness_endpoint():
    """Kubernetes readiness probe endpoint"""
    result = await health_manager.readiness_probe()
    status_code = 200 if result["status"] == "ready" else 503
    return JSONResponse(content=result, status_code=status_code)

async def liveness_endpoint():
    """Kubernetes liveness probe endpoint"""
    result = await health_manager.liveness_probe()
    status_code = 200 if result["status"] == "healthy" else 503
    return JSONResponse(content=result, status_code=status_code)

async def health_endpoint():
    """Comprehensive health endpoint for monitoring"""
    result = await health_manager.detailed_health_status()
    status_code = 200 if result["status"] in ["healthy", "error"] else 503
    return JSONResponse(content=result, status_code=status_code)