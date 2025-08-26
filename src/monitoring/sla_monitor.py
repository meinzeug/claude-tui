"""
SLA Monitoring and Validation System
Ensures 99.9% uptime, <100ms response time, and <0.1% error rate
"""

import asyncio
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import statistics
import json
import httpx
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)


class SLAStatus(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    BREACH = "breach"


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SLAMetrics:
    """SLA metrics tracking"""
    timestamp: datetime
    uptime_percentage: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate_percentage: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    availability_status: SLAStatus
    performance_status: SLAStatus
    error_rate_status: SLAStatus
    overall_status: SLAStatus


@dataclass
class SLAAlert:
    """SLA alert notification"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    metric: str
    current_value: float
    threshold_value: float
    message: str
    recommendations: List[str]


class SLAMonitor:
    """Comprehensive SLA monitoring and validation"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1440)  # 24 hours of minute-by-minute data
        self.response_times = deque(maxlen=10000)  # Last 10k requests
        self.error_counts = defaultdict(int)
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = datetime.now(timezone.utc)
        self.last_downtime = None
        self.downtime_periods = []
        
        # SLA thresholds
        self.sla_thresholds = {
            "uptime_target": 99.9,           # 99.9% uptime
            "response_time_target": 100,      # <100ms average
            "p95_response_time_target": 200,  # <200ms 95th percentile
            "p99_response_time_target": 500,  # <500ms 99th percentile
            "error_rate_target": 0.1         # <0.1% error rate
        }
        
        # Alert thresholds (when to send alerts before SLA breach)
        self.alert_thresholds = {
            "uptime_warning": 99.95,     # Alert at 99.95%
            "response_time_warning": 80,  # Alert at 80ms
            "error_rate_warning": 0.05   # Alert at 0.05%
        }
        
        self.active_alerts = {}
        self.metrics_file = Path("sla_metrics.json")
        
    async def record_request(self, response_time_ms: float, success: bool) -> None:
        """Record a request for SLA tracking"""
        self.total_requests += 1
        self.response_times.append(response_time_ms)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
    async def record_downtime_start(self) -> None:
        """Record the start of a downtime period"""
        self.last_downtime = datetime.now(timezone.utc)
        logger.warning("Downtime period started")
        
    async def record_downtime_end(self) -> None:
        """Record the end of a downtime period"""
        if self.last_downtime:
            downtime_duration = datetime.now(timezone.utc) - self.last_downtime
            self.downtime_periods.append({
                "start": self.last_downtime,
                "end": datetime.now(timezone.utc),
                "duration_seconds": downtime_duration.total_seconds()
            })
            self.last_downtime = None
            logger.info(f"Downtime period ended, duration: {downtime_duration.total_seconds():.2f}s")
    
    async def calculate_uptime_percentage(self) -> float:
        """Calculate current uptime percentage"""
        now = datetime.now(timezone.utc)
        total_time = (now - self.start_time).total_seconds()
        
        if total_time == 0:
            return 100.0
            
        total_downtime = sum(period["duration_seconds"] for period in self.downtime_periods)
        
        # Add current downtime if system is down
        if self.last_downtime:
            current_downtime = (now - self.last_downtime).total_seconds()
            total_downtime += current_downtime
            
        uptime_seconds = total_time - total_downtime
        uptime_percentage = (uptime_seconds / total_time) * 100
        
        return min(100.0, max(0.0, uptime_percentage))
    
    async def calculate_response_time_metrics(self) -> Dict[str, float]:
        """Calculate response time percentiles"""
        if not self.response_times:
            return {
                "average": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0
            }
            
        sorted_times = sorted(self.response_times)
        
        return {
            "average": statistics.mean(sorted_times),
            "p95": sorted_times[int(len(sorted_times) * 0.95)],
            "p99": sorted_times[int(len(sorted_times) * 0.99)],
            "min": min(sorted_times),
            "max": max(sorted_times)
        }
    
    async def calculate_error_rate(self) -> float:
        """Calculate current error rate percentage"""
        if self.total_requests == 0:
            return 0.0
            
        return (self.failed_requests / self.total_requests) * 100
    
    async def get_current_metrics(self) -> SLAMetrics:
        """Get current SLA metrics"""
        uptime_percentage = await self.calculate_uptime_percentage()
        response_metrics = await self.calculate_response_time_metrics()
        error_rate = await self.calculate_error_rate()
        
        # Determine status for each SLA component
        availability_status = self.get_sla_status(uptime_percentage, self.sla_thresholds["uptime_target"], higher_is_better=True)
        performance_status = self.get_sla_status(response_metrics["average"], self.sla_thresholds["response_time_target"], higher_is_better=False)
        error_rate_status = self.get_sla_status(error_rate, self.sla_thresholds["error_rate_target"], higher_is_better=False)
        
        # Overall status is the worst of all components
        overall_status = self.get_worst_status([availability_status, performance_status, error_rate_status])
        
        return SLAMetrics(
            timestamp=datetime.now(timezone.utc),
            uptime_percentage=uptime_percentage,
            avg_response_time_ms=response_metrics["average"],
            p95_response_time_ms=response_metrics["p95"],
            p99_response_time_ms=response_metrics["p99"],
            error_rate_percentage=error_rate,
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            availability_status=availability_status,
            performance_status=performance_status,
            error_rate_status=error_rate_status,
            overall_status=overall_status
        )
    
    def get_sla_status(self, current_value: float, threshold: float, higher_is_better: bool = True) -> SLAStatus:
        """Determine SLA status based on current value vs threshold"""
        if higher_is_better:
            if current_value >= threshold:
                return SLAStatus.COMPLIANT
            elif current_value >= threshold * 0.99:  # Within 1% of threshold
                return SLAStatus.WARNING
            else:
                return SLAStatus.BREACH
        else:
            if current_value <= threshold:
                return SLAStatus.COMPLIANT
            elif current_value <= threshold * 1.5:  # Within 50% of threshold
                return SLAStatus.WARNING
            else:
                return SLAStatus.BREACH
    
    def get_worst_status(self, statuses: List[SLAStatus]) -> SLAStatus:
        """Get the worst status from a list of statuses"""
        if SLAStatus.BREACH in statuses:
            return SLAStatus.BREACH
        elif SLAStatus.WARNING in statuses:
            return SLAStatus.WARNING
        else:
            return SLAStatus.COMPLIANT
    
    async def check_and_generate_alerts(self, metrics: SLAMetrics) -> List[SLAAlert]:
        """Check metrics and generate alerts if thresholds are breached"""
        alerts = []
        
        # Uptime alert
        if metrics.uptime_percentage < self.alert_thresholds["uptime_warning"]:
            alert = SLAAlert(
                alert_id=f"uptime_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                level=AlertLevel.CRITICAL if metrics.uptime_percentage < self.sla_thresholds["uptime_target"] else AlertLevel.WARNING,
                metric="uptime",
                current_value=metrics.uptime_percentage,
                threshold_value=self.sla_thresholds["uptime_target"],
                message=f"Uptime {metrics.uptime_percentage:.2f}% is below target {self.sla_thresholds['uptime_target']}%",
                recommendations=[
                    "Check system health and resource utilization",
                    "Review recent deployments or configuration changes",
                    "Investigate infrastructure issues",
                    "Consider implementing auto-scaling"
                ]
            )
            alerts.append(alert)
        
        # Response time alert
        if metrics.avg_response_time_ms > self.alert_thresholds["response_time_warning"]:
            alert = SLAAlert(
                alert_id=f"response_time_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                level=AlertLevel.CRITICAL if metrics.avg_response_time_ms > self.sla_thresholds["response_time_target"] else AlertLevel.WARNING,
                metric="response_time",
                current_value=metrics.avg_response_time_ms,
                threshold_value=self.sla_thresholds["response_time_target"],
                message=f"Average response time {metrics.avg_response_time_ms:.1f}ms exceeds target {self.sla_thresholds['response_time_target']}ms",
                recommendations=[
                    "Check database query performance",
                    "Review application resource usage",
                    "Optimize slow API endpoints",
                    "Consider caching implementation",
                    "Scale application instances"
                ]
            )
            alerts.append(alert)
        
        # Error rate alert
        if metrics.error_rate_percentage > self.alert_thresholds["error_rate_warning"]:
            alert = SLAAlert(
                alert_id=f"error_rate_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                level=AlertLevel.CRITICAL if metrics.error_rate_percentage > self.sla_thresholds["error_rate_target"] else AlertLevel.WARNING,
                metric="error_rate",
                current_value=metrics.error_rate_percentage,
                threshold_value=self.sla_thresholds["error_rate_target"],
                message=f"Error rate {metrics.error_rate_percentage:.2f}% exceeds target {self.sla_thresholds['error_rate_target']}%",
                recommendations=[
                    "Review application logs for error patterns",
                    "Check external dependency health",
                    "Validate input validation and error handling",
                    "Monitor database connection pool",
                    "Investigate recent code deployments"
                ]
            )
            alerts.append(alert)
        
        return alerts
    
    async def update_metrics_history(self) -> None:
        """Update metrics history with current snapshot"""
        current_metrics = await self.get_current_metrics()
        self.metrics_history.append(current_metrics)
        
        # Generate alerts
        alerts = await self.check_and_generate_alerts(current_metrics)
        for alert in alerts:
            self.active_alerts[alert.alert_id] = alert
            logger.warning(f"SLA Alert: {alert.message}")
        
        # Save metrics to file
        await self.save_metrics_to_file(current_metrics)
    
    async def save_metrics_to_file(self, metrics: SLAMetrics) -> None:
        """Save current metrics to file"""
        try:
            metrics_data = asdict(metrics)
            metrics_data["timestamp"] = metrics.timestamp.isoformat()
            
            # Load existing data
            existing_data = []
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Add new metrics
            existing_data.append(metrics_data)
            
            # Keep only last 24 hours of data
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            existing_data = [
                data for data in existing_data 
                if datetime.fromisoformat(data["timestamp"]) > cutoff_time
            ]
            
            # Save updated data
            with open(self.metrics_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save SLA metrics: {e}")
    
    async def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get SLA metrics summary for specified time period"""
        try:
            if not self.metrics_file.exists():
                return {"error": "No metrics data available"}
            
            with open(self.metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # Filter data for specified time period
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            filtered_data = [
                data for data in metrics_data 
                if datetime.fromisoformat(data["timestamp"]) > cutoff_time
            ]
            
            if not filtered_data:
                return {"error": "No metrics data for specified time period"}
            
            # Calculate summary statistics
            uptimes = [data["uptime_percentage"] for data in filtered_data]
            response_times = [data["avg_response_time_ms"] for data in filtered_data]
            error_rates = [data["error_rate_percentage"] for data in filtered_data]
            
            summary = {
                "time_period_hours": hours,
                "data_points": len(filtered_data),
                "uptime": {
                    "average": statistics.mean(uptimes),
                    "minimum": min(uptimes),
                    "current": filtered_data[-1]["uptime_percentage"],
                    "sla_compliant": all(uptime >= self.sla_thresholds["uptime_target"] for uptime in uptimes)
                },
                "response_time": {
                    "average_ms": statistics.mean(response_times),
                    "maximum_ms": max(response_times),
                    "current_ms": filtered_data[-1]["avg_response_time_ms"],
                    "sla_compliant": all(rt <= self.sla_thresholds["response_time_target"] for rt in response_times)
                },
                "error_rate": {
                    "average_percentage": statistics.mean(error_rates),
                    "maximum_percentage": max(error_rates),
                    "current_percentage": filtered_data[-1]["error_rate_percentage"],
                    "sla_compliant": all(er <= self.sla_thresholds["error_rate_target"] for er in error_rates)
                },
                "overall_sla_compliance": {
                    "compliant": all([
                        all(uptime >= self.sla_thresholds["uptime_target"] for uptime in uptimes),
                        all(rt <= self.sla_thresholds["response_time_target"] for rt in response_times),
                        all(er <= self.sla_thresholds["error_rate_target"] for er in error_rates)
                    ]),
                    "current_status": filtered_data[-1]["overall_status"]
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate metrics summary: {e}")
            return {"error": str(e)}
    
    async def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous SLA monitoring"""
        logger.info(f"Starting SLA monitoring with {interval_seconds}s interval")
        
        while True:
            try:
                await self.update_metrics_history()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in SLA monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def simulate_load_for_testing(self, duration_seconds: int = 300, requests_per_second: int = 10) -> Dict[str, Any]:
        """Simulate load for SLA testing purposes"""
        logger.info(f"Starting SLA load simulation: {requests_per_second} RPS for {duration_seconds}s")
        
        start_time = time.time()
        test_results = {
            "test_duration": duration_seconds,
            "target_rps": requests_per_second,
            "actual_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": []
        }
        
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < duration_seconds:
                batch_start = time.time()
                
                # Send batch of requests
                tasks = []
                for _ in range(requests_per_second):
                    task = self.make_test_request(client)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        test_results["failed_requests"] += 1
                        await self.record_request(5000, False)  # 5s timeout = failure
                    else:
                        test_results["actual_requests"] += 1
                        success, response_time = result
                        if success:
                            test_results["successful_requests"] += 1
                        else:
                            test_results["failed_requests"] += 1
                        
                        test_results["response_times"].append(response_time)
                        await self.record_request(response_time, success)
                
                # Wait for next second
                batch_duration = time.time() - batch_start
                if batch_duration < 1.0:
                    await asyncio.sleep(1.0 - batch_duration)
        
        # Calculate final statistics
        if test_results["response_times"]:
            test_results["avg_response_time"] = statistics.mean(test_results["response_times"])
            test_results["p95_response_time"] = sorted(test_results["response_times"])[int(len(test_results["response_times"]) * 0.95)]
        
        test_results["error_rate"] = (test_results["failed_requests"] / max(1, test_results["actual_requests"])) * 100
        
        logger.info(f"Load simulation completed: {test_results['actual_requests']} requests, {test_results['error_rate']:.2f}% error rate")
        return test_results
    
    async def make_test_request(self, client: httpx.AsyncClient) -> Tuple[bool, float]:
        """Make a test request for load simulation"""
        start_time = time.time()
        try:
            response = await client.get("http://localhost:8000/health/liveness", timeout=5.0)
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            success = response.status_code == 200
            return success, response_time
        except Exception:
            response_time = (time.time() - start_time) * 1000
            return False, response_time


# Global SLA monitor instance
sla_monitor = SLAMonitor()