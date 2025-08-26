"""
API Gateway monitoring and analytics system.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import redis.asyncio as redis
from fastapi import Request, Response

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Individual request metrics."""
    timestamp: float
    method: str
    path: str
    status_code: int
    response_time: float
    request_size: int
    response_size: int
    client_ip: str
    user_agent: str
    auth_type: Optional[str]
    api_key_id: Optional[str]
    backend_server: Optional[str]
    error_message: Optional[str]


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over time period."""
    period_start: datetime
    period_end: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    total_bytes_in: int
    total_bytes_out: int
    unique_clients: int
    error_rate: float
    top_endpoints: Dict[str, int]
    top_errors: Dict[str, int]
    status_codes: Dict[int, int]


class MetricsCollector:
    """Real-time metrics collector."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.request_history = deque(maxlen=max_history)
        self.current_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'active_connections': 0,
            'avg_response_time': 0.0,
            'bytes_in': 0,
            'bytes_out': 0
        }
        
        # Real-time tracking
        self.response_times = deque(maxlen=1000)
        self.status_codes = defaultdict(int)
        self.endpoints = defaultdict(int)
        self.clients = set()
        self.errors = defaultdict(int)
    
    def record_request(self, metrics: RequestMetrics):
        """Record request metrics."""
        self.request_history.append(metrics)
        
        # Update current metrics
        self.current_metrics['total_requests'] += 1
        
        if 200 <= metrics.status_code < 400:
            self.current_metrics['successful_requests'] += 1
        else:
            self.current_metrics['failed_requests'] += 1
            if metrics.error_message:
                self.errors[metrics.error_message] += 1
        
        # Update response time
        self.response_times.append(metrics.response_time)
        if self.response_times:
            self.current_metrics['avg_response_time'] = sum(self.response_times) / len(self.response_times)
        
        # Update other metrics
        self.current_metrics['bytes_in'] += metrics.request_size
        self.current_metrics['bytes_out'] += metrics.response_size
        self.status_codes[metrics.status_code] += 1
        self.endpoints[f"{metrics.method} {metrics.path}"] += 1
        self.clients.add(metrics.client_ip)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        response_times_list = list(self.response_times)
        
        metrics = self.current_metrics.copy()
        metrics.update({
            'unique_clients': len(self.clients),
            'error_rate': self.current_metrics['failed_requests'] / max(self.current_metrics['total_requests'], 1) * 100,
            'p95_response_time': self._calculate_percentile(response_times_list, 0.95),
            'p99_response_time': self._calculate_percentile(response_times_list, 0.99),
            'top_endpoints': dict(sorted(self.endpoints.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_errors': dict(sorted(self.errors.items(), key=lambda x: x[1], reverse=True)[:10]),
            'status_codes': dict(self.status_codes),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return metrics
    
    def get_aggregated_metrics(self, start_time: datetime, end_time: datetime) -> AggregatedMetrics:
        """Get aggregated metrics for time period."""
        period_requests = [
            req for req in self.request_history
            if start_time.timestamp() <= req.timestamp <= end_time.timestamp()
        ]
        
        if not period_requests:
            return AggregatedMetrics(
                period_start=start_time,
                period_end=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                total_bytes_in=0,
                total_bytes_out=0,
                unique_clients=0,
                error_rate=0.0,
                top_endpoints={},
                top_errors={},
                status_codes={}
            )
        
        # Calculate aggregations
        total_requests = len(period_requests)
        successful = len([r for r in period_requests if 200 <= r.status_code < 400])
        failed = total_requests - successful
        
        response_times = [r.response_time for r in period_requests]
        avg_response_time = sum(response_times) / len(response_times)
        
        endpoints_count = defaultdict(int)
        errors_count = defaultdict(int)
        status_count = defaultdict(int)
        unique_clients = set()
        
        total_bytes_in = 0
        total_bytes_out = 0
        
        for req in period_requests:
            endpoints_count[f"{req.method} {req.path}"] += 1
            status_count[req.status_code] += 1
            unique_clients.add(req.client_ip)
            total_bytes_in += req.request_size
            total_bytes_out += req.response_size
            
            if req.error_message:
                errors_count[req.error_message] += 1
        
        return AggregatedMetrics(
            period_start=start_time,
            period_end=end_time,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            avg_response_time=avg_response_time,
            p95_response_time=self._calculate_percentile(response_times, 0.95),
            p99_response_time=self._calculate_percentile(response_times, 0.99),
            total_bytes_in=total_bytes_in,
            total_bytes_out=total_bytes_out,
            unique_clients=len(unique_clients),
            error_rate=(failed / total_requests) * 100,
            top_endpoints=dict(sorted(endpoints_count.items(), key=lambda x: x[1], reverse=True)[:10]),
            top_errors=dict(sorted(errors_count.items(), key=lambda x: x[1], reverse=True)[:10]),
            status_codes=dict(status_count)
        )
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        values_sorted = sorted(values)
        k = (len(values_sorted) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f == len(values_sorted) - 1:
            return values_sorted[f]
        
        return values_sorted[f] * (1 - c) + values_sorted[f + 1] * c


class GatewayMonitor:
    """Gateway monitoring system."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        metrics_retention_hours: int = 24,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_thresholds = alert_thresholds or {
            'error_rate_threshold': 5.0,  # 5% error rate
            'response_time_threshold': 2000.0,  # 2 seconds
            'active_connections_threshold': 1000
        }
        
        self.metrics_collector = MetricsCollector()
        self.alerts = []
        self.monitoring_task = None
    
    async def initialize(self):
        """Initialize monitoring system."""
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        # Start background monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Gateway Monitor initialized")
    
    async def shutdown(self):
        """Shutdown monitoring system."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def record_request(
        self,
        request: Request,
        response: Response,
        response_time: float,
        error_message: Optional[str] = None
    ):
        """Record request metrics."""
        # Extract request info
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get('user-agent', '')
        request_size = len(await request.body()) if hasattr(request, 'body') else 0
        response_size = len(response.body) if hasattr(response, 'body') else 0
        
        # Get auth info from request state if available
        auth_info = getattr(request.state, 'auth', {})
        auth_type = auth_info.get('type')
        api_key_id = auth_info.get('key_id')
        
        # Get backend server info
        backend_server = getattr(request.state, 'selected_backend', None)
        
        # Create metrics object
        metrics = RequestMetrics(
            timestamp=time.time(),
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            response_time=response_time,
            request_size=request_size,
            response_size=response_size,
            client_ip=client_ip,
            user_agent=user_agent,
            auth_type=auth_type,
            api_key_id=api_key_id,
            backend_server=backend_server,
            error_message=error_message
        )
        
        # Record in collector
        self.metrics_collector.record_request(metrics)
        
        # Store in Redis for persistence
        if self.redis_client:
            await self._store_metrics(metrics)
        
        # Check for alerts
        await self._check_alerts()
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        return self.metrics_collector.get_current_metrics()
    
    async def get_historical_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        resolution: str = "hour"
    ) -> List[AggregatedMetrics]:
        """Get historical metrics with specified resolution."""
        if not self.redis_client:
            return []
        
        try:
            # Get time buckets based on resolution
            time_buckets = self._generate_time_buckets(start_time, end_time, resolution)
            historical_metrics = []
            
            for bucket_start, bucket_end in time_buckets:
                # Try to get from Redis cache first
                cache_key = f"metrics:{resolution}:{bucket_start.isoformat()}"
                cached_data = await self.redis_client.get(cache_key)
                
                if cached_data:
                    metrics_data = json.loads(cached_data)
                    historical_metrics.append(AggregatedMetrics(**metrics_data))
                else:
                    # Calculate from raw data
                    metrics = self.metrics_collector.get_aggregated_metrics(bucket_start, bucket_end)
                    historical_metrics.append(metrics)
                    
                    # Cache the result
                    await self.redis_client.setex(
                        cache_key,
                        3600,  # 1 hour cache
                        json.dumps(asdict(metrics), default=str)
                    )
            
            return historical_metrics
            
        except Exception as e:
            logger.error(f"Historical metrics error: {e}")
            return []
    
    async def get_top_endpoints(self, limit: int = 10, time_window_hours: int = 1) -> Dict[str, int]:
        """Get top endpoints by request count."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        metrics = self.metrics_collector.get_aggregated_metrics(start_time, end_time)
        return dict(list(metrics.top_endpoints.items())[:limit])
    
    async def get_error_summary(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """Get error summary for time window."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        metrics = self.metrics_collector.get_aggregated_metrics(start_time, end_time)
        
        return {
            'total_errors': metrics.failed_requests,
            'error_rate': metrics.error_rate,
            'top_errors': metrics.top_errors,
            'status_codes': {k: v for k, v in metrics.status_codes.items() if k >= 400},
            'time_window': f"{time_window_hours} hours"
        }
    
    async def get_performance_summary(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for time window."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        metrics = self.metrics_collector.get_aggregated_metrics(start_time, end_time)
        
        return {
            'avg_response_time': metrics.avg_response_time,
            'p95_response_time': metrics.p95_response_time,
            'p99_response_time': metrics.p99_response_time,
            'total_requests': metrics.total_requests,
            'throughput': metrics.total_requests / max(time_window_hours, 1),  # requests per hour
            'bandwidth_in': metrics.total_bytes_in,
            'bandwidth_out': metrics.total_bytes_out,
            'unique_clients': metrics.unique_clients
        }
    
    async def get_alerts(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get system alerts."""
        if active_only:
            return [alert for alert in self.alerts if alert.get('status') == 'active']
        return self.alerts
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return 'unknown'
    
    async def _store_metrics(self, metrics: RequestMetrics):
        """Store metrics in Redis for persistence."""
        try:
            # Store individual request metrics
            timestamp = int(metrics.timestamp)
            key = f"request_metrics:{timestamp}:{id(metrics)}"
            
            await self.redis_client.setex(
                key,
                self.metrics_retention_hours * 3600,  # Convert to seconds
                json.dumps(asdict(metrics))
            )
            
            # Update hourly aggregates
            hour_key = f"hourly_metrics:{timestamp // 3600}"
            await self.redis_client.hincrby(hour_key, 'total_requests', 1)
            await self.redis_client.hincrbyfloat(hour_key, 'response_time_sum', metrics.response_time)
            await self.redis_client.expire(hour_key, self.metrics_retention_hours * 3600)
            
            if 200 <= metrics.status_code < 400:
                await self.redis_client.hincrby(hour_key, 'successful_requests', 1)
            else:
                await self.redis_client.hincrby(hour_key, 'failed_requests', 1)
            
        except Exception as e:
            logger.error(f"Metrics storage error: {e}")
    
    async def _check_alerts(self):
        """Check for alert conditions."""
        try:
            current_metrics = self.metrics_collector.get_current_metrics()
            
            # Check error rate threshold
            if current_metrics['error_rate'] > self.alert_thresholds['error_rate_threshold']:
                await self._create_alert(
                    'high_error_rate',
                    f"Error rate is {current_metrics['error_rate']:.2f}% (threshold: {self.alert_thresholds['error_rate_threshold']}%)",
                    'critical'
                )
            
            # Check response time threshold
            if current_metrics['p95_response_time'] > self.alert_thresholds['response_time_threshold']:
                await self._create_alert(
                    'high_response_time',
                    f"95th percentile response time is {current_metrics['p95_response_time']:.2f}ms (threshold: {self.alert_thresholds['response_time_threshold']}ms)",
                    'warning'
                )
            
            # Check active connections
            if current_metrics['active_connections'] > self.alert_thresholds['active_connections_threshold']:
                await self._create_alert(
                    'high_connections',
                    f"Active connections is {current_metrics['active_connections']} (threshold: {self.alert_thresholds['active_connections_threshold']})",
                    'warning'
                )
                
        except Exception as e:
            logger.error(f"Alert checking error: {e}")
    
    async def _create_alert(self, alert_type: str, message: str, severity: str):
        """Create a new alert."""
        alert = {
            'id': f"{alert_type}_{int(time.time())}",
            'type': alert_type,
            'message': message,
            'severity': severity,
            'status': 'active',
            'created_at': datetime.utcnow().isoformat(),
            'resolved_at': None
        }
        
        # Check if similar alert already exists
        existing = [a for a in self.alerts if a['type'] == alert_type and a['status'] == 'active']
        if not existing:
            self.alerts.append(alert)
            logger.warning(f"Alert created: {message}")
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.lpush('gateway_alerts', json.dumps(alert))
                await self.redis_client.ltrim('gateway_alerts', 0, 99)  # Keep last 100 alerts
    
    def _generate_time_buckets(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        resolution: str
    ) -> List[tuple[datetime, datetime]]:
        """Generate time buckets for aggregation."""
        buckets = []
        current = start_time
        
        if resolution == "minute":
            delta = timedelta(minutes=1)
        elif resolution == "hour":
            delta = timedelta(hours=1)
        elif resolution == "day":
            delta = timedelta(days=1)
        else:
            delta = timedelta(hours=1)  # Default to hour
        
        while current < end_time:
            bucket_end = min(current + delta, end_time)
            buckets.append((current, bucket_end))
            current = bucket_end
        
        return buckets
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Resolve old alerts
                await self._resolve_old_alerts()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old metrics data."""
        if not self.redis_client:
            return
        
        try:
            cutoff_time = time.time() - (self.metrics_retention_hours * 3600)
            pattern = "request_metrics:*"
            
            keys = await self.redis_client.keys(pattern)
            old_keys = []
            
            for key in keys:
                # Extract timestamp from key
                try:
                    timestamp_str = key.decode().split(':')[1]
                    timestamp = int(timestamp_str)
                    if timestamp < cutoff_time:
                        old_keys.append(key)
                except (IndexError, ValueError):
                    continue
            
            if old_keys:
                await self.redis_client.delete(*old_keys)
                logger.info(f"Cleaned up {len(old_keys)} old metric records")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def _resolve_old_alerts(self):
        """Resolve alerts that are no longer active."""
        current_time = datetime.utcnow()
        
        for alert in self.alerts:
            if alert['status'] == 'active':
                created_at = datetime.fromisoformat(alert['created_at'])
                if current_time - created_at > timedelta(hours=1):  # Auto-resolve after 1 hour
                    alert['status'] = 'resolved'
                    alert['resolved_at'] = current_time.isoformat()
                    logger.info(f"Auto-resolved alert: {alert['id']}")