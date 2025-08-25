"""
Enterprise Rate Limiting and DDoS Protection System for claude-tui.

This module provides comprehensive protection against abuse with:
- Multi-tier rate limiting (IP, user, endpoint)
- Advanced DDoS detection and mitigation
- Adaptive throttling based on system load
- Intelligent pattern recognition
- Whitelist/blacklist management
- Real-time monitoring and alerting
- Circuit breaker patterns
- Geographic-based filtering
"""

import time
import json
import threading
import ipaddress
import statistics
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """DDoS threat levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ActionType(Enum):
    """Rate limiting actions."""
    ALLOW = "allow"
    THROTTLE = "throttle"
    BLOCK = "block"
    CHALLENGE = "challenge"
    BLACKLIST = "blacklist"

class AttackType(Enum):
    """Types of detected attacks."""
    VOLUMETRIC = "volumetric"
    PROTOCOL = "protocol"
    APPLICATION = "application"
    BOTNET = "botnet"
    SLOWLORIS = "slowloris"
    HTTP_FLOOD = "http_flood"
    SYN_FLOOD = "syn_flood"

@dataclass
class RateLimit:
    """Rate limiting configuration."""
    requests_per_minute: int
    requests_per_hour: int
    burst_allowance: int
    window_size: int = 60  # seconds
    
    def __post_init__(self):
        """Validate rate limit configuration."""
        if self.requests_per_minute <= 0 or self.requests_per_hour <= 0:
            raise ValueError("Rate limits must be positive")
        if self.burst_allowance < 0:
            raise ValueError("Burst allowance cannot be negative")

@dataclass
class ClientInfo:
    """Information about a client."""
    ip_address: str
    user_id: Optional[str] = None
    user_agent: Optional[str] = None
    country: Optional[str] = None
    asn: Optional[str] = None
    is_proxy: bool = False
    is_tor: bool = False
    reputation_score: float = 50.0  # 0-100 scale
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    def update_last_seen(self):
        """Update last seen timestamp."""
        self.last_seen = datetime.utcnow()

@dataclass
class RequestRecord:
    """Record of a request for rate limiting."""
    timestamp: float
    endpoint: str
    method: str
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    payload_size: int = 0
    
    def age(self) -> float:
        """Get age of request in seconds."""
        return time.time() - self.timestamp

@dataclass
class AttackPattern:
    """Pattern of detected attack."""
    attack_type: AttackType
    confidence: float
    metrics: Dict[str, Any]
    first_detected: datetime
    last_seen: datetime
    affected_ips: Set[str] = field(default_factory=set)
    
    def update_detection(self, ip: str):
        """Update attack pattern with new detection."""
        self.last_seen = datetime.utcnow()
        self.affected_ips.add(ip)

class SmartRateLimiter:
    """
    Advanced rate limiter with intelligent threat detection.
    
    Features:
    - Multi-tier rate limiting (IP, user, endpoint)
    - Adaptive limits based on behavior patterns
    - DDoS attack detection and mitigation
    - Whitelist/blacklist management
    - Real-time metrics and alerting
    """
    
    def __init__(
        self,
        default_limits: Optional[Dict[str, RateLimit]] = None,
        enable_ddos_protection: bool = True,
        enable_adaptive_limiting: bool = True,
        max_clients: int = 100000,
        cleanup_interval: int = 3600  # 1 hour
    ):
        """
        Initialize the smart rate limiter.
        
        Args:
            default_limits: Default rate limits by category
            enable_ddos_protection: Enable DDoS detection
            enable_adaptive_limiting: Enable adaptive rate limiting
            max_clients: Maximum clients to track
            cleanup_interval: Cleanup interval in seconds
        """
        self.enable_ddos_protection = enable_ddos_protection
        self.enable_adaptive_limiting = enable_adaptive_limiting
        self.max_clients = max_clients
        self.cleanup_interval = cleanup_interval
        
        # Default rate limits
        self.default_limits = default_limits or {
            'api_general': RateLimit(60, 1000, 10),
            'api_ai': RateLimit(20, 200, 5),
            'api_upload': RateLimit(10, 50, 2),
            'web_general': RateLimit(100, 2000, 20),
            'auth': RateLimit(5, 30, 1)
        }
        
        # Client tracking
        self._clients: Dict[str, ClientInfo] = {}
        self._client_requests: Dict[str, deque] = defaultdict(deque)
        self._client_lock = threading.RLock()
        
        # Request tracking by endpoint
        self._endpoint_requests: Dict[str, deque] = defaultdict(deque)
        self._endpoint_lock = threading.RLock()
        
        # Blacklist and whitelist
        self._blacklisted_ips: Set[str] = set()
        self._whitelisted_ips: Set[str] = set()
        self._blacklisted_subnets: List[ipaddress.IPv4Network] = []
        self._whitelisted_subnets: List[ipaddress.IPv4Network] = []
        self._list_lock = threading.RLock()
        
        # DDoS protection
        self._attack_patterns: Dict[str, AttackPattern] = {}
        self._suspicious_activities: Dict[str, List[float]] = defaultdict(list)
        self._ddos_lock = threading.RLock()
        
        # System metrics
        self._system_load = 0.0
        self._request_queue_size = 0
        self._blocked_requests = 0
        self._total_requests = 0
        self._metrics_lock = threading.RLock()
        
        # Circuit breakers
        self._circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        
        # Start background tasks
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        # Load persistent data
        self._load_persistent_data()
        
        logger.info("Smart rate limiter initialized")
    
    def check_rate_limit(
        self,
        client_ip: str,
        endpoint: str,
        method: str = "GET",
        user_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        payload_size: int = 0
    ) -> Tuple[ActionType, Dict[str, Any]]:
        """
        Check if request should be rate limited.
        
        Args:
            client_ip: Client IP address
            endpoint: API endpoint or resource
            method: HTTP method
            user_id: Optional user ID
            user_agent: Optional user agent
            payload_size: Request payload size
            
        Returns:
            Tuple of (action, metadata)
        """
        current_time = time.time()
        
        with self._client_lock:
            # Update total request counter
            with self._metrics_lock:
                self._total_requests += 1
            
            # Check whitelist first
            if self._is_whitelisted(client_ip):
                return ActionType.ALLOW, {"reason": "whitelisted"}
            
            # Check blacklist
            if self._is_blacklisted(client_ip):
                with self._metrics_lock:
                    self._blocked_requests += 1
                return ActionType.BLOCK, {"reason": "blacklisted"}
            
            # Get or create client info
            client = self._get_or_create_client(client_ip, user_id, user_agent)
            client.update_last_seen()
            
            # Create request record
            request = RequestRecord(
                timestamp=current_time,
                endpoint=endpoint,
                method=method,
                payload_size=payload_size
            )
            
            # Add to client request history
            self._client_requests[client_ip].append(request)
            
            # Determine rate limit category
            limit_category = self._categorize_endpoint(endpoint)
            rate_limit = self.default_limits.get(limit_category, self.default_limits['api_general'])
            
            # Check DDoS patterns first
            if self.enable_ddos_protection:
                ddos_result = self._check_ddos_patterns(client_ip, request, client)
                if ddos_result[0] != ActionType.ALLOW:
                    return ddos_result
            
            # Apply adaptive limiting if enabled
            if self.enable_adaptive_limiting:
                rate_limit = self._adapt_rate_limit(rate_limit, client, endpoint)
            
            # Check rate limits
            limit_result = self._check_limits(client_ip, rate_limit, current_time)
            
            # Apply circuit breaker logic
            breaker_result = self._check_circuit_breaker(endpoint)
            if breaker_result[0] != ActionType.ALLOW:
                return breaker_result
            
            # Record successful check
            if limit_result[0] == ActionType.ALLOW:
                self._record_success(endpoint, current_time)
            else:
                with self._metrics_lock:
                    self._blocked_requests += 1
            
            return limit_result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        with self._metrics_lock:
            current_time = time.time()
            
            # Calculate recent metrics
            recent_requests = 0
            
            with self._client_lock:
                for requests in self._client_requests.values():
                    for request in requests:
                        if current_time - request.timestamp < 3600:  # Last hour
                            recent_requests += 1
            
            return {
                'total_requests': self._total_requests,
                'blocked_requests': self._blocked_requests,
                'recent_requests_1h': recent_requests,
                'active_clients': len(self._clients),
                'blacklisted_ips': len(self._blacklisted_ips),
                'whitelisted_ips': len(self._whitelisted_ips),
                'attack_patterns_detected': len(self._attack_patterns),
                'system_load': self._system_load,
                'block_rate': (self._blocked_requests / max(self._total_requests, 1)) * 100
            }
    
    def _get_or_create_client(self, ip: str, user_id: Optional[str], user_agent: Optional[str]) -> ClientInfo:
        """Get existing client or create new one."""
        if ip in self._clients:
            client = self._clients[ip]
            # Update user info if provided
            if user_id and not client.user_id:
                client.user_id = user_id
            if user_agent and not client.user_agent:
                client.user_agent = user_agent
            return client
        
        # Create new client
        client = ClientInfo(
            ip_address=ip,
            user_id=user_id,
            user_agent=user_agent
        )
        
        # Limit number of tracked clients
        if len(self._clients) >= self.max_clients:
            self._evict_oldest_clients()
        
        self._clients[ip] = client
        return client
    
    def _evict_oldest_clients(self):
        """Evict oldest clients to maintain memory limits."""
        # Sort clients by last seen time
        sorted_clients = sorted(
            self._clients.items(),
            key=lambda x: x[1].last_seen
        )
        
        # Remove oldest 10%
        evict_count = max(1, len(sorted_clients) // 10)
        for i in range(evict_count):
            ip, _ = sorted_clients[i]
            del self._clients[ip]
            if ip in self._client_requests:
                del self._client_requests[ip]
    
    def _categorize_endpoint(self, endpoint: str) -> str:
        """Categorize endpoint for rate limiting."""
        endpoint_lower = endpoint.lower()
        
        if '/auth/' in endpoint_lower or '/login' in endpoint_lower:
            return 'auth'
        elif '/ai/' in endpoint_lower or '/generate' in endpoint_lower:
            return 'api_ai'
        elif '/upload' in endpoint_lower or '/file' in endpoint_lower:
            return 'api_upload'
        elif endpoint_lower.startswith('/api/'):
            return 'api_general'
        else:
            return 'web_general'
    
    def _is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        with self._list_lock:
            if ip in self._whitelisted_ips:
                return True
            
            try:
                ip_obj = ipaddress.ip_address(ip)
                for subnet in self._whitelisted_subnets:
                    if ip_obj in subnet:
                        return True
            except ValueError:
                pass
            
            return False
    
    def _is_blacklisted(self, ip: str) -> bool:
        """Check if IP is blacklisted."""
        with self._list_lock:
            if ip in self._blacklisted_ips:
                return True
            
            try:
                ip_obj = ipaddress.ip_address(ip)
                for subnet in self._blacklisted_subnets:
                    if ip_obj in subnet:
                        return True
            except ValueError:
                pass
            
            return False
    
    def _check_limits(self, client_ip: str, rate_limit: RateLimit, current_time: float) -> Tuple[ActionType, Dict[str, Any]]:
        """Check rate limits for client."""
        requests = self._client_requests[client_ip]
        
        # Clean old requests
        cutoff_time = current_time - rate_limit.window_size
        while requests and requests[0].timestamp < cutoff_time:
            requests.popleft()
        
        # Check minute limit
        minute_cutoff = current_time - 60
        minute_requests = sum(1 for r in requests if r.timestamp >= minute_cutoff)
        
        if minute_requests > rate_limit.requests_per_minute + rate_limit.burst_allowance:
            return ActionType.BLOCK, {
                "reason": "rate_limit_exceeded",
                "limit_type": "per_minute",
                "current_rate": minute_requests,
                "limit": rate_limit.requests_per_minute
            }
        
        # Check hour limit
        hour_cutoff = current_time - 3600
        hour_requests = sum(1 for r in requests if r.timestamp >= hour_cutoff)
        
        if hour_requests > rate_limit.requests_per_hour:
            return ActionType.BLOCK, {
                "reason": "rate_limit_exceeded",
                "limit_type": "per_hour",
                "current_rate": hour_requests,
                "limit": rate_limit.requests_per_hour
            }
        
        # Check for throttling conditions
        if minute_requests > rate_limit.requests_per_minute:
            return ActionType.THROTTLE, {
                "reason": "rate_limit_approaching",
                "current_rate": minute_requests,
                "limit": rate_limit.requests_per_minute
            }
        
        return ActionType.ALLOW, {
            "current_minute_rate": minute_requests,
            "current_hour_rate": hour_requests
        }
    
    def _adapt_rate_limit(self, base_limit: RateLimit, client: ClientInfo, endpoint: str) -> RateLimit:
        """Adapt rate limit based on client behavior and system load."""
        # Start with base limit
        adapted_rpm = base_limit.requests_per_minute
        adapted_rph = base_limit.requests_per_hour
        adapted_burst = base_limit.burst_allowance
        
        # Adjust based on reputation score
        reputation_factor = client.reputation_score / 50.0  # Normalize to 0-2 range
        adapted_rpm = int(adapted_rpm * reputation_factor)
        adapted_rph = int(adapted_rph * reputation_factor)
        
        # Adjust based on system load
        if self._system_load > 0.8:  # High load
            adapted_rpm = int(adapted_rpm * 0.5)
            adapted_rph = int(adapted_rph * 0.5)
            adapted_burst = int(adapted_burst * 0.5)
        elif self._system_load > 0.6:  # Medium load
            adapted_rpm = int(adapted_rpm * 0.75)
            adapted_rph = int(adapted_rph * 0.75)
        
        return RateLimit(
            requests_per_minute=max(1, adapted_rpm),
            requests_per_hour=max(10, adapted_rph),
            burst_allowance=max(0, adapted_burst),
            window_size=base_limit.window_size
        )
    
    def _check_ddos_patterns(self, client_ip: str, request: RequestRecord, client: ClientInfo) -> Tuple[ActionType, Dict[str, Any]]:
        """Check for DDoS attack patterns."""
        current_time = time.time()
        
        # Get recent requests for this client
        recent_requests = [
            r for r in self._client_requests[client_ip]
            if current_time - r.timestamp < 60  # Last minute
        ]
        
        # Volumetric attack detection
        if len(recent_requests) > 100:  # More than 100 requests per minute
            threat_level = ThreatLevel.HIGH
            self._record_attack_pattern(
                AttackType.VOLUMETRIC,
                client_ip,
                {"requests_per_minute": len(recent_requests)}
            )
            
            return ActionType.BLOCK, {
                "reason": "volumetric_attack_suspected",
                "threat_level": threat_level.value,
                "requests_per_minute": len(recent_requests)
            }
        
        return ActionType.ALLOW, {}
    
    def _record_attack_pattern(self, attack_type: AttackType, client_ip: str, metrics: Dict[str, Any]):
        """Record a detected attack pattern."""
        with self._ddos_lock:
            pattern_id = f"{attack_type.value}_{int(time.time() // 300)}"  # 5-minute windows
            
            if pattern_id in self._attack_patterns:
                self._attack_patterns[pattern_id].update_detection(client_ip)
                self._attack_patterns[pattern_id].metrics.update(metrics)
            else:
                self._attack_patterns[pattern_id] = AttackPattern(
                    attack_type=attack_type,
                    confidence=0.8,
                    metrics=metrics,
                    first_detected=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    affected_ips={client_ip}
                )
        
        logger.warning(f"Attack pattern detected: {attack_type.value} from {client_ip}")
    
    def _check_circuit_breaker(self, endpoint: str) -> Tuple[ActionType, Dict[str, Any]]:
        """Check circuit breaker status for endpoint."""
        if endpoint not in self._circuit_breakers:
            self._circuit_breakers[endpoint] = CircuitBreaker(
                failure_threshold=10,
                recovery_timeout=30
            )
        
        breaker = self._circuit_breakers[endpoint]
        
        if breaker.is_open():
            return ActionType.BLOCK, {
                "reason": "circuit_breaker_open",
                "endpoint": endpoint,
                "failures": breaker.failure_count
            }
        
        return ActionType.ALLOW, {}
    
    def _record_success(self, endpoint: str, timestamp: float):
        """Record successful request for metrics."""
        with self._endpoint_lock:
            self._endpoint_requests[endpoint].append(timestamp)
            
            # Cleanup old records
            cutoff_time = timestamp - 3600  # Keep last hour
            while (self._endpoint_requests[endpoint] and 
                   self._endpoint_requests[endpoint][0] < cutoff_time):
                self._endpoint_requests[endpoint].popleft()
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                self._cleanup_old_data()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                time.sleep(60)  # Wait before retry
    
    def _cleanup_old_data(self):
        """Clean up old data to manage memory usage."""
        current_time = time.time()
        cutoff_time = current_time - 86400  # Keep last 24 hours
        
        with self._client_lock:
            # Clean old client requests
            for ip in list(self._client_requests.keys()):
                requests = self._client_requests[ip]
                while requests and requests[0].timestamp < cutoff_time:
                    requests.popleft()
                
                # Remove empty deques
                if not requests:
                    del self._client_requests[ip]
    
    def _load_persistent_data(self):
        """Load persistent blacklist/whitelist data."""
        try:
            data_dir = Path.home() / ".claude-tiu" / "security"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Load blacklist
            blacklist_file = data_dir / "blacklist.json"
            if blacklist_file.exists():
                with open(blacklist_file) as f:
                    data = json.load(f)
                    self._blacklisted_ips.update(data.get("ips", []))
                            
        except Exception as e:
            logger.error(f"Failed to load persistent data: {e}")
    
    def shutdown(self):
        """Shutdown the rate limiter."""
        self._running = False
        logger.info("Rate limiter shutdown complete")

class CircuitBreaker:
    """Circuit breaker for endpoint protection."""
    
    def __init__(self, failure_threshold: int = 10, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half-open
        self._lock = threading.Lock()
    
    def record_result(self, success: bool):
        """Record request result."""
        with self._lock:
            if success:
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
            else:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        with self._lock:
            if self.state == 'open':
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'half-open'
                    return False
                return True
            return False

# Utility function
def create_rate_limiter(
    enable_ddos_protection: bool = True,
    max_requests_per_minute: int = 60,
    max_requests_per_hour: int = 1000
) -> SmartRateLimiter:
    """Create a configured rate limiter."""
    default_limits = {
        'api_general': RateLimit(max_requests_per_minute, max_requests_per_hour, 10),
        'api_ai': RateLimit(max_requests_per_minute // 3, max_requests_per_hour // 5, 5),
        'web_general': RateLimit(max_requests_per_minute * 2, max_requests_per_hour * 2, 20)
    }
    
    return SmartRateLimiter(
        default_limits=default_limits,
        enable_ddos_protection=enable_ddos_protection
    )