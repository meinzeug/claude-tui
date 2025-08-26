"""Integration Manager - Centralized integration orchestration.

Provides centralized management of all AI service integrations with:
- Smart routing between Claude Code/Flow
- Circuit breaker pattern for resilience
- Comprehensive health monitoring
- Performance optimization and caching
- Automatic failover mechanisms
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.integrations.ai_interface import AIInterface
from src.claude_tui.integrations.claude_code_client import ClaudeCodeClient
from src.claude_tui.integrations.claude_flow_client import ClaudeFlowClient
try:
    from src.claude_tui.integrations.anti_hallucination_integration import AntiHallucinationIntegration
except ImportError:
    # Fallback if anti-hallucination integration is not available
    AntiHallucinationIntegration = None
from src.claude_tui.models.project import Project
from src.claude_tui.models.task import DevelopmentTask, TaskResult
from src.claude_tui.models.ai_models import CodeResult, WorkflowRequest

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Available AI service types."""
    CLAUDE_CODE = "claude_code"
    CLAUDE_FLOW = "claude_flow"
    HYBRID = "hybrid"


class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ServiceHealthMetrics:
    """Health metrics for a service."""
    status: ServiceStatus = ServiceStatus.HEALTHY
    response_time: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    timeout_duration: int = 60  # seconds
    success_threshold: int = 3
    max_retry_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_backoff: float = 1.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """Circuit breaker implementation for service resilience."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                await self._on_success()
                return result
            except Exception as e:
                await self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout_duration
    
    async def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
        else:
            self.failure_count = 0
    
    async def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            if self.state == CircuitState.CLOSED:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} OPENED due to failures")
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit breaker {self.name} re-OPENED during recovery")


class IntegrationCache:
    """High-performance caching layer with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        async with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if datetime.now() > entry.expires_at:
                del self.cache[key]
                self.miss_count += 1
                return None
            
            # Update access stats
            entry.hit_count += 1
            entry.last_accessed = datetime.now()
            self.hit_count += 1
            
            return entry.data
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        async with self._lock:
            # Evict if at max size
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            self.cache[key] = CacheEntry(
                data=value,
                created_at=datetime.now(),
                expires_at=expires_at
            )
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )
        
        del self.cache[lru_key]
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class SmartRouter:
    """Intelligent routing between AI services based on context and performance."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager
        self.service_metrics: Dict[ServiceType, ServiceHealthMetrics] = {
            ServiceType.CLAUDE_CODE: ServiceHealthMetrics(),
            ServiceType.CLAUDE_FLOW: ServiceHealthMetrics()
        }
        self.routing_rules: List[Dict[str, Any]] = []
        if config_manager:
            self._load_routing_config()
    
    def _load_routing_config(self) -> None:
        """Load routing configuration."""
        # Default routing rules
        self.routing_rules = [
            {
                'condition': lambda context: context.get('task_type') in ['code_generation', 'code_review'],
                'preferred_service': ServiceType.CLAUDE_CODE,
                'fallback_service': ServiceType.CLAUDE_FLOW
            },
            {
                'condition': lambda context: context.get('task_type') in ['workflow', 'orchestration'],
                'preferred_service': ServiceType.CLAUDE_FLOW,
                'fallback_service': ServiceType.CLAUDE_CODE
            },
            {
                'condition': lambda context: context.get('complexity', 'medium') == 'high',
                'preferred_service': ServiceType.HYBRID,
                'fallback_service': ServiceType.CLAUDE_CODE
            }
        ]
    
    async def route_request(
        self,
        context: Dict[str, Any],
        prefer_performance: bool = True
    ) -> Tuple[ServiceType, Optional[ServiceType]]:
        """Route request to optimal service."""
        # Apply routing rules
        for rule in self.routing_rules:
            if rule['condition'](context):
                preferred = rule['preferred_service']
                fallback = rule.get('fallback_service')
                
                # Check service health if performance is prioritized
                if prefer_performance and preferred != ServiceType.HYBRID:
                    metrics = self.service_metrics[preferred]
                    if metrics.status not in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
                        logger.warning(f"Service {preferred.value} unhealthy, using fallback")
                        return fallback, preferred
                
                return preferred, fallback
        
        # Default routing
        return ServiceType.CLAUDE_CODE, ServiceType.CLAUDE_FLOW
    
    async def update_service_metrics(
        self,
        service: ServiceType,
        response_time: float,
        success: bool
    ) -> None:
        """Update service performance metrics."""
        if service == ServiceType.HYBRID:
            return  # Skip metrics for hybrid service
        
        metrics = self.service_metrics[service]
        metrics.total_requests += 1
        metrics.last_check = datetime.now()
        
        if success:
            metrics.successful_requests += 1
            metrics.consecutive_failures = 0
        else:
            metrics.failed_requests += 1
            metrics.consecutive_failures += 1
        
        # Update rates
        metrics.success_rate = metrics.successful_requests / metrics.total_requests
        metrics.error_rate = metrics.failed_requests / metrics.total_requests
        
        # Update response time (moving average)
        if metrics.total_requests == 1:
            metrics.avg_response_time = response_time
        else:
            alpha = 0.1  # Smoothing factor
            metrics.avg_response_time = (
                alpha * response_time + (1 - alpha) * metrics.avg_response_time
            )
        
        metrics.response_time = response_time
        
        # Update health status
        await self._update_health_status(service)
    
    async def _update_health_status(self, service: ServiceType) -> None:
        """Update service health status based on metrics."""
        metrics = self.service_metrics[service]
        
        # Determine status based on metrics
        if metrics.consecutive_failures >= 5:
            metrics.status = ServiceStatus.UNHEALTHY
        elif metrics.error_rate > 0.2:  # >20% error rate
            metrics.status = ServiceStatus.DEGRADED
        elif metrics.avg_response_time > 10.0:  # >10s response time
            metrics.status = ServiceStatus.DEGRADED
        else:
            metrics.status = ServiceStatus.HEALTHY


class IntegrationManager:
    """Centralized integration manager for all AI services."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager
        
        # Core components
        self.claude_code_client: Optional[ClaudeCodeClient] = None
        self.claude_flow_client: Optional[ClaudeFlowClient] = None
        self.ai_interface: Optional[AIInterface] = None
        self.anti_hallucination: Optional[AntiHallucinationIntegration] = None
        
        # Integration infrastructure
        self.router = SmartRouter(config_manager) if config_manager else None
        self.cache = IntegrationCache(max_size=2000, default_ttl=600)
        self.circuit_breakers: Dict[ServiceType, CircuitBreaker] = {}
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0,
            'reliability_score': 1.0
        }
        
        # Configuration
        self.circuit_breaker_config = CircuitBreakerConfig()
        self.enable_caching = True
        self.enable_auto_retry = True
        self.max_concurrent_requests = 50
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Runtime state
        self.is_initialized = False
        self.initialization_lock = asyncio.Lock()
        
        logger.info("Integration Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the Integration Manager."""
        async with self.initialization_lock:
            if self.is_initialized:
                return
            
            logger.info("Initializing Integration Manager")
            
            try:
                # Initialize AI service clients
                await self._initialize_clients()
                
                # Initialize circuit breakers
                await self._initialize_circuit_breakers()
                
                # Initialize AI interface
                self.ai_interface = AIInterface(self.config_manager)
                await self.ai_interface.initialize()
                
                # Initialize anti-hallucination system if available
                if AntiHallucinationIntegration:
                    self.anti_hallucination = AntiHallucinationIntegration(self.config_manager)
                    await self.anti_hallucination.initialize()
                else:
                    logger.warning("Anti-hallucination integration not available")
                
                # Start health monitoring
                await self._start_health_monitoring()
                
                # Load configuration
                await self._load_integration_config()
                
                self.is_initialized = True
                logger.info("Integration Manager ready")
                
            except Exception as e:
                logger.error(f"Failed to initialize Integration Manager: {e}")
                raise
    
    async def execute_request(
        self,
        request_type: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute AI service request with smart routing and resilience."""
        if not self.is_initialized:
            await self.initialize()
        
        async with self.request_semaphore:
            request_id = self._generate_request_id()
            start_time = time.time()
            
            logger.debug(f"Executing request {request_id}: {request_type}")
            
            try:
                # Check cache first
                if self.enable_caching:
                    cache_key = self._generate_cache_key(request_type, context, kwargs)
                    cached_result = await self.cache.get(cache_key)
                    if cached_result is not None:
                        logger.debug(f"Cache hit for request {request_id}")
                        self.performance_metrics['cache_hits'] += 1
                        return cached_result
                    
                    self.performance_metrics['cache_misses'] += 1
                
                # Route request to appropriate service
                primary_service, fallback_service = await self.router.route_request(context)
                
                # Execute request with fallback
                result = await self._execute_with_fallback(
                    request_type, primary_service, fallback_service,
                    context, request_id, **kwargs
                )
                
                # Cache result
                if self.enable_caching and result is not None:
                    await self.cache.set(cache_key, result)
                
                # Update performance metrics
                execution_time = time.time() - start_time
                await self._update_performance_metrics(True, execution_time)
                
                logger.debug(f"Request {request_id} completed in {execution_time:.3f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                await self._update_performance_metrics(False, execution_time)
                logger.error(f"Request {request_id} failed: {e}")
                raise
    
    async def execute_coding_task(
        self,
        prompt: str,
        context: Dict[str, Any] = None,
        project: Optional[Project] = None
    ) -> CodeResult:
        """Execute coding task with optimal routing."""
        context = context or {}
        context.update({
            'task_type': 'code_generation',
            'prompt': prompt,
            'project': project
        })
        
        return await self.execute_request(
            'coding_task',
            context,
            prompt=prompt,
            project=project
        )
    
    async def execute_workflow(
        self,
        workflow_request: WorkflowRequest,
        project: Optional[Project] = None
    ) -> Any:
        """Execute workflow with optimal routing."""
        context = {
            'task_type': 'workflow',
            'workflow_name': workflow_request.workflow_name,
            'project': project
        }
        
        return await self.execute_request(
            'workflow',
            context,
            workflow_request=workflow_request,
            project=project
        )
    
    async def validate_content(
        self,
        content: str,
        context: Dict[str, Any] = None,
        project: Optional[Project] = None
    ) -> Any:
        """Validate content using anti-hallucination system."""
        if not self.anti_hallucination:
            raise RuntimeError("Anti-hallucination system not initialized")
        
        return await self.anti_hallucination.validate_ai_generated_content(
            content, context, project=project
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        service_health = {}
        for service_type, metrics in self.router.service_metrics.items():
            service_health[service_type.value] = {
                'status': metrics.status.value,
                'response_time': metrics.response_time,
                'success_rate': metrics.success_rate,
                'error_rate': metrics.error_rate,
                'consecutive_failures': metrics.consecutive_failures,
                'total_requests': metrics.total_requests
            }
        
        circuit_breaker_status = {}
        for service_type, breaker in self.circuit_breakers.items():
            circuit_breaker_status[service_type.value] = {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'success_count': breaker.success_count
            }
        
        cache_stats = self.cache.get_stats()
        
        overall_health = self._calculate_overall_health(service_health)
        
        return {
            'overall_health': overall_health,
            'reliability_score': self.performance_metrics['reliability_score'],
            'services': service_health,
            'circuit_breakers': circuit_breaker_status,
            'cache': cache_stats,
            'performance': self.performance_metrics,
            'last_updated': datetime.now().isoformat()
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        health_status = await self.get_health_status()
        
        return {
            'integration_performance': self.performance_metrics,
            'service_metrics': health_status['services'],
            'cache_performance': health_status['cache'],
            'reliability_metrics': {
                'uptime_percentage': self._calculate_uptime_percentage(),
                'mttr': self._calculate_mean_time_to_recovery(),
                'error_rate_threshold': 0.01,  # 1% target
                'response_time_p95': self._calculate_p95_response_time(),
                'availability_sla': 99.9  # 99.9% target
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup Integration Manager resources."""
        logger.info("Cleaning up Integration Manager")
        
        # Stop health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup clients
        if self.claude_code_client:
            await self.claude_code_client.cleanup()
        
        if self.claude_flow_client:
            await self.claude_flow_client.cleanup()
        
        if self.ai_interface:
            await self.ai_interface.cleanup()
        
        if self.anti_hallucination:
            await self.anti_hallucination.cleanup()
        
        self.is_initialized = False
        logger.info("Integration Manager cleanup completed")
    
    # Private implementation methods
    
    async def _initialize_clients(self) -> None:
        """Initialize AI service clients."""
        self.claude_code_client = ClaudeCodeClient(self.config_manager)
        self.claude_flow_client = ClaudeFlowClient(self.config_manager)
        await self.claude_flow_client.initialize()
    
    async def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for services."""
        for service_type in [ServiceType.CLAUDE_CODE, ServiceType.CLAUDE_FLOW]:
            self.circuit_breakers[service_type] = CircuitBreaker(
                name=service_type.value,
                config=self.circuit_breaker_config
            )
    
    async def _load_integration_config(self) -> None:
        """Load integration configuration."""
        config = await self.config_manager.get_setting('integration_manager', {})
        
        self.enable_caching = config.get('enable_caching', True)
        self.enable_auto_retry = config.get('enable_auto_retry', True)
        self.max_concurrent_requests = config.get('max_concurrent_requests', 50)
        self.health_check_interval = config.get('health_check_interval', 30)
        
        # Update semaphore if needed
        if self.request_semaphore._value != self.max_concurrent_requests:
            self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
    
    async def _execute_with_fallback(
        self,
        request_type: str,
        primary_service: ServiceType,
        fallback_service: Optional[ServiceType],
        context: Dict[str, Any],
        request_id: str,
        **kwargs
    ) -> Any:
        """Execute request with fallback mechanism."""
        # Try primary service
        try:
            result = await self._execute_on_service(
                request_type, primary_service, context, **kwargs
            )
            
            # Update router metrics
            await self.router.update_service_metrics(
                primary_service, context.get('_execution_time', 0.0), True
            )
            
            return result
            
        except Exception as e:
            logger.warning(
                f"Request {request_id} failed on {primary_service.value}: {e}"
            )
            
            # Update router metrics
            await self.router.update_service_metrics(
                primary_service, context.get('_execution_time', 0.0), False
            )
            
            # Try fallback service if available
            if fallback_service and fallback_service != primary_service:
                logger.info(f"Attempting fallback to {fallback_service.value}")
                
                try:
                    result = await self._execute_on_service(
                        request_type, fallback_service, context, **kwargs
                    )
                    
                    await self.router.update_service_metrics(
                        fallback_service, context.get('_execution_time', 0.0), True
                    )
                    
                    return result
                    
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback also failed for {request_id}: {fallback_error}"
                    )
                    
                    await self.router.update_service_metrics(
                        fallback_service, context.get('_execution_time', 0.0), False
                    )
            
            # No successful execution
            raise e
    
    async def _execute_on_service(
        self,
        request_type: str,
        service: ServiceType,
        context: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute request on specific service."""
        start_time = time.time()
        
        try:
            if service == ServiceType.CLAUDE_CODE:
                result = await self._execute_claude_code_request(
                    request_type, context, **kwargs
                )
            elif service == ServiceType.CLAUDE_FLOW:
                result = await self._execute_claude_flow_request(
                    request_type, context, **kwargs
                )
            elif service == ServiceType.HYBRID:
                result = await self._execute_hybrid_request(
                    request_type, context, **kwargs
                )
            else:
                raise ValueError(f"Unknown service type: {service}")
            
            execution_time = time.time() - start_time
            context['_execution_time'] = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            context['_execution_time'] = execution_time
            raise
    
    async def _execute_claude_code_request(
        self,
        request_type: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute request using Claude Code service."""
        if not self.claude_code_client:
            raise RuntimeError("Claude Code client not initialized")
        
        circuit_breaker = self.circuit_breakers[ServiceType.CLAUDE_CODE]
        
        if request_type == 'coding_task':
            return await circuit_breaker.call(
                self.claude_code_client.execute_coding_task,
                kwargs['prompt'],
                context,
                kwargs.get('project')
            )
        elif request_type == 'validate_output':
            return await circuit_breaker.call(
                self.claude_code_client.validate_output,
                kwargs['output'],
                context
            )
        else:
            raise ValueError(f"Unknown request type for Claude Code: {request_type}")
    
    async def _execute_claude_flow_request(
        self,
        request_type: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute request using Claude Flow service."""
        if not self.claude_flow_client:
            raise RuntimeError("Claude Flow client not initialized")
        
        circuit_breaker = self.circuit_breakers[ServiceType.CLAUDE_FLOW]
        
        if request_type == 'workflow':
            return await circuit_breaker.call(
                self.claude_flow_client.execute_workflow,
                kwargs['workflow_request'],
                kwargs.get('project')
            )
        else:
            raise ValueError(f"Unknown request type for Claude Flow: {request_type}")
    
    async def _execute_hybrid_request(
        self,
        request_type: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute request using hybrid approach."""
        # For complex requests, use AI Interface which handles hybrid execution
        if not self.ai_interface:
            raise RuntimeError("AI Interface not initialized")
        
        if request_type == 'coding_task':
            return await self.ai_interface.execute_claude_code(
                kwargs['prompt'],
                context,
                kwargs.get('project')
            )
        else:
            # Fallback to Claude Code
            return await self._execute_claude_code_request(request_type, context, **kwargs)
    
    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all services."""
        # Check Claude Code health
        if self.claude_code_client:
            try:
                start_time = time.time()
                is_healthy = await self.claude_code_client.health_check()
                response_time = time.time() - start_time
                
                await self.router.update_service_metrics(
                    ServiceType.CLAUDE_CODE, response_time, is_healthy
                )
            except Exception as e:
                logger.warning(f"Claude Code health check failed: {e}")
                await self.router.update_service_metrics(
                    ServiceType.CLAUDE_CODE, 10.0, False
                )
        
        # Check Claude Flow health
        if self.claude_flow_client:
            try:
                start_time = time.time()
                # Claude Flow doesn't have direct health check, so we'll use a simple call
                await self.claude_flow_client._test_connection()
                response_time = time.time() - start_time
                is_healthy = True
                
                await self.router.update_service_metrics(
                    ServiceType.CLAUDE_FLOW, response_time, is_healthy
                )
            except Exception as e:
                logger.warning(f"Claude Flow health check failed: {e}")
                await self.router.update_service_metrics(
                    ServiceType.CLAUDE_FLOW, 10.0, False
                )
    
    async def _update_performance_metrics(
        self,
        success: bool,
        execution_time: float
    ) -> None:
        """Update overall performance metrics."""
        self.performance_metrics['total_requests'] += 1
        
        if success:
            self.performance_metrics['successful_requests'] += 1
        else:
            self.performance_metrics['failed_requests'] += 1
        
        # Update average response time (moving average)
        total_requests = self.performance_metrics['total_requests']
        if total_requests == 1:
            self.performance_metrics['avg_response_time'] = execution_time
        else:
            alpha = 0.1
            self.performance_metrics['avg_response_time'] = (
                alpha * execution_time + 
                (1 - alpha) * self.performance_metrics['avg_response_time']
            )
        
        # Update reliability score
        success_rate = (
            self.performance_metrics['successful_requests'] / 
            max(total_requests, 1)
        )
        self.performance_metrics['reliability_score'] = success_rate
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _generate_cache_key(
        self,
        request_type: str,
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> str:
        """Generate cache key for request."""
        # Create deterministic hash of request parameters
        cache_data = {
            'request_type': request_type,
            'context': {k: v for k, v in context.items() if k != 'project'},
            'kwargs': {k: str(v) for k, v in kwargs.items() if k != 'project'}
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:16]
    
    def _calculate_overall_health(self, service_health: Dict[str, Any]) -> str:
        """Calculate overall system health."""
        healthy_services = sum(
            1 for service in service_health.values()
            if service['status'] == 'healthy'
        )
        total_services = len(service_health)
        
        if healthy_services == total_services:
            return 'healthy'
        elif healthy_services >= total_services * 0.5:
            return 'degraded'
        else:
            return 'unhealthy'
    
    def _calculate_uptime_percentage(self) -> float:
        """Calculate system uptime percentage."""
        # Simplified calculation based on success rate
        return self.performance_metrics['reliability_score'] * 100
    
    def _calculate_mean_time_to_recovery(self) -> float:
        """Calculate mean time to recovery in seconds."""
        # Simplified MTTR calculation
        return 30.0  # Placeholder - would track actual recovery times
    
    def _calculate_p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        # Simplified calculation - would use histogram in production
        return self.performance_metrics['avg_response_time'] * 1.5
    
    # Context managers
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
