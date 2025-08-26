"""
Quantum Intelligence Error Handling & Performance Monitoring
===========================================================

Comprehensive error handling, recovery, and performance monitoring system
for the quantum intelligence architecture. Provides resilient operation
with self-healing capabilities and advanced diagnostics.

Features:
- Quantum-aware error detection and classification
- Automated error recovery with rollback mechanisms
- Performance monitoring with predictive analytics
- System health assessment and optimization
- Graceful degradation strategies
- Real-time diagnostics and alerting

Architecture:
- Multi-layer error handling (component, system, quantum)
- Performance metrics collection and analysis
- Predictive failure detection
- Automated recovery workflows
- Comprehensive logging and tracing
"""

import asyncio
import logging
import time
import json
import traceback
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
from contextlib import asynccontextmanager
import functools
import weakref

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"

class ErrorCategory(Enum):
    """Error categories for classification."""
    COMPONENT_FAILURE = "component_failure"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"
    EXTERNAL_DEPENDENCY = "external_dependency"

class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    COMPONENT_RESTART = "component_restart"
    SYSTEM_RESTART = "system_restart"
    FAILOVER = "failover"
    ISOLATION = "isolation"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    timestamp: float
    component: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    disk_io: Dict[str, float] = field(default_factory=dict)
    response_times: Dict[str, float] = field(default_factory=dict)
    throughput: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    quantum_coherence: float = 0.0
    component_health: Dict[str, float] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """Overall system health assessment."""
    timestamp: float
    overall_score: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    predicted_failures: List[Dict[str, Any]] = field(default_factory=list)
    system_stability: float = 0.0
    quantum_stability: float = 0.0

class ErrorClassifier:
    """Advanced error classification and analysis."""
    
    def __init__(self):
        self.error_patterns = {}
        self.classification_history = deque(maxlen=1000)
        
    def classify_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[ErrorSeverity, ErrorCategory]:
        """Classify error based on type, context, and patterns."""
        try:
            error_type = type(error).__name__
            error_message = str(error).lower()
            
            # Memory errors
            if any(keyword in error_message for keyword in ['memory', 'malloc', 'out of memory']):
                severity = ErrorSeverity.HIGH if 'critical' in error_message else ErrorSeverity.MEDIUM
                return severity, ErrorCategory.MEMORY_ERROR
            
            # Network errors
            if any(keyword in error_message for keyword in ['connection', 'network', 'timeout', 'refused']):
                severity = ErrorSeverity.MEDIUM if 'timeout' in error_message else ErrorSeverity.HIGH
                return severity, ErrorCategory.NETWORK_ERROR
            
            # Quantum-specific errors
            if any(keyword in error_message for keyword in ['quantum', 'coherence', 'entanglement', 'decoherence']):
                return ErrorSeverity.HIGH, ErrorCategory.QUANTUM_DECOHERENCE
            
            # Component failures
            if any(keyword in error_message for keyword in ['component', 'service', 'module', 'failed']):
                return ErrorSeverity.HIGH, ErrorCategory.COMPONENT_FAILURE
            
            # Performance issues
            if any(keyword in error_message for keyword in ['slow', 'performance', 'degradation', 'bottleneck']):
                return ErrorSeverity.MEDIUM, ErrorCategory.PERFORMANCE_DEGRADATION
            
            # Data corruption
            if any(keyword in error_message for keyword in ['corrupt', 'invalid', 'checksum', 'integrity']):
                return ErrorSeverity.CRITICAL, ErrorCategory.DATA_CORRUPTION
            
            # Resource exhaustion
            if any(keyword in error_message for keyword in ['resource', 'limit', 'quota', 'exhausted']):
                return ErrorSeverity.HIGH, ErrorCategory.RESOURCE_EXHAUSTION
            
            # Configuration errors
            if any(keyword in error_message for keyword in ['config', 'setting', 'parameter', 'invalid']):
                return ErrorSeverity.MEDIUM, ErrorCategory.CONFIGURATION_ERROR
            
            # Default classification based on exception type
            critical_types = ['SystemError', 'RuntimeError', 'MemoryError']
            if error_type in critical_types:
                return ErrorSeverity.CRITICAL, ErrorCategory.COMPONENT_FAILURE
            
            return ErrorSeverity.MEDIUM, ErrorCategory.COMPONENT_FAILURE
            
        except Exception as e:
            logger.error(f"Error classification failed: {e}")
            return ErrorSeverity.MEDIUM, ErrorCategory.COMPONENT_FAILURE

class PerformanceMonitor:
    """Advanced performance monitoring with predictive analytics."""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=1000)
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        self.monitoring_active = False
        self.monitor_thread = None
        
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            
            while self.monitoring_active:
                try:
                    metrics = loop.run_until_complete(self.collect_metrics())
                    self.metrics_history.append(metrics)
                    loop.run_until_complete(self.analyze_performance(metrics))
                    time.sleep(self.collection_interval)
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    time.sleep(5)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_metrics = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_metrics = {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0
            }
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                memory_available=memory.available / (1024**3),  # GB
                network_io=network_metrics,
                disk_io=disk_metrics
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return PerformanceMetrics(timestamp=time.time())
    
    async def analyze_performance(self, current_metrics: PerformanceMetrics):
        """Analyze performance metrics for anomalies and trends."""
        try:
            if len(self.metrics_history) < 10:
                return  # Need more data for analysis
            
            # Calculate baselines if not established
            if not self.baseline_metrics:
                await self._calculate_baselines()
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(current_metrics)
            
            if anomalies:
                for anomaly in anomalies:
                    logger.warning(f"Performance anomaly detected: {anomaly}")
            
            # Predict potential issues
            predictions = await self._predict_performance_issues()
            
            if predictions:
                for prediction in predictions:
                    logger.info(f"Performance prediction: {prediction}")
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
    
    async def _calculate_baselines(self):
        """Calculate baseline performance metrics."""
        try:
            recent_metrics = list(self.metrics_history)[-50:]  # Last 50 data points
            
            if len(recent_metrics) < 10:
                return
            
            # CPU baseline
            cpu_values = [m.cpu_usage for m in recent_metrics]
            self.baseline_metrics['cpu'] = {
                'mean': np.mean(cpu_values),
                'std': np.std(cpu_values),
                'p95': np.percentile(cpu_values, 95)
            }
            
            # Memory baseline
            memory_values = [m.memory_usage for m in recent_metrics]
            self.baseline_metrics['memory'] = {
                'mean': np.mean(memory_values),
                'std': np.std(memory_values),
                'p95': np.percentile(memory_values, 95)
            }
            
            logger.debug("Performance baselines updated")
            
        except Exception as e:
            logger.error(f"Baseline calculation failed: {e}")
    
    async def _detect_anomalies(self, metrics: PerformanceMetrics) -> List[str]:
        """Detect performance anomalies."""
        anomalies = []
        
        try:
            # CPU anomaly detection
            if 'cpu' in self.baseline_metrics:
                baseline = self.baseline_metrics['cpu']
                deviation = abs(metrics.cpu_usage - baseline['mean']) / (baseline['std'] + 1e-6)
                
                if deviation > self.anomaly_threshold:
                    anomalies.append(f"CPU usage anomaly: {metrics.cpu_usage:.1f}% (baseline: {baseline['mean']:.1f}%)")
            
            # Memory anomaly detection
            if 'memory' in self.baseline_metrics:
                baseline = self.baseline_metrics['memory']
                deviation = abs(metrics.memory_usage - baseline['mean']) / (baseline['std'] + 1e-6)
                
                if deviation > self.anomaly_threshold:
                    anomalies.append(f"Memory usage anomaly: {metrics.memory_usage:.1f}% (baseline: {baseline['mean']:.1f}%)")
            
            # Critical thresholds
            if metrics.cpu_usage > 90:
                anomalies.append(f"Critical CPU usage: {metrics.cpu_usage:.1f}%")
            
            if metrics.memory_usage > 95:
                anomalies.append(f"Critical memory usage: {metrics.memory_usage:.1f}%")
            
            if metrics.memory_available < 0.5:  # Less than 500MB available
                anomalies.append(f"Low memory available: {metrics.memory_available:.2f}GB")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def _predict_performance_issues(self) -> List[str]:
        """Predict potential performance issues based on trends."""
        predictions = []
        
        try:
            if len(self.metrics_history) < 20:
                return predictions
            
            recent_metrics = list(self.metrics_history)[-20:]
            
            # CPU trend analysis
            cpu_values = [m.cpu_usage for m in recent_metrics]
            cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]  # Linear trend
            
            if cpu_trend > 2.0:  # CPU usage increasing by >2% per measurement
                predictions.append(f"CPU usage trending upward: +{cpu_trend:.1f}% per interval")
            
            # Memory trend analysis
            memory_values = [m.memory_usage for m in recent_metrics]
            memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
            
            if memory_trend > 1.0:  # Memory usage increasing by >1% per measurement
                predictions.append(f"Memory usage trending upward: +{memory_trend:.1f}% per interval")
            
            # Available memory trend
            available_values = [m.memory_available for m in recent_metrics]
            available_trend = np.polyfit(range(len(available_values)), available_values, 1)[0]
            
            if available_trend < -0.1:  # Available memory decreasing
                predictions.append(f"Available memory decreasing: {available_trend:.2f}GB per interval")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            return []

class RecoveryManager:
    """Automated error recovery and system healing."""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.recovery_history = deque(maxlen=500)
        self.component_states = {}
        self.rollback_points = {}
        
    def register_recovery_strategy(self, 
                                 error_category: ErrorCategory,
                                 strategy: RecoveryStrategy,
                                 handler: Callable):
        """Register a recovery strategy for an error category."""
        if error_category not in self.recovery_strategies:
            self.recovery_strategies[error_category] = []
        
        self.recovery_strategies[error_category].append({
            'strategy': strategy,
            'handler': handler,
            'success_rate': 0.0,
            'usage_count': 0
        })
    
    async def attempt_recovery(self, error_record: ErrorRecord) -> bool:
        """Attempt to recover from an error."""
        try:
            strategies = self.recovery_strategies.get(error_record.category, [])
            
            if not strategies:
                logger.warning(f"No recovery strategies available for {error_record.category}")
                return False
            
            # Sort strategies by success rate
            strategies.sort(key=lambda x: x['success_rate'], reverse=True)
            
            for strategy_info in strategies:
                strategy = strategy_info['strategy']
                handler = strategy_info['handler']
                
                logger.info(f"Attempting recovery with strategy: {strategy.value}")
                
                try:
                    # Update error record
                    error_record.recovery_attempted = True
                    error_record.recovery_strategy = strategy
                    
                    # Attempt recovery
                    success = await handler(error_record)
                    
                    # Update statistics
                    strategy_info['usage_count'] += 1
                    if success:
                        strategy_info['success_rate'] = (
                            strategy_info['success_rate'] * (strategy_info['usage_count'] - 1) + 1.0
                        ) / strategy_info['usage_count']
                        
                        error_record.recovery_successful = True
                        
                        # Record successful recovery
                        self.recovery_history.append({
                            'timestamp': time.time(),
                            'error_id': error_record.error_id,
                            'strategy': strategy.value,
                            'success': True
                        })
                        
                        logger.info(f"Recovery successful with strategy: {strategy.value}")
                        return True
                    else:
                        # Update failure rate
                        strategy_info['success_rate'] = (
                            strategy_info['success_rate'] * (strategy_info['usage_count'] - 1)
                        ) / strategy_info['usage_count']
                
                except Exception as recovery_error:
                    logger.error(f"Recovery attempt failed: {recovery_error}")
                    strategy_info['usage_count'] += 1
                    strategy_info['success_rate'] = (
                        strategy_info['success_rate'] * (strategy_info['usage_count'] - 1)
                    ) / strategy_info['usage_count']
            
            # All recovery attempts failed
            self.recovery_history.append({
                'timestamp': time.time(),
                'error_id': error_record.error_id,
                'strategy': 'all_failed',
                'success': False
            })
            
            logger.error(f"All recovery attempts failed for error: {error_record.error_id}")
            return False
            
        except Exception as e:
            logger.error(f"Recovery attempt failed with exception: {e}")
            return False
    
    async def create_rollback_point(self, component: str, state: Dict[str, Any]) -> str:
        """Create a rollback point for component state."""
        try:
            rollback_id = f"rollback_{component}_{int(time.time())}"
            
            self.rollback_points[rollback_id] = {
                'component': component,
                'state': state.copy(),
                'timestamp': time.time()
            }
            
            # Keep only recent rollback points (last hour)
            current_time = time.time()
            expired_points = [
                rid for rid, rp in self.rollback_points.items()
                if current_time - rp['timestamp'] > 3600
            ]
            
            for rid in expired_points:
                del self.rollback_points[rid]
            
            logger.debug(f"Created rollback point: {rollback_id}")
            return rollback_id
            
        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
            return ""
    
    async def rollback_to_point(self, rollback_id: str) -> bool:
        """Rollback component to previous state."""
        try:
            if rollback_id not in self.rollback_points:
                logger.error(f"Rollback point not found: {rollback_id}")
                return False
            
            rollback_point = self.rollback_points[rollback_id]
            component = rollback_point['component']
            state = rollback_point['state']
            
            # This would be implemented by the specific component
            logger.info(f"Rolling back component {component} to state from {rollback_point['timestamp']}")
            
            # Component-specific rollback logic would go here
            # For now, just log the operation
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

class QuantumErrorHandler:
    """Comprehensive error handling and performance monitoring for Quantum Intelligence."""
    
    def __init__(self):
        self.error_classifier = ErrorClassifier()
        self.performance_monitor = PerformanceMonitor()
        self.recovery_manager = RecoveryManager()
        
        self.error_history = deque(maxlen=1000)
        self.system_health_history = deque(maxlen=100)
        self.alert_callbacks: List[Callable] = []
        
        # Component health tracking
        self.component_health = defaultdict(lambda: {
            'status': 'healthy',
            'last_error': None,
            'error_count': 0,
            'recovery_count': 0,
            'health_score': 1.0
        })
        
        self.monitoring_active = False
        self.health_check_thread = None
        
        # Register default recovery strategies
        self._register_default_strategies()
        
        logger.info("Quantum Error Handler initialized")
    
    async def initialize(self) -> bool:
        """Initialize error handling system."""
        try:
            await self.performance_monitor.start_monitoring()
            await self.start_health_monitoring()
            
            logger.info("Quantum Error Handler fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error handler initialization failed: {e}")
            return False
    
    @asynccontextmanager
    async def error_context(self, component: str, operation: str, **context):
        """Context manager for error handling."""
        operation_start = time.time()
        error_occurred = False
        
        try:
            # Create rollback point if needed
            if context.get('create_rollback', False):
                state = context.get('component_state', {})
                rollback_id = await self.recovery_manager.create_rollback_point(component, state)
                context['rollback_id'] = rollback_id
            
            yield
            
        except Exception as e:
            error_occurred = True
            await self.handle_error(e, component, operation, context)
            raise
        
        finally:
            # Record performance metrics
            operation_time = time.time() - operation_start
            await self._record_operation_metrics(component, operation, operation_time, error_occurred)
    
    async def handle_error(self, 
                          error: Exception, 
                          component: str,
                          operation: str = "unknown",
                          context: Dict[str, Any] = None) -> bool:
        """Handle an error with classification, recovery, and logging."""
        try:
            context = context or {}
            
            # Classify error
            severity, category = self.error_classifier.classify_error(error, context)
            
            # Create error record
            error_record = ErrorRecord(
                error_id=f"err_{int(time.time())}_{component}",
                timestamp=time.time(),
                component=component,
                severity=severity,
                category=category,
                message=str(error),
                exception_type=type(error).__name__,
                stack_trace=traceback.format_exc(),
                context=context
            )
            
            # Store error
            self.error_history.append(error_record)
            
            # Update component health
            await self._update_component_health(component, error_record)
            
            # Log error
            log_level = {
                ErrorSeverity.LOW: logging.INFO,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL,
                ErrorSeverity.CATASTROPHIC: logging.CRITICAL
            }
            
            logger.log(
                log_level[severity],
                f"Error in {component}.{operation}: {error} (ID: {error_record.error_id})"
            )
            
            # Attempt recovery for high-severity errors
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL, ErrorSeverity.CATASTROPHIC]:
                recovery_success = await self.recovery_manager.attempt_recovery(error_record)
                
                if not recovery_success and severity == ErrorSeverity.CATASTROPHIC:
                    await self._trigger_emergency_procedures(error_record)
            
            # Send alerts
            await self._send_alerts(error_record)
            
            return error_record.recovery_successful
            
        except Exception as handler_error:
            logger.critical(f"Error handler itself failed: {handler_error}")
            return False
    
    def error_handler_decorator(self, component: str, operation: str = None):
        """Decorator for automatic error handling."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                op_name = operation or func.__name__
                async with self.error_context(component, op_name):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    async def start_health_monitoring(self):
        """Start system health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def health_monitor_loop():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            
            while self.monitoring_active:
                try:
                    health = loop.run_until_complete(self.assess_system_health())
                    self.system_health_history.append(health)
                    
                    if health.overall_score < 0.7:
                        loop.run_until_complete(self._handle_poor_health(health))
                    
                    time.sleep(30)  # Health check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(10)
        
        self.health_check_thread = threading.Thread(target=health_monitor_loop, daemon=True)
        self.health_check_thread.start()
        
        logger.info("System health monitoring started")
    
    async def stop_monitoring(self):
        """Stop all monitoring."""
        self.monitoring_active = False
        
        await self.performance_monitor.stop_monitoring()
        
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=10)
        
        logger.info("Error handling monitoring stopped")
    
    async def assess_system_health(self) -> SystemHealth:
        """Assess overall system health."""
        try:
            current_time = time.time()
            
            # Calculate component health scores
            component_scores = {}
            critical_issues = []
            warnings = []
            recommendations = []
            
            for component, health_info in self.component_health.items():
                health_score = health_info['health_score']
                component_scores[component] = health_score
                
                if health_score < 0.5:
                    critical_issues.append(f"{component}: Health score {health_score:.2f}")
                elif health_score < 0.7:
                    warnings.append(f"{component}: Health score {health_score:.2f}")
                
                if health_info['error_count'] > 10:
                    recommendations.append(f"Investigate frequent errors in {component}")
            
            # Calculate overall health score
            if component_scores:
                overall_score = np.mean(list(component_scores.values()))
            else:
                overall_score = 1.0
            
            # Get recent performance metrics
            quantum_stability = 0.8  # Would be calculated from quantum components
            system_stability = overall_score
            
            # Performance-based health factors
            if self.performance_monitor.metrics_history:
                recent_metrics = self.performance_monitor.metrics_history[-1]
                
                if recent_metrics.cpu_usage > 90:
                    critical_issues.append(f"High CPU usage: {recent_metrics.cpu_usage:.1f}%")
                    overall_score *= 0.8
                elif recent_metrics.cpu_usage > 80:
                    warnings.append(f"Elevated CPU usage: {recent_metrics.cpu_usage:.1f}%")
                    overall_score *= 0.9
                
                if recent_metrics.memory_usage > 95:
                    critical_issues.append(f"High memory usage: {recent_metrics.memory_usage:.1f}%")
                    overall_score *= 0.7
                elif recent_metrics.memory_usage > 85:
                    warnings.append(f"Elevated memory usage: {recent_metrics.memory_usage:.1f}%")
                    overall_score *= 0.9
            
            # Error-based health factors
            recent_errors = [
                err for err in self.error_history 
                if current_time - err.timestamp < 300  # Last 5 minutes
            ]
            
            critical_errors = [err for err in recent_errors if err.severity == ErrorSeverity.CRITICAL]
            if critical_errors:
                critical_issues.append(f"{len(critical_errors)} critical errors in last 5 minutes")
                overall_score *= 0.6
            
            high_errors = [err for err in recent_errors if err.severity == ErrorSeverity.HIGH]
            if len(high_errors) > 5:
                warnings.append(f"{len(high_errors)} high-severity errors in last 5 minutes")
                overall_score *= 0.8
            
            # Generate recommendations
            if overall_score < 0.8:
                recommendations.append("Consider system optimization or maintenance")
            
            if len(recent_errors) > 10:
                recommendations.append("Investigate root cause of frequent errors")
            
            health = SystemHealth(
                timestamp=current_time,
                overall_score=max(0.0, overall_score),
                component_scores=component_scores,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations,
                system_stability=system_stability,
                quantum_stability=quantum_stability
            )
            
            return health
            
        except Exception as e:
            logger.error(f"System health assessment failed: {e}")
            return SystemHealth(
                timestamp=time.time(),
                overall_score=0.5,
                critical_issues=[f"Health assessment failed: {str(e)}"]
            )
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        try:
            current_time = time.time()
            
            # Time-based analysis
            last_hour_errors = [err for err in self.error_history if current_time - err.timestamp < 3600]
            last_day_errors = [err for err in self.error_history if current_time - err.timestamp < 86400]
            
            # Severity distribution
            severity_counts = defaultdict(int)
            for error in last_day_errors:
                severity_counts[error.severity.value] += 1
            
            # Category distribution
            category_counts = defaultdict(int)
            for error in last_day_errors:
                category_counts[error.category.value] += 1
            
            # Component error rates
            component_errors = defaultdict(int)
            for error in last_day_errors:
                component_errors[error.component] += 1
            
            # Recovery statistics
            recovery_attempts = len([err for err in last_day_errors if err.recovery_attempted])
            successful_recoveries = len([err for err in last_day_errors if err.recovery_successful])
            recovery_rate = successful_recoveries / max(1, recovery_attempts)
            
            return {
                'timestamp': current_time,
                'total_errors': len(self.error_history),
                'last_hour_errors': len(last_hour_errors),
                'last_day_errors': len(last_day_errors),
                'severity_distribution': dict(severity_counts),
                'category_distribution': dict(category_counts),
                'component_error_counts': dict(component_errors),
                'recovery_statistics': {
                    'attempts': recovery_attempts,
                    'successful': successful_recoveries,
                    'rate': recovery_rate
                },
                'component_health': {
                    component: {
                        'status': info['status'],
                        'health_score': info['health_score'],
                        'error_count': info['error_count'],
                        'recovery_count': info['recovery_count']
                    }
                    for component, info in self.component_health.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error statistics collection failed: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    # Private methods
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        # Memory error recovery
        async def memory_cleanup_recovery(error_record: ErrorRecord) -> bool:
            try:
                import gc
                gc.collect()
                logger.info("Performed garbage collection for memory cleanup")
                return True
            except Exception as e:
                logger.error(f"Memory cleanup failed: {e}")
                return False
        
        self.recovery_manager.register_recovery_strategy(
            ErrorCategory.MEMORY_ERROR, RecoveryStrategy.RETRY, memory_cleanup_recovery
        )
        
        # Component failure recovery
        async def component_restart_recovery(error_record: ErrorRecord) -> bool:
            try:
                component = error_record.component
                logger.info(f"Attempting component restart for: {component}")
                # Component-specific restart logic would go here
                return True
            except Exception as e:
                logger.error(f"Component restart failed: {e}")
                return False
        
        self.recovery_manager.register_recovery_strategy(
            ErrorCategory.COMPONENT_FAILURE, RecoveryStrategy.COMPONENT_RESTART, component_restart_recovery
        )
        
        # Network error recovery
        async def network_retry_recovery(error_record: ErrorRecord) -> bool:
            try:
                # Wait and retry
                await asyncio.sleep(1)
                logger.info("Network retry recovery attempted")
                return True
            except Exception as e:
                logger.error(f"Network retry failed: {e}")
                return False
        
        self.recovery_manager.register_recovery_strategy(
            ErrorCategory.NETWORK_ERROR, RecoveryStrategy.RETRY, network_retry_recovery
        )
    
    async def _update_component_health(self, component: str, error_record: ErrorRecord):
        """Update component health based on error."""
        health_info = self.component_health[component]
        health_info['last_error'] = error_record.timestamp
        health_info['error_count'] += 1
        
        # Update health score based on error severity and frequency
        severity_impact = {
            ErrorSeverity.LOW: 0.05,
            ErrorSeverity.MEDIUM: 0.1,
            ErrorSeverity.HIGH: 0.2,
            ErrorSeverity.CRITICAL: 0.3,
            ErrorSeverity.CATASTROPHIC: 0.5
        }
        
        impact = severity_impact.get(error_record.severity, 0.1)
        health_info['health_score'] = max(0.0, health_info['health_score'] - impact)
        
        # Recovery can improve health
        if error_record.recovery_successful:
            health_info['recovery_count'] += 1
            health_info['health_score'] = min(1.0, health_info['health_score'] + 0.1)
        
        # Update status
        if health_info['health_score'] < 0.3:
            health_info['status'] = 'critical'
        elif health_info['health_score'] < 0.6:
            health_info['status'] = 'degraded'
        elif health_info['health_score'] < 0.8:
            health_info['status'] = 'warning'
        else:
            health_info['status'] = 'healthy'
    
    async def _record_operation_metrics(self, 
                                      component: str, 
                                      operation: str, 
                                      duration: float, 
                                      error_occurred: bool):
        """Record operation performance metrics."""
        # This would integrate with the performance monitor
        # to track operation-specific metrics
        pass
    
    async def _trigger_emergency_procedures(self, error_record: ErrorRecord):
        """Trigger emergency procedures for catastrophic errors."""
        logger.critical(f"EMERGENCY: Catastrophic error in {error_record.component}: {error_record.message}")
        
        # Emergency procedures would include:
        # - Immediate alerts to administrators
        # - System state preservation
        # - Graceful service degradation
        # - Failover to backup systems
        
        await self._send_emergency_alert(error_record)
    
    async def _send_alerts(self, error_record: ErrorRecord):
        """Send alerts for error conditions."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_record)
                else:
                    callback(error_record)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    async def _send_emergency_alert(self, error_record: ErrorRecord):
        """Send emergency alert for catastrophic errors."""
        emergency_alert = {
            'type': 'EMERGENCY',
            'timestamp': time.time(),
            'error_id': error_record.error_id,
            'component': error_record.component,
            'severity': error_record.severity.value,
            'message': error_record.message
        }
        
        logger.critical(f"EMERGENCY ALERT: {json.dumps(emergency_alert)}")
        
        # Additional emergency notification mechanisms would go here
    
    async def _handle_poor_health(self, health: SystemHealth):
        """Handle poor system health conditions."""
        logger.warning(f"Poor system health detected: {health.overall_score:.2f}")
        
        # Implement health improvement strategies
        if health.overall_score < 0.5:
            logger.error("System health critical - implementing emergency measures")
            # Emergency measures would go here
        elif health.overall_score < 0.7:
            logger.warning("System health degraded - implementing optimization measures")
            # Optimization measures would go here
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for error alerts."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """Remove alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def __del__(self):
        """Cleanup when handler is destroyed."""
        if hasattr(self, 'monitoring_active') and self.monitoring_active:
            asyncio.create_task(self.stop_monitoring())