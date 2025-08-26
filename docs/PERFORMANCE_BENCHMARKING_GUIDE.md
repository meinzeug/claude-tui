# Performance Benchmarking Guide

This guide covers the comprehensive performance benchmarking system for Claude-TUI production deployment.

## Overview

The Performance Benchmarker provides comprehensive performance validation and optimization capabilities including:

- **Load Testing**: Automated load testing with realistic user scenarios
- **Performance Regression**: Baseline comparison and regression detection
- **Memory & Resource Optimization**: Memory leak detection and resource monitoring
- **Quantum Intelligence Performance**: Validation of quantum modules under load
- **Database Performance**: Query optimization and caching performance
- **Real-time Monitoring**: Performance dashboards and alerting

## Quick Start

### 1. Basic Benchmark Execution

```bash
# Run comprehensive performance benchmark
python scripts/performance/run_comprehensive_benchmark.py

# Run with custom configuration
python scripts/performance/run_comprehensive_benchmark.py --config scripts/performance/benchmark_config.json

# Enable monitoring and optimization
python scripts/performance/run_comprehensive_benchmark.py --monitoring --optimization
```

### 2. Performance Regression Testing

```bash
# Create performance baseline
python scripts/performance/run_performance_regression_tests.py baseline --output baseline.json

# Run regression tests against baseline
python scripts/performance/run_performance_regression_tests.py test --baseline baseline.json

# Strict mode (fail on warnings)
python scripts/performance/run_performance_regression_tests.py test --baseline baseline.json --strict
```

## Architecture

### Core Components

```
src/performance/
├── benchmarking/
│   └── comprehensive_benchmarker.py     # Main benchmarking orchestrator
├── monitoring/
│   ├── realtime_monitor.py             # Real-time performance monitoring
│   ├── performance_dashboard.py        # Advanced dashboard with SQLite storage
│   └── memory_leak_detector.py         # Memory leak detection system
└── optimization/
    └── adaptive_optimizer.py           # ML-based performance optimization
```

### Key Classes

1. **ComprehensivePerformanceBenchmarker**: Main orchestrator for all benchmarking activities
2. **LoadTestEngine**: HTTP load testing with concurrent user simulation
3. **QuantumIntelligencePerformanceTester**: Performance testing for quantum modules
4. **DatabasePerformanceOptimizer**: Database and caching performance analysis
5. **RealtimePerformanceMonitor**: Live monitoring with SLA tracking and alerting
6. **MemoryLeakDetector**: Advanced memory leak detection with heap profiling
7. **AdaptivePerformanceOptimizer**: ML-based parameter tuning and optimization

## Load Testing Scenarios

### Predefined Scenarios

The system includes several predefined load testing scenarios:

```json
{
  "Baseline Load": {
    "concurrent_users": 5,
    "duration": 60,
    "description": "Minimal load for baseline measurements"
  },
  "Light Load": {
    "concurrent_users": 10,
    "duration": 120,
    "description": "Light production load simulation"
  },
  "Medium Load": {
    "concurrent_users": 50,
    "duration": 180,
    "description": "Medium production load with realistic patterns"
  },
  "Heavy Load": {
    "concurrent_users": 100,
    "duration": 300,
    "description": "Heavy load testing system limits"
  },
  "Peak Load": {
    "concurrent_users": 200,
    "duration": 240,
    "description": "Peak load for extreme scenarios"
  },
  "Spike Test": {
    "concurrent_users": 300,
    "duration": 120,
    "description": "Traffic spike simulation"
  }
}
```

### Custom Scenarios

Create custom load test scenarios by modifying the configuration:

```json
{
  "name": "Custom API Test",
  "duration": 300,
  "concurrent_users": 75,
  "ramp_up_time": 90,
  "think_time": 1.0,
  "endpoints": [
    "/api/v1/projects",
    "/api/v1/tasks",
    "/api/v1/ai/generate"
  ],
  "request_patterns": {
    "distribution": "weighted",
    "weights": {
      "/api/v1/projects": 0.4,
      "/api/v1/tasks": 0.4,
      "/api/v1/ai/generate": 0.2
    }
  }
}
```

## Quantum Intelligence Performance Testing

The system validates performance of all 4 quantum intelligence modules:

### Quantum Modules

1. **Quantum Consciousness**: Advanced AI decision making
2. **Quantum Memory**: Distributed memory management
3. **Quantum Reasoning**: Complex problem solving
4. **Quantum Optimization**: Parameter optimization

### Performance Targets

```json
{
  "quantum_consciousness": {
    "min_ops_per_second": 50,
    "max_latency_ms": 200,
    "scalability_threshold": 0.8
  },
  "quantum_memory": {
    "min_ops_per_second": 100,
    "max_latency_ms": 100,
    "scalability_threshold": 0.9
  },
  "quantum_reasoning": {
    "min_ops_per_second": 30,
    "max_latency_ms": 500,
    "scalability_threshold": 0.7
  },
  "quantum_optimization": {
    "min_ops_per_second": 40,
    "max_latency_ms": 300,
    "scalability_threshold": 0.8
  }
}
```

## Memory Leak Detection

### Features

- **Heap Profiling**: Using `tracemalloc` for detailed memory tracking
- **Object Lifecycle Tracking**: Monitor object creation/destruction patterns
- **Growth Pattern Analysis**: Detect suspicious memory growth trends
- **Automatic GC Triggering**: Emergency garbage collection for critical situations
- **Stack Trace Collection**: Identify leak sources with stack traces

### Usage

```python
from src.performance.monitoring.memory_leak_detector import MemoryLeakDetector

# Initialize detector
detector = MemoryLeakDetector({
    'detection_interval': 60,
    'memory_growth_threshold_mb': 50,
    'enable_heap_profiling': True
})

# Start detection
await detector.start_detection()

# Export report
detector.export_detection_report("memory_leak_report.json")
```

### Leak Detection Thresholds

- **Memory Growth**: >50MB growth in analysis window
- **Object Growth**: >1000 objects of same type
- **Confidence Scoring**: ML-based confidence assessment
- **Emergency Actions**: Auto-GC when memory usage >90%

## Real-time Performance Monitoring

### Dashboard Widgets

1. **System Metrics**: CPU, Memory, Disk, Network I/O
2. **Application Metrics**: Response times, throughput, error rates
3. **Database Metrics**: Connection pools, query times
4. **Alert Summary**: Active alerts and SLA violations

### SLA Tracking

```python
# Default SLA Thresholds
sla_thresholds = [
    SLAThreshold('cpu.percent', 80.0, 'lt', 'warning'),
    SLAThreshold('memory.percent', 85.0, 'lt', 'critical'),
    SLAThreshold('response_time.p95', 2.0, 'lt', 'warning'),
    SLAThreshold('error_rate', 0.05, 'lt', 'critical'),
    SLAThreshold('throughput', 100.0, 'gt', 'warning')
]
```

### Alerting Rules

```python
# Example Alert Rules
alert_rules = [
    AlertRule('high_cpu', 'cpu.percent', 90.0, 30, 'critical', 'scale_up'),
    AlertRule('high_memory', 'memory.percent', 90.0, 30, 'critical', 'restart_service'),
    AlertRule('high_error_rate', 'error_rate', 0.1, 60, 'critical', 'investigate')
]
```

## Database Performance Optimization

### Connection Pool Testing

Tests various pool sizes to find optimal configuration:

```python
# Test pool sizes: 5, 10, 20, 50, 100 connections
optimal_size = await db_optimizer.benchmark_connection_pool()
print(f"Optimal pool size: {optimal_size['optimal_pool_size']}")
```

### Query Performance Analysis

Benchmarks common query patterns:

- Simple SELECT statements
- Complex JOINs
- Aggregation queries
- Full-text search
- Transaction-heavy operations

### Cache Performance Testing

Evaluates different caching strategies:

- Redis with LRU eviction
- Redis with LFU eviction
- In-memory cache with TTL
- Cache hit rate optimization

## Performance Regression Testing

### Creating Baselines

```bash
# Create baseline for current codebase
python scripts/performance/run_performance_regression_tests.py baseline

# Create baseline with custom name
python scripts/performance/run_performance_regression_tests.py baseline --output my_baseline.json
```

### Running Regression Tests

```bash
# Test against existing baseline
python scripts/performance/run_performance_regression_tests.py test --baseline baseline.json

# CI/CD integration with exit codes
python scripts/performance/run_performance_regression_tests.py test --baseline baseline.json
echo $?  # 0 = pass, 1 = fail
```

### Regression Thresholds

```json
{
  "regression_thresholds": {
    "throughput_degradation_percent": 10.0,
    "latency_increase_percent": 20.0,
    "error_rate_increase_percent": 50.0,
    "memory_increase_percent": 15.0,
    "cpu_increase_percent": 25.0
  }
}
```

## Adaptive Performance Optimization

### ML-Based Parameter Tuning

The system uses machine learning to optimize performance parameters:

```python
# Parameters that can be optimized
optimization_parameters = [
    'db_pool_size',      # Database connection pool size
    'cache_size_mb',     # Cache memory allocation
    'worker_threads',    # Application worker threads
    'request_timeout',   # Request timeout values
    'batch_size',        # Batch processing sizes
    'compression_enabled' # Response compression
]
```

### Optimization Targets

```python
# Multi-objective optimization
optimization_targets = [
    ('throughput', 'maximize', 0.4),      # 40% weight
    ('latency_p95', 'minimize', 0.3),     # 30% weight
    ('error_rate', 'minimize', 0.2),      # 20% weight
    ('cpu_usage', 'minimize', 0.1)        # 10% weight
]
```

### Bayesian Optimization

Uses Bayesian optimization for efficient parameter space exploration:

- **Exploration**: Initial random parameter sampling
- **Exploitation**: Focus on promising parameter regions
- **Convergence**: Stop when improvement <5%
- **Safety**: Rollback on performance degradation

## Configuration

### Benchmark Configuration

```json
{
  "benchmark_type": "comprehensive",
  "enable_monitoring": true,
  "enable_optimization": false,
  "monitoring_duration": 300,
  "alert_thresholds": {
    "error_rate_max": 0.05,
    "latency_p95_max": 2.0,
    "cpu_usage_max": 85.0,
    "memory_usage_max": 90.0
  }
}
```

### Environment Configuration

```json
{
  "environment_config": {
    "target_environment": "production",
    "api_base_url": "http://localhost:8000",
    "required_services": [
      {
        "name": "api_server",
        "host": "localhost",
        "port": 8000
      },
      {
        "name": "database",
        "host": "localhost",
        "port": 5432
      }
    ]
  }
}
```

## CI/CD Integration

### GitHub Actions Integration

```yaml
name: Performance Regression Tests
on: [push, pull_request]
jobs:
  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run performance tests
        run: |
          python scripts/performance/run_performance_regression_tests.py test \
            --baseline performance_baseline.json \
            --strict
```

### Exit Codes

- **0**: All tests passed
- **1**: Performance regressions detected
- **130**: Interrupted by user

## Reporting

### Report Types

1. **JSON Report**: Complete machine-readable results
2. **HTML Report**: Visual dashboard with charts
3. **Executive Summary**: High-level text summary
4. **Alert Notifications**: Real-time alerts and notifications

### Report Structure

```json
{
  "benchmark_run_id": "benchmark_1234567890",
  "status": "completed",
  "load_tests": {
    "Light Load": {
      "throughput": 245.5,
      "latency": {
        "p50": 0.45,
        "p95": 1.2,
        "p99": 2.1
      },
      "error_rate": 0.02
    }
  },
  "quantum_performance": {
    "quantum_consciousness": {
      "scalability_score": 0.85
    }
  },
  "alerts": [
    {
      "type": "high_latency",
      "severity": "warning",
      "message": "P95 latency above threshold"
    }
  ],
  "recommendations": [
    "Consider optimizing database queries",
    "Review caching strategy"
  ]
}
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks
   - Optimize object lifecycle management
   - Tune garbage collection

2. **Database Connection Issues**
   - Verify database connectivity
   - Check connection pool configuration
   - Monitor active connections

3. **Load Test Failures**
   - Ensure target endpoints are accessible
   - Check for rate limiting
   - Verify network connectivity

4. **Quantum Module Issues**
   - Check quantum module initialization
   - Verify quantum parameter configuration
   - Monitor quantum processing load

### Debug Mode

```bash
# Enable verbose logging
python scripts/performance/run_comprehensive_benchmark.py --verbose

# Enable debug mode in configuration
{
  "debug_mode": true,
  "log_level": "DEBUG",
  "save_raw_data": true
}
```

## Best Practices

### Performance Testing

1. **Consistent Environment**: Use identical hardware and software configurations
2. **Warm-up Periods**: Allow system warm-up before measurement
3. **Multiple Samples**: Run multiple test iterations for statistical significance
4. **Baseline Management**: Regularly update performance baselines
5. **Resource Monitoring**: Monitor system resources during tests

### Memory Management

1. **Regular Profiling**: Schedule regular memory leak detection
2. **Object Lifecycle**: Implement proper object cleanup
3. **Cache Management**: Monitor and tune cache sizes
4. **GC Tuning**: Optimize garbage collection parameters

### Monitoring

1. **SLA Definition**: Define clear performance SLAs
2. **Alert Tuning**: Avoid alert fatigue with proper thresholds
3. **Dashboard Design**: Create actionable dashboards
4. **Historical Data**: Maintain performance history for trend analysis

## API Reference

### Core Classes

#### ComprehensivePerformanceBenchmarker

```python
class ComprehensivePerformanceBenchmarker:
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Execute complete performance benchmark suite"""
        pass
    
    async def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        pass
```

#### LoadTestEngine

```python
class LoadTestEngine:
    async def run_load_test(self, scenario: LoadTestScenario) -> Dict[str, Any]:
        """Execute load test scenario"""
        pass
    
    async def _simulate_user(self, scenario: LoadTestScenario, delay: float):
        """Simulate individual user behavior"""
        pass
```

#### RealtimePerformanceMonitor

```python
class RealtimePerformanceMonitor:
    async def start_monitoring(self):
        """Start real-time performance monitoring"""
        pass
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        pass
```

### Configuration Classes

#### LoadTestScenario

```python
@dataclass
class LoadTestScenario:
    name: str
    duration: int
    concurrent_users: int
    ramp_up_time: int
    think_time: float
    endpoints: List[str]
    request_patterns: Dict[str, Any]
```

#### SLAThreshold

```python
@dataclass
class SLAThreshold:
    metric_name: str
    threshold_value: float
    comparison: str  # 'lt', 'gt', 'eq'
    severity: str    # 'critical', 'warning', 'info'
    description: str
```

This comprehensive performance benchmarking system provides production-ready performance validation, regression testing, and optimization capabilities for Claude-TUI deployment.