# Performance Scripts

This directory contains production-ready performance benchmarking and regression testing scripts.

## Scripts Overview

### 1. run_comprehensive_benchmark.py

Comprehensive performance benchmarking script with load testing, quantum intelligence validation, and system optimization.

**Usage:**
```bash
# Basic benchmark execution
python run_comprehensive_benchmark.py

# With custom configuration
python run_comprehensive_benchmark.py --config benchmark_config.json

# Enable monitoring and optimization
python run_comprehensive_benchmark.py --monitoring --optimization

# Specific duration override
python run_comprehensive_benchmark.py --duration 600

# Verbose logging
python run_comprehensive_benchmark.py --verbose
```

**Features:**
- Automated load testing scenarios
- Quantum intelligence performance validation
- Database and caching optimization
- Memory leak detection
- Real-time performance monitoring
- SLA compliance tracking
- Automated alerting and reporting

### 2. run_performance_regression_tests.py

Performance regression testing with baseline comparison and CI/CD integration.

**Usage:**
```bash
# Create performance baseline
python run_performance_regression_tests.py baseline --output baseline.json

# Run regression tests
python run_performance_regression_tests.py test --baseline baseline.json

# CI/CD integration with strict mode
python run_performance_regression_tests.py test --baseline baseline.json --strict

# With custom configuration
python run_performance_regression_tests.py test --config config.json --baseline baseline.json
```

**Features:**
- Statistical significance testing
- Automated baseline management
- Regression threshold configuration
- CI/CD pipeline integration
- Detailed regression analysis
- Performance trend tracking

### 3. benchmark_config.json

Comprehensive configuration file for all performance testing scenarios.

**Key Sections:**
- **Load Test Scenarios**: Predefined and custom load patterns
- **Quantum Intelligence Config**: Performance targets for quantum modules
- **Database Performance Config**: Connection pooling and query optimization
- **Monitoring Config**: SLA thresholds and alerting rules
- **Optimization Config**: ML-based parameter tuning settings
- **Environment Config**: Target environment and service dependencies

## Quick Start Guide

### 1. Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure services are running
systemctl status postgresql
systemctl status redis
systemctl status your-api-service
```

### 2. Create Performance Baseline

```bash
# Create baseline for current system
python run_performance_regression_tests.py baseline

# This creates: performance_baseline_<timestamp>.json
```

### 3. Run Comprehensive Benchmark

```bash
# Run full performance suite
python run_comprehensive_benchmark.py --config benchmark_config.json --monitoring

# Check the generated reports:
# - comprehensive_benchmark_report_<timestamp>.json
# - comprehensive_benchmark_report_<timestamp>.html
# - comprehensive_benchmark_report_<timestamp>_summary.txt
```

### 4. Set Up Regression Testing

```bash
# Add to your CI/CD pipeline
python run_performance_regression_tests.py test \
  --baseline performance_baseline.json \
  --strict

# Exit code 0 = pass, 1 = regression detected
```

## Configuration Examples

### Load Test Scenario

```json
{
  "name": "API Load Test",
  "duration": 300,
  "concurrent_users": 50,
  "ramp_up_time": 60,
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

### SLA Configuration

```json
{
  "sla_thresholds": [
    {
      "metric_name": "cpu.percent",
      "threshold_value": 80.0,
      "comparison": "lt",
      "severity": "warning",
      "description": "CPU usage should be below 80%"
    },
    {
      "metric_name": "response_time.p95",
      "threshold_value": 2.0,
      "comparison": "lt",
      "severity": "critical",
      "description": "P95 response time should be below 2s"
    }
  ]
}
```

### Alert Rules

```json
{
  "alert_rules": [
    {
      "name": "high_cpu",
      "condition": "cpu.percent",
      "threshold": 90.0,
      "duration": 30,
      "severity": "critical",
      "action": "scale_up"
    },
    {
      "name": "high_error_rate",
      "condition": "error_rate",
      "threshold": 0.1,
      "duration": 60,
      "severity": "critical",
      "action": "investigate"
    }
  ]
}
```

## CI/CD Integration

### GitHub Actions

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
        
      - name: Start services
        run: |
          docker-compose up -d postgres redis
          python -m src.api.main &
          sleep 30
          
      - name: Run performance tests
        run: |
          cd scripts/performance
          python run_performance_regression_tests.py test \
            --baseline ../../baselines/performance_baseline.json \
            --strict
            
      - name: Upload reports
        uses: actions/upload-artifact@v2
        if: always()
        with:
          name: performance-reports
          path: scripts/performance/regression_report_*.json
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('Performance Tests') {
            steps {
                script {
                    sh '''
                        cd scripts/performance
                        python run_comprehensive_benchmark.py \
                          --config benchmark_config.json \
                          --monitoring
                        
                        python run_performance_regression_tests.py test \
                          --baseline performance_baseline.json
                    '''
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'scripts/performance/*_report_*.json'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'scripts/performance',
                        reportFiles: '*_report_*.html',
                        reportName: 'Performance Report'
                    ])
                }
            }
        }
    }
}
```

## Report Analysis

### Benchmark Report Structure

```json
{
  "benchmark_run_id": "benchmark_1234567890",
  "status": "completed",
  "total_duration": 1205.5,
  "load_tests": {
    "Light Load": {
      "throughput": 245.5,
      "latency": {
        "p50": 0.45,
        "p95": 1.2,
        "p99": 2.1
      },
      "error_rate": 0.02,
      "resource_usage": {
        "cpu": {"avg": 45.2, "max": 67.8},
        "memory": {"avg": 52.1, "max": 71.3}
      }
    }
  },
  "quantum_performance": {
    "quantum_consciousness": {
      "scalability_score": 0.85,
      "operations_per_second": 67.3
    }
  },
  "alerts": [
    {
      "type": "high_latency",
      "severity": "warning",
      "message": "P95 latency exceeded threshold in Heavy Load scenario"
    }
  ],
  "recommendations": [
    "Consider optimizing database queries",
    "Review caching strategy for high-load scenarios"
  ]
}
```

### Key Metrics to Monitor

1. **Throughput**: Requests per second under various loads
2. **Latency**: P50, P95, P99 response times
3. **Error Rate**: Percentage of failed requests
4. **Resource Usage**: CPU, memory, disk, network utilization
5. **Quantum Performance**: Scalability scores for AI modules
6. **Database Performance**: Query times, connection pool efficiency

## Troubleshooting

### Common Issues

1. **Connection Failures**
   ```bash
   # Check service status
   curl http://localhost:8000/health
   
   # Verify database connectivity
   psql -h localhost -p 5432 -U username -d database
   ```

2. **High Memory Usage**
   ```bash
   # Enable memory leak detection
   python run_comprehensive_benchmark.py --config benchmark_config.json --verbose
   
   # Check reports for memory growth patterns
   ```

3. **Test Timeouts**
   ```json
   {
     "load_test_scenarios": [{
       "request_timeout": 60,
       "connection_timeout": 30
     }]
   }
   ```

4. **Insufficient Baseline Data**
   ```bash
   # Recreate baseline with more samples
   python run_performance_regression_tests.py baseline \
     --config extended_config.json
   ```

### Debug Mode

```bash
# Enable verbose logging
python run_comprehensive_benchmark.py --verbose

# Enable debug configuration
{
  "debug_mode": true,
  "log_level": "DEBUG",
  "save_raw_data": true,
  "extended_metrics": true
}
```

## Performance Optimization Tips

### 1. Database Optimization

- Monitor connection pool utilization
- Optimize slow queries identified in reports
- Tune cache hit rates
- Consider read replicas for heavy read workloads

### 2. Application Optimization

- Review high-latency endpoints
- Implement caching for expensive operations
- Optimize memory usage patterns
- Tune garbage collection parameters

### 3. Infrastructure Optimization

- Scale based on CPU/memory thresholds
- Implement auto-scaling policies
- Optimize network configurations
- Monitor disk I/O patterns

### 4. Monitoring and Alerting

- Set up proactive alerting
- Create performance dashboards
- Implement SLA monitoring
- Schedule regular performance reviews

## Advanced Usage

### Custom Load Patterns

```python
# Custom load pattern implementation
class CustomLoadPattern:
    def generate_requests(self, scenario):
        # Implement custom request distribution
        pass
```

### Custom Metrics

```python
# Add custom performance metrics
def custom_metric_collector():
    return {
        'custom_metric': measure_custom_performance(),
        'business_metric': calculate_business_kpi()
    }

benchmarker.add_custom_metric_collector(custom_metric_collector)
```

### Integration with APM Tools

```python
# Integrate with external monitoring
from performance.monitoring import APMIntegration

apm = APMIntegration('datadog')  # or 'newrelic', 'prometheus'
benchmarker.add_apm_integration(apm)
```

This performance benchmarking system provides comprehensive production-ready performance validation, regression testing, and optimization capabilities for Claude-TUI deployment.