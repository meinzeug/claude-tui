#!/usr/bin/env python3
"""
Analytics Test Runner.

Comprehensive test runner for the performance analytics system.
Provides detailed test execution, coverage reporting, and benchmarking.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def run_command(cmd: str, capture_output: bool = True) -> Dict[str, Any]:
    """Run a shell command and return results."""
    print(f"🔧 Running: {cmd}")
    
    start_time = time.time()
    try:
        if capture_output:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=300
            )
        else:
            result = subprocess.run(cmd, shell=True, timeout=300)
        
        execution_time = time.time() - start_time
        
        return {
            'command': cmd,
            'returncode': result.returncode,
            'stdout': result.stdout if capture_output else '',
            'stderr': result.stderr if capture_output else '',
            'execution_time': execution_time,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'command': cmd,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Command timed out',
            'execution_time': time.time() - start_time,
            'success': False
        }
    except Exception as e:
        return {
            'command': cmd,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'execution_time': time.time() - start_time,
            'success': False
        }


def setup_test_environment():
    """Set up the test environment."""
    print("🔧 Setting up test environment...")
    
    # Create necessary directories
    dirs_to_create = [
        'tests/analytics',
        'logs',
        'tmp',
        'coverage'
    ]
    
    for directory in dirs_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created directory: {directory}")
    
    # Install test dependencies
    dependencies = [
        'pytest>=7.0.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.0.0',
        'pytest-xdist>=3.0.0',
        'pytest-html>=3.1.0',
        'pytest-benchmark>=4.0.0',
        'coverage>=7.0.0',
        'numpy>=1.20.0',
        'psutil>=5.9.0'
    ]
    
    print("📦 Installing test dependencies...")
    for dep in dependencies:
        result = run_command(f"pip install {dep}")
        if result['success']:
            print(f"  ✓ Installed: {dep}")
        else:
            print(f"  ❌ Failed to install: {dep}")
            print(f"     Error: {result['stderr']}")


def run_unit_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run unit tests for analytics components."""
    print("\n🧪 Running Unit Tests...")
    
    test_files = [
        'tests/analytics/test_engine.py',
        'tests/analytics/test_collector.py',
        'tests/analytics/test_optimizer.py'
    ]
    
    # Check if test files exist
    existing_tests = []
    for test_file in test_files:
        if Path(test_file).exists():
            existing_tests.append(test_file)
        else:
            print(f"  ⚠️  Test file not found: {test_file}")
    
    if not existing_tests:
        return {
            'success': False,
            'error': 'No test files found',
            'execution_time': 0,
            'test_count': 0
        }
    
    # Run pytest with coverage
    pytest_cmd = [
        'python', '-m', 'pytest',
        '--cov=src/analytics',
        '--cov-report=html:coverage/html',
        '--cov-report=json:coverage/coverage.json',
        '--cov-report=term-missing',
        '--html=tmp/test_report.html',
        '--self-contained-html',
        '-v' if verbose else '',
        '--tb=short',
        '--durations=10'
    ]
    
    # Add test files
    pytest_cmd.extend(existing_tests)
    
    # Remove empty strings
    pytest_cmd = [arg for arg in pytest_cmd if arg]
    
    result = run_command(' '.join(pytest_cmd))
    
    # Parse results
    test_results = {
        'success': result['success'],
        'execution_time': result['execution_time'],
        'stdout': result['stdout'],
        'stderr': result['stderr']
    }
    
    # Extract test count from output
    if 'collected' in result['stdout']:
        try:
            collected_line = [line for line in result['stdout'].split('\n') if 'collected' in line][0]
            test_count = int(collected_line.split()[0])
            test_results['test_count'] = test_count
        except:
            test_results['test_count'] = 0
    
    # Extract coverage information
    if Path('coverage/coverage.json').exists():
        try:
            with open('coverage/coverage.json', 'r') as f:
                coverage_data = json.load(f)
                test_results['coverage'] = coverage_data['totals']['percent_covered']
        except:
            test_results['coverage'] = 0
    
    print(f"  {'✅' if result['success'] else '❌'} Unit tests {'passed' if result['success'] else 'failed'}")
    if 'test_count' in test_results:
        print(f"  📊 Tests run: {test_results['test_count']}")
    if 'coverage' in test_results:
        print(f"  📈 Coverage: {test_results['coverage']:.1f}%")
    
    return test_results


def run_integration_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run integration tests."""
    print("\n🔗 Running Integration Tests...")
    
    integration_test = 'tests/analytics/test_integration.py'
    
    if not Path(integration_test).exists():
        return {
            'success': False,
            'error': f'Integration test file not found: {integration_test}',
            'execution_time': 0
        }
    
    pytest_cmd = [
        'python', '-m', 'pytest',
        integration_test,
        '-m', 'integration',
        '-v' if verbose else '',
        '--tb=short',
        '--durations=5'
    ]
    
    pytest_cmd = [arg for arg in pytest_cmd if arg]
    
    result = run_command(' '.join(pytest_cmd))
    
    integration_results = {
        'success': result['success'],
        'execution_time': result['execution_time'],
        'stdout': result['stdout'],
        'stderr': result['stderr']
    }
    
    print(f"  {'✅' if result['success'] else '❌'} Integration tests {'passed' if result['success'] else 'failed'}")
    
    return integration_results


def run_performance_benchmarks() -> Dict[str, Any]:
    """Run performance benchmarks."""
    print("\n⚡ Running Performance Benchmarks...")
    
    # Create a simple benchmark script
    benchmark_script = """
import pytest
import time
from datetime import datetime, timedelta
import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.analytics.engine import PerformanceAnalyticsEngine
from src.analytics.models import AnalyticsConfiguration, PerformanceMetrics
from src.core.types import SystemMetrics

def create_test_data(size: int):
    metrics = []
    for i in range(size):
        system_metrics = SystemMetrics(
            cpu_usage=50 + np.random.normal(0, 15),
            memory_usage=60 + np.random.normal(0, 10),
            disk_io=40 + np.random.normal(0, 12),
            network_io=30 + np.random.normal(0, 8),
            timestamp=datetime.now() - timedelta(hours=size-i)
        )
        
        perf_metrics = PerformanceMetrics(
            base_metrics=system_metrics,
            execution_time=2.0 + np.random.normal(0, 0.5),
            throughput=100 + np.random.normal(0, 20),
            error_rate=max(0, 0.01 + np.random.normal(0, 0.005))
        )
        metrics.append(perf_metrics)
    return metrics

def test_engine_performance_small():
    '''Benchmark engine with small dataset (100 metrics)'''
    config = AnalyticsConfiguration()
    engine = PerformanceAnalyticsEngine(config)
    metrics = create_test_data(100)
    
    start = time.time()
    bottlenecks = engine.analyze_bottlenecks(metrics)
    anomalies = engine.detect_anomalies(metrics)
    end = time.time()
    
    assert len(metrics) == 100
    assert isinstance(bottlenecks, list)
    assert isinstance(anomalies, list)
    print(f"Small dataset (100): {end - start:.3f}s")

def test_engine_performance_medium():
    '''Benchmark engine with medium dataset (1000 metrics)'''
    config = AnalyticsConfiguration()
    engine = PerformanceAnalyticsEngine(config)
    metrics = create_test_data(1000)
    
    start = time.time()
    bottlenecks = engine.analyze_bottlenecks(metrics)
    anomalies = engine.detect_anomalies(metrics)
    end = time.time()
    
    assert len(metrics) == 1000
    assert isinstance(bottlenecks, list)
    assert isinstance(anomalies, list)
    print(f"Medium dataset (1000): {end - start:.3f}s")

def test_engine_performance_large():
    '''Benchmark engine with large dataset (5000 metrics)'''
    config = AnalyticsConfiguration()
    engine = PerformanceAnalyticsEngine(config)
    metrics = create_test_data(5000)
    
    start = time.time()
    bottlenecks = engine.analyze_bottlenecks(metrics)
    anomalies = engine.detect_anomalies(metrics)
    end = time.time()
    
    assert len(metrics) == 5000
    assert isinstance(bottlenecks, list)
    assert isinstance(anomalies, list)
    print(f"Large dataset (5000): {end - start:.3f}s")
    
    # Performance assertion - should complete within reasonable time
    assert end - start < 60.0, f"Large dataset processing too slow: {end - start:.3f}s"

if __name__ == "__main__":
    test_engine_performance_small()
    test_engine_performance_medium()
    test_engine_performance_large()
    """
    
    # Write benchmark script
    with open('tmp/benchmark_tests.py', 'w') as f:
        f.write(benchmark_script)
    
    # Run benchmarks
    result = run_command('python tmp/benchmark_tests.py')
    
    benchmark_results = {
        'success': result['success'],
        'execution_time': result['execution_time'],
        'stdout': result['stdout'],
        'stderr': result['stderr']
    }
    
    print(f"  {'✅' if result['success'] else '❌'} Performance benchmarks {'passed' if result['success'] else 'failed'}")
    
    if result['success'] and result['stdout']:
        print("  📊 Benchmark Results:")
        for line in result['stdout'].split('\n'):
            if 'dataset' in line and 's' in line:
                print(f"    {line}")
    
    return benchmark_results


def run_example_scripts() -> Dict[str, Any]:
    """Run example usage scripts."""
    print("\n📚 Running Example Scripts...")
    
    example_script = 'examples/analytics_usage_examples.py'
    
    if not Path(example_script).exists():
        return {
            'success': False,
            'error': f'Example script not found: {example_script}',
            'execution_time': 0
        }
    
    # Run examples (with timeout to prevent hanging)
    result = run_command(f'timeout 120 python {example_script}', capture_output=True)
    
    example_results = {
        'success': result['success'],
        'execution_time': result['execution_time'],
        'stdout': result['stdout'],
        'stderr': result['stderr']
    }
    
    print(f"  {'✅' if result['success'] else '❌'} Example scripts {'completed' if result['success'] else 'failed'}")
    
    return example_results


def generate_test_report(results: Dict[str, Dict[str, Any]]):
    """Generate comprehensive test report."""
    print("\n📊 Generating Test Report...")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_tests_run': 0,
            'total_execution_time': 0,
            'overall_success': True,
            'coverage_percentage': 0
        },
        'detailed_results': results
    }
    
    # Calculate summary statistics
    for test_type, result in results.items():
        if result.get('execution_time'):
            report['summary']['total_execution_time'] += result['execution_time']
        
        if result.get('test_count'):
            report['summary']['total_tests_run'] += result['test_count']
        
        if not result.get('success', True):
            report['summary']['overall_success'] = False
        
        if result.get('coverage'):
            report['summary']['coverage_percentage'] = result['coverage']
    
    # Write JSON report
    with open('tmp/test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Write human-readable report
    with open('tmp/test_report.txt', 'w') as f:
        f.write("ANALYTICS SYSTEM TEST REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Timestamp: {report['timestamp']}\n")
        f.write(f"Overall Success: {'✅ PASS' if report['summary']['overall_success'] else '❌ FAIL'}\n")
        f.write(f"Total Tests Run: {report['summary']['total_tests_run']}\n")
        f.write(f"Total Execution Time: {report['summary']['total_execution_time']:.2f}s\n")
        f.write(f"Test Coverage: {report['summary']['coverage_percentage']:.1f}%\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 30 + "\n\n")
        
        for test_type, result in results.items():
            f.write(f"{test_type.upper()}:\n")
            f.write(f"  Success: {'✅' if result.get('success') else '❌'}\n")
            f.write(f"  Execution Time: {result.get('execution_time', 0):.2f}s\n")
            
            if result.get('test_count'):
                f.write(f"  Tests Run: {result['test_count']}\n")
            
            if result.get('coverage'):
                f.write(f"  Coverage: {result['coverage']:.1f}%\n")
            
            if result.get('error'):
                f.write(f"  Error: {result['error']}\n")
            
            f.write("\n")
    
    print("  ✅ Test report generated:")
    print("    - tmp/test_report.json")
    print("    - tmp/test_report.txt")
    
    if Path('coverage/html/index.html').exists():
        print("    - coverage/html/index.html")


def cleanup_test_artifacts():
    """Clean up test artifacts."""
    print("\n🧹 Cleaning up test artifacts...")
    
    cleanup_patterns = [
        'tmp/benchmark_tests.py',
        '**/__pycache__',
        '**/*.pyc',
        '.pytest_cache'
    ]
    
    for pattern in cleanup_patterns:
        result = run_command(f'find . -path "*{pattern}*" -exec rm -rf {{}} \\; 2>/dev/null || true')
        print(f"  ✓ Cleaned: {pattern}")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Analytics System Test Runner')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--skip-setup', action='store_true', help='Skip environment setup')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--benchmarks-only', action='store_true', help='Run only benchmarks')
    parser.add_argument('--examples-only', action='store_true', help='Run only examples')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup')
    
    args = parser.parse_args()
    
    print("🚀 Analytics System Test Runner")
    print("=" * 50)
    
    # Set up environment
    if not args.skip_setup:
        setup_test_environment()
    
    results = {}
    
    # Run tests based on arguments
    if args.unit_only:
        results['unit_tests'] = run_unit_tests(args.verbose)
    elif args.integration_only:
        results['integration_tests'] = run_integration_tests(args.verbose)
    elif args.benchmarks_only:
        results['performance_benchmarks'] = run_performance_benchmarks()
    elif args.examples_only:
        results['example_scripts'] = run_example_scripts()
    else:
        # Run all tests
        results['unit_tests'] = run_unit_tests(args.verbose)
        results['integration_tests'] = run_integration_tests(args.verbose)
        results['performance_benchmarks'] = run_performance_benchmarks()
        results['example_scripts'] = run_example_scripts()
    
    # Generate report
    generate_test_report(results)
    
    # Cleanup
    if not args.no_cleanup:
        cleanup_test_artifacts()
    
    # Print final summary
    print("\n" + "=" * 50)
    overall_success = all(result.get('success', True) for result in results.values())
    print(f"🏁 Test Suite {'PASSED' if overall_success else 'FAILED'}")
    
    total_time = sum(result.get('execution_time', 0) for result in results.values())
    print(f"⏱️  Total execution time: {total_time:.2f}s")
    
    if not overall_success:
        print("\n❌ Some tests failed. Check the detailed report for more information.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")


if __name__ == "__main__":
    main()