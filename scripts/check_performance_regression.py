#!/usr/bin/env python3
"""
Performance regression check script for Claude-TIU
Analyzes benchmark results and fails if performance degrades beyond acceptable thresholds
"""

import json
import sys
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path


class PerformanceRegression:
    """Performance regression analysis for Claude-TIU benchmarks"""
    
    # Performance thresholds (adjust based on requirements)
    THRESHOLDS = {
        'response_time_ms': 500,      # Max 500ms response time
        'memory_usage_mb': 400,       # Max 400MB memory usage
        'cpu_usage_percent': 70,      # Max 70% CPU usage
        'throughput_rps': 100,        # Min 100 requests per second
        'error_rate_percent': 1.0,    # Max 1% error rate
    }
    
    # Regression thresholds (percentage degradation from baseline)
    REGRESSION_THRESHOLDS = {
        'response_time_ms': 20,       # 20% increase is regression
        'memory_usage_mb': 15,        # 15% increase is regression
        'cpu_usage_percent': 10,      # 10% increase is regression
        'throughput_rps': -15,        # 15% decrease is regression
        'error_rate_percent': 100,    # 100% increase is regression
    }
    
    def __init__(self, baseline_file: Optional[Path] = None):
        self.baseline_file = baseline_file
        self.baseline_data = self._load_baseline() if baseline_file else None
    
    def _load_baseline(self) -> Optional[Dict]:
        """Load baseline performance data"""
        if not self.baseline_file or not self.baseline_file.exists():
            print(f"⚠️  Baseline file {self.baseline_file} not found, skipping regression analysis")
            return None
            
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Error loading baseline file: {e}")
            return None
    
    def analyze_benchmark_results(self, results_file: Path) -> bool:
        """
        Analyze benchmark results and check for performance regressions
        Returns True if performance is acceptable, False otherwise
        """
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Error loading benchmark results: {e}")
            return False
        
        print("🔍 Analyzing Claude-TIU Performance Benchmarks")
        print("=" * 60)
        
        # Extract performance metrics
        metrics = self._extract_metrics(results)
        if not metrics:
            print("❌ No performance metrics found in results")
            return False
        
        # Check absolute thresholds
        threshold_passed = self._check_absolute_thresholds(metrics)
        
        # Check regression if baseline exists
        regression_passed = True
        if self.baseline_data:
            regression_passed = self._check_regression(metrics)
        
        # Overall result
        passed = threshold_passed and regression_passed
        
        print("\n" + "=" * 60)
        if passed:
            print("✅ Performance validation PASSED")
            print("📈 All metrics within acceptable thresholds")
        else:
            print("❌ Performance validation FAILED")
            print("🚨 Performance degradation detected")
        
        return passed
    
    def _extract_metrics(self, results: Dict) -> Dict[str, float]:
        """Extract relevant performance metrics from benchmark results"""
        metrics = {}
        
        # Handle pytest-benchmark format
        if 'benchmarks' in results:
            for benchmark in results['benchmarks']:
                name = benchmark.get('name', '')
                stats = benchmark.get('stats', {})
                
                # Response time (mean execution time)
                if 'mean' in stats:
                    metrics['response_time_ms'] = stats['mean'] * 1000  # Convert to ms
                
                # Memory usage (if available in params)
                params = benchmark.get('params', {})
                if 'memory_mb' in params:
                    metrics['memory_usage_mb'] = params['memory_mb']
        
        # Handle custom format
        if 'performance_metrics' in results:
            perf_metrics = results['performance_metrics']
            
            for key, value in perf_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        
        # Handle load testing results (locust/artillery format)
        if 'stats' in results:
            stats = results['stats']
            
            # Response time
            if 'response_time' in stats:
                metrics['response_time_ms'] = stats['response_time']
            elif 'avg_response_time' in stats:
                metrics['response_time_ms'] = stats['avg_response_time']
            
            # Throughput
            if 'requests_per_sec' in stats:
                metrics['throughput_rps'] = stats['requests_per_sec']
            elif 'rps' in stats:
                metrics['throughput_rps'] = stats['rps']
            
            # Error rate
            if 'error_rate' in stats:
                metrics['error_rate_percent'] = stats['error_rate'] * 100
        
        return metrics
    
    def _check_absolute_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check metrics against absolute performance thresholds"""
        print("\n📊 Absolute Threshold Analysis:")
        print("-" * 40)
        
        passed = True
        
        for metric, value in metrics.items():
            if metric not in self.THRESHOLDS:
                continue
                
            threshold = self.THRESHOLDS[metric]
            
            # Determine if metric should be below or above threshold
            if metric in ['response_time_ms', 'memory_usage_mb', 'cpu_usage_percent', 'error_rate_percent']:
                # Lower is better
                metric_passed = value <= threshold
                comparison = "≤"
            else:
                # Higher is better (throughput)
                metric_passed = value >= threshold
                comparison = "≥"
            
            status = "✅" if metric_passed else "❌"
            print(f"{status} {metric}: {value:.2f} {comparison} {threshold}")
            
            if not metric_passed:
                passed = False
        
        return passed
    
    def _check_regression(self, current_metrics: Dict[str, float]) -> bool:
        """Check for performance regression against baseline"""
        print("\n📈 Regression Analysis:")
        print("-" * 40)
        
        baseline_metrics = self._extract_metrics(self.baseline_data)
        if not baseline_metrics:
            print("⚠️  No baseline metrics available for regression analysis")
            return True
        
        passed = True
        
        for metric, current_value in current_metrics.items():
            if metric not in baseline_metrics or metric not in self.REGRESSION_THRESHOLDS:
                continue
            
            baseline_value = baseline_metrics[metric]
            threshold = self.REGRESSION_THRESHOLDS[metric]
            
            # Calculate percentage change
            if baseline_value == 0:
                continue
            
            change_percent = ((current_value - baseline_value) / baseline_value) * 100
            
            # Check if change exceeds regression threshold
            if metric in ['response_time_ms', 'memory_usage_mb', 'cpu_usage_percent', 'error_rate_percent']:
                # Increase is bad
                regression = change_percent > threshold
            else:
                # Decrease is bad (throughput)
                regression = change_percent < threshold
            
            status = "❌" if regression else "✅"
            trend = "↗️" if change_percent > 0 else "↘️"
            
            print(f"{status} {metric}: {baseline_value:.2f} → {current_value:.2f} "
                  f"({change_percent:+.1f}%) {trend}")
            
            if regression:
                passed = False
                print(f"   🚨 Regression detected: {abs(change_percent):.1f}% change exceeds {abs(threshold)}% threshold")
        
        return passed
    
    def save_as_baseline(self, results_file: Path, baseline_file: Path):
        """Save current results as new baseline"""
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            with open(baseline_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"💾 Saved new baseline to {baseline_file}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Error saving baseline: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Check Claude-TIU performance for regressions"
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Benchmark results file (JSON format)"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline benchmark file for regression analysis"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as new baseline"
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("benchmark_baseline.json"),
        help="Baseline file path (default: benchmark_baseline.json)"
    )
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"❌ Benchmark results file not found: {args.results_file}")
        sys.exit(1)
    
    # Initialize performance checker
    baseline_file = args.baseline or args.baseline_file
    checker = PerformanceRegression(baseline_file)
    
    # Analyze results
    passed = checker.analyze_benchmark_results(args.results_file)
    
    # Save as baseline if requested
    if args.save_baseline:
        checker.save_as_baseline(args.results_file, args.baseline_file)
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()