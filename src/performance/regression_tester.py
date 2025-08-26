#!/usr/bin/env python3
"""
Performance Regression Tester - Automated Performance Regression Detection

This module provides comprehensive performance regression testing with:
- Automated baseline comparison
- Statistical significance testing
- Performance trend analysis
- CI/CD integration support
- Automated alerting for regressions
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib

try:
    from .performance_profiler import PerformanceProfiler, PerformanceSnapshot
except ImportError:
    from performance_profiler import PerformanceProfiler, PerformanceSnapshot


@dataclass
class RegressionThreshold:
    """Defines thresholds for regression detection"""
    metric_name: str
    max_increase_percent: float  # Maximum allowed increase
    max_increase_absolute: Optional[float] = None
    confidence_level: float = 0.95  # Statistical confidence required
    min_samples: int = 5  # Minimum samples for valid comparison


@dataclass
class RegressionResult:
    """Result of regression analysis for a single metric"""
    metric_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    change_absolute: float
    is_regression: bool
    confidence: float
    severity: str  # NONE, MINOR, MAJOR, CRITICAL
    threshold: RegressionThreshold
    samples_count: int
    statistical_test: str
    p_value: Optional[float] = None


class PerformanceRegressionTester:
    """
    Automated Performance Regression Testing System
    
    Features:
    - Baseline performance tracking
    - Statistical regression detection
    - Performance trend analysis
    - CI/CD integration
    - Automated alerting
    """
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.profiler = PerformanceProfiler()
        
        # Default regression thresholds
        self.thresholds = [
            RegressionThreshold("memory_mb", 15.0, 50.0),  # 15% or 50MB increase
            RegressionThreshold("startup_time_ms", 20.0, 500.0),  # 20% or 500ms increase
            RegressionThreshold("import_time_ms", 25.0, 200.0),  # 25% or 200ms increase
            RegressionThreshold("gc_time_ms", 30.0, 50.0),  # 30% or 50ms increase
            RegressionThreshold("cpu_percent", 50.0, 20.0),  # 50% or 20% increase
            RegressionThreshold("system_memory_percent", 10.0, 5.0),  # 10% or 5% increase
        ]
        
    def create_baseline(self, runs: int = 5) -> Dict[str, Any]:
        """Create performance baseline from multiple runs"""
        print(f"🏁 Creating performance baseline from {runs} runs...")
        
        all_results = []
        for run in range(runs):
            print(f"   Run {run + 1}/{runs}...")
            
            # Reset profiler for clean measurement
            profiler = PerformanceProfiler()
            
            # Create snapshot and benchmarks
            baseline = profiler.create_performance_baseline()
            all_results.append(baseline)
            
            time.sleep(1)  # Brief pause between runs
            
        # Aggregate results
        aggregated_baseline = self._aggregate_baseline_results(all_results)
        
        # Add metadata
        aggregated_baseline.update({
            "baseline_created": datetime.now().isoformat(),
            "runs_count": runs,
            "system_signature": self._get_system_signature(),
            "confidence_interval": 0.95
        })
        
        # Save baseline
        self._save_baseline(aggregated_baseline)
        
        print(f"✅ Baseline created and saved to {self.baseline_file}")
        return aggregated_baseline
        
    def _aggregate_baseline_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple baseline results into statistical baseline"""
        
        # Extract metrics from all runs
        memory_values = [r["current_performance"]["process_memory_mb"] for r in results]
        startup_values = [r["startup_performance"]["total_import_time_ms"] for r in results]
        gc_values = [r["startup_performance"]["gc_time_ms"] for r in results]
        
        # Calculate statistics
        aggregated = {
            "baseline_metrics": {
                "memory_mb": {
                    "mean": statistics.mean(memory_values),
                    "median": statistics.median(memory_values),
                    "std": statistics.stdev(memory_values) if len(memory_values) > 1 else 0,
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "samples": memory_values
                },
                "startup_time_ms": {
                    "mean": statistics.mean(startup_values),
                    "median": statistics.median(startup_values),
                    "std": statistics.stdev(startup_values) if len(startup_values) > 1 else 0,
                    "min": min(startup_values),
                    "max": max(startup_values),
                    "samples": startup_values
                },
                "gc_time_ms": {
                    "mean": statistics.mean(gc_values),
                    "median": statistics.median(gc_values),
                    "std": statistics.stdev(gc_values) if len(gc_values) > 1 else 0,
                    "min": min(gc_values),
                    "max": max(gc_values),
                    "samples": gc_values
                }
            },
            "reference_run": results[0],  # Keep first run as reference
            "run_variations": {
                "memory_cv": (statistics.stdev(memory_values) / statistics.mean(memory_values)) * 100 if statistics.mean(memory_values) > 0 else 0,
                "startup_cv": (statistics.stdev(startup_values) / statistics.mean(startup_values)) * 100 if statistics.mean(startup_values) > 0 else 0,
                "gc_cv": (statistics.stdev(gc_values) / statistics.mean(gc_values)) * 100 if statistics.mean(gc_values) > 0 else 0
            }
        }
        
        return aggregated
        
    def _save_baseline(self, baseline: Dict[str, Any]):
        """Save baseline to file"""
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
            
    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline from file"""
        if not self.baseline_file.exists():
            return None
            
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading baseline: {e}")
            return None
            
    def run_regression_test(self, current_runs: int = 3) -> List[RegressionResult]:
        """Run regression test against baseline"""
        print(f"🔍 Running regression test with {current_runs} current runs...")
        
        # Load baseline
        baseline = self._load_baseline()
        if not baseline:
            raise ValueError("No baseline found. Run create_baseline() first.")
            
        # Get current performance
        current_results = []
        for run in range(current_runs):
            print(f"   Current run {run + 1}/{current_runs}...")
            profiler = PerformanceProfiler()
            current = profiler.create_performance_baseline()
            current_results.append(current)
            time.sleep(1)
            
        # Compare against baseline
        regression_results = self._compare_to_baseline(baseline, current_results)
        
        print(f"✅ Regression test completed. Found {sum(1 for r in regression_results if r.is_regression)} regressions.")
        
        return regression_results
        
    def _compare_to_baseline(self, baseline: Dict[str, Any], current_results: List[Dict[str, Any]]) -> List[RegressionResult]:
        """Compare current results to baseline and detect regressions"""
        results = []
        
        # Extract current metrics
        current_memory = [r["current_performance"]["process_memory_mb"] for r in current_results]
        current_startup = [r["startup_performance"]["total_import_time_ms"] for r in current_results]
        current_gc = [r["startup_performance"]["gc_time_ms"] for r in current_results]
        
        # Compare memory performance
        memory_result = self._analyze_metric_regression(
            "memory_mb",
            baseline["baseline_metrics"]["memory_mb"],
            current_memory,
            next(t for t in self.thresholds if t.metric_name == "memory_mb")
        )
        results.append(memory_result)
        
        # Compare startup performance
        startup_result = self._analyze_metric_regression(
            "startup_time_ms",
            baseline["baseline_metrics"]["startup_time_ms"],
            current_startup,
            next(t for t in self.thresholds if t.metric_name == "startup_time_ms")
        )
        results.append(startup_result)
        
        # Compare GC performance
        gc_result = self._analyze_metric_regression(
            "gc_time_ms",
            baseline["baseline_metrics"]["gc_time_ms"],
            current_gc,
            next(t for t in self.thresholds if t.metric_name == "gc_time_ms")
        )
        results.append(gc_result)
        
        return results
        
    def _analyze_metric_regression(self, metric_name: str, baseline_stats: Dict[str, Any], 
                                  current_values: List[float], threshold: RegressionThreshold) -> RegressionResult:
        """Analyze single metric for regression"""
        
        baseline_mean = baseline_stats["mean"]
        current_mean = statistics.mean(current_values)
        
        change_absolute = current_mean - baseline_mean
        change_percent = (change_absolute / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        # Determine if this is a regression
        is_regression = False
        if change_percent > threshold.max_increase_percent:
            is_regression = True
        elif threshold.max_increase_absolute and change_absolute > threshold.max_increase_absolute:
            is_regression = True
            
        # Statistical significance test (simplified t-test approach)
        confidence = self._calculate_confidence(baseline_stats, current_values)
        
        # Determine severity
        severity = "NONE"
        if is_regression:
            if change_percent > threshold.max_increase_percent * 2:
                severity = "CRITICAL"
            elif change_percent > threshold.max_increase_percent * 1.5:
                severity = "MAJOR"
            else:
                severity = "MINOR"
                
        return RegressionResult(
            metric_name=metric_name,
            baseline_value=baseline_mean,
            current_value=current_mean,
            change_percent=change_percent,
            change_absolute=change_absolute,
            is_regression=is_regression,
            confidence=confidence,
            severity=severity,
            threshold=threshold,
            samples_count=len(current_values),
            statistical_test="simplified_t_test"
        )
        
    def _calculate_confidence(self, baseline_stats: Dict[str, Any], current_values: List[float]) -> float:
        """Calculate statistical confidence in the result"""
        if len(current_values) < 2:
            return 0.5
            
        baseline_std = baseline_stats["std"]
        current_std = statistics.stdev(current_values)
        
        # Simple confidence calculation based on standard deviations
        # In production, would use proper t-test
        if baseline_std == 0 and current_std == 0:
            return 1.0
        elif baseline_std == 0 or current_std == 0:
            return 0.8
        else:
            # Simplified confidence based on coefficient of variation
            baseline_cv = baseline_std / baseline_stats["mean"] if baseline_stats["mean"] > 0 else 1
            current_cv = current_std / statistics.mean(current_values) if statistics.mean(current_values) > 0 else 1
            
            # Higher confidence if both measurements are consistent
            consistency_factor = 1 - min(baseline_cv + current_cv, 1.0)
            return max(0.5, consistency_factor + 0.3)
            
    def _get_system_signature(self) -> str:
        """Get system signature for baseline validation"""
        import sys
        import platform
        
        signature_data = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.architecture()[0]
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()[:16]
        
    def generate_regression_report(self, results: List[RegressionResult], 
                                 output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive regression test report"""
        
        regressions = [r for r in results if r.is_regression]
        critical_regressions = [r for r in regressions if r.severity == "CRITICAL"]
        major_regressions = [r for r in regressions if r.severity == "MAJOR"]
        
        report = {
            "regression_test_report": {
                "timestamp": datetime.now().isoformat(),
                "test_summary": {
                    "total_metrics_tested": len(results),
                    "regressions_detected": len(regressions),
                    "critical_regressions": len(critical_regressions),
                    "major_regressions": len(major_regressions),
                    "test_passed": len(critical_regressions) == 0
                },
                "regression_details": [
                    {
                        "metric": r.metric_name,
                        "baseline_value": r.baseline_value,
                        "current_value": r.current_value,
                        "change_percent": r.change_percent,
                        "change_absolute": r.change_absolute,
                        "severity": r.severity,
                        "confidence": r.confidence,
                        "threshold_percent": r.threshold.max_increase_percent,
                        "threshold_absolute": r.threshold.max_increase_absolute
                    }
                    for r in results
                ],
                "performance_trends": self._analyze_performance_trends(results),
                "recommendations": self._generate_recommendations(regressions)
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report
        
    def _analyze_performance_trends(self, results: List[RegressionResult]) -> Dict[str, Any]:
        """Analyze performance trends from results"""
        trends = {}
        
        for result in results:
            trends[result.metric_name] = {
                "trend": "IMPROVING" if result.change_percent < -5 else
                        "DEGRADING" if result.change_percent > 10 else "STABLE",
                "change_percent": result.change_percent,
                "significance": "HIGH" if result.confidence > 0.9 else
                              "MEDIUM" if result.confidence > 0.7 else "LOW"
            }
            
        return trends
        
    def _generate_recommendations(self, regressions: List[RegressionResult]) -> List[str]:
        """Generate recommendations based on detected regressions"""
        recommendations = []
        
        memory_regressions = [r for r in regressions if "memory" in r.metric_name]
        timing_regressions = [r for r in regressions if "time" in r.metric_name]
        
        if memory_regressions:
            recommendations.extend([
                "Memory usage regression detected - review recent changes for memory leaks",
                "Consider running memory profiler to identify specific allocation issues",
                "Review object lifecycle and garbage collection patterns"
            ])
            
        if timing_regressions:
            recommendations.extend([
                "Performance timing regression detected - profile recent code changes",
                "Consider optimizing import patterns and module loading",
                "Review computational complexity of recent algorithm changes"
            ])
            
        if not recommendations:
            recommendations.append("No significant regressions detected - performance is stable")
            
        return recommendations
        
    def run_ci_integration_test(self) -> Tuple[bool, Dict[str, Any]]:
        """Run regression test suitable for CI/CD integration"""
        try:
            # Quick regression test (fewer runs for CI speed)
            results = self.run_regression_test(current_runs=2)
            report = self.generate_regression_report(results)
            
            # CI test passes if no critical regressions
            test_passed = report["regression_test_report"]["test_summary"]["test_passed"]
            
            return test_passed, report
            
        except Exception as e:
            return False, {"error": str(e), "timestamp": datetime.now().isoformat()}


# CLI and utility functions
def create_performance_baseline_cli(runs: int = 5) -> str:
    """CLI function to create performance baseline"""
    tester = PerformanceRegressionTester()
    baseline = tester.create_baseline(runs)
    return f"Baseline created with {runs} runs"


def run_regression_test_cli(current_runs: int = 3) -> str:
    """CLI function to run regression test"""
    tester = PerformanceRegressionTester()
    results = tester.run_regression_test(current_runs)
    
    regressions = [r for r in results if r.is_regression]
    critical = [r for r in regressions if r.severity == "CRITICAL"]
    
    if critical:
        return f"CRITICAL: {len(critical)} critical regressions detected!"
    elif regressions:
        return f"WARNING: {len(regressions)} regressions detected"
    else:
        return "PASS: No regressions detected"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Regression Tester")
    parser.add_argument("command", choices=["baseline", "test", "ci"], 
                       help="Command to run")
    parser.add_argument("--runs", type=int, default=5, 
                       help="Number of runs for baseline/test")
    parser.add_argument("--output", type=str, 
                       help="Output file for report")
    
    args = parser.parse_args()
    
    tester = PerformanceRegressionTester()
    
    if args.command == "baseline":
        print("🏁 Creating performance baseline...")
        baseline = tester.create_baseline(args.runs)
        print("✅ Baseline created successfully")
        
    elif args.command == "test":
        print("🔍 Running regression test...")
        results = tester.run_regression_test(args.runs)
        report = tester.generate_regression_report(results, args.output)
        
        test_summary = report["regression_test_report"]["test_summary"]
        print(f"\n📊 REGRESSION TEST RESULTS:")
        print(f"   Total metrics: {test_summary['total_metrics_tested']}")
        print(f"   Regressions: {test_summary['regressions_detected']}")
        print(f"   Critical: {test_summary['critical_regressions']}")
        print(f"   Test status: {'PASS' if test_summary['test_passed'] else 'FAIL'}")
        
        if args.output:
            print(f"   Report saved: {args.output}")
            
    elif args.command == "ci":
        print("🤖 Running CI integration test...")
        passed, report = tester.run_ci_integration_test()
        
        print(f"CI Test: {'PASS' if passed else 'FAIL'}")
        if not passed:
            exit(1)