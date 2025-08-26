#!/usr/bin/env python3
"""
ML Validation Test Runner
Comprehensive test runner for all ML validation tests with detailed reporting,
performance benchmarking, and accuracy validation for the Anti-Hallucination Engine.
"""

import pytest
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestRunResult:
    """Test run result with comprehensive metrics."""
    test_suite: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    accuracy_tests_passed: int
    performance_tests_passed: int
    integration_tests_passed: int
    coverage_percentage: Optional[float] = None
    detailed_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLValidationReport:
    """Comprehensive ML validation report."""
    timestamp: str
    total_execution_time: float
    overall_pass_rate: float
    accuracy_validation_score: float
    performance_benchmark_score: float
    integration_test_score: float
    test_run_results: List[TestRunResult]
    summary: Dict[str, Any]
    recommendations: List[str]


class MLTestRunner:
    """Comprehensive ML validation test runner."""
    
    def __init__(self):
        self.test_suites = {
            "accuracy": "test_anti_hallucination_accuracy.py",
            "semantic": "test_semantic_analysis.py", 
            "performance": "../performance/test_ml_inference_speed.py",
            "training": "test_ml_training_validation.py"
        }
        
        self.markers = {
            "ml": "ML model tests",
            "performance": "Performance benchmark tests",
            "integration": "Integration tests",
            "benchmark": "Benchmark tests",
            "slow": "Slow-running tests"
        }
    
    def run_all_tests(
        self, 
        include_slow: bool = False,
        generate_report: bool = True,
        output_file: Optional[str] = None
    ) -> MLValidationReport:
        """Run all ML validation tests with comprehensive reporting."""
        
        logger.info("Starting comprehensive ML validation test suite")
        start_time = time.perf_counter()
        
        test_results = []
        
        # Run each test suite
        for suite_name, suite_file in self.test_suites.items():
            logger.info(f"Running {suite_name} test suite: {suite_file}")
            
            try:
                result = self._run_test_suite(suite_name, suite_file, include_slow)
                test_results.append(result)
                
                logger.info(f"Suite {suite_name}: {result.passed_tests}/{result.total_tests} tests passed")
                
            except Exception as e:
                logger.error(f"Error running {suite_name} suite: {e}")
                # Create failed result
                test_results.append(TestRunResult(
                    test_suite=suite_name,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=1,
                    skipped_tests=0,
                    execution_time=0.0,
                    accuracy_tests_passed=0,
                    performance_tests_passed=0,
                    integration_tests_passed=0,
                    detailed_results={"error": str(e)}
                ))
        
        total_execution_time = time.perf_counter() - start_time
        
        # Generate comprehensive report
        report = self._generate_ml_validation_report(test_results, total_execution_time)
        
        if generate_report:
            self._save_report(report, output_file)
        
        return report
    
    def run_accuracy_tests_only(self) -> TestRunResult:
        """Run only accuracy validation tests."""
        logger.info("Running accuracy validation tests only")
        return self._run_test_suite("accuracy", self.test_suites["accuracy"], include_slow=False)
    
    def run_performance_benchmarks(self) -> TestRunResult:
        """Run only performance benchmark tests."""
        logger.info("Running performance benchmark tests only")
        return self._run_test_suite("performance", self.test_suites["performance"], include_slow=True)
    
    def run_quick_validation(self) -> MLValidationReport:
        """Run quick validation subset for CI/CD."""
        logger.info("Running quick ML validation suite")
        start_time = time.perf_counter()
        
        # Run subset of critical tests
        test_results = []
        
        # Accuracy tests (critical)
        accuracy_result = self._run_test_suite(
            "accuracy", 
            self.test_suites["accuracy"], 
            include_slow=False,
            quick_mode=True
        )
        test_results.append(accuracy_result)
        
        # Performance tests (subset)
        performance_result = self._run_test_suite(
            "performance",
            self.test_suites["performance"],
            include_slow=False,
            quick_mode=True
        )
        test_results.append(performance_result)
        
        total_execution_time = time.perf_counter() - start_time
        
        report = self._generate_ml_validation_report(test_results, total_execution_time)
        report.summary["test_mode"] = "quick_validation"
        
        return report
    
    def _run_test_suite(
        self, 
        suite_name: str, 
        suite_file: str, 
        include_slow: bool = False,
        quick_mode: bool = False
    ) -> TestRunResult:
        """Run individual test suite with detailed metrics."""
        
        suite_start_time = time.perf_counter()
        
        # Build pytest arguments
        pytest_args = [
            Path(__file__).parent / suite_file,
            "-v",
            "--tb=short",
            "-m", "ml",
            "--durations=10",
            "--json-report",
            "--json-report-file=/tmp/pytest_report.json"
        ]
        
        if not include_slow:
            pytest_args.extend(["-m", "not slow"])
        
        if quick_mode:
            pytest_args.extend(["-x", "--maxfail=3"])  # Stop early for quick mode
        
        # Run tests
        exit_code = pytest.main(pytest_args)
        
        execution_time = time.perf_counter() - suite_start_time
        
        # Parse pytest JSON report
        try:
            with open("/tmp/pytest_report.json", "r") as f:
                pytest_report = json.load(f)
            
            # Extract test metrics
            summary = pytest_report.get("summary", {})
            total_tests = summary.get("total", 0)
            passed_tests = summary.get("passed", 0)
            failed_tests = summary.get("failed", 0)
            skipped_tests = summary.get("skipped", 0)
            
            # Count specific test types
            accuracy_tests_passed = self._count_test_type(pytest_report, "accuracy")
            performance_tests_passed = self._count_test_type(pytest_report, "performance")
            integration_tests_passed = self._count_test_type(pytest_report, "integration")
            
        except Exception as e:
            logger.warning(f"Could not parse pytest report: {e}")
            # Fallback to basic metrics
            total_tests = 1
            passed_tests = 1 if exit_code == 0 else 0
            failed_tests = 0 if exit_code == 0 else 1
            skipped_tests = 0
            accuracy_tests_passed = 0
            performance_tests_passed = 0
            integration_tests_passed = 0
        
        return TestRunResult(
            test_suite=suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            execution_time=execution_time,
            accuracy_tests_passed=accuracy_tests_passed,
            performance_tests_passed=performance_tests_passed,
            integration_tests_passed=integration_tests_passed,
            detailed_results={"exit_code": exit_code}
        )
    
    def _count_test_type(self, pytest_report: Dict[str, Any], test_type: str) -> int:
        """Count tests of specific type from pytest report."""
        tests = pytest_report.get("tests", [])
        count = 0
        
        for test in tests:
            test_name = test.get("nodeid", "").lower()
            if test_type in test_name and test.get("outcome") == "passed":
                count += 1
        
        return count
    
    def _generate_ml_validation_report(
        self, 
        test_results: List[TestRunResult], 
        total_execution_time: float
    ) -> MLValidationReport:
        """Generate comprehensive ML validation report."""
        
        # Calculate overall metrics
        total_tests = sum(result.total_tests for result in test_results)
        total_passed = sum(result.passed_tests for result in test_results)
        total_failed = sum(result.failed_tests for result in test_results)
        
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Calculate specialized scores
        accuracy_validation_score = self._calculate_accuracy_score(test_results)
        performance_benchmark_score = self._calculate_performance_score(test_results)
        integration_test_score = self._calculate_integration_score(test_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            test_results, 
            overall_pass_rate,
            accuracy_validation_score,
            performance_benchmark_score
        )
        
        # Summary statistics
        summary = {
            "total_test_suites": len(test_results),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "execution_time_seconds": total_execution_time,
            "average_test_time_ms": (total_execution_time / total_tests * 1000) if total_tests > 0 else 0,
            "test_suites_passed": sum(1 for r in test_results if r.failed_tests == 0),
            "critical_accuracy_threshold_met": accuracy_validation_score >= 0.958,
            "performance_requirements_met": performance_benchmark_score >= 0.95,
            "integration_tests_stable": integration_test_score >= 0.90
        }
        
        return MLValidationReport(
            timestamp=datetime.now().isoformat(),
            total_execution_time=total_execution_time,
            overall_pass_rate=overall_pass_rate,
            accuracy_validation_score=accuracy_validation_score,
            performance_benchmark_score=performance_benchmark_score,
            integration_test_score=integration_test_score,
            test_run_results=test_results,
            summary=summary,
            recommendations=recommendations
        )
    
    def _calculate_accuracy_score(self, test_results: List[TestRunResult]) -> float:
        """Calculate accuracy validation score."""
        accuracy_results = [r for r in test_results if r.test_suite == "accuracy"]
        
        if not accuracy_results:
            return 0.0
        
        result = accuracy_results[0]
        if result.total_tests == 0:
            return 0.0
        
        # Weight accuracy tests more heavily
        accuracy_score = (result.passed_tests / result.total_tests) * 0.9
        
        # Bonus for specific accuracy tests
        if result.accuracy_tests_passed > 0:
            accuracy_score += 0.1 * (result.accuracy_tests_passed / max(result.total_tests, 1))
        
        return min(1.0, accuracy_score)
    
    def _calculate_performance_score(self, test_results: List[TestRunResult]) -> float:
        """Calculate performance benchmark score."""
        performance_results = [r for r in test_results if r.test_suite == "performance"]
        
        if not performance_results:
            return 0.0
        
        result = performance_results[0]
        if result.total_tests == 0:
            return 0.0
        
        # Performance tests must pass to meet <200ms requirement
        base_score = result.passed_tests / result.total_tests
        
        # Penalty for slow execution
        if result.execution_time > 60:  # More than 1 minute for performance tests
            base_score *= 0.8
        
        return base_score
    
    def _calculate_integration_score(self, test_results: List[TestRunResult]) -> float:
        """Calculate integration test score."""
        integration_counts = sum(r.integration_tests_passed for r in test_results)
        total_integration_possible = sum(r.total_tests for r in test_results if r.test_suite in ["semantic", "training"])
        
        if total_integration_possible == 0:
            return 1.0  # No integration tests expected
        
        return integration_counts / total_integration_possible
    
    def _generate_recommendations(
        self, 
        test_results: List[TestRunResult],
        overall_pass_rate: float,
        accuracy_score: float,
        performance_score: float
    ) -> List[str]:
        """Generate actionable recommendations based on test results."""
        
        recommendations = []
        
        # Overall pass rate recommendations
        if overall_pass_rate < 0.95:
            recommendations.append(
                f"Overall pass rate ({overall_pass_rate:.3f}) below 95%. "
                "Review failed tests and address underlying issues."
            )
        
        # Accuracy recommendations
        if accuracy_score < 0.958:
            recommendations.append(
                f"Accuracy score ({accuracy_score:.4f}) below 95.8% target. "
                "Review ML model training data and feature extraction quality."
            )
        
        # Performance recommendations
        if performance_score < 0.95:
            recommendations.append(
                f"Performance score ({performance_score:.3f}) below 95%. "
                "Optimize ML inference pipeline to meet <200ms requirement."
            )
        
        # Test suite specific recommendations
        for result in test_results:
            if result.failed_tests > 0:
                failure_rate = result.failed_tests / result.total_tests
                if failure_rate > 0.1:
                    recommendations.append(
                        f"Test suite '{result.test_suite}' has high failure rate ({failure_rate:.2%}). "
                        "Investigate and fix failing test cases."
                    )
            
            if result.execution_time > 300:  # 5 minutes
                recommendations.append(
                    f"Test suite '{result.test_suite}' execution time ({result.execution_time:.1f}s) is too high. "
                    "Consider optimizing test performance or running in parallel."
                )
        
        # Positive recommendations
        if overall_pass_rate >= 0.99:
            recommendations.append(
                "Excellent test coverage and pass rate. "
                "Consider adding more edge case tests to further improve robustness."
            )
        
        if accuracy_score >= 0.97:
            recommendations.append(
                "ML model accuracy exceeds targets. "
                "Consider documenting best practices and model configuration for reproducibility."
            )
        
        if not recommendations:
            recommendations.append(
                "All ML validation tests are passing within acceptable thresholds. "
                "System appears ready for production deployment."
            )
        
        return recommendations
    
    def _save_report(self, report: MLValidationReport, output_file: Optional[str] = None) -> None:
        """Save ML validation report to file."""
        
        if output_file is None:
            output_file = f"ml_validation_report_{int(time.time())}.json"
        
        output_path = Path(output_file)
        
        # Convert dataclass to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "total_execution_time": report.total_execution_time,
            "overall_pass_rate": report.overall_pass_rate,
            "accuracy_validation_score": report.accuracy_validation_score,
            "performance_benchmark_score": report.performance_benchmark_score,
            "integration_test_score": report.integration_test_score,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "test_run_results": [
                {
                    "test_suite": result.test_suite,
                    "total_tests": result.total_tests,
                    "passed_tests": result.passed_tests,
                    "failed_tests": result.failed_tests,
                    "skipped_tests": result.skipped_tests,
                    "execution_time": result.execution_time,
                    "accuracy_tests_passed": result.accuracy_tests_passed,
                    "performance_tests_passed": result.performance_tests_passed,
                    "integration_tests_passed": result.integration_tests_passed,
                    "detailed_results": result.detailed_results
                }
                for result in report.test_run_results
            ]
        }
        
        try:
            with open(output_path, "w") as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"ML validation report saved to: {output_path}")
            
            # Also create human-readable summary
            self._create_human_readable_report(report, output_path.with_suffix(".txt"))
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def _create_human_readable_report(self, report: MLValidationReport, output_path: Path) -> None:
        """Create human-readable text report."""
        
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ML VALIDATION TEST REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {report.timestamp}\n")
            f.write(f"Total Execution Time: {report.total_execution_time:.2f} seconds\n\n")
            
            # Overall results
            f.write("OVERALL RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Pass Rate: {report.overall_pass_rate:.3f} ({report.overall_pass_rate*100:.1f}%)\n")
            f.write(f"Accuracy Validation Score: {report.accuracy_validation_score:.4f}\n")
            f.write(f"Performance Benchmark Score: {report.performance_benchmark_score:.3f}\n")
            f.write(f"Integration Test Score: {report.integration_test_score:.3f}\n\n")
            
            # Test suite results
            f.write("TEST SUITE RESULTS\n")
            f.write("-" * 40 + "\n")
            for result in report.test_run_results:
                f.write(f"\n{result.test_suite.upper()} SUITE:\n")
                f.write(f"  Total Tests: {result.total_tests}\n")
                f.write(f"  Passed: {result.passed_tests}\n")
                f.write(f"  Failed: {result.failed_tests}\n")
                f.write(f"  Skipped: {result.skipped_tests}\n")
                f.write(f"  Execution Time: {result.execution_time:.2f}s\n")
                if result.accuracy_tests_passed > 0:
                    f.write(f"  Accuracy Tests Passed: {result.accuracy_tests_passed}\n")
                if result.performance_tests_passed > 0:
                    f.write(f"  Performance Tests Passed: {result.performance_tests_passed}\n")
            
            # Summary
            f.write("\nSUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            for key, value in report.summary.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            for i, recommendation in enumerate(report.recommendations, 1):
                f.write(f"{i}. {recommendation}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Human-readable report saved to: {output_path}")


def main():
    """Main function for command-line execution."""
    
    parser = argparse.ArgumentParser(description="Run ML validation tests")
    parser.add_argument(
        "--mode",
        choices=["all", "accuracy", "performance", "quick"],
        default="all",
        help="Test mode to run"
    )
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow-running tests"
    )
    parser.add_argument(
        "--output",
        help="Output file for report (default: auto-generated)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = MLTestRunner()
    
    try:
        if args.mode == "all":
            report = runner.run_all_tests(
                include_slow=args.include_slow,
                output_file=args.output
            )
        elif args.mode == "accuracy":
            result = runner.run_accuracy_tests_only()
            logger.info(f"Accuracy tests: {result.passed_tests}/{result.total_tests} passed")
            return 0 if result.failed_tests == 0 else 1
        elif args.mode == "performance":
            result = runner.run_performance_benchmarks()
            logger.info(f"Performance tests: {result.passed_tests}/{result.total_tests} passed")
            return 0 if result.failed_tests == 0 else 1
        elif args.mode == "quick":
            report = runner.run_quick_validation()
        
        # Print final results
        print(f"\nML VALIDATION RESULTS:")
        print(f"Overall Pass Rate: {report.overall_pass_rate:.3f}")
        print(f"Accuracy Score: {report.accuracy_validation_score:.4f}")
        print(f"Performance Score: {report.performance_benchmark_score:.3f}")
        print(f"Integration Score: {report.integration_test_score:.3f}")
        
        # Exit code based on critical thresholds
        if (report.accuracy_validation_score >= 0.958 and 
            report.performance_benchmark_score >= 0.90 and
            report.overall_pass_rate >= 0.95):
            print("✅ ALL CRITICAL ML VALIDATION TESTS PASSED")
            return 0
        else:
            print("❌ SOME CRITICAL ML VALIDATION TESTS FAILED")
            return 1
    
    except Exception as e:
        logger.error(f"ML validation test run failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())