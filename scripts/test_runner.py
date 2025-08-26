#!/usr/bin/env python3
"""
Comprehensive Test Runner for Claude TUI

This script orchestrates the complete test suite execution with detailed
reporting, performance benchmarking, and quality metrics analysis.
"""

import asyncio
import json
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pytest


@dataclass
class TestResults:
    """Container for test execution results."""
    category: str
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    coverage: Optional[float] = None
    details: Optional[Dict] = None


class TestRunner:
    """Comprehensive test runner with reporting and analysis."""
    
    def __init__(self, project_root: Path = None):
        """Initialize test runner."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test categories and their specific configurations
        self.test_categories = {
            "unit": {
                "path": "tests/unit",
                "markers": "unit and not slow",
                "coverage_target": 90,
                "timeout": 60
            },
            "integration": {
                "path": "tests/integration",
                "markers": "integration",
                "coverage_target": 80,
                "timeout": 300
            },
            "performance": {
                "path": "tests/performance",
                "markers": "performance",
                "coverage_target": 70,
                "timeout": 600,
                "benchmark": True
            },
            "validation": {
                "path": "tests/validation",
                "markers": "validation",
                "coverage_target": 95,
                "timeout": 120
            },
            "ui": {
                "path": "tests/ui",
                "markers": "tui",
                "coverage_target": 75,
                "timeout": 180
            },
            "security": {
                "path": "tests/security",
                "markers": "security",
                "coverage_target": 95,
                "timeout": 300
            }
        }
    
    async def run_all_tests(self, 
                           categories: Optional[List[str]] = None,
                           verbose: bool = True,
                           parallel: bool = True,
                           generate_report: bool = True) -> Dict[str, TestResults]:
        """Run all test categories and generate comprehensive report."""
        print("🚀 Starting Claude TUI Comprehensive Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        categories = categories or list(self.test_categories.keys())
        results = {}
        
        # Run tests by category
        for category in categories:
            if category not in self.test_categories:
                print(f"❌ Unknown test category: {category}")
                continue
            
            print(f"\n📋 Running {category.title()} Tests")
            print("-" * 40)
            
            result = await self._run_test_category(
                category, verbose=verbose, parallel=parallel
            )
            results[category] = result
            
            # Print immediate results
            self._print_category_results(category, result)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        if generate_report:
            await self._generate_comprehensive_report(results, total_time)
        
        # Print summary
        self._print_test_summary(results, total_time)
        
        return results
    
    async def _run_test_category(self, 
                                category: str,
                                verbose: bool = True,
                                parallel: bool = True) -> TestResults:
        """Run tests for a specific category."""
        config = self.test_categories[category]
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            config["path"],
            f"--timeout={config['timeout']}",
            "--tb=short",
            "--json-report",
            f"--json-report-file={self.reports_dir}/{category}_report.json"
        ]
        
        # Add markers
        if "markers" in config:
            cmd.extend(["-m", config["markers"]])
        
        # Add coverage if not performance tests
        if category != "performance":
            cmd.extend([
                f"--cov=src",
                f"--cov-report=json:{self.reports_dir}/{category}_coverage.json",
                f"--cov-fail-under={config['coverage_target']}"
            ])
        
        # Add benchmark support for performance tests
        if config.get("benchmark", False):
            cmd.extend([
                "--benchmark-only",
                f"--benchmark-json={self.reports_dir}/{category}_benchmark.json"
            ])
        
        # Add parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])
        
        # Add verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Execute tests
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=config["timeout"]
            )
            
            duration = time.time() - start_time
            
            # Parse results
            return await self._parse_test_results(
                category, result, duration
            )
            
        except subprocess.TimeoutExpired:
            return TestResults(
                category=category,
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration=config["timeout"],
                details={"error": "Test execution timed out"}
            )
        
        except Exception as e:
            return TestResults(
                category=category,
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration=time.time() - start_time,
                details={"error": str(e)}
            )
    
    async def _parse_test_results(self,
                                 category: str,
                                 result: subprocess.CompletedProcess,
                                 duration: float) -> TestResults:
        """Parse pytest execution results."""
        # Default values
        passed = failed = skipped = errors = 0
        coverage = None
        details = {}
        
        try:
            # Parse JSON report if available
            report_file = self.reports_dir / f"{category}_report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                summary = report_data.get("summary", {})
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                skipped = summary.get("skipped", 0)
                errors = summary.get("error", 0)
                
                details.update({
                    "total": summary.get("total", 0),
                    "collected": report_data.get("collected", 0),
                    "deselected": summary.get("deselected", 0)
                })
            
            # Parse coverage report if available
            coverage_file = self.reports_dir / f"{category}_coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                coverage = coverage_data.get("totals", {}).get("percent_covered")
                details["coverage_details"] = coverage_data.get("files", {})
            
            # Parse benchmark report for performance tests
            if category == "performance":
                benchmark_file = self.reports_dir / f"{category}_benchmark.json"
                if benchmark_file.exists():
                    with open(benchmark_file, 'r') as f:
                        benchmark_data = json.load(f)
                    
                    details["benchmarks"] = benchmark_data.get("benchmarks", [])
            
            # Add execution details
            details.update({
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
        except Exception as e:
            details["parse_error"] = str(e)
            # Fallback to parsing stdout/stderr
            if "failed" in result.stdout.lower():
                failed = 1
            elif "passed" in result.stdout.lower():
                passed = 1
        
        return TestResults(
            category=category,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            coverage=coverage,
            details=details
        )
    
    def _print_category_results(self, category: str, result: TestResults):
        """Print results for a test category."""
        total = result.passed + result.failed + result.skipped + result.errors
        
        # Status emoji
        if result.failed > 0 or result.errors > 0:
            status = "❌ FAILED"
            color = "\033[91m"  # Red
        elif result.passed > 0:
            status = "✅ PASSED"
            color = "\033[92m"  # Green
        else:
            status = "⚠️  SKIPPED"
            color = "\033[93m"  # Yellow
        
        reset = "\033[0m"
        
        print(f"{color}{status}{reset} {category.title()} Tests")
        print(f"  📊 Results: {result.passed} passed, {result.failed} failed, "
              f"{result.skipped} skipped, {result.errors} errors")
        print(f"  ⏱️  Duration: {result.duration:.2f}s")
        
        if result.coverage is not None:
            coverage_color = "\033[92m" if result.coverage >= 80 else "\033[93m"
            print(f"  🎯 Coverage: {coverage_color}{result.coverage:.1f}%{reset}")
        
        # Show benchmark summary for performance tests
        if (category == "performance" and result.details and 
            "benchmarks" in result.details):
            benchmarks = result.details["benchmarks"]
            if benchmarks:
                avg_time = sum(b.get("stats", {}).get("mean", 0) for b in benchmarks) / len(benchmarks)
                print(f"  🏃 Avg Benchmark: {avg_time:.4f}s")
    
    async def _generate_comprehensive_report(self, 
                                           results: Dict[str, TestResults],
                                           total_time: float):
        """Generate comprehensive HTML and JSON reports."""
        timestamp = datetime.now().isoformat()
        
        # Aggregate statistics
        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        total_skipped = sum(r.skipped for r in results.values())
        total_errors = sum(r.errors for r in results.values())
        total_tests = total_passed + total_failed + total_skipped + total_errors
        
        # Calculate average coverage
        coverages = [r.coverage for r in results.values() if r.coverage is not None]
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0
        
        # Generate JSON report
        report_data = {
            "timestamp": timestamp,
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "errors": total_errors,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "average_coverage": avg_coverage,
                "total_duration": total_time
            },
            "categories": {
                category: {
                    "passed": result.passed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                    "errors": result.errors,
                    "duration": result.duration,
                    "coverage": result.coverage,
                    "details": result.details
                }
                for category, result in results.items()
            }
        }
        
        # Save JSON report
        json_report_path = self.reports_dir / "comprehensive_test_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate HTML report
        await self._generate_html_report(report_data)
        
        print(f"\n📄 Reports generated:")
        print(f"  📊 JSON: {json_report_path}")
        print(f"  🌐 HTML: {self.reports_dir / 'comprehensive_test_report.html'}")
    
    async def _generate_html_report(self, report_data: Dict):
        """Generate HTML test report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Claude TUI - Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .summary { display: flex; justify-content: space-around; margin: 20px 0; }
        .metric { text-align: center; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        .metric h3 { margin: 0 0 10px 0; color: #333; }
        .metric .value { font-size: 24px; font-weight: bold; }
        .passed { color: #28a745; }
        .failed { color: #dc3545; }
        .skipped { color: #ffc107; }
        .category { margin: 20px 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
        .category-header { background: #007bff; color: white; padding: 15px; font-weight: bold; }
        .category-content { padding: 15px; }
        .progress-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #28a745, #17a2b8); }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 8px 12px; border: 1px solid #ddd; text-align: left; }
        th { background: #f8f9fa; font-weight: bold; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Claude TUI - Comprehensive Test Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>
        <p>Complete test suite execution results with coverage analysis and performance benchmarks.</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Total Tests</h3>
            <div class="value">{total_tests}</div>
        </div>
        <div class="metric">
            <h3>Success Rate</h3>
            <div class="value passed">{success_rate:.1f}%</div>
        </div>
        <div class="metric">
            <h3>Coverage</h3>
            <div class="value">{avg_coverage:.1f}%</div>
        </div>
        <div class="metric">
            <h3>Duration</h3>
            <div class="value">{total_duration:.1f}s</div>
        </div>
    </div>
    
    <div class="progress-bar">
        <div class="progress-fill" style="width: {success_rate:.1f}%"></div>
    </div>
    
    {categories_html}
    
</body>
</html>
        """
        
        # Generate categories HTML
        categories_html = ""
        for category, data in report_data["categories"].items():
            total = data["passed"] + data["failed"] + data["skipped"] + data["errors"]
            success_rate = (data["passed"] / total * 100) if total > 0 else 0
            
            categories_html += f"""
            <div class="category">
                <div class="category-header">{category.title()} Tests</div>
                <div class="category-content">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Passed</td>
                            <td class="passed">{data["passed"]}</td>
                        </tr>
                        <tr>
                            <td>Failed</td>
                            <td class="failed">{data["failed"]}</td>
                        </tr>
                        <tr>
                            <td>Skipped</td>
                            <td class="skipped">{data["skipped"]}</td>
                        </tr>
                        <tr>
                            <td>Duration</td>
                            <td>{data["duration"]:.2f}s</td>
                        </tr>
            """
            
            if data["coverage"]:
                categories_html += f"""
                        <tr>
                            <td>Coverage</td>
                            <td>{data["coverage"]:.1f}%</td>
                        </tr>
                """
            
            categories_html += """
                    </table>
                </div>
            </div>
            """
        
        # Fill template
        html_content = html_template.format(
            timestamp=report_data["timestamp"],
            total_tests=report_data["summary"]["total_tests"],
            success_rate=report_data["summary"]["success_rate"],
            avg_coverage=report_data["summary"]["average_coverage"],
            total_duration=report_data["summary"]["total_duration"],
            categories_html=categories_html
        )
        
        # Save HTML report
        html_report_path = self.reports_dir / "comprehensive_test_report.html"
        with open(html_report_path, 'w') as f:
            f.write(html_content)
    
    def _print_test_summary(self, results: Dict[str, TestResults], total_time: float):
        """Print overall test execution summary."""
        print("\n" + "=" * 60)
        print("🎯 TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        # Calculate totals
        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        total_skipped = sum(r.skipped for r in results.values())
        total_errors = sum(r.errors for r in results.values())
        total_tests = total_passed + total_failed + total_skipped + total_errors
        
        # Calculate success rate
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Overall status
        if total_failed > 0 or total_errors > 0:
            status = "❌ OVERALL: FAILED"
            color = "\033[91m"
        elif total_passed > 0:
            status = "✅ OVERALL: PASSED"
            color = "\033[92m"
        else:
            status = "⚠️  OVERALL: NO TESTS RUN"
            color = "\033[93m"
        
        reset = "\033[0m"
        
        print(f"{color}{status}{reset}")
        print(f"📊 Total Results: {total_passed} passed, {total_failed} failed, "
              f"{total_skipped} skipped, {total_errors} errors")
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Total Duration: {total_time:.2f}s")
        
        # Coverage summary
        coverages = [r.coverage for r in results.values() if r.coverage is not None]
        if coverages:
            avg_coverage = sum(coverages) / len(coverages)
            coverage_color = "\033[92m" if avg_coverage >= 80 else "\033[93m"
            print(f"🎯 Average Coverage: {coverage_color}{avg_coverage:.1f}%{reset}")
        
        # Quality assessment
        print("\n🔍 Quality Assessment:")
        if success_rate >= 95:
            print("  ✅ Excellent test results!")
        elif success_rate >= 85:
            print("  ✅ Good test results")
        elif success_rate >= 70:
            print("  ⚠️  Acceptable test results - some improvements needed")
        else:
            print("  ❌ Poor test results - significant issues to address")
        
        print("=" * 60)
    
    async def run_specific_tests(self, 
                                test_pattern: str,
                                verbose: bool = True) -> TestResults:
        """Run specific tests matching a pattern."""
        print(f"🔍 Running specific tests: {test_pattern}")
        
        cmd = [
            sys.executable, "-m", "pytest",
            test_pattern,
            "--tb=short",
            "--json-report",
            f"--json-report-file={self.reports_dir}/specific_test_report.json",
            "--cov=src",
            f"--cov-report=json:{self.reports_dir}/specific_coverage.json"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        return await self._parse_test_results("specific", result, duration)


async def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Claude TUI Comprehensive Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_runner.py                    # Run all tests
  python scripts/test_runner.py -c unit integration # Run specific categories
  python scripts/test_runner.py -t tests/unit/test_*.py # Run specific tests
  python scripts/test_runner.py --quick            # Run fast tests only
  python scripts/test_runner.py --no-parallel      # Disable parallel execution
        """
    )
    
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        choices=["unit", "integration", "performance", "validation", "ui", "security"],
        help="Test categories to run"
    )
    
    parser.add_argument(
        "-t", "--tests",
        help="Specific test pattern to run"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only fast tests (excludes slow tests)"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel test execution"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    try:
        if args.tests:
            # Run specific tests
            result = await runner.run_specific_tests(
                args.tests,
                verbose=args.verbose
            )
            print(f"\n📊 Specific Test Results:")
            runner._print_category_results("specific", result)
            
        else:
            # Run test categories
            categories = args.categories
            if args.quick:
                categories = ["unit"]  # Only run fast unit tests
            
            results = await runner.run_all_tests(
                categories=categories,
                verbose=args.verbose,
                parallel=not args.no_parallel,
                generate_report=not args.no_report
            )
            
            # Exit with appropriate code
            failed_categories = [cat for cat, result in results.items() 
                               if result.failed > 0 or result.errors > 0]
            
            if failed_categories:
                print(f"\n❌ Categories with failures: {', '.join(failed_categories)}")
                sys.exit(1)
            else:
                print("\n✅ All test categories passed!")
                sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n💥 Test runner failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())