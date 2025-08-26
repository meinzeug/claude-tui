#!/usr/bin/env python3
"""
Test Automation Framework for Claude-TUI Hive Mind
Comprehensive CI/CD pipeline and automated testing orchestration.
"""

import subprocess
import json
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    test_patterns: List[str]
    markers: List[str] = field(default_factory=list)
    timeout: int = 300  # 5 minutes default
    retry_count: int = 1
    parallel: bool = True
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1=highest, 10=lowest

@dataclass
class TestResult:
    """Test execution result."""
    suite_name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    exit_code: int = 0
    output: str = ""
    coverage: float = 0.0

class TestOrchestrator:
    """Orchestrates test execution across the entire project."""
    
    def __init__(self, project_root: str = "/home/tekkadmin/claude-tui"):
        self.project_root = Path(project_root)
        self.test_suites = self._define_test_suites()
        self.results: Dict[str, TestResult] = {}
        
    def _define_test_suites(self) -> List[TestSuite]:
        """Define comprehensive test suites."""
        return [
            # Unit Tests - Highest Priority
            TestSuite(
                name="unit_critical",
                test_patterns=[
                    "tests/unit/validation/test_anti_hallucination_engine_comprehensive.py",
                    "tests/unit/ai/test_swarm_coordination_comprehensive.py",
                    "tests/unit/core/test_*.py"
                ],
                markers=["unit", "critical"],
                timeout=120,
                parallel=True,
                priority=1
            ),
            
            TestSuite(
                name="unit_fast",
                test_patterns=[
                    "tests/unit/**/*test*.py"
                ],
                markers=["unit", "fast"],
                timeout=60,
                parallel=True,
                priority=2
            ),
            
            # Integration Tests
            TestSuite(
                name="integration_core",
                test_patterns=[
                    "tests/integration/test_*.py"
                ],
                markers=["integration"],
                timeout=300,
                parallel=False,  # Integration tests may interfere with each other
                dependencies=["unit_critical"],
                priority=3
            ),
            
            # UI Tests
            TestSuite(
                name="ui_components",
                test_patterns=[
                    "tests/ui/test_textual_components_enhanced.py",
                    "tests/ui/test_*.py"
                ],
                markers=["ui"],
                timeout=180,
                parallel=True,
                dependencies=["unit_fast"],
                priority=4
            ),
            
            # Performance Tests
            TestSuite(
                name="performance_benchmarks", 
                test_patterns=[
                    "tests/performance/test_ai_performance_benchmarks.py",
                    "tests/performance/test_*.py"
                ],
                markers=["performance"],
                timeout=600,  # 10 minutes for performance tests
                parallel=False,
                dependencies=["unit_critical", "integration_core"],
                priority=5
            ),
            
            # Accuracy Validation
            TestSuite(
                name="accuracy_validation",
                test_patterns=[
                    "tests/validation/test_accuracy_validation_suite.py"
                ],
                markers=["accuracy", "critical"],
                timeout=300,
                parallel=False,
                dependencies=["unit_critical"],
                priority=1  # Critical for 95.8% target
            ),
            
            # Swarm Testing
            TestSuite(
                name="swarm_integration",
                test_patterns=[
                    "tests/swarm_testing/test_*.py"
                ],
                markers=["swarm", "integration"],
                timeout=240,
                parallel=False,
                dependencies=["integration_core"],
                priority=3
            ),
            
            # Security Tests
            TestSuite(
                name="security",
                test_patterns=[
                    "tests/security/test_*.py"
                ],
                markers=["security"],
                timeout=180,
                parallel=True,
                priority=4
            ),
            
            # End-to-End Tests
            TestSuite(
                name="e2e",
                test_patterns=[
                    "tests/e2e/test_*.py"
                ],
                markers=["e2e", "slow"],
                timeout=900,  # 15 minutes
                parallel=False,
                dependencies=["unit_critical", "integration_core", "ui_components"],
                priority=6
            ),
        ]
    
    async def run_test_suite(self, suite: TestSuite) -> TestResult:
        """Run a single test suite."""
        print(f"🚀 Running test suite: {suite.name}")
        
        result = TestResult(suite_name=suite.name)
        start_time = time.perf_counter()
        
        try:
            # Build pytest command
            cmd = self._build_pytest_command(suite)
            
            print(f"   Command: {' '.join(cmd)}")
            
            # Execute tests
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.project_root
            )
            
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=suite.timeout
            )
            
            result.exit_code = process.returncode
            result.output = stdout.decode('utf-8') if stdout else ""
            
            # Parse test results from output
            self._parse_test_output(result)
            
        except asyncio.TimeoutError:
            print(f"   ⏰ Timeout after {suite.timeout} seconds")
            result.exit_code = 124  # Timeout exit code
            result.errors = 1
            
        except Exception as e:
            print(f"   ❌ Error running suite: {e}")
            result.exit_code = 1
            result.errors = 1
            result.output = str(e)
        
        result.duration = time.perf_counter() - start_time
        
        # Log results
        status = "✅ PASSED" if result.exit_code == 0 else "❌ FAILED"
        print(f"   {status} in {result.duration:.1f}s")
        print(f"   Tests: {result.passed} passed, {result.failed} failed, {result.skipped} skipped")
        
        return result
    
    def _build_pytest_command(self, suite: TestSuite) -> List[str]:
        """Build pytest command for test suite."""
        cmd = ["python", "-m", "pytest"]
        
        # Add test patterns
        for pattern in suite.test_patterns:
            test_path = self.project_root / pattern
            if test_path.exists():
                cmd.append(str(test_path))
            else:
                # Pattern matching
                cmd.append(pattern)
        
        # Add markers
        if suite.markers:
            marker_expr = " and ".join(suite.markers)
            cmd.extend(["-m", marker_expr])
        
        # Parallel execution
        if suite.parallel:
            cmd.extend(["-n", "auto"])  # pytest-xdist
        
        # Verbose output
        cmd.append("-v")
        
        # Coverage if applicable
        if "unit" in suite.name:
            cmd.extend(["--cov=src", "--cov-report=json"])
        
        # Output format
        cmd.extend(["--tb=short", "--durations=10"])
        
        return cmd
    
    def _parse_test_output(self, result: TestResult):
        """Parse pytest output to extract test statistics."""
        output = result.output
        
        # Parse test results line
        # Example: "collected 45 items" and "= 42 passed, 2 failed, 1 skipped in 30.2s ="
        
        # Extract collected items
        import re
        collected_match = re.search(r'collected (\d+) items', output)
        if collected_match:
            total_collected = int(collected_match.group(1))
        
        # Extract final results
        results_pattern = r'=+ (?:(\d+) passed)?(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) error)?.*in ([\d.]+)s'
        results_match = re.search(results_pattern, output)
        
        if results_match:
            groups = results_match.groups()
            result.passed = int(groups[0]) if groups[0] else 0
            result.failed = int(groups[1]) if groups[1] else 0
            result.skipped = int(groups[2]) if groups[2] else 0
            result.errors = int(groups[3]) if groups[3] else 0
        
        # Extract coverage if present
        coverage_match = re.search(r'TOTAL.*?(\d+)%', output)
        if coverage_match:
            result.coverage = float(coverage_match.group(1))
    
    async def run_all_suites(self) -> Dict[str, TestResult]:
        """Run all test suites in dependency order."""
        print("🎯 Starting comprehensive test execution")
        
        # Sort suites by priority and dependencies
        sorted_suites = self._sort_suites_by_dependencies()
        
        # Track completion for dependency resolution
        completed_suites = set()
        
        for suite in sorted_suites:
            # Check dependencies
            if suite.dependencies:
                missing_deps = [dep for dep in suite.dependencies if dep not in completed_suites]
                if missing_deps:
                    print(f"⏸️  Skipping {suite.name}: missing dependencies {missing_deps}")
                    continue
            
            # Run suite
            result = await self.run_test_suite(suite)
            self.results[suite.name] = result
            
            # Mark as completed if successful
            if result.exit_code == 0:
                completed_suites.add(suite.name)
            else:
                print(f"⚠️  Suite {suite.name} failed - may affect dependent suites")
        
        return self.results
    
    def _sort_suites_by_dependencies(self) -> List[TestSuite]:
        """Sort test suites by priority and dependencies."""
        # Simple topological sort by priority and dependencies
        sorted_suites = []
        remaining_suites = self.test_suites.copy()
        
        while remaining_suites:
            # Find suites with no unmet dependencies
            ready_suites = []
            for suite in remaining_suites:
                if not suite.dependencies or all(
                    dep in [s.name for s in sorted_suites] 
                    for dep in suite.dependencies
                ):
                    ready_suites.append(suite)
            
            if not ready_suites:
                # Add remaining suites anyway to avoid infinite loop
                ready_suites = remaining_suites
            
            # Sort ready suites by priority
            ready_suites.sort(key=lambda s: s.priority)
            
            # Add first ready suite
            next_suite = ready_suites[0]
            sorted_suites.append(next_suite)
            remaining_suites.remove(next_suite)
        
        return sorted_suites
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test execution report."""
        if not self.results:
            return {"error": "No test results available"}
        
        # Aggregate statistics
        total_passed = sum(r.passed for r in self.results.values())
        total_failed = sum(r.failed for r in self.results.values())
        total_skipped = sum(r.skipped for r in self.results.values())
        total_errors = sum(r.errors for r in self.results.values())
        total_duration = sum(r.duration for r in self.results.values())
        
        total_tests = total_passed + total_failed + total_skipped + total_errors
        success_rate = (total_passed / total_tests) if total_tests > 0 else 0.0
        
        # Suite-level analysis
        suite_results = []
        for suite_name, result in self.results.items():
            suite_results.append({
                "name": suite_name,
                "status": "PASSED" if result.exit_code == 0 else "FAILED",
                "passed": result.passed,
                "failed": result.failed,
                "skipped": result.skipped,
                "errors": result.errors,
                "duration": result.duration,
                "coverage": result.coverage,
                "exit_code": result.exit_code
            })
        
        # Critical test analysis
        critical_suites = ["unit_critical", "accuracy_validation"]
        critical_passed = all(
            self.results[suite].exit_code == 0 
            for suite in critical_suites 
            if suite in self.results
        )
        
        # Coverage analysis
        coverage_results = {
            suite: result.coverage 
            for suite, result in self.results.items() 
            if result.coverage > 0
        }
        avg_coverage = statistics.mean(coverage_results.values()) if coverage_results else 0.0
        
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_suites": len(self.results),
                "total_duration": total_duration
            },
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "errors": total_errors,
                "success_rate": success_rate,
                "critical_tests_passed": critical_passed,
                "average_coverage": avg_coverage
            },
            "suite_results": suite_results,
            "coverage_by_suite": coverage_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not self.results:
            recommendations.append("No test results available - ensure tests are running")
            return recommendations
        
        # Check critical test failures
        critical_suites = ["unit_critical", "accuracy_validation"]
        for suite_name in critical_suites:
            if suite_name in self.results and self.results[suite_name].exit_code != 0:
                recommendations.append(f"🚨 Critical: Fix failures in {suite_name} - these are essential for system reliability")
        
        # Check overall success rate
        total_tests = sum(r.passed + r.failed + r.errors for r in self.results.values())
        total_passed = sum(r.passed for r in self.results.values())
        success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        if success_rate < 0.95:
            recommendations.append(f"📈 Improve overall test success rate from {success_rate:.1%} to 95%+")
        
        # Check performance
        slow_suites = [
            (name, result.duration) 
            for name, result in self.results.items() 
            if result.duration > 300  # 5 minutes
        ]
        
        if slow_suites:
            recommendations.append(f"⚡ Optimize slow test suites: {', '.join(name for name, _ in slow_suites)}")
        
        # Check coverage
        coverage_results = {
            suite: result.coverage 
            for suite, result in self.results.items() 
            if result.coverage > 0
        }
        
        if coverage_results:
            avg_coverage = statistics.mean(coverage_results.values())
            if avg_coverage < 80:
                recommendations.append(f"🎯 Increase test coverage from {avg_coverage:.1f}% to 80%+")
        
        # Check flaky tests
        error_suites = [name for name, result in self.results.items() if result.errors > 0]
        if error_suites:
            recommendations.append(f"🔧 Investigate test errors in: {', '.join(error_suites)}")
        
        return recommendations[:10]  # Limit to top 10
    
    def save_report(self, filename: str = None) -> Path:
        """Save test execution report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_execution_report_{timestamp}.json"
        
        report_path = self.project_root / "tests" / "reports" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def print_summary(self):
        """Print test execution summary."""
        if not self.results:
            print("❌ No test results available")
            return
        
        report = self.generate_report()
        summary = report["summary"]
        
        print("\n" + "="*80)
        print("🎯 TEST EXECUTION SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL RESULTS")
        print(f"   Total Tests: {summary['total_tests']:,}")
        print(f"   Passed: {summary['passed']:,} ({summary['success_rate']:.1%})")
        print(f"   Failed: {summary['failed']:,}")
        print(f"   Skipped: {summary['skipped']:,}")
        print(f"   Errors: {summary['errors']:,}")
        print(f"   Average Coverage: {summary['average_coverage']:.1f}%")
        
        # Critical tests status
        critical_status = "✅ PASSED" if summary['critical_tests_passed'] else "❌ FAILED"
        print(f"   Critical Tests: {critical_status}")
        
        print(f"\n🧪 SUITE RESULTS")
        for suite_result in report["suite_results"]:
            status_icon = "✅" if suite_result["status"] == "PASSED" else "❌"
            duration = suite_result["duration"]
            coverage = f" ({suite_result['coverage']:.1f}% cov)" if suite_result['coverage'] > 0 else ""
            
            print(f"   {status_icon} {suite_result['name']}: "
                  f"{suite_result['passed']}P/{suite_result['failed']}F "
                  f"in {duration:.1f}s{coverage}")
        
        # Recommendations
        recommendations = report["recommendations"]
        if recommendations:
            print(f"\n🎯 RECOMMENDATIONS")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)

class CICDPipeline:
    """CI/CD pipeline configuration generator."""
    
    def __init__(self, project_root: str = "/home/tekkadmin/claude-tui"):
        self.project_root = Path(project_root)
    
    def generate_github_actions_workflow(self) -> str:
        """Generate GitHub Actions workflow for automated testing."""
        workflow = {
            "name": "Claude-TUI Test Suite",
            "on": {
                "push": {
                    "branches": ["main", "develop"]
                },
                "pull_request": {
                    "branches": ["main", "develop"]
                },
                "schedule": [
                    {"cron": "0 2 * * *"}  # Daily at 2 AM
                ]
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "strategy": {
                        "matrix": {
                            "python-version": ["3.9", "3.10", "3.11"]
                        }
                    },
                    "steps": [
                        {
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Python ${{ matrix.python-version }}",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "${{ matrix.python-version }}"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run unit tests",
                            "run": "python -m pytest tests/unit/ -m 'unit and critical' --cov=src --cov-report=xml"
                        },
                        {
                            "name": "Run accuracy validation",
                            "run": "python -m pytest tests/validation/test_accuracy_validation_suite.py -m accuracy"
                        },
                        {
                            "name": "Run integration tests", 
                            "run": "python -m pytest tests/integration/ -m integration --tb=short"
                        },
                        {
                            "name": "Upload coverage",
                            "uses": "codecov/codecov-action@v3",
                            "with": {
                                "file": "./coverage.xml"
                            }
                        }
                    ]
                },
                "performance": {
                    "runs-on": "ubuntu-latest",
                    "needs": "test",
                    "steps": [
                        {
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "3.10"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run performance benchmarks",
                            "run": "python -m pytest tests/performance/ -m performance --durations=0"
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False)
    
    def save_github_workflow(self) -> Path:
        """Save GitHub Actions workflow to file."""
        workflow_dir = self.project_root / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_path = workflow_dir / "test-suite.yml"
        
        workflow_content = self.generate_github_actions_workflow()
        
        with open(workflow_path, 'w', encoding='utf-8') as f:
            f.write(workflow_content)
        
        return workflow_path

async def main():
    """Main function to run test automation."""
    print("🚀 Claude-TUI Test Automation Framework")
    
    # Initialize orchestrator
    orchestrator = TestOrchestrator()
    
    # Run all test suites
    print("📋 Running comprehensive test suite...")
    results = await orchestrator.run_all_suites()
    
    # Print summary
    orchestrator.print_summary()
    
    # Save detailed report
    report_path = orchestrator.save_report()
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    # Generate CI/CD configuration
    print("\n🔧 Generating CI/CD pipeline configuration...")
    cicd = CICDPipeline()
    workflow_path = cicd.save_github_workflow()
    print(f"   GitHub Actions workflow saved to: {workflow_path}")
    
    # Final status
    report = orchestrator.generate_report()
    success_rate = report["summary"]["success_rate"]
    critical_passed = report["summary"]["critical_tests_passed"]
    
    print(f"\n🎯 FINAL STATUS")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Critical Tests: {'✅ PASSED' if critical_passed else '❌ FAILED'}")
    
    if critical_passed and success_rate >= 0.95:
        print("   🚀 READY FOR DEPLOYMENT")
    else:
        print("   ⚠️  NEEDS ATTENTION BEFORE DEPLOYMENT")
    
    return results

if __name__ == "__main__":
    import statistics
    asyncio.run(main())