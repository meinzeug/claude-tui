#!/usr/bin/env python3
"""
Comprehensive Test Runner for Claude-TIU.

Provides convenient interface for running different test suites with
appropriate configurations for development, CI, and validation scenarios.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


class TestRunner:
    """Comprehensive test runner with multiple execution modes."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        
    def run_unit_tests(self, coverage: bool = True, verbose: bool = False) -> int:
        """Run unit tests with optional coverage."""
        cmd = ["python3", "-m", "pytest"]
        
        # Test selection
        cmd.extend(["-m", "unit and not slow"])
        cmd.extend(["tests/unit/"])
        
        # Coverage options
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit",
                "--cov-fail-under=85"
            ])
        
        # Verbosity
        if verbose:
            cmd.extend(["-v", "-s"])
        else:
            cmd.extend(["-q"])
        
        return self._run_command(cmd, "Unit Tests")
    
    def run_integration_tests(self, coverage: bool = True, verbose: bool = False) -> int:
        """Run integration tests."""
        cmd = ["python3", "-m", "pytest"]
        
        # Test selection
        cmd.extend(["-m", "integration"])
        cmd.extend(["tests/integration/"])
        
        # Extended timeout for integration tests
        cmd.extend(["--timeout=600"])
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing", 
                "--cov-report=html:htmlcov/integration",
                "--cov-fail-under=75"
            ])
        
        if verbose:
            cmd.extend(["-v", "-s"])
        
        return self._run_command(cmd, "Integration Tests")
    
    def run_validation_tests(self, comprehensive: bool = False) -> int:
        """Run anti-hallucination validation tests."""
        cmd = ["python3", "-m", "pytest"]
        
        # Test selection
        if comprehensive:
            cmd.extend(["-m", "validation or antihallucination"])
        else:
            cmd.extend(["-m", "validation and not slow"])
        
        cmd.extend(["tests/validation/"])
        
        # Validation-specific options
        cmd.extend([
            "--tb=long",  # Detailed output for validation failures
            "--durations=0",  # Show all durations
            "-v"
        ])
        
        return self._run_command(cmd, "Validation Tests")
    
    def run_performance_tests(self, quick: bool = False) -> int:
        """Run performance tests."""
        cmd = ["python3", "-m", "pytest"]
        
        # Test selection
        if quick:
            cmd.extend(["-m", "performance and not slow and not stress"])
        else:
            cmd.extend(["-m", "performance"])
        
        cmd.extend(["tests/performance/"])
        
        # Performance-specific options
        cmd.extend([
            "--timeout=1200",  # 20 minutes for performance tests
            "--tb=short",
            "--durations=20"
        ])
        
        return self._run_command(cmd, "Performance Tests")
    
    def run_security_tests(self, comprehensive: bool = False) -> int:
        """Run security tests."""
        cmd = ["python3", "-m", "pytest"]
        
        # Test selection
        if comprehensive:
            cmd.extend(["-m", "security"])
        else:
            cmd.extend(["-m", "security and not slow"])
        
        cmd.extend(["tests/security/"])
        
        # Security-specific options
        cmd.extend([
            "--tb=long",  # Detailed output for security issues
            "-v"
        ])
        
        return self._run_command(cmd, "Security Tests")
    
    def run_tui_tests(self) -> int:
        """Run TUI component tests."""
        cmd = ["python3", "-m", "pytest"]
        
        cmd.extend(["-m", "tui or ui"])
        cmd.extend(["tests/ui/", "tests/tui/"])
        
        # TUI-specific options
        cmd.extend([
            "--tb=short",
            "-v"
        ])
        
        return self._run_command(cmd, "TUI Tests")
    
    def run_smoke_tests(self) -> int:
        """Run smoke tests for basic functionality."""
        cmd = ["python3", "-m", "pytest"]
        
        cmd.extend(["-m", "smoke"])
        cmd.extend(["--tb=short", "-q"])
        
        return self._run_command(cmd, "Smoke Tests")
    
    def run_all_tests(self, coverage: bool = True, parallel: bool = False) -> int:
        """Run complete test suite."""
        cmd = ["python3", "-m", "pytest"]
        
        # Include all test directories
        cmd.extend(["tests/"])
        
        # Exclude slow tests by default for full runs
        cmd.extend(["-m", "not slow or critical"])
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                "--cov-fail-under=80"
            ])
        
        if parallel:
            # Use pytest-xdist for parallel execution
            cmd.extend(["-n", "auto"])
        
        # Comprehensive options
        cmd.extend([
            "--tb=short",
            "--maxfail=10",
            "--durations=10"
        ])
        
        return self._run_command(cmd, "Complete Test Suite")
    
    def run_ci_tests(self) -> int:
        """Run tests optimized for CI environment."""
        cmd = ["python3", "-m", "pytest"]
        
        cmd.extend(["tests/"])
        
        # CI-optimized selection
        cmd.extend(["-m", "not slow and not requires_gpu and not requires_docker"])
        
        # CI-specific options
        cmd.extend([
            "--cov=src",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term",
            "--cov-fail-under=80",
            "--tb=short",
            "--maxfail=3",
            "--durations=10",
            "--junit-xml=test-results.xml"
        ])
        
        return self._run_command(cmd, "CI Test Suite")
    
    def run_failing_tests(self) -> int:
        """Re-run only previously failed tests."""
        cmd = ["python", "-m", "pytest", "--lf"]
        return self._run_command(cmd, "Failed Tests")
    
    def run_custom_tests(self, markers: str, paths: List[str] = None) -> int:
        """Run tests with custom markers and paths."""
        cmd = ["python3", "-m", "pytest"]
        
        if markers:
            cmd.extend(["-m", markers])
        
        if paths:
            cmd.extend(paths)
        else:
            cmd.extend(["tests/"])
        
        cmd.extend(["-v"])
        
        return self._run_command(cmd, f"Custom Tests ({markers})")
    
    def _run_command(self, cmd: List[str], test_name: str) -> int:
        """Execute command and handle output."""
        print(f"\n{'='*60}")
        print(f"Running {test_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root)
            
            # Run the command
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                capture_output=False
            )
            
            print(f"\n{test_name} completed with exit code: {result.returncode}")
            return result.returncode
            
        except KeyboardInterrupt:
            print(f"\n{test_name} interrupted by user")
            return 130
        except Exception as e:
            print(f"\nError running {test_name}: {e}")
            return 1
    
    def generate_coverage_report(self) -> int:
        """Generate comprehensive coverage report."""
        print("Generating coverage report...")
        
        commands = [
            ["python3", "-m", "coverage", "combine"],
            ["python3", "-m", "coverage", "report", "--show-missing"],
            ["python3", "-m", "coverage", "html", "--directory=htmlcov"],
            ["python3", "-m", "coverage", "xml", "--output=coverage.xml"]
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd, cwd=self.project_root)
            if result.returncode != 0:
                return result.returncode
        
        print("Coverage report generated successfully!")
        print("HTML report: htmlcov/index.html")
        print("XML report: coverage.xml")
        return 0


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Claude-TIU Test Runner")
    
    parser.add_argument("test_type", 
                       choices=["unit", "integration", "validation", "performance", 
                               "security", "tui", "smoke", "all", "ci", "failed", "custom"],
                       help="Type of tests to run")
    
    parser.add_argument("--no-coverage", action="store_true",
                       help="Disable coverage reporting")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    parser.add_argument("--parallel", "-n", action="store_true",
                       help="Run tests in parallel")
    
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive tests (including slow ones)")
    
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only")
    
    parser.add_argument("--markers", "-m", type=str,
                       help="Custom test markers for custom test type")
    
    parser.add_argument("--paths", "-p", nargs="*",
                       help="Custom test paths")
    
    parser.add_argument("--coverage-report", action="store_true",
                       help="Generate coverage report only")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Handle coverage report generation
    if args.coverage_report:
        return runner.generate_coverage_report()
    
    # Coverage setting
    coverage = not args.no_coverage
    
    # Route to appropriate test method
    if args.test_type == "unit":
        return runner.run_unit_tests(coverage=coverage, verbose=args.verbose)
    elif args.test_type == "integration":
        return runner.run_integration_tests(coverage=coverage, verbose=args.verbose)
    elif args.test_type == "validation":
        return runner.run_validation_tests(comprehensive=args.comprehensive)
    elif args.test_type == "performance":
        return runner.run_performance_tests(quick=args.quick)
    elif args.test_type == "security":
        return runner.run_security_tests(comprehensive=args.comprehensive)
    elif args.test_type == "tui":
        return runner.run_tui_tests()
    elif args.test_type == "smoke":
        return runner.run_smoke_tests()
    elif args.test_type == "all":
        return runner.run_all_tests(coverage=coverage, parallel=args.parallel)
    elif args.test_type == "ci":
        return runner.run_ci_tests()
    elif args.test_type == "failed":
        return runner.run_failing_tests()
    elif args.test_type == "custom":
        if not args.markers:
            print("Error: --markers required for custom test type")
            return 1
        return runner.run_custom_tests(args.markers, args.paths)
    else:
        print(f"Unknown test type: {args.test_type}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)