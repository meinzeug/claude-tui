#!/usr/bin/env python3
"""
Enhanced test runner for claude-tiu project with comprehensive testing capabilities.

This module provides a centralized test runner that:
- Runs all test categories (unit, integration, performance, security)
- Generates comprehensive coverage reports
- Provides quality gates and thresholds
- Integrates with CI/CD pipelines
- Supports parallel test execution
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import pytest


class TestRunner:
    """Comprehensive test runner with quality gates and reporting."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test runner."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        self.coverage_threshold = 80
        self.performance_threshold = 30  # seconds for performance tests
        
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run unit tests with coverage analysis."""
        print("🧪 Running unit tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "unit"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--asyncio-mode=auto",
        ]
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit",
                "--cov-report=xml:coverage-unit.xml",
                f"--cov-fail-under={self.coverage_threshold}"
            ])
        
        return self._run_command(cmd, "Unit tests")
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "integration"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--timeout=60",
            "--asyncio-mode=auto",
        ]
        
        return self._run_command(cmd, "Integration tests")
    
    def run_tui_tests(self, verbose: bool = False) -> int:
        """Run TUI component tests."""
        print("🖥️  Running TUI tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "ui"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--timeout=30",
            "--asyncio-mode=auto",
        ]
        
        return self._run_command(cmd, "TUI tests")
    
    def run_performance_tests(self, verbose: bool = False) -> int:
        """Run performance and load tests."""
        print("⚡ Running performance tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "performance"),
            "-v" if verbose else "-q",
            "--tb=short",
            f"--timeout={self.performance_threshold}",
            "-m", "performance",
        ]
        
        return self._run_command(cmd, "Performance tests")
    
    def run_security_tests(self, verbose: bool = False) -> int:
        """Run security and vulnerability tests."""
        print("🔒 Running security tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "security"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--timeout=60",
        ]
        
        return self._run_command(cmd, "Security tests")
    
    def run_e2e_tests(self, verbose: bool = False) -> int:
        """Run end-to-end workflow tests."""
        print("🎯 Running E2E tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "e2e"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--timeout=120",
            "--asyncio-mode=auto",
        ]
        
        return self._run_command(cmd, "E2E tests")
    
    def run_validation_tests(self, verbose: bool = False) -> int:
        """Run anti-hallucination validation tests."""
        print("🔍 Running validation tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "validation"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--timeout=30",
            "--asyncio-mode=auto",
        ]
        
        return self._run_command(cmd, "Validation tests")
    
    def run_all_tests(self, verbose: bool = False, parallel: bool = False) -> Dict[str, int]:
        """Run all test suites and return results."""
        print("🚀 Running comprehensive test suite...")
        start_time = time.time()
        
        results = {}
        
        if parallel:
            # Run tests in parallel using pytest-xdist
            cmd = [
                sys.executable, "-m", "pytest",
                str(self.tests_dir),
                "-v" if verbose else "-q",
                "--tb=short",
                "--asyncio-mode=auto",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                f"--cov-fail-under={self.coverage_threshold}",
                "-n", "auto",  # Use all available CPUs
            ]
            results["all"] = self._run_command(cmd, "All tests (parallel)")
        else:
            # Run test suites sequentially
            results["unit"] = self.run_unit_tests(verbose)
            results["integration"] = self.run_integration_tests(verbose)
            results["tui"] = self.run_tui_tests(verbose)
            results["validation"] = self.run_validation_tests(verbose)
            results["security"] = self.run_security_tests(verbose)
            results["performance"] = self.run_performance_tests(verbose)
            results["e2e"] = self.run_e2e_tests(verbose)
        
        duration = time.time() - start_time
        self._print_summary(results, duration)
        
        return results
    
    def run_smoke_tests(self, verbose: bool = False) -> int:
        """Run smoke tests for basic functionality."""
        print("💨 Running smoke tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "smoke",
            "--timeout=10",
        ]
        
        return self._run_command(cmd, "Smoke tests")
    
    def run_coverage_analysis(self) -> int:
        """Generate comprehensive coverage analysis."""
        print("📊 Running coverage analysis...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir),
            "--cov=src",
            "--cov-report=term-missing:skip-covered",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            f"--cov-fail-under={self.coverage_threshold}",
            "--cov-branch",
            "-q",
        ]
        
        result = self._run_command(cmd, "Coverage analysis")
        
        if result == 0:
            print("✅ Coverage analysis completed successfully")
            print(f"📁 HTML report: {self.project_root}/htmlcov/index.html")
        
        return result
    
    def run_linting(self) -> int:
        """Run code linting and quality checks."""
        print("🧹 Running code quality checks...")
        
        results = []
        
        # Run black formatting check
        print("  • Checking code formatting (black)...")
        results.append(self._run_command([
            sys.executable, "-m", "black", "--check", "--diff", "src", "tests"
        ], "Black formatting"))
        
        # Run isort import sorting check
        print("  • Checking import sorting (isort)...")
        results.append(self._run_command([
            sys.executable, "-m", "isort", "--check-only", "--diff", "src", "tests"
        ], "Import sorting"))
        
        # Run flake8 linting
        print("  • Running linting (flake8)...")
        results.append(self._run_command([
            sys.executable, "-m", "flake8", "src", "tests"
        ], "Flake8 linting"))
        
        # Run mypy type checking
        print("  • Running type checking (mypy)...")
        results.append(self._run_command([
            sys.executable, "-m", "mypy", "src"
        ], "Type checking"))
        
        return max(results) if results else 0
    
    def run_security_scan(self) -> int:
        """Run security vulnerability scanning."""
        print("🛡️  Running security scans...")
        
        results = []
        
        # Run bandit security linting
        print("  • Running security linting (bandit)...")
        results.append(self._run_command([
            sys.executable, "-m", "bandit", "-r", "src", "-f", "json", "-o", "bandit-report.json"
        ], "Bandit security scan"))
        
        # Run safety dependency check
        print("  • Checking dependencies (safety)...")
        results.append(self._run_command([
            sys.executable, "-m", "safety", "check", "--json", "--output", "safety-report.json"
        ], "Safety dependency check"))
        
        return max(results) if results else 0
    
    def _run_command(self, cmd: List[str], description: str) -> int:
        """Run a command and handle the result."""
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=False)
            if result.returncode == 0:
                print(f"✅ {description} passed")
            else:
                print(f"❌ {description} failed")
            return result.returncode
        except FileNotFoundError as e:
            print(f"❌ {description} failed: {e}")
            return 1
        except Exception as e:
            print(f"❌ {description} failed with error: {e}")
            return 1
    
    def _print_summary(self, results: Dict[str, int], duration: float):
        """Print test execution summary."""
        print("\n" + "="*60)
        print("📋 TEST EXECUTION SUMMARY")
        print("="*60)
        
        passed = sum(1 for code in results.values() if code == 0)
        total = len(results)
        
        print(f"⏱️  Total execution time: {duration:.2f} seconds")
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        
        print("\n📊 Test Suite Results:")
        for suite, code in results.items():
            status = "✅ PASS" if code == 0 else "❌ FAIL"
            print(f"  • {suite.ljust(15)} {status}")
        
        if all(code == 0 for code in results.values()):
            print("\n🎉 All tests passed! Ready for deployment.")
        else:
            print("\n⚠️  Some tests failed. Please review and fix issues.")
        
        print("="*60)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="claude-tiu Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--lint", "-l", action="store_true", help="Run linting checks")
    parser.add_argument("--security", "-s", action="store_true", help="Run security scans")
    
    subparsers = parser.add_subparsers(dest="command", help="Test commands")
    
    # Test suite commands
    subparsers.add_parser("unit", help="Run unit tests")
    subparsers.add_parser("integration", help="Run integration tests")
    subparsers.add_parser("tui", help="Run TUI tests")
    subparsers.add_parser("performance", help="Run performance tests")
    subparsers.add_parser("security-tests", help="Run security tests")
    subparsers.add_parser("e2e", help="Run E2E tests")
    subparsers.add_parser("validation", help="Run validation tests")
    subparsers.add_parser("smoke", help="Run smoke tests")
    subparsers.add_parser("all", help="Run all tests")
    
    # Analysis commands
    subparsers.add_parser("coverage", help="Generate coverage analysis")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Run additional checks if requested
    if args.lint:
        lint_result = runner.run_linting()
        if lint_result != 0:
            sys.exit(lint_result)
    
    if args.security:
        security_result = runner.run_security_scan()
        if security_result != 0:
            sys.exit(security_result)
    
    # Run tests based on command
    if args.command == "unit":
        result = runner.run_unit_tests(args.verbose, args.coverage)
    elif args.command == "integration":
        result = runner.run_integration_tests(args.verbose)
    elif args.command == "tui":
        result = runner.run_tui_tests(args.verbose)
    elif args.command == "performance":
        result = runner.run_performance_tests(args.verbose)
    elif args.command == "security-tests":
        result = runner.run_security_tests(args.verbose)
    elif args.command == "e2e":
        result = runner.run_e2e_tests(args.verbose)
    elif args.command == "validation":
        result = runner.run_validation_tests(args.verbose)
    elif args.command == "smoke":
        result = runner.run_smoke_tests(args.verbose)
    elif args.command == "coverage":
        result = runner.run_coverage_analysis()
    elif args.command == "all" or args.command is None:
        results = runner.run_all_tests(args.verbose, args.parallel)
        result = max(results.values()) if results else 0
    else:
        parser.print_help()
        result = 1
    
    sys.exit(result)


if __name__ == "__main__":
    main()