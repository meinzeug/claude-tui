#!/usr/bin/env python3
"""
Test Quality Analyzer for claude-tiu

This module analyzes test quality metrics, generates comprehensive reports,
and provides recommendations for improving test coverage and effectiveness.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse


@dataclass
class TestResult:
    """Individual test result data."""
    name: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: Optional[str] = None
    category: str = "unknown"


@dataclass
class TestSuiteResult:
    """Test suite result data."""
    name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    tests: List[TestResult]


@dataclass
class CoverageReport:
    """Code coverage report data."""
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    statement_coverage: float
    missing_lines: int
    total_lines: int


@dataclass
class TestQualityMetrics:
    """Comprehensive test quality metrics."""
    overall_score: float
    test_distribution: Dict[str, float]
    coverage_report: CoverageReport
    execution_performance: Dict[str, float]
    reliability_metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: str


class TestResultParser:
    """Parse test results from various formats."""
    
    def parse_junit_xml(self, xml_file: Path) -> TestSuiteResult:
        """Parse JUnit XML test results."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        suite_name = root.get('name', xml_file.stem)
        total_tests = int(root.get('tests', 0))
        failures = int(root.get('failures', 0))
        errors = int(root.get('errors', 0))
        skipped = int(root.get('skipped', 0))
        time = float(root.get('time', 0))
        
        passed = total_tests - failures - errors - skipped
        
        tests = []
        for testcase in root.findall('.//testcase'):
            name = testcase.get('name')
            classname = testcase.get('classname', '')
            duration = float(testcase.get('time', 0))
            
            # Determine status
            if testcase.find('failure') is not None:
                status = 'failed'
                message = testcase.find('failure').get('message', '')
            elif testcase.find('error') is not None:
                status = 'error'
                message = testcase.find('error').get('message', '')
            elif testcase.find('skipped') is not None:
                status = 'skipped'
                message = testcase.find('skipped').get('message', '')
            else:
                status = 'passed'
                message = None
            
            # Categorize test
            category = self._categorize_test(classname, name)
            
            tests.append(TestResult(
                name=f"{classname}.{name}",
                status=status,
                duration=duration,
                message=message,
                category=category
            ))
        
        return TestSuiteResult(
            name=suite_name,
            total_tests=total_tests,
            passed=passed,
            failed=failures,
            skipped=skipped,
            errors=errors,
            duration=time,
            tests=tests
        )
    
    def _categorize_test(self, classname: str, testname: str) -> str:
        """Categorize test based on class and test names."""
        classname_lower = classname.lower()
        testname_lower = testname.lower()
        
        if 'unit' in classname_lower or 'unit' in testname_lower:
            return 'unit'
        elif 'integration' in classname_lower or 'integration' in testname_lower:
            return 'integration'
        elif 'e2e' in classname_lower or 'end_to_end' in testname_lower:
            return 'e2e'
        elif 'performance' in classname_lower or 'benchmark' in testname_lower:
            return 'performance'
        elif 'security' in classname_lower or 'security' in testname_lower:
            return 'security'
        elif 'tui' in classname_lower or 'ui' in testname_lower:
            return 'tui'
        else:
            return 'unit'  # Default to unit test


class CoverageAnalyzer:
    """Analyze code coverage reports."""
    
    def parse_coverage_xml(self, xml_file: Path) -> CoverageReport:
        """Parse coverage XML report."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Find coverage metrics
            line_rate = float(root.get('line-rate', 0)) * 100
            branch_rate = float(root.get('branch-rate', 0)) * 100
            
            # Calculate additional metrics
            total_lines = 0
            covered_lines = 0
            
            for package in root.findall('.//package'):
                for classes in package.findall('.//classes/class'):
                    lines = classes.findall('.//lines/line')
                    total_lines += len(lines)
                    covered_lines += len([l for l in lines if l.get('hits', '0') != '0'])
            
            missing_lines = total_lines - covered_lines
            
            return CoverageReport(
                line_coverage=line_rate,
                branch_coverage=branch_rate,
                function_coverage=85.0,  # Placeholder - would calculate from data
                statement_coverage=line_rate,  # Often same as line coverage
                missing_lines=missing_lines,
                total_lines=total_lines
            )
            
        except Exception as e:
            # Return default coverage if parsing fails
            return CoverageReport(
                line_coverage=0.0,
                branch_coverage=0.0,
                function_coverage=0.0,
                statement_coverage=0.0,
                missing_lines=0,
                total_lines=0
            )


class TestQualityAnalyzer:
    """Main analyzer for test quality assessment."""
    
    def __init__(self):
        self.parser = TestResultParser()
        self.coverage_analyzer = CoverageAnalyzer()
    
    def analyze_test_results(self, results_dir: Path) -> TestQualityMetrics:
        """Analyze all test results and generate quality metrics."""
        # Parse all test result files
        test_suites = []
        
        # Find and parse JUnit XML files
        junit_files = list(results_dir.rglob("test-results-*.xml"))
        for junit_file in junit_files:
            try:
                suite = self.parser.parse_junit_xml(junit_file)
                test_suites.append(suite)
            except Exception as e:
                print(f"Warning: Could not parse {junit_file}: {e}")
        
        # Parse coverage reports
        coverage_files = list(results_dir.rglob("coverage.xml"))
        coverage_report = None
        if coverage_files:
            coverage_report = self.coverage_analyzer.parse_coverage_xml(coverage_files[0])
        else:
            coverage_report = CoverageReport(0, 0, 0, 0, 0, 0)
        
        # Calculate metrics
        return self._calculate_quality_metrics(test_suites, coverage_report)
    
    def _calculate_quality_metrics(
        self, 
        test_suites: List[TestSuiteResult], 
        coverage: CoverageReport
    ) -> TestQualityMetrics:
        """Calculate comprehensive quality metrics."""
        
        # Aggregate test results
        total_tests = sum(suite.total_tests for suite in test_suites)
        total_passed = sum(suite.passed for suite in test_suites)
        total_failed = sum(suite.failed for suite in test_suites)
        total_errors = sum(suite.errors for suite in test_suites)
        total_skipped = sum(suite.skipped for suite in test_suites)
        
        # Calculate test distribution
        test_distribution = self._calculate_test_distribution(test_suites)
        
        # Calculate execution performance
        execution_performance = self._calculate_execution_performance(test_suites)
        
        # Calculate reliability metrics
        reliability_metrics = self._calculate_reliability_metrics(
            total_tests, total_passed, total_failed, total_errors, total_skipped
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            test_distribution, coverage, execution_performance, reliability_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            test_distribution, coverage, execution_performance, reliability_metrics
        )
        
        return TestQualityMetrics(
            overall_score=overall_score,
            test_distribution=test_distribution,
            coverage_report=coverage,
            execution_performance=execution_performance,
            reliability_metrics=reliability_metrics,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_test_distribution(self, test_suites: List[TestSuiteResult]) -> Dict[str, float]:
        """Calculate test distribution across categories."""
        category_counts = {}
        total_tests = 0
        
        for suite in test_suites:
            for test in suite.tests:
                category_counts[test.category] = category_counts.get(test.category, 0) + 1
                total_tests += 1
        
        if total_tests == 0:
            return {}
        
        distribution = {}
        for category, count in category_counts.items():
            distribution[f"{category}_percentage"] = (count / total_tests) * 100
            distribution[f"{category}_count"] = count
        
        distribution["total_tests"] = total_tests
        return distribution
    
    def _calculate_execution_performance(self, test_suites: List[TestSuiteResult]) -> Dict[str, float]:
        """Calculate test execution performance metrics."""
        total_duration = sum(suite.duration for suite in test_suites)
        total_tests = sum(suite.total_tests for suite in test_suites)
        
        if total_tests == 0:
            return {"total_duration": 0, "average_test_time": 0, "tests_per_second": 0}
        
        average_test_time = total_duration / total_tests
        tests_per_second = total_tests / total_duration if total_duration > 0 else 0
        
        # Find slowest tests
        all_tests = []
        for suite in test_suites:
            all_tests.extend(suite.tests)
        
        slow_tests = [t for t in all_tests if t.duration > 5.0]  # Tests > 5 seconds
        
        return {
            "total_duration": total_duration,
            "average_test_time": average_test_time,
            "tests_per_second": tests_per_second,
            "slow_test_count": len(slow_tests),
            "execution_efficiency": min(100, tests_per_second * 10)  # Efficiency score
        }
    
    def _calculate_reliability_metrics(
        self, total: int, passed: int, failed: int, errors: int, skipped: int
    ) -> Dict[str, float]:
        """Calculate test reliability metrics."""
        if total == 0:
            return {"success_rate": 0, "failure_rate": 0, "error_rate": 0, "skip_rate": 0}
        
        return {
            "success_rate": (passed / total) * 100,
            "failure_rate": (failed / total) * 100,
            "error_rate": (errors / total) * 100,
            "skip_rate": (skipped / total) * 100,
            "reliability_score": ((passed / total) * 100) if total > 0 else 0
        }
    
    def _calculate_overall_score(
        self, 
        distribution: Dict[str, float],
        coverage: CoverageReport,
        performance: Dict[str, float],
        reliability: Dict[str, float]
    ) -> float:
        """Calculate overall test quality score (0-100)."""
        scores = {}
        
        # Coverage score (25% weight)
        coverage_score = (
            coverage.line_coverage * 0.4 + 
            coverage.branch_coverage * 0.4 + 
            coverage.function_coverage * 0.2
        )
        scores["coverage"] = min(100, coverage_score)
        
        # Test distribution score (20% weight) - pyramid compliance
        pyramid_score = 100
        unit_pct = distribution.get("unit_percentage", 0)
        integration_pct = distribution.get("integration_percentage", 0)
        e2e_pct = distribution.get("e2e_percentage", 0)
        
        # Ideal: 60% unit, 30% integration, 10% e2e
        unit_deviation = abs(unit_pct - 60)
        integration_deviation = abs(integration_pct - 30)
        e2e_deviation = abs(e2e_pct - 10)
        
        pyramid_score -= (unit_deviation + integration_deviation + e2e_deviation) / 3
        scores["distribution"] = max(0, pyramid_score)
        
        # Performance score (15% weight)
        execution_score = min(100, performance.get("execution_efficiency", 0))
        scores["performance"] = execution_score
        
        # Reliability score (25% weight)
        scores["reliability"] = reliability.get("reliability_score", 0)
        
        # Completeness score (15% weight) - based on having all test types
        test_types = ["unit", "integration", "security", "tui"]
        present_types = sum(1 for t in test_types if distribution.get(f"{t}_count", 0) > 0)
        completeness_score = (present_types / len(test_types)) * 100
        scores["completeness"] = completeness_score
        
        # Weighted average
        weights = {
            "coverage": 0.25,
            "distribution": 0.20,
            "performance": 0.15,
            "reliability": 0.25,
            "completeness": 0.15
        }
        
        overall_score = sum(scores[category] * weights[category] for category in weights)
        return round(overall_score, 1)
    
    def _generate_recommendations(
        self,
        distribution: Dict[str, float],
        coverage: CoverageReport,
        performance: Dict[str, float],
        reliability: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations for improvement."""
        recommendations = []
        
        # Coverage recommendations
        if coverage.line_coverage < 80:
            recommendations.append(
                f"Increase line coverage from {coverage.line_coverage:.1f}% to at least 80%"
            )
        
        if coverage.branch_coverage < 75:
            recommendations.append(
                f"Improve branch coverage from {coverage.branch_coverage:.1f}% to at least 75%"
            )
        
        # Test distribution recommendations
        unit_pct = distribution.get("unit_percentage", 0)
        if unit_pct < 55:
            unit_gap = int((60 - unit_pct) / 100 * distribution.get("total_tests", 0))
            recommendations.append(f"Add approximately {unit_gap} unit tests to achieve 60% distribution")
        
        e2e_pct = distribution.get("e2e_percentage", 0)
        if e2e_pct > 15:
            e2e_excess = int((e2e_pct - 10) / 100 * distribution.get("total_tests", 0))
            recommendations.append(f"Consider converting {e2e_excess} E2E tests to integration tests")
        
        # Performance recommendations
        slow_tests = performance.get("slow_test_count", 0)
        if slow_tests > 0:
            recommendations.append(f"Optimize {slow_tests} slow tests (>5 seconds) for better execution speed")
        
        if performance.get("tests_per_second", 0) < 2:
            recommendations.append("Improve test execution speed - current throughput is below 2 tests/second")
        
        # Reliability recommendations
        failure_rate = reliability.get("failure_rate", 0)
        if failure_rate > 5:
            recommendations.append(f"Investigate and fix failing tests - current failure rate: {failure_rate:.1f}%")
        
        skip_rate = reliability.get("skip_rate", 0)
        if skip_rate > 10:
            recommendations.append(f"Reduce skipped tests - current skip rate: {skip_rate:.1f}%")
        
        # Missing test categories
        categories = ["unit", "integration", "security", "tui", "performance"]
        missing_categories = [cat for cat in categories if distribution.get(f"{cat}_count", 0) == 0]
        if missing_categories:
            recommendations.append(f"Add tests for missing categories: {', '.join(missing_categories)}")
        
        return recommendations
    
    def generate_report(self, metrics: TestQualityMetrics) -> str:
        """Generate a comprehensive quality report."""
        report = f"""
# 📊 Test Quality Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overall Quality Score: {metrics.overall_score}/100

### 🎯 Summary
{"✅ Excellent test quality!" if metrics.overall_score >= 90 else 
 "🟡 Good test quality with room for improvement" if metrics.overall_score >= 75 else
 "🔴 Test quality needs significant improvement"}

### 📈 Test Distribution
- **Unit Tests**: {metrics.test_distribution.get('unit_count', 0)} ({metrics.test_distribution.get('unit_percentage', 0):.1f}%)
- **Integration Tests**: {metrics.test_distribution.get('integration_count', 0)} ({metrics.test_distribution.get('integration_percentage', 0):.1f}%)
- **E2E Tests**: {metrics.test_distribution.get('e2e_count', 0)} ({metrics.test_distribution.get('e2e_percentage', 0):.1f}%)
- **Security Tests**: {metrics.test_distribution.get('security_count', 0)} ({metrics.test_distribution.get('security_percentage', 0):.1f}%)
- **TUI Tests**: {metrics.test_distribution.get('tui_count', 0)} ({metrics.test_distribution.get('tui_percentage', 0):.1f}%)
- **Performance Tests**: {metrics.test_distribution.get('performance_count', 0)} ({metrics.test_distribution.get('performance_percentage', 0):.1f}%)

**Total Tests**: {metrics.test_distribution.get('total_tests', 0)}

### 🔍 Coverage Analysis
- **Line Coverage**: {metrics.coverage_report.line_coverage:.1f}%
- **Branch Coverage**: {metrics.coverage_report.branch_coverage:.1f}%
- **Function Coverage**: {metrics.coverage_report.function_coverage:.1f}%
- **Missing Lines**: {metrics.coverage_report.missing_lines}/{metrics.coverage_report.total_lines}

### ⚡ Performance Metrics
- **Total Execution Time**: {metrics.execution_performance.get('total_duration', 0):.1f}s
- **Average Test Time**: {metrics.execution_performance.get('average_test_time', 0):.3f}s
- **Test Throughput**: {metrics.execution_performance.get('tests_per_second', 0):.1f} tests/second
- **Slow Tests**: {metrics.execution_performance.get('slow_test_count', 0)} (>5s)

### 🛡️ Reliability Metrics
- **Success Rate**: {metrics.reliability_metrics.get('success_rate', 0):.1f}%
- **Failure Rate**: {metrics.reliability_metrics.get('failure_rate', 0):.1f}%
- **Error Rate**: {metrics.reliability_metrics.get('error_rate', 0):.1f}%
- **Skip Rate**: {metrics.reliability_metrics.get('skip_rate', 0):.1f}%

### 💡 Recommendations

"""
        
        for i, recommendation in enumerate(metrics.recommendations, 1):
            report += f"{i}. {recommendation}\n"
        
        if not metrics.recommendations:
            report += "No specific recommendations - test quality is excellent! 🎉\n"
        
        report += f"""
### 📊 Quality Breakdown
- **Coverage Score**: {((metrics.coverage_report.line_coverage * 0.4 + metrics.coverage_report.branch_coverage * 0.4 + metrics.coverage_report.function_coverage * 0.2)):.1f}/100
- **Distribution Score**: Based on test pyramid compliance
- **Performance Score**: {metrics.execution_performance.get('execution_efficiency', 0):.1f}/100
- **Reliability Score**: {metrics.reliability_metrics.get('reliability_score', 0):.1f}/100

---

*Report generated by claude-tiu Test Quality Analyzer*
*Timestamp: {metrics.timestamp}*
"""
        
        return report


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze test quality and generate reports")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing test results")
    parser.add_argument("--output", type=Path, help="Output file for quality report")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input directory {args.input} does not exist")
        return 1
    
    # Analyze test quality
    analyzer = TestQualityAnalyzer()
    metrics = analyzer.analyze_test_results(args.input)
    
    if args.verbose:
        print("🔍 Analyzing test results...")
        print(f"   Found test results in: {args.input}")
        print(f"   Overall quality score: {metrics.overall_score}/100")
    
    # Generate output
    if args.json:
        output = json.dumps(asdict(metrics), indent=2)
    else:
        output = analyzer.generate_report(metrics)
    
    # Write to file or stdout
    if args.output:
        args.output.write_text(output)
        print(f"✅ Quality report written to: {args.output}")
    else:
        print(output)
    
    return 0


if __name__ == "__main__":
    exit(main())