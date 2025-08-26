#!/usr/bin/env python3
"""
Test Coverage Analysis - Identify modules needing tests
Generate comprehensive coverage report and create test templates
"""

import ast
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pytest

logger = logging.getLogger(__name__)

class TestCoverageAnalyzer:
    """Analyze test coverage and identify gaps"""
    
    def __init__(self, src_dir: str = "src", tests_dir: str = "tests"):
        self.src_dir = Path(src_dir)
        self.tests_dir = Path(tests_dir)
        self.uncovered_modules: List[str] = []
        self.coverage_report: Dict[str, Dict] = {}
        
    def analyze_source_files(self) -> Dict[str, Dict]:
        """Analyze all Python source files"""
        source_info = {}
        
        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                # Extract classes and functions
                classes = []
                functions = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):  # Skip private methods
                            functions.append(node.name)
                
                relative_path = py_file.relative_to(self.src_dir)
                source_info[str(relative_path)] = {
                    'path': str(py_file),
                    'classes': classes,
                    'functions': functions,
                    'lines': len(content.splitlines()),
                    'has_tests': self._has_corresponding_test(relative_path)
                }
                
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")
                
        return source_info
    
    def _has_corresponding_test(self, src_path: Path) -> bool:
        """Check if a source file has corresponding tests"""
        test_patterns = [
            f"test_{src_path.stem}.py",
            f"test_{src_path.stem}_unit.py",
            f"test_{src_path.stem}_integration.py",
            f"test_{src_path.stem}_comprehensive.py"
        ]
        
        # Check in multiple test directories
        test_dirs = [
            self.tests_dir,
            self.tests_dir / "unit",
            self.tests_dir / "integration",
            self.tests_dir / src_path.parent.name if src_path.parent.name != "." else self.tests_dir
        ]
        
        for test_dir in test_dirs:
            if not test_dir.exists():
                continue
                
            for pattern in test_patterns:
                if (test_dir / pattern).exists():
                    return True
                    
        return False
    
    def identify_untested_modules(self, source_info: Dict) -> List[str]:
        """Identify modules that lack adequate test coverage"""
        untested = []
        
        for file_path, info in source_info.items():
            if not info['has_tests']:
                # Priority scoring based on complexity
                priority_score = len(info['classes']) * 3 + len(info['functions']) * 1
                
                untested.append({
                    'file': file_path,
                    'priority_score': priority_score,
                    'classes': len(info['classes']),
                    'functions': len(info['functions']),
                    'lines': info['lines']
                })
        
        # Sort by priority (most important first)
        untested.sort(key=lambda x: x['priority_score'], reverse=True)
        return untested
    
    def run_coverage_analysis(self) -> Dict:
        """Run pytest with coverage to get detailed report"""
        try:
            # Run pytest with coverage but skip the problematic tests for now
            cmd = [
                "python3", "-m", "pytest", 
                "--cov=src", 
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing",
                "-x",  # Stop on first failure
                "tests/test_coverage_analysis.py",  # Only run this test initially
                "--tb=short"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            # Try to load coverage data
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                import json
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                return coverage_data
                    
        except Exception as e:
            logger.warning(f"Coverage analysis failed: {e}")
            
        return {}

    def generate_priority_test_plan(self) -> Dict:
        """Generate a prioritized test implementation plan"""
        source_info = self.analyze_source_files()
        untested = self.identify_untested_modules(source_info)
        
        # Categorize by priority
        critical = [m for m in untested if m['priority_score'] >= 10]
        high = [m for m in untested if 5 <= m['priority_score'] < 10]
        medium = [m for m in untested if 2 <= m['priority_score'] < 5]
        low = [m for m in untested if m['priority_score'] < 2]
        
        return {
            'total_source_files': len(source_info),
            'total_untested': len(untested),
            'priority_breakdown': {
                'critical': len(critical),
                'high': len(high),
                'medium': len(medium),
                'low': len(low)
            },
            'critical_modules': critical[:10],  # Top 10 most critical
            'recommendations': self._generate_recommendations(critical, high)
        }
    
    def _generate_recommendations(self, critical: List, high: List) -> List[str]:
        """Generate testing recommendations"""
        recommendations = []
        
        if critical:
            recommendations.append("IMMEDIATE: Create unit tests for critical business logic modules")
            recommendations.append("Focus on modules with highest class/function count first")
            
        if high:
            recommendations.append("HIGH PRIORITY: Add integration tests for service layers")
            
        recommendations.extend([
            "Create comprehensive test fixtures and mocks",
            "Implement performance benchmarks for critical paths",
            "Add end-to-end tests for complete user workflows",
            "Set up continuous integration with coverage requirements"
        ])
        
        return recommendations


class TestCoverageAnalysis:
    """Test the coverage analysis functionality itself"""
    
    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly"""
        analyzer = TestCoverageAnalyzer()
        assert analyzer.src_dir == Path("src")
        assert analyzer.tests_dir == Path("tests")
        assert isinstance(analyzer.uncovered_modules, list)
        assert isinstance(analyzer.coverage_report, dict)
    
    def test_source_file_analysis(self):
        """Test source file analysis"""
        analyzer = TestCoverageAnalyzer()
        source_info = analyzer.analyze_source_files()
        
        # Should find source files
        assert len(source_info) > 0
        
        # Each entry should have required fields
        for file_path, info in source_info.items():
            assert 'path' in info
            assert 'classes' in info
            assert 'functions' in info
            assert 'lines' in info
            assert 'has_tests' in info
            assert isinstance(info['classes'], list)
            assert isinstance(info['functions'], list)
            assert isinstance(info['lines'], int)
            assert isinstance(info['has_tests'], bool)
    
    def test_priority_test_plan_generation(self):
        """Test priority test plan generation"""
        analyzer = TestCoverageAnalyzer()
        plan = analyzer.generate_priority_test_plan()
        
        # Should have required structure
        assert 'total_source_files' in plan
        assert 'total_untested' in plan
        assert 'priority_breakdown' in plan
        assert 'critical_modules' in plan
        assert 'recommendations' in plan
        
        # Priority breakdown should have all categories
        breakdown = plan['priority_breakdown']
        assert 'critical' in breakdown
        assert 'high' in breakdown
        assert 'medium' in breakdown
        assert 'low' in breakdown
        
        # Recommendations should be non-empty
        assert len(plan['recommendations']) > 0


if __name__ == "__main__":
    # Run analysis and print results
    analyzer = TestCoverageAnalyzer()
    plan = analyzer.generate_priority_test_plan()
    
    print("="*80)
    print("CLAUDE-TUI TEST COVERAGE ANALYSIS")
    print("="*80)
    print(f"Total source files: {plan['total_source_files']}")
    print(f"Total untested modules: {plan['total_untested']}")
    print()
    
    print("Priority Breakdown:")
    for priority, count in plan['priority_breakdown'].items():
        print(f"  {priority.upper()}: {count} modules")
    print()
    
    print("CRITICAL MODULES NEEDING TESTS:")
    print("-" * 40)
    for module in plan['critical_modules']:
        print(f"📁 {module['file']}")
        print(f"   Classes: {module['classes']}, Functions: {module['functions']}")
        print(f"   Lines: {module['lines']}, Priority Score: {module['priority_score']}")
        print()
    
    print("RECOMMENDATIONS:")
    print("-" * 40)
    for i, rec in enumerate(plan['recommendations'], 1):
        print(f"{i}. {rec}")
    print()