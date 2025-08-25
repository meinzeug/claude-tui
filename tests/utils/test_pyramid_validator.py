#!/usr/bin/env python3
"""
Test Pyramid Validator for claude-tiu

This module validates and monitors test distribution according to the testing pyramid
principles, ensuring optimal balance between unit, integration, and e2e tests.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class TestType(Enum):
    """Test type classifications."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class TestDistribution:
    """Test distribution metrics."""
    unit_count: int
    integration_count: int
    e2e_count: int
    total_count: int
    
    @property
    def unit_percentage(self) -> float:
        return (self.unit_count / self.total_count) * 100 if self.total_count > 0 else 0
    
    @property
    def integration_percentage(self) -> float:
        return (self.integration_count / self.total_count) * 100 if self.total_count > 0 else 0
    
    @property
    def e2e_percentage(self) -> float:
        return (self.e2e_count / self.total_count) * 100 if self.total_count > 0 else 0


@dataclass
class PyramidTarget:
    """Target distribution for test pyramid."""
    unit_target: float = 60.0
    integration_target: float = 30.0
    e2e_target: float = 10.0
    tolerance: float = 5.0  # Acceptable deviation in percentage points


class TestPyramidValidator:
    """Validate and monitor test pyramid compliance."""
    
    def __init__(self, test_root: Path = None):
        self.test_root = test_root or Path(__file__).parent.parent
        self.targets = PyramidTarget()
        
    def analyze_test_distribution(self) -> TestDistribution:
        """Analyze current test distribution."""
        test_files = list(self.test_root.rglob("test_*.py"))
        
        unit_count = 0
        integration_count = 0
        e2e_count = 0
        
        for test_file in test_files:
            test_type = self._classify_test_file(test_file)
            test_functions = self._count_test_functions(test_file)
            
            if test_type == TestType.UNIT:
                unit_count += test_functions
            elif test_type == TestType.INTEGRATION:
                integration_count += test_functions
            elif test_type == TestType.E2E:
                e2e_count += test_functions
        
        total = unit_count + integration_count + e2e_count
        
        return TestDistribution(
            unit_count=unit_count,
            integration_count=integration_count,
            e2e_count=e2e_count,
            total_count=total
        )
    
    def _classify_test_file(self, test_file: Path) -> TestType:
        """Classify test file based on path and markers."""
        path_str = str(test_file)
        
        # Check directory structure first
        if "/unit/" in path_str or "/tests/unit/" in path_str:
            return TestType.UNIT
        elif "/integration/" in path_str or "/tests/integration/" in path_str:
            return TestType.INTEGRATION
        elif "/e2e/" in path_str or "/tests/e2e/" in path_str:
            return TestType.E2E
        elif "/performance/" in path_str:
            return TestType.PERFORMANCE
        elif "/security/" in path_str:
            return TestType.SECURITY
        
        # Analyze file content for markers
        try:
            content = test_file.read_text()
            
            # Count different marker types
            unit_markers = len(re.findall(r'@pytest\.mark\.unit', content))
            integration_markers = len(re.findall(r'@pytest\.mark\.integration', content))
            e2e_markers = len(re.findall(r'@pytest\.mark\.e2e', content))
            
            # Classify based on predominant marker type
            if unit_markers > integration_markers and unit_markers > e2e_markers:
                return TestType.UNIT
            elif integration_markers > e2e_markers:
                return TestType.INTEGRATION
            elif e2e_markers > 0:
                return TestType.E2E
            
            # Default classification based on complexity indicators
            if self._has_unit_indicators(content):
                return TestType.UNIT
            elif self._has_integration_indicators(content):
                return TestType.INTEGRATION
            else:
                return TestType.E2E
                
        except Exception:
            # Default to unit test if unable to classify
            return TestType.UNIT
    
    def _has_unit_indicators(self, content: str) -> bool:
        """Check for unit test indicators."""
        unit_indicators = [
            "Mock(",
            "MagicMock(",
            "@patch",
            "monkeypatch",
            "assert_called_once",
            "return_value =",
            # Fast, isolated test patterns
            "def test_.*_unit",
            "TestCase",
        ]
        
        return any(indicator in content for indicator in unit_indicators)
    
    def _has_integration_indicators(self, content: str) -> bool:
        """Check for integration test indicators."""
        integration_indicators = [
            "create_engine",
            "sessionmaker", 
            "TestClient",
            "requests.",
            "subprocess.run",
            "docker",
            "database",
            "api_client",
            # Component interaction patterns
            "def test_.*_integration",
            "test_.*_workflow",
        ]
        
        return any(indicator in content for indicator in integration_indicators)
    
    def _count_test_functions(self, test_file: Path) -> int:
        """Count test functions in a file."""
        try:
            content = test_file.read_text()
            # Count functions starting with 'test_' or decorated with @pytest markers
            test_functions = len(re.findall(r'def test_\w+\s*\(', content))
            return test_functions
        except Exception:
            return 0
    
    def validate_pyramid_compliance(self) -> Dict:
        """Validate current distribution against pyramid targets."""
        distribution = self.analyze_test_distribution()
        
        compliance = {
            "compliant": True,
            "issues": [],
            "recommendations": [],
            "metrics": {
                "unit_percentage": distribution.unit_percentage,
                "integration_percentage": distribution.integration_percentage,
                "e2e_percentage": distribution.e2e_percentage,
                "total_tests": distribution.total_count
            }
        }
        
        # Check unit test compliance
        unit_deviation = abs(distribution.unit_percentage - self.targets.unit_target)
        if unit_deviation > self.targets.tolerance:
            compliance["compliant"] = False
            if distribution.unit_percentage < self.targets.unit_target:
                shortage = int((self.targets.unit_target - distribution.unit_percentage) / 100 * distribution.total_count)
                compliance["issues"].append(f"Unit test shortage: {unit_deviation:.1f}% below target")
                compliance["recommendations"].append(f"Add approximately {shortage} unit tests")
            else:
                compliance["issues"].append(f"Unit test excess: {unit_deviation:.1f}% above target")
        
        # Check integration test compliance
        integration_deviation = abs(distribution.integration_percentage - self.targets.integration_target)
        if integration_deviation > self.targets.tolerance:
            compliance["compliant"] = False
            compliance["issues"].append(f"Integration test imbalance: {integration_deviation:.1f}% deviation")
        
        # Check e2e test compliance
        e2e_deviation = abs(distribution.e2e_percentage - self.targets.e2e_target)
        if e2e_deviation > self.targets.tolerance:
            compliance["compliant"] = False
            if distribution.e2e_percentage > self.targets.e2e_target:
                excess = int((distribution.e2e_percentage - self.targets.e2e_target) / 100 * distribution.total_count)
                compliance["issues"].append(f"E2E test excess: {e2e_deviation:.1f}% above target")
                compliance["recommendations"].append(f"Consider converting {excess} E2E tests to integration tests")
        
        return compliance
    
    def generate_improvement_plan(self) -> Dict:
        """Generate specific improvement plan for pyramid compliance."""
        distribution = self.analyze_test_distribution()
        compliance = self.validate_pyramid_compliance()
        
        plan = {
            "current_state": {
                "unit": distribution.unit_count,
                "integration": distribution.integration_count,
                "e2e": distribution.e2e_count,
                "total": distribution.total_count
            },
            "target_state": {},
            "actions": [],
            "timeline": "2-6 weeks"
        }
        
        # Calculate target numbers
        target_total = distribution.total_count
        plan["target_state"] = {
            "unit": int(target_total * (self.targets.unit_target / 100)),
            "integration": int(target_total * (self.targets.integration_target / 100)),
            "e2e": int(target_total * (self.targets.e2e_target / 100))
        }
        
        # Generate specific actions
        unit_gap = plan["target_state"]["unit"] - distribution.unit_count
        if unit_gap > 0:
            plan["actions"].append({
                "action": "Add unit tests",
                "count": unit_gap,
                "priority": "HIGH",
                "areas": [
                    "Core components (config, project_manager, task_engine)",
                    "AI services (agent_coordinator, neural_trainer)",
                    "Security layer (input_validator, sandbox)",
                    "Database repositories"
                ]
            })
        
        e2e_excess = distribution.e2e_count - plan["target_state"]["e2e"]
        if e2e_excess > 0:
            plan["actions"].append({
                "action": "Optimize E2E tests",
                "count": e2e_excess,
                "priority": "MEDIUM", 
                "strategy": "Convert to faster integration tests where possible"
            })
        
        return plan
    
    def export_metrics(self, output_file: Path = None) -> None:
        """Export test metrics to JSON file."""
        distribution = self.analyze_test_distribution()
        compliance = self.validate_pyramid_compliance()
        improvement_plan = self.generate_improvement_plan()
        
        metrics = {
            "timestamp": "2025-08-25T11:30:00Z",
            "distribution": {
                "unit_count": distribution.unit_count,
                "integration_count": distribution.integration_count,
                "e2e_count": distribution.e2e_count,
                "total_count": distribution.total_count,
                "unit_percentage": distribution.unit_percentage,
                "integration_percentage": distribution.integration_percentage,
                "e2e_percentage": distribution.e2e_percentage
            },
            "compliance": compliance,
            "improvement_plan": improvement_plan
        }
        
        if output_file is None:
            output_file = self.test_root / "pyramid_metrics.json"
        
        output_file.write_text(json.dumps(metrics, indent=2))
        print(f"Test pyramid metrics exported to: {output_file}")


def main():
    """Main execution for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate test pyramid compliance")
    parser.add_argument("--test-root", type=Path, help="Root directory of tests")
    parser.add_argument("--export", type=Path, help="Export metrics to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = TestPyramidValidator(test_root=args.test_root)
    
    print("🔍 Analyzing test distribution...")
    distribution = validator.analyze_test_distribution()
    
    print(f"\n📊 Current Test Distribution:")
    print(f"   Unit Tests:        {distribution.unit_count:4d} ({distribution.unit_percentage:5.1f}%)")
    print(f"   Integration Tests: {distribution.integration_count:4d} ({distribution.integration_percentage:5.1f}%)")
    print(f"   E2E Tests:         {distribution.e2e_count:4d} ({distribution.e2e_percentage:5.1f}%)")
    print(f"   Total Tests:       {distribution.total_count:4d}")
    
    compliance = validator.validate_pyramid_compliance()
    
    if compliance["compliant"]:
        print("\n✅ Test pyramid is compliant!")
    else:
        print("\n⚠️  Test pyramid compliance issues:")
        for issue in compliance["issues"]:
            print(f"   • {issue}")
        
        if compliance["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in compliance["recommendations"]:
                print(f"   • {rec}")
    
    if args.verbose:
        plan = validator.generate_improvement_plan()
        print(f"\n📋 Improvement Plan:")
        for action in plan["actions"]:
            print(f"   {action['priority']}: {action['action']} ({action['count']} tests)")
    
    if args.export:
        validator.export_metrics(args.export)


if __name__ == "__main__":
    main()