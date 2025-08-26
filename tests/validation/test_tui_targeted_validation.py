#!/usr/bin/env python3
"""
Targeted TUI Validation Test Suite
Testing with correct import paths and comprehensive validation
"""

import pytest
import asyncio
import subprocess
import sys
import os
import time
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class TargetedTUIValidator:
    """Targeted validator for TUI functionality with correct paths."""
    
    def __init__(self):
        self.project_root = project_root
        self.results = {
            "core_imports": {},
            "ui_imports": {},
            "startup_validation": {},
            "component_tests": {},
            "integration_tests": {}
        }
        self.test_timeout = 30
        
    def validate_core_imports(self):
        """Validate core module imports with correct paths."""
        print("🔍 Validating Core Imports...")
        
        core_modules = [
            ("core.config_manager", "ConfigManager"),
            ("core.project_manager", "ProjectManager"), 
            ("core.task_engine", "TaskEngine"),
            ("core.logger", "Logger"),
            ("claude_tui.main", "main"),
            ("claude_tui.core.config_manager", "ConfigManager"),
        ]
        
        for module_path, class_name in core_modules:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    self.results["core_imports"][module_path] = {
                        "success": True,
                        "details": f"Successfully imported {class_name} from {module_path}"
                    }
                    print(f"  ✅ {module_path}.{class_name}")
                else:
                    self.results["core_imports"][module_path] = {
                        "success": False,
                        "details": f"Module {module_path} imported but {class_name} not found"
                    }
                    print(f"  ⚠️  {module_path}.{class_name} - class not found")
                    
            except ImportError as e:
                self.results["core_imports"][module_path] = {
                    "success": False,
                    "details": f"Import failed: {str(e)}"
                }
                print(f"  ❌ {module_path} - {str(e)}")
                
    def validate_ui_imports(self):
        """Validate UI module imports."""
        print("🎨 Validating UI Imports...")
        
        ui_modules = [
            ("ui.main_app", "MainApp"),
            ("ui.widgets.task_dashboard", "TaskDashboard"),
            ("ui.widgets.progress_intelligence", "ProgressIntelligence"),
            ("ui.widgets.git_workflow_widget", "GitWorkflowWidget"),
            ("claude_tui.ui.main_app", "MainApp"),
        ]
        
        for module_path, class_name in ui_modules:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    self.results["ui_imports"][module_path] = {
                        "success": True,
                        "details": f"Successfully imported {class_name} from {module_path}"
                    }
                    print(f"  ✅ {module_path}.{class_name}")
                else:
                    self.results["ui_imports"][module_path] = {
                        "success": False,
                        "details": f"Module {module_path} imported but {class_name} not found"
                    }
                    print(f"  ⚠️  {module_path}.{class_name} - class not found")
                    
            except ImportError as e:
                self.results["ui_imports"][module_path] = {
                    "success": False,
                    "details": f"Import failed: {str(e)}"
                }
                print(f"  ❌ {module_path} - {str(e)}")
                
    def test_main_entry_points(self):
        """Test main entry points work correctly."""
        print("🚀 Testing Main Entry Points...")
        
        # Test main TUI entry point
        try:
            result = subprocess.run([
                sys.executable, "-c", 
                """
import sys
sys.path.insert(0, 'src')

try:
    # Test run_tui.py
    import run_tui
    print("RUN_TUI_IMPORTED")
    
    # Test main entry
    from claude_tui.main import main
    print("MAIN_IMPORTED")
    
    # Test TUI creation without running
    if hasattr(run_tui, 'create_tui_app'):
        print("CREATE_TUI_APP_AVAILABLE")
    
    print("ENTRY_POINT_SUCCESS")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=self.test_timeout,
                cwd=str(self.project_root)
            )
            
            success = "ENTRY_POINT_SUCCESS" in result.stdout
            self.results["startup_validation"]["main_entry_points"] = {
                "success": success,
                "details": result.stdout if success else result.stderr
            }
            
            if success:
                print("  ✅ Main entry points working")
            else:
                print(f"  ❌ Entry points failed: {result.stderr}")
                
        except Exception as e:
            self.results["startup_validation"]["main_entry_points"] = {
                "success": False,
                "details": f"Test failed: {str(e)}"
            }
            print(f"  ❌ Entry point test failed: {str(e)}")
            
    def test_component_initialization(self):
        """Test component initialization without UI."""
        print("🔧 Testing Component Initialization...")
        
        try:
            result = subprocess.run([
                sys.executable, "-c", 
                """
import sys
sys.path.insert(0, 'src')

# Test component initialization
try:
    from core.config_manager import ConfigManager
    config = ConfigManager()
    print("CONFIG_MANAGER_INIT")
    
    from core.project_manager import ProjectManager
    project_mgr = ProjectManager()
    print("PROJECT_MANAGER_INIT")
    
    # Test fallback systems
    try:
        from claude_tui.core.fallback_implementations import FallbackManager
        fallback = FallbackManager()
        print("FALLBACK_MANAGER_INIT")
    except ImportError:
        print("FALLBACK_MANAGER_NOT_AVAILABLE")
    
    print("COMPONENT_INIT_SUCCESS")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=self.test_timeout,
                cwd=str(self.project_root)
            )
            
            success = "COMPONENT_INIT_SUCCESS" in result.stdout
            self.results["component_tests"]["initialization"] = {
                "success": success,
                "details": result.stdout if success else result.stderr
            }
            
            if success:
                print("  ✅ Component initialization working")
            else:
                print(f"  ❌ Component initialization failed: {result.stderr}")
                
        except Exception as e:
            self.results["component_tests"]["initialization"] = {
                "success": False,
                "details": f"Test failed: {str(e)}"
            }
            print(f"  ❌ Component test failed: {str(e)}")
            
    def test_tui_mock_startup(self):
        """Test TUI startup with complete mocking."""
        print("🎮 Testing TUI Mock Startup...")
        
        try:
            result = subprocess.run([
                sys.executable, "-c", 
                """
import sys
sys.path.insert(0, 'src')

# Complete textual mocking
class MockWidget:
    def __init__(self, *args, **kwargs):
        self.id = kwargs.get('id', 'mock')
        
    def compose(self):
        return []
        
    def render(self):
        return "Mock content"

class MockApp:
    def __init__(self, *args, **kwargs):
        self.title = kwargs.get('title', 'Mock App')
        
    def run(self):
        print("MOCK_APP_RUN")
        
    async def run_async(self):
        print("MOCK_APP_RUN_ASYNC")

# Mock all textual components
textual_mock = type(sys)('textual')
textual_mock.Widget = MockWidget
textual_mock.App = MockApp

sys.modules['textual'] = textual_mock
sys.modules['textual.app'] = type(sys)('textual.app')
sys.modules['textual.widgets'] = type(sys)('textual.widgets')
sys.modules['textual.containers'] = type(sys)('textual.containers')

for module in ['textual.app', 'textual.widgets', 'textual.containers']:
    mock_module = sys.modules[module]
    mock_module.App = MockApp
    mock_module.Widget = MockWidget
    mock_module.Static = MockWidget
    mock_module.Button = MockWidget
    mock_module.Container = MockWidget
    mock_module.ListView = MockWidget

try:
    # Now test TUI creation
    from ui.main_app import MainApp
    app = MainApp()
    print("MAIN_APP_CREATED")
    
    print("TUI_MOCK_SUCCESS")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=self.test_timeout,
                cwd=str(self.project_root)
            )
            
            success = "TUI_MOCK_SUCCESS" in result.stdout
            self.results["integration_tests"]["tui_mock_startup"] = {
                "success": success,
                "details": result.stdout if success else result.stderr
            }
            
            if success:
                print("  ✅ TUI mock startup working")
            else:
                print(f"  ❌ TUI mock startup failed: {result.stderr}")
                
        except Exception as e:
            self.results["integration_tests"]["tui_mock_startup"] = {
                "success": False,
                "details": f"Test failed: {str(e)}"
            }
            print(f"  ❌ TUI mock test failed: {str(e)}")
            
    def test_dependency_resolution(self):
        """Test dependency resolution and fallback systems."""
        print("📦 Testing Dependency Resolution...")
        
        try:
            result = subprocess.run([
                sys.executable, "-c", 
                """
import sys
sys.path.insert(0, 'src')

try:
    # Test dependency checker
    from claude_tui.core.dependency_checker import DependencyChecker
    checker = DependencyChecker()
    print("DEPENDENCY_CHECKER_OK")
    
    # Test if textual is available
    try:
        import textual
        print("TEXTUAL_AVAILABLE")
    except ImportError:
        print("TEXTUAL_NOT_AVAILABLE")
    
    print("DEPENDENCY_TEST_SUCCESS")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=self.test_timeout,
                cwd=str(self.project_root)
            )
            
            success = "DEPENDENCY_TEST_SUCCESS" in result.stdout
            self.results["integration_tests"]["dependency_resolution"] = {
                "success": success,
                "details": result.stdout if success else result.stderr
            }
            
            if success:
                print("  ✅ Dependency resolution working")
            else:
                print(f"  ❌ Dependency resolution failed: {result.stderr}")
                
        except Exception as e:
            self.results["integration_tests"]["dependency_resolution"] = {
                "success": False,
                "details": f"Test failed: {str(e)}"
            }
            print(f"  ❌ Dependency test failed: {str(e)}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive validation report."""
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.results.items():
            for test_name, result in tests.items():
                total_tests += 1
                if result["success"]:
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        overall_status = "PASSED" if success_rate >= 80 else "FAILED"
        
        report = {
            "validation_type": "Targeted TUI Validation",
            "timestamp": time.time(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": f"{success_rate:.1f}%",
                "overall_status": overall_status
            },
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations(),
            "operational_status": self._determine_operational_status()
        }
        
        return report
    
    def _generate_recommendations(self):
        """Generate specific recommendations based on failures."""
        recommendations = []
        
        # Check for specific failure patterns
        for category, tests in self.results.items():
            for test_name, result in tests.items():
                if not result["success"]:
                    if "ImportError" in result["details"] or "No module named" in result["details"]:
                        recommendations.append({
                            "issue": f"{category}.{test_name} - Import failure",
                            "recommendation": "Check PYTHONPATH and module structure. Ensure all __init__.py files exist.",
                            "priority": "HIGH"
                        })
                    elif "not found" in result["details"]:
                        recommendations.append({
                            "issue": f"{category}.{test_name} - Missing class/function",
                            "recommendation": "Verify class/function definitions and naming consistency.",
                            "priority": "MEDIUM"
                        })
                    else:
                        recommendations.append({
                            "issue": f"{category}.{test_name} - General failure",
                            "recommendation": "Review implementation and dependencies for this component.",
                            "priority": "MEDIUM"
                        })
        
        return recommendations
    
    def _determine_operational_status(self):
        """Determine if TUI is operationally ready."""
        critical_components = [
            ("core_imports", "core.config_manager"),
            ("startup_validation", "main_entry_points"),
            ("component_tests", "initialization")
        ]
        
        critical_failures = []
        for category, component in critical_components:
            if category in self.results and component in self.results[category]:
                if not self.results[category][component]["success"]:
                    critical_failures.append(f"{category}.{component}")
        
        if not critical_failures:
            return {
                "status": "OPERATIONAL",
                "message": "TUI is ready for use with all critical components working"
            }
        else:
            return {
                "status": "NOT_OPERATIONAL", 
                "message": f"Critical failures in: {', '.join(critical_failures)}",
                "critical_failures": critical_failures
            }
    
    async def save_results(self):
        """Save validation results to hive memory."""
        try:
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            # Save to hive memory directory
            results_dir = self.project_root / ".swarm" / "memory" / "tester"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = results_dir / "targeted_validation_results.json"
            with open(results_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Notify via hooks
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--file", str(results_path),
                "--memory-key", "swarm/tester/targeted_validation"
            ], check=False)
            
            return report
            
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
            return self.generate_comprehensive_report()

async def run_targeted_validation():
    """Run the targeted TUI validation suite."""
    print("🎯 Starting Targeted TUI Validation Suite")
    print("=" * 55)
    
    validator = TargetedTUIValidator()
    
    # Run all validation tests
    validator.validate_core_imports()
    validator.validate_ui_imports()
    validator.test_main_entry_points()
    validator.test_component_initialization()
    validator.test_tui_mock_startup()
    validator.test_dependency_resolution()
    
    # Generate and save report
    report = await validator.save_results()
    
    # Print summary
    print(f"\n📊 Targeted Validation Summary")
    print("=" * 45)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']}")
    print(f"Overall Status: {report['summary']['overall_status']}")
    
    # Print operational status
    op_status = report['operational_status']
    print(f"\n🚀 Operational Status: {op_status['status']}")
    print(f"   {op_status['message']}")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations ({len(report['recommendations'])}):")
        for i, rec in enumerate(report['recommendations'][:5], 1):  # Show top 5
            print(f"  {i}. [{rec['priority']}] {rec['recommendation']}")
        if len(report['recommendations']) > 5:
            print(f"     ... and {len(report['recommendations']) - 5} more")
    
    return report

if __name__ == "__main__":
    asyncio.run(run_targeted_validation())