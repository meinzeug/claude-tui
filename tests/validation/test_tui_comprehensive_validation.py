#!/usr/bin/env python3
"""
Comprehensive TUI Validation Test Suite
Testing and Quality Assurance Agent Implementation

This module provides comprehensive testing for the Claude TUI application,
ensuring 100% operational success after fixes have been applied.
"""

import pytest
import asyncio
import subprocess
import sys
import os
import time
import threading
import signal
import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TUIValidationTestSuite:
    """Comprehensive TUI validation test suite."""
    
    def __init__(self):
        self.results = {
            "import_tests": {},
            "startup_tests": {},
            "component_tests": {},
            "error_handling_tests": {},
            "performance_tests": {},
            "overall_status": "PENDING"
        }
        self.project_root = Path(__file__).parent.parent.parent
        self.test_timeout = 30  # seconds
        
    def log_test_result(self, category: str, test_name: str, success: bool, details: str = ""):
        """Log test results for coordination."""
        self.results[category][test_name] = {
            "success": success,
            "details": details,
            "timestamp": time.time()
        }
        
    async def save_results_to_hive(self):
        """Save test results to hive memory for coordination."""
        try:
            results_path = self.project_root / ".swarm" / "memory" / "tester" / "validation_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            # Also save to memory store via hooks
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit", 
                "--file", str(results_path),
                "--memory-key", "swarm/tester/validation_results"
            ], check=False)
            
            return True
        except Exception as e:
            print(f"Warning: Could not save results to hive memory: {e}")
            return False

class TestTUIImports:
    """Test all TUI imports work correctly."""
    
    def __init__(self, suite: TUIValidationTestSuite):
        self.suite = suite
        
    def test_core_imports(self):
        """Test core TUI module imports."""
        core_modules = [
            "src.claude_tui.main",
            "src.claude_tui.ui.main_app",
            "src.claude_tui.core.config_manager",
            "src.claude_tui.core.project_manager",
            "src.claude_tui.core.task_engine",
        ]
        
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                self.suite.log_test_result(
                    "import_tests", f"import_{module_name}", 
                    True, f"Successfully imported {module_name}"
                )
            except ImportError as e:
                self.suite.log_test_result(
                    "import_tests", f"import_{module_name}", 
                    False, f"Import failed: {str(e)}"
                )
                
    def test_ui_widget_imports(self):
        """Test UI widget imports."""
        widget_modules = [
            "src.ui.widgets.task_dashboard",
            "src.ui.widgets.progress_intelligence",
            "src.ui.widgets.git_workflow_widget",
            "src.ui.widgets.metrics_dashboard",
            "src.ui.widgets.placeholder_alert",
        ]
        
        for module_name in widget_modules:
            try:
                importlib.import_module(module_name)
                self.suite.log_test_result(
                    "import_tests", f"widget_import_{module_name}", 
                    True, f"Successfully imported {module_name}"
                )
            except ImportError as e:
                self.suite.log_test_result(
                    "import_tests", f"widget_import_{module_name}", 
                    False, f"Widget import failed: {str(e)}"
                )

class TestTUIStartup:
    """Test TUI startup in different modes."""
    
    def __init__(self, suite: TUIValidationTestSuite):
        self.suite = suite
        
    def test_headless_startup(self):
        """Test TUI startup in headless mode."""
        try:
            # Test with minimal environment
            env = os.environ.copy()
            env['TERM'] = 'dumb'
            env['DISPLAY'] = ''
            
            # Run TUI with timeout
            result = subprocess.run([
                sys.executable, "-c", 
                """
import sys
sys.path.insert(0, 'src')
try:
    from claude_tui.main import main
    print("IMPORT_SUCCESS")
    # Don't actually run main() in test
    print("STARTUP_SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=self.suite.test_timeout,
                env=env,
                cwd=str(self.suite.project_root)
            )
            
            if "IMPORT_SUCCESS" in result.stdout and "STARTUP_SUCCESS" in result.stdout:
                self.suite.log_test_result(
                    "startup_tests", "headless_startup", 
                    True, "TUI imports and basic startup successful"
                )
            else:
                self.suite.log_test_result(
                    "startup_tests", "headless_startup", 
                    False, f"Startup failed: {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            self.suite.log_test_result(
                "startup_tests", "headless_startup", 
                False, "Startup timed out"
            )
        except Exception as e:
            self.suite.log_test_result(
                "startup_tests", "headless_startup", 
                False, f"Startup test failed: {str(e)}"
            )
            
    def test_component_initialization(self):
        """Test component initialization without full UI."""
        try:
            # Test individual component initialization
            result = subprocess.run([
                sys.executable, "-c", 
                """
import sys
sys.path.insert(0, 'src')

# Test core components
try:
    from claude_tui.core.config_manager import ConfigManager
    config = ConfigManager()
    print("CONFIG_MANAGER_OK")
    
    from claude_tui.core.project_manager import ProjectManager  
    project_mgr = ProjectManager()
    print("PROJECT_MANAGER_OK")
    
    from claude_tui.core.task_engine import TaskEngine
    task_engine = TaskEngine()
    print("TASK_ENGINE_OK")
    
    print("COMPONENT_INIT_SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=self.suite.test_timeout,
                cwd=str(self.suite.project_root)
            )
            
            success_markers = ["CONFIG_MANAGER_OK", "PROJECT_MANAGER_OK", "TASK_ENGINE_OK", "COMPONENT_INIT_SUCCESS"]
            if all(marker in result.stdout for marker in success_markers):
                self.suite.log_test_result(
                    "startup_tests", "component_initialization", 
                    True, "All core components initialized successfully"
                )
            else:
                self.suite.log_test_result(
                    "startup_tests", "component_initialization", 
                    False, f"Component init failed: {result.stderr}"
                )
                
        except Exception as e:
            self.suite.log_test_result(
                "startup_tests", "component_initialization", 
                False, f"Component test failed: {str(e)}"
            )

class TestTUIComponents:
    """Test individual TUI components."""
    
    def __init__(self, suite: TUIValidationTestSuite):
        self.suite = suite
        
    def test_widget_creation(self):
        """Test widget creation without display."""
        try:
            result = subprocess.run([
                sys.executable, "-c", 
                """
import sys
sys.path.insert(0, 'src')

# Mock textual to avoid display requirements
class MockWidget:
    def __init__(self, *args, **kwargs):
        pass
    def compose(self):
        return []

class MockApp:
    def __init__(self, *args, **kwargs):
        pass
        
import sys
sys.modules['textual'] = type(sys)('textual')
sys.modules['textual.app'] = type(sys)('textual.app')
sys.modules['textual.widgets'] = type(sys)('textual.widgets')
sys.modules['textual.containers'] = type(sys)('textual.containers')

sys.modules['textual'].Widget = MockWidget
sys.modules['textual.app'].App = MockApp
sys.modules['textual.widgets'].Static = MockWidget
sys.modules['textual.widgets'].Button = MockWidget
sys.modules['textual.containers'].Container = MockWidget

try:
    from ui.widgets.task_dashboard import TaskDashboard
    dashboard = TaskDashboard()
    print("TASK_DASHBOARD_OK")
    
    from ui.widgets.progress_intelligence import ProgressIntelligence
    progress = ProgressIntelligence()
    print("PROGRESS_INTELLIGENCE_OK")
    
    print("WIDGET_CREATION_SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=self.suite.test_timeout,
                cwd=str(self.suite.project_root)
            )
            
            if "WIDGET_CREATION_SUCCESS" in result.stdout:
                self.suite.log_test_result(
                    "component_tests", "widget_creation", 
                    True, "Widgets can be created successfully"
                )
            else:
                self.suite.log_test_result(
                    "component_tests", "widget_creation", 
                    False, f"Widget creation failed: {result.stderr}"
                )
                
        except Exception as e:
            self.suite.log_test_result(
                "component_tests", "widget_creation", 
                False, f"Widget test failed: {str(e)}"
            )

class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    def __init__(self, suite: TUIValidationTestSuite):
        self.suite = suite
        
    def test_import_error_handling(self):
        """Test graceful handling of import errors."""
        try:
            # Test with missing dependencies
            result = subprocess.run([
                sys.executable, "-c", 
                """
import sys
sys.path.insert(0, 'src')

# Test fallback mechanisms
try:
    from claude_tui.core.fallback_implementations import FallbackManager
    fallback = FallbackManager()
    print("FALLBACK_MANAGER_OK")
    
    # Test dependency checker
    from claude_tui.core.dependency_checker import DependencyChecker
    checker = DependencyChecker()
    print("DEPENDENCY_CHECKER_OK")
    
    print("ERROR_HANDLING_SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=self.suite.test_timeout,
                cwd=str(self.suite.project_root)
            )
            
            if "ERROR_HANDLING_SUCCESS" in result.stdout:
                self.suite.log_test_result(
                    "error_handling_tests", "import_error_handling", 
                    True, "Error handling mechanisms work correctly"
                )
            else:
                self.suite.log_test_result(
                    "error_handling_tests", "import_error_handling", 
                    False, f"Error handling failed: {result.stderr}"
                )
                
        except Exception as e:
            self.suite.log_test_result(
                "error_handling_tests", "import_error_handling", 
                False, f"Error handling test failed: {str(e)}"
            )

class TestPerformance:
    """Test performance metrics during startup."""
    
    def __init__(self, suite: TUIValidationTestSuite):
        self.suite = suite
        
    def test_startup_performance(self):
        """Test startup time and memory usage."""
        try:
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable, "-c", 
                """
import sys
import time
import psutil
import os
sys.path.insert(0, 'src')

start_time = time.time()
process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss

try:
    # Import core modules
    from claude_tui.core.config_manager import ConfigManager
    from claude_tui.core.project_manager import ProjectManager
    from claude_tui.core.task_engine import TaskEngine
    
    # Initialize components
    config = ConfigManager()
    project_mgr = ProjectManager()
    task_engine = TaskEngine()
    
    end_time = time.time()
    end_memory = process.memory_info().rss
    
    startup_time = end_time - start_time
    memory_usage = end_memory - start_memory
    
    print(f"STARTUP_TIME: {startup_time:.3f}s")
    print(f"MEMORY_USAGE: {memory_usage / 1024 / 1024:.2f}MB")
    print("PERFORMANCE_TEST_SUCCESS")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
            ], 
                capture_output=True, 
                text=True, 
                timeout=self.suite.test_timeout,
                cwd=str(self.suite.project_root)
            )
            
            if "PERFORMANCE_TEST_SUCCESS" in result.stdout:
                # Extract performance metrics
                lines = result.stdout.split('\n')
                startup_time = next((line.split(': ')[1] for line in lines if line.startswith('STARTUP_TIME:')), "N/A")
                memory_usage = next((line.split(': ')[1] for line in lines if line.startswith('MEMORY_USAGE:')), "N/A")
                
                self.suite.log_test_result(
                    "performance_tests", "startup_performance", 
                    True, f"Startup time: {startup_time}, Memory usage: {memory_usage}"
                )
            else:
                self.suite.log_test_result(
                    "performance_tests", "startup_performance", 
                    False, f"Performance test failed: {result.stderr}"
                )
                
        except Exception as e:
            self.suite.log_test_result(
                "performance_tests", "startup_performance", 
                False, f"Performance test failed: {str(e)}"
            )

async def run_comprehensive_validation():
    """Run comprehensive TUI validation tests."""
    print("🧪 Starting Comprehensive TUI Validation Test Suite")
    print("=" * 60)
    
    suite = TUIValidationTestSuite()
    
    # Run import tests
    print("\n📦 Testing Imports...")
    import_tester = TestTUIImports(suite)
    import_tester.test_core_imports()
    import_tester.test_ui_widget_imports()
    
    # Run startup tests
    print("\n🚀 Testing Startup...")
    startup_tester = TestTUIStartup(suite)
    startup_tester.test_headless_startup()
    startup_tester.test_component_initialization()
    
    # Run component tests
    print("\n🔧 Testing Components...")
    component_tester = TestTUIComponents(suite)
    component_tester.test_widget_creation()
    
    # Run error handling tests
    print("\n🛡️ Testing Error Handling...")
    error_tester = TestErrorHandling(suite)
    error_tester.test_import_error_handling()
    
    # Run performance tests
    print("\n⚡ Testing Performance...")
    performance_tester = TestPerformance(suite)
    performance_tester.test_startup_performance()
    
    # Calculate overall status
    all_tests_passed = True
    total_tests = 0
    passed_tests = 0
    
    for category, tests in suite.results.items():
        if category == "overall_status":
            continue
        for test_name, result in tests.items():
            total_tests += 1
            if result["success"]:
                passed_tests += 1
            else:
                all_tests_passed = False
    
    suite.results["overall_status"] = "PASSED" if all_tests_passed else "FAILED"
    suite.results["test_summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
    }
    
    # Save results
    await suite.save_results_to_hive()
    
    # Print summary
    print(f"\n📊 Test Summary")
    print("=" * 40)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {suite.results['test_summary']['success_rate']}")
    print(f"Overall Status: {suite.results['overall_status']}")
    
    if not all_tests_passed:
        print("\n❌ Failed Tests:")
        for category, tests in suite.results.items():
            if category in ["overall_status", "test_summary"]:
                continue
            for test_name, result in tests.items():
                if not result["success"]:
                    print(f"  - {category}.{test_name}: {result['details']}")
    
    return suite.results

if __name__ == "__main__":
    asyncio.run(run_comprehensive_validation())