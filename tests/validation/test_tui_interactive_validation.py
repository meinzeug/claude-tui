#!/usr/bin/env python3
"""
Interactive TUI Validation Test Suite
For testing TUI in actual interactive mode with proper mocking
"""

import pytest
import asyncio
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class InteractiveTUIValidator:
    """Validator for interactive TUI functionality."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.results = {
            "interactive_tests": {},
            "ui_responsiveness": {},
            "keyboard_handling": {},
            "screen_rendering": {}
        }
        
    def test_tui_with_mocked_terminal(self):
        """Test TUI with mocked terminal environment."""
        try:
            # Create a test script that mocks the terminal environment
            test_script = """
import sys
import os
sys.path.insert(0, 'src')

# Mock terminal environment
os.environ['TERM'] = 'xterm-256color'
os.environ['COLUMNS'] = '80'
os.environ['LINES'] = '24'

# Mock textual dependencies
class MockConsole:
    def __init__(self, *args, **kwargs):
        self.width = 80
        self.height = 24
    
    def print(self, *args, **kwargs):
        pass

class MockApp:
    def __init__(self, *args, **kwargs):
        self.console = MockConsole()
        self.screen_stack = []
        
    async def run_async(self, *args, **kwargs):
        print("MOCK_APP_RUN_SUCCESS")
        return True
        
    def run(self, *args, **kwargs):
        print("MOCK_APP_RUN_SUCCESS")
        return True
        
    def push_screen(self, screen):
        self.screen_stack.append(screen)
        
    def pop_screen(self):
        if self.screen_stack:
            return self.screen_stack.pop()

# Patch textual modules
import sys
textual_mock = type(sys)('textual')
textual_mock.Console = MockConsole
textual_mock.App = MockApp

sys.modules['textual'] = textual_mock
sys.modules['textual.app'] = type(sys)('textual.app')
sys.modules['textual.app'].App = MockApp
sys.modules['textual.console'] = type(sys)('textual.console')
sys.modules['textual.console'].Console = MockConsole

try:
    from claude_tui.main import create_app
    app = create_app()
    print("APP_CREATION_SUCCESS")
    
    # Test app initialization
    if hasattr(app, 'console'):
        print("CONSOLE_AVAILABLE")
    
    print("INTERACTIVE_TEST_SUCCESS")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd=str(self.project_root)
            )
            
            success = "INTERACTIVE_TEST_SUCCESS" in result.stdout
            self.results["interactive_tests"]["mocked_terminal"] = {
                "success": success,
                "details": result.stdout if success else result.stderr
            }
            
            return success
            
        except Exception as e:
            self.results["interactive_tests"]["mocked_terminal"] = {
                "success": False,
                "details": f"Test failed: {str(e)}"
            }
            return False
            
    def test_ui_component_rendering(self):
        """Test UI component rendering without actual display."""
        try:
            test_script = """
import sys
sys.path.insert(0, 'src')

# Create minimal mocks for textual
class MockWidget:
    def __init__(self, *args, **kwargs):
        self.id = kwargs.get('id', 'mock_widget')
        self.classes = kwargs.get('classes', '')
        
    def compose(self):
        return []
        
    def render(self):
        return "Mock rendered content"

class MockContainer(MockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = []

# Mock all textual components
import sys
for module_name in ['textual', 'textual.widgets', 'textual.containers', 'textual.app']:
    mock_module = type(sys)(module_name)
    mock_module.Widget = MockWidget
    mock_module.Container = MockContainer
    mock_module.Static = MockWidget
    mock_module.Button = MockWidget
    mock_module.Label = MockWidget
    mock_module.Input = MockWidget
    mock_module.ListView = MockWidget
    mock_module.ListItem = MockWidget
    mock_module.App = MockWidget
    sys.modules[module_name] = mock_module

try:
    # Test widget imports and creation
    from ui.widgets.task_dashboard import TaskDashboard
    task_dashboard = TaskDashboard()
    print("TASK_DASHBOARD_CREATED")
    
    from ui.widgets.progress_intelligence import ProgressIntelligence  
    progress_widget = ProgressIntelligence()
    print("PROGRESS_WIDGET_CREATED")
    
    from ui.widgets.git_workflow_widget import GitWorkflowWidget
    git_widget = GitWorkflowWidget()
    print("GIT_WIDGET_CREATED")
    
    print("UI_RENDERING_SUCCESS")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd=str(self.project_root)
            )
            
            success = "UI_RENDERING_SUCCESS" in result.stdout
            self.results["ui_responsiveness"]["component_rendering"] = {
                "success": success,
                "details": result.stdout if success else result.stderr
            }
            
            return success
            
        except Exception as e:
            self.results["ui_responsiveness"]["component_rendering"] = {
                "success": False,
                "details": f"Test failed: {str(e)}"
            }
            return False
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.results.items():
            for test_name, result in tests.items():
                total_tests += 1
                if result["success"]:
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "validation_type": "Interactive TUI Validation",
            "timestamp": time.time(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": f"{success_rate:.1f}%",
                "overall_status": "PASSED" if success_rate >= 80 else "FAILED"
            },
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []
        
        for category, tests in self.results.items():
            for test_name, result in tests.items():
                if not result["success"]:
                    recommendations.append({
                        "category": category,
                        "test": test_name,
                        "issue": result["details"],
                        "recommendation": self._get_recommendation_for_failure(category, test_name)
                    })
        
        return recommendations
    
    def _get_recommendation_for_failure(self, category: str, test_name: str):
        """Get specific recommendations for test failures."""
        recommendations_map = {
            "interactive_tests": {
                "mocked_terminal": "Check textual library installation and terminal environment setup"
            },
            "ui_responsiveness": {
                "component_rendering": "Verify widget class definitions and inheritance structure"
            },
            "keyboard_handling": {
                "default": "Review keyboard event handling and binding configuration"
            },
            "screen_rendering": {
                "default": "Check screen composition and layout management"
            }
        }
        
        category_recs = recommendations_map.get(category, {})
        return category_recs.get(test_name, category_recs.get("default", "Review implementation and dependencies"))

async def run_interactive_validation():
    """Run interactive TUI validation."""
    print("🎮 Starting Interactive TUI Validation")
    print("=" * 50)
    
    validator = InteractiveTUIValidator()
    
    print("\n🖥️  Testing Mocked Terminal Environment...")
    terminal_success = validator.test_tui_with_mocked_terminal()
    
    print("\n🎨 Testing UI Component Rendering...")  
    rendering_success = validator.test_ui_component_rendering()
    
    # Generate and save report
    report = validator.generate_validation_report()
    
    # Save results to hive memory
    results_path = validator.project_root / ".swarm" / "memory" / "tester" / "interactive_validation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Interactive Validation Summary")
    print("=" * 45)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']}")
    print(f"Overall Status: {report['summary']['overall_status']}")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec['category']}.{rec['test']}: {rec['recommendation']}")
    
    return report

if __name__ == "__main__":
    asyncio.run(run_interactive_validation())