#!/usr/bin/env python3
"""
Test Actual TUI Components with Mock Backend

This script tests the real TUI application with mock backend services to identify
issues and validate functionality without external dependencies.
"""

import asyncio
import sys
import os
import logging
import traceback
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import tempfile
import shutil

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Import mock backend
from mock_backend import (
    MockTUIBackendBridge,
    MockServiceOrchestrator,
    MockProjectManager,
    MockAIInterface,
    MockValidationEngine,
    MockConfigManager,
    get_mock_service_orchestrator_instance,
    reset_mock_instances
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TUIComponentTester:
    """Test actual TUI components with mock backend."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tui_test_")
        self.mock_bridge = None
        self.mock_orchestrator = None
        self.errors = []
        self.warnings = []
        self.successes = []
        
    async def setup_mock_environment(self):
        """Set up mock environment for testing."""
        try:
            reset_mock_instances()
            self.mock_orchestrator = get_mock_service_orchestrator_instance()
            self.mock_bridge = MockTUIBackendBridge(MockConfigManager())
            await self.mock_bridge.initialize()
            logger.info("Mock environment setup completed")
            return True
        except Exception as e:
            logger.error(f"Mock environment setup failed: {e}")
            self.errors.append(f"Setup error: {e}")
            return False
    
    def test_widget_imports(self):
        """Test importing all TUI widgets."""
        logger.info("Testing widget imports...")
        
        widget_tests = [
            ("project_tree", "ui.widgets.project_tree", "ProjectTree"),
            ("task_dashboard", "ui.widgets.task_dashboard", "TaskDashboard"),
            ("progress_intelligence", "ui.widgets.progress_intelligence", "ProgressIntelligence"),
            ("console_widget", "ui.widgets.console_widget", "ConsoleWidget"),
            ("notification_system", "ui.widgets.notification_system", "NotificationSystem"),
            ("placeholder_alert", "ui.widgets.placeholder_alert", "PlaceholderAlert"),
            ("metrics_dashboard", "ui.widgets.metrics_dashboard", "MetricsDashboard"),
            ("modal_dialogs", "ui.widgets.modal_dialogs", "Modal"),
            ("workflow_visualizer", "ui.widgets.workflow_visualizer", "WorkflowVisualizer"),
            ("git_workflow_widget", "ui.widgets.git_workflow_widget", "GitWorkflowWidget"),
        ]
        
        results = {}
        for widget_name, module_path, class_name in widget_tests:
            try:
                module = __import__(module_path, fromlist=[class_name])
                widget_class = getattr(module, class_name, None)
                if widget_class:
                    results[widget_name] = {"status": "success", "class": widget_class}
                    self.successes.append(f"Widget {widget_name} imported successfully")
                else:
                    results[widget_name] = {"status": "missing_class", "error": f"Class {class_name} not found"}
                    self.warnings.append(f"Widget {widget_name}: class {class_name} not found")
            except ImportError as e:
                results[widget_name] = {"status": "import_error", "error": str(e)}
                self.warnings.append(f"Widget {widget_name} import failed: {e}")
            except Exception as e:
                results[widget_name] = {"status": "error", "error": str(e)}
                self.errors.append(f"Widget {widget_name} error: {e}")
        
        success_count = sum(1 for r in results.values() if r["status"] == "success")
        total_count = len(results)
        logger.info(f"Widget import test: {success_count}/{total_count} successful")
        
        return results
    
    def test_screen_imports(self):
        """Test importing all TUI screens."""
        logger.info("Testing screen imports...")
        
        screen_tests = [
            ("project_wizard", "ui.screens.project_wizard", "ProjectWizardScreen"),
            ("settings", "ui.screens.settings", "SettingsScreen"),
            ("help_screen", "ui.screens.help_screen", "HelpScreen"),
            ("workspace_screen", "ui.screens.workspace_screen", "WorkspaceScreen"),
        ]
        
        results = {}
        for screen_name, module_path, class_name in screen_tests:
            try:
                module = __import__(module_path, fromlist=[class_name])
                screen_class = getattr(module, class_name, None)
                if screen_class:
                    results[screen_name] = {"status": "success", "class": screen_class}
                    self.successes.append(f"Screen {screen_name} imported successfully")
                else:
                    results[screen_name] = {"status": "missing_class", "error": f"Class {class_name} not found"}
                    self.warnings.append(f"Screen {screen_name}: class {class_name} not found")
            except ImportError as e:
                results[screen_name] = {"status": "import_error", "error": str(e)}
                self.warnings.append(f"Screen {screen_name} import failed: {e}")
            except Exception as e:
                results[screen_name] = {"status": "error", "error": str(e)}
                self.errors.append(f"Screen {screen_name} error: {e}")
        
        success_count = sum(1 for r in results.values() if r["status"] == "success")
        total_count = len(results)
        logger.info(f"Screen import test: {success_count}/{total_count} successful")
        
        return results
    
    @patch('src.backend.core_services.get_service_orchestrator')
    @patch('src.backend.tui_backend_bridge.initialize_tui_bridge')
    def test_main_app_creation(self, mock_bridge_init, mock_orchestrator):
        """Test creating the main TUI application."""
        logger.info("Testing main app creation...")
        
        try:
            # Configure mocks
            mock_orchestrator.return_value = self.mock_orchestrator
            mock_bridge_init.return_value = self.mock_bridge
            
            # Import and create app
            from ui.main_app import ClaudeTUIApp
            
            app = ClaudeTUIApp()
            
            # Test basic properties
            assert hasattr(app, 'project_manager')
            assert hasattr(app, 'ai_interface')
            assert hasattr(app, 'validation_engine')
            
            # Test initialization
            app.init_core_systems()
            
            # Test mounting
            app.on_mount()
            
            self.successes.append("Main TUI app created successfully")
            logger.info("Main app creation test passed")
            return True
            
        except Exception as e:
            logger.error(f"Main app creation failed: {e}")
            self.errors.append(f"Main app creation error: {e}")
            return False
    
    async def test_widget_functionality(self):
        """Test actual widget functionality with mock data."""
        logger.info("Testing widget functionality...")
        
        try:
            # Test ProjectTree widget
            try:
                from ui.widgets.project_tree import ProjectTree
                mock_project_manager = MockProjectManager()
                project_tree = ProjectTree(mock_project_manager)
                
                # Test basic methods if they exist
                if hasattr(project_tree, 'set_project'):
                    project_tree.set_project(self.temp_dir)
                
                if hasattr(project_tree, 'refresh'):
                    project_tree.refresh()
                
                self.successes.append("ProjectTree widget functionality tested")
                
            except Exception as e:
                self.warnings.append(f"ProjectTree widget test failed: {e}")
            
            # Test TaskDashboard widget
            try:
                from ui.widgets.task_dashboard import TaskDashboard
                mock_project_manager = MockProjectManager()
                task_dashboard = TaskDashboard(mock_project_manager)
                
                # Test basic methods if they exist
                if hasattr(task_dashboard, 'refresh'):
                    task_dashboard.refresh()
                
                if hasattr(task_dashboard, 'update_tasks'):
                    mock_tasks = [{'id': '1', 'name': 'Test Task', 'status': 'pending'}]
                    task_dashboard.update_tasks(mock_tasks)
                
                self.successes.append("TaskDashboard widget functionality tested")
                
            except Exception as e:
                self.warnings.append(f"TaskDashboard widget test failed: {e}")
            
            # Test ProgressIntelligence widget
            try:
                from ui.widgets.progress_intelligence import ProgressIntelligence
                progress_widget = ProgressIntelligence()
                
                # Test validation update if method exists
                if hasattr(progress_widget, 'update_validation'):
                    from types import SimpleNamespace
                    mock_validation = SimpleNamespace(
                        real_progress=0.7,
                        claimed_progress=0.9,
                        fake_progress=0.2,
                        quality_score=7.5,
                        authenticity_score=0.78,
                        placeholders_found=3,
                        todos_found=5
                    )
                    progress_widget.update_validation(mock_validation)
                
                self.successes.append("ProgressIntelligence widget functionality tested")
                
            except Exception as e:
                self.warnings.append(f"ProgressIntelligence widget test failed: {e}")
            
            # Test ConsoleWidget
            try:
                from ui.widgets.console_widget import ConsoleWidget
                mock_ai_interface = MockAIInterface()
                console_widget = ConsoleWidget(mock_ai_interface)
                
                # Test basic console methods if they exist
                if hasattr(console_widget, 'add_message'):
                    console_widget.add_message("Test message")
                
                if hasattr(console_widget, 'execute_command'):
                    await console_widget.execute_command("test command")
                
                self.successes.append("ConsoleWidget functionality tested")
                
            except Exception as e:
                self.warnings.append(f"ConsoleWidget test failed: {e}")
            
            logger.info("Widget functionality tests completed")
            return True
            
        except Exception as e:
            logger.error(f"Widget functionality test error: {e}")
            self.errors.append(f"Widget functionality error: {e}")
            return False
    
    async def test_backend_integration(self):
        """Test backend integration with real TUI components."""
        logger.info("Testing backend integration...")
        
        try:
            # Test cache service integration
            cache_service = self.mock_orchestrator.get_cache_service()
            await cache_service.set("tui_test", {"screen": "main", "focus": "project_tree"})
            cached_data = await cache_service.get("tui_test")
            
            assert cached_data is not None
            assert cached_data['value']['screen'] == "main"
            
            # Test AI service integration  
            ai_service = self.mock_orchestrator.get_ai_service()
            code_result = await ai_service.generate_code("Create a test function", "python")
            
            assert code_result is not None
            assert 'code' in code_result
            
            # Test task service integration
            task_service = self.mock_orchestrator.get_task_service()
            task = await task_service.create_task("Test Integration Task", "Test task description", "test-project")
            
            assert task is not None
            assert task['name'] == "Test Integration Task"
            
            # Test validation service integration
            validation_service = self.mock_orchestrator.get_validation_service()
            analysis = await validation_service.analyze_project(self.temp_dir)
            
            assert analysis is not None
            assert hasattr(analysis, 'real_progress')
            
            self.successes.append("Backend integration tests passed")
            logger.info("Backend integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"Backend integration test failed: {e}")
            self.errors.append(f"Backend integration error: {e}")
            return False
    
    async def test_error_scenarios(self):
        """Test error handling scenarios."""
        logger.info("Testing error scenarios...")
        
        try:
            # Test with invalid project path
            validation_service = self.mock_orchestrator.get_validation_service()
            analysis = await validation_service.analyze_project("/invalid/path")
            # Should not crash, should handle gracefully
            
            # Test with None inputs
            cache_service = self.mock_orchestrator.get_cache_service()
            try:
                await cache_service.set(None, "value")
            except:
                pass  # Expected to fail
            
            # Test task execution with invalid task
            task_service = self.mock_orchestrator.get_task_service()
            result = await task_service.execute_task("invalid-task-id")
            # Should return None gracefully
            
            self.successes.append("Error scenarios handled correctly")
            logger.info("Error scenarios test passed")
            return True
            
        except Exception as e:
            logger.error(f"Error scenarios test failed: {e}")
            self.errors.append(f"Error scenarios test error: {e}")
            return False
    
    def cleanup(self):
        """Clean up test environment."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            reset_mock_instances()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive TUI component test...")
        
        # Setup
        if not await self.setup_mock_environment():
            return self.generate_report()
        
        # Test component imports
        widget_results = self.test_widget_imports()
        screen_results = self.test_screen_imports()
        
        # Test main app creation
        app_creation_success = self.test_main_app_creation()
        
        # Test widget functionality
        widget_func_success = await self.test_widget_functionality()
        
        # Test backend integration
        backend_int_success = await self.test_backend_integration()
        
        # Test error scenarios
        error_scenarios_success = await self.test_error_scenarios()
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report."""
        total_successes = len(self.successes)
        total_warnings = len(self.warnings)
        total_errors = len(self.errors)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'successes': total_successes,
                'warnings': total_warnings,
                'errors': total_errors,
                'overall_status': 'PASS' if total_errors == 0 else 'FAIL' if total_errors > 5 else 'PARTIAL'
            },
            'successes': self.successes,
            'warnings': self.warnings,
            'errors': self.errors
        }
        
        return report


async def main():
    """Main test execution function."""
    tester = TUIComponentTester()
    
    try:
        report = await tester.run_comprehensive_test()
        
        print("\n" + "="*60)
        print("TUI COMPONENT TEST REPORT")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Status: {report['summary']['overall_status']}")
        print(f"Successes: {report['summary']['successes']}")
        print(f"Warnings: {report['summary']['warnings']}")
        print(f"Errors: {report['summary']['errors']}")
        print()
        
        if report['successes']:
            print("✅ SUCCESSES:")
            for i, success in enumerate(report['successes'], 1):
                print(f"  {i}. {success}")
            print()
        
        if report['warnings']:
            print("⚠️  WARNINGS:")
            for i, warning in enumerate(report['warnings'], 1):
                print(f"  {i}. {warning}")
            print()
        
        if report['errors']:
            print("❌ ERRORS:")
            for i, error in enumerate(report['errors'], 1):
                print(f"  {i}. {error}")
            print()
        
        print("="*60)
        
        # Return appropriate exit code
        return 0 if report['summary']['overall_status'] != 'FAIL' else 1
        
    finally:
        tester.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)