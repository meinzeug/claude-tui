#!/usr/bin/env python3
"""
End-to-End Integration Test for Automatic Programming Workflow
============================================================

This script tests the complete automatic programming integration:
1. Initializes all components
2. Creates a test workflow
3. Monitors progress in real-time
4. Validates results
5. Generates a comprehensive report

Run with: python scripts/test_automatic_programming_integration.py
"""

import asyncio
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import json
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from claude_tui.integrations.automatic_programming_workflow import (
        AutomaticProgrammingWorkflow, ProgressUpdate
    )
    from claude_tui.integrations.demo_workflows import DemoWorkflowGenerator
    from ui.integration_bridge import UIIntegrationBridge
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomaticProgrammingTester:
    """
    Comprehensive tester for automatic programming integration
    """
    
    def __init__(self):
        self.results = {
            "timestamp": time.time(),
            "tests_run": [],
            "successes": 0,
            "failures": 0,
            "errors": [],
            "performance_metrics": {},
            "generated_files": []
        }
        
        self.bridge = None
        self.workflow_manager = None
        self.demo_generator = None
        self.progress_updates = []
        
    def log_progress(self, update: ProgressUpdate):
        """Log progress updates for analysis"""
        self.progress_updates.append({
            "workflow_id": update.workflow_id,
            "step_id": update.step_id,
            "step_name": update.step_name,
            "progress": update.progress,
            "message": update.message,
            "timestamp": update.timestamp,
            "step_status": update.step_status
        })
        
        print(f"[{update.step_status.upper()}] {update.step_name}: {update.message} ({update.progress:.1%})")
    
    async def setup(self):
        """Initialize all components"""
        print("🔧 Setting up automatic programming integration...")
        
        try:
            # Initialize integration bridge
            self.bridge = UIIntegrationBridge()
            if not self.bridge.initialize():
                raise RuntimeError("Failed to initialize integration bridge")
            
            # Get workflow manager
            self.workflow_manager = self.bridge.workflow_manager
            if not self.workflow_manager:
                raise RuntimeError("Workflow manager not available")
            
            # Add progress callback
            self.workflow_manager.add_progress_callback(self.log_progress)
            
            # Initialize demo generator
            self.demo_generator = DemoWorkflowGenerator(self.workflow_manager)
            
            print("✅ Setup completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            self.results["errors"].append(f"Setup failed: {str(e)}")
            return False
    
    async def test_workflow_creation(self):
        """Test workflow creation from templates"""
        print("\n📝 Testing workflow creation...")
        
        test_name = "workflow_creation"
        self.results["tests_run"].append(test_name)
        
        try:
            # Test template-based workflow creation
            with tempfile.TemporaryDirectory() as temp_dir:
                project_path = Path(temp_dir) / "test_fastapi_project"
                
                workflow_id = await self.workflow_manager.create_workflow_from_template(
                    template_name="fastapi_app",
                    project_name="test_fastapi_project",
                    project_path=project_path
                )
                
                # Verify workflow was created
                workflow_status = self.workflow_manager.get_workflow_status(workflow_id)
                
                if workflow_status["status"] == "pending":
                    print(f"✅ Workflow created successfully: {workflow_id}")
                    self.results["successes"] += 1
                    return True
                else:
                    raise RuntimeError(f"Unexpected workflow status: {workflow_status['status']}")
                    
        except Exception as e:
            print(f"❌ Workflow creation failed: {e}")
            self.results["failures"] += 1
            self.results["errors"].append(f"{test_name}: {str(e)}")
            return False
    
    async def test_custom_workflow_creation(self):
        """Test custom workflow creation from natural language"""
        print("\n🎯 Testing custom workflow creation...")
        
        test_name = "custom_workflow_creation"
        self.results["tests_run"].append(test_name)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                project_path = Path(temp_dir) / "test_custom_project"
                
                workflow_id = await self.workflow_manager.create_custom_workflow(
                    name="Test Custom App",
                    description="A simple test application",
                    prompt="Create a simple Python Flask web application with a basic HTML template and a few routes",
                    project_path=project_path,
                    workflow_type="python"
                )
                
                # Verify workflow was created
                workflow_status = self.workflow_manager.get_workflow_status(workflow_id)
                
                if workflow_status["status"] == "pending":
                    print(f"✅ Custom workflow created successfully: {workflow_id}")
                    self.results["successes"] += 1
                    return True
                else:
                    raise RuntimeError(f"Unexpected workflow status: {workflow_status['status']}")
                    
        except Exception as e:
            print(f"❌ Custom workflow creation failed: {e}")
            self.results["failures"] += 1
            self.results["errors"].append(f"{test_name}: {str(e)}")
            return False
    
    async def test_workflow_execution(self):
        """Test full workflow execution with progress monitoring"""
        print("\n⚡ Testing workflow execution with real-time monitoring...")
        
        test_name = "workflow_execution"
        self.results["tests_run"].append(test_name)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                project_path = Path(temp_dir) / "test_execution_project"
                
                # Create a simple workflow
                workflow_id = await self.workflow_manager.create_workflow_from_template(
                    template_name="fastapi_app",
                    project_name="test_execution_project", 
                    project_path=project_path
                )
                
                print(f"📊 Starting workflow execution: {workflow_id}")
                
                # Track execution time
                start_time = time.time()
                
                # Execute workflow
                result = await self.workflow_manager.execute_workflow(workflow_id)
                
                execution_time = time.time() - start_time
                self.results["performance_metrics"]["execution_time"] = execution_time
                
                # Analyze results
                print(f"📈 Workflow completed in {execution_time:.2f}s")
                print(f"Status: {result.status}")
                print(f"Steps completed: {result.steps_completed}/{result.steps_total}")
                print(f"Errors: {len(result.errors)}")
                print(f"Created files: {len(result.created_files)}")
                
                # Check if execution was successful (at least partially)
                if result.steps_completed > 0:
                    print("✅ Workflow execution completed")
                    self.results["successes"] += 1
                    
                    # Store file information
                    self.results["generated_files"] = result.created_files
                    self.results["performance_metrics"]["steps_completed"] = result.steps_completed
                    self.results["performance_metrics"]["steps_total"] = result.steps_total
                    
                    return True
                else:
                    raise RuntimeError("No steps were completed")
                    
        except Exception as e:
            print(f"❌ Workflow execution failed: {e}")
            self.results["failures"] += 1
            self.results["errors"].append(f"{test_name}: {str(e)}")
            return False
    
    async def test_demo_workflow(self):
        """Test the demo workflow generator"""
        print("\n🎭 Testing demo workflow generation...")
        
        test_name = "demo_workflow"
        self.results["tests_run"].append(test_name)
        
        try:
            if not self.demo_generator:
                raise RuntimeError("Demo generator not initialized")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                project_path = Path(temp_dir) / "demo_fastapi_project"
                
                # Create demo workflow
                workflow_id = await self.demo_generator.create_fastapi_demo(
                    project_name="demo_fastapi_project",
                    project_path=project_path
                )
                
                # Get workflow info
                workflow_status = self.workflow_manager.get_workflow_status(workflow_id)
                
                print(f"✅ Demo workflow created: {workflow_status['name']}")
                print(f"Steps: {workflow_status['steps_total']}")
                
                self.results["successes"] += 1
                return True
                
        except Exception as e:
            print(f"❌ Demo workflow test failed: {e}")
            self.results["failures"] += 1
            self.results["errors"].append(f"{test_name}: {str(e)}")
            return False
    
    async def test_template_listing(self):
        """Test template listing functionality"""
        print("\n📚 Testing template listing...")
        
        test_name = "template_listing"
        self.results["tests_run"].append(test_name)
        
        try:
            templates = self.workflow_manager.get_available_templates()
            
            print(f"Available templates: {len(templates)}")
            for template_id, template_info in templates.items():
                print(f"  • {template_info['name']}: {template_info['description']}")
            
            if len(templates) > 0:
                print("✅ Template listing successful")
                self.results["successes"] += 1
                return True
            else:
                raise RuntimeError("No templates available")
                
        except Exception as e:
            print(f"❌ Template listing failed: {e}")
            self.results["failures"] += 1
            self.results["errors"].append(f"{test_name}: {str(e)}")
            return False
    
    async def test_progress_monitoring(self):
        """Test progress monitoring functionality"""
        print("\n📊 Testing progress monitoring...")
        
        test_name = "progress_monitoring"
        self.results["tests_run"].append(test_name)
        
        try:
            # Check if we received progress updates from previous tests
            if len(self.progress_updates) > 0:
                print(f"✅ Received {len(self.progress_updates)} progress updates")
                
                # Analyze progress updates
                steps_started = len([u for u in self.progress_updates if u["step_status"] == "starting"])
                steps_completed = len([u for u in self.progress_updates if u["step_status"] == "completed"])
                steps_failed = len([u for u in self.progress_updates if u["step_status"] == "failed"])
                
                print(f"  Steps started: {steps_started}")
                print(f"  Steps completed: {steps_completed}")
                print(f"  Steps failed: {steps_failed}")
                
                self.results["performance_metrics"]["progress_updates"] = len(self.progress_updates)
                self.results["performance_metrics"]["steps_started"] = steps_started
                self.results["performance_metrics"]["steps_completed_via_progress"] = steps_completed
                self.results["performance_metrics"]["steps_failed_via_progress"] = steps_failed
                
                self.results["successes"] += 1
                return True
            else:
                raise RuntimeError("No progress updates received")
                
        except Exception as e:
            print(f"❌ Progress monitoring test failed: {e}")
            self.results["failures"] += 1
            self.results["errors"].append(f"{test_name}: {str(e)}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        print("\n📋 Generating test report...")
        
        # Calculate summary statistics
        total_tests = len(self.results["tests_run"])
        success_rate = (self.results["successes"] / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "test_summary": {
                "timestamp": self.results["timestamp"],
                "total_tests": total_tests,
                "successes": self.results["successes"],
                "failures": self.results["failures"],
                "success_rate": success_rate
            },
            "tests_executed": self.results["tests_run"],
            "performance_metrics": self.results["performance_metrics"],
            "errors": self.results["errors"],
            "generated_files_count": len(self.results["generated_files"]),
            "progress_tracking": {
                "total_updates": len(self.progress_updates),
                "update_frequency": len(self.progress_updates) / self.results["performance_metrics"].get("execution_time", 1)
            }
        }
        
        return report
    
    async def cleanup(self):
        """Clean up test resources"""
        print("\n🧹 Cleaning up...")
        
        try:
            if self.workflow_manager:
                # Remove progress callback
                self.workflow_manager.remove_progress_callback(self.log_progress)
                
                # Cancel any active workflows
                workflows = self.workflow_manager.list_workflows()
                for workflow in workflows:
                    if workflow["is_active"]:
                        await self.workflow_manager.cancel_workflow(workflow["id"])
                
                # Cleanup workflow manager
                await self.workflow_manager.cleanup()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("🚀 Starting Automatic Programming Integration Tests")
        print("=" * 60)
        
        try:
            # Setup
            if not await self.setup():
                return self.generate_report()
            
            # Run individual tests
            tests = [
                self.test_template_listing,
                self.test_workflow_creation,
                self.test_custom_workflow_creation,
                self.test_demo_workflow,
                self.test_workflow_execution,
                self.test_progress_monitoring
            ]
            
            for test_func in tests:
                try:
                    await test_func()
                except Exception as e:
                    print(f"❌ Test {test_func.__name__} failed with exception: {e}")
                    self.results["failures"] += 1
                    self.results["errors"].append(f"{test_func.__name__}: {str(e)}")
                
                # Small delay between tests
                await asyncio.sleep(0.5)
            
            return self.generate_report()
            
        finally:
            await self.cleanup()


async def main():
    """Main test execution function"""
    print("🧪 Automatic Programming Integration Test Suite")
    print("===============================================")
    
    tester = AutomaticProgrammingTester()
    
    try:
        report = await tester.run_all_tests()
        
        # Print final report
        print("\n" + "=" * 60)
        print("📊 FINAL TEST REPORT")
        print("=" * 60)
        
        print(f"Total Tests: {report['test_summary']['total_tests']}")
        print(f"Successes: {report['test_summary']['successes']}")
        print(f"Failures: {report['test_summary']['failures']}")
        print(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
        
        if report["performance_metrics"]:
            print(f"\nPerformance Metrics:")
            for metric, value in report["performance_metrics"].items():
                print(f"  • {metric}: {value}")
        
        if report["errors"]:
            print(f"\nErrors Encountered:")
            for error in report["errors"]:
                print(f"  • {error}")
        
        # Save detailed report
        report_file = Path("automatic_programming_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Return appropriate exit code
        if report['test_summary']['failures'] == 0:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️ Some tests failed. Check the report for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))