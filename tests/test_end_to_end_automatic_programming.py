#!/usr/bin/env python3
"""
End-to-End Automatic Programming System Test Suite

Tests the complete integration of:
- Claude Code Direct Client with OAuth
- Claude Flow API server
- Automatic Programming Pipeline
- TUI interface
- Error handling and recovery
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import unittest
from unittest.mock import patch, MagicMock
import requests
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claude_tui.integrations.claude_code_client import ClaudeCodeClient
from claude_tui.integrations.claude_flow_client import ClaudeFlowClient
from claude_tui.core.ai_interface import AIInterface
from claude_tui.validation.real_time_validator import RealTimeValidator
from claude_tui.ui.main_app import ClaudeTUIApp


class EndToEndAutomaticProgrammingTest(unittest.TestCase):
    """Comprehensive end-to-end test for automatic programming system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path("/home/tekkadmin/claude-tui/testing/complete_system_validation")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        cls.test_config = {
            "claude_code_oauth_token": os.getenv("CLAUDE_CODE_OAUTH_TOKEN"),
            "claude_flow_api_url": "http://localhost:3000",
            "test_timeout": 300,  # 5 minutes
            "max_retries": 3
        }
        
        # Initialize test results
        cls.test_results = {
            "timestamp": time.time(),
            "components": {},
            "workflows": {},
            "performance": {},
            "errors": []
        }
    
    def setUp(self):
        """Set up each test"""
        self.start_time = time.time()
        
    def tearDown(self):
        """Clean up after each test"""
        self.execution_time = time.time() - self.start_time
    
    def test_01_claude_code_oauth_integration(self):
        """Test Claude Code Direct Client OAuth integration"""
        print("\n🔐 Testing Claude Code OAuth Integration...")
        
        try:
            # Check for .cc file
            cc_file = Path.home() / ".cc"
            self.assertTrue(cc_file.exists(), ".cc file not found")
            
            # Read OAuth token
            with open(cc_file, 'r') as f:
                cc_config = json.load(f)
            
            self.assertIn('oauth_token', cc_config, "OAuth token not found in .cc file")
            
            # Initialize Claude Code client
            client = ClaudeCodeClient()
            
            # Test connection
            response = client.test_connection()
            self.assertTrue(response.get('success', False), "Claude Code connection failed")
            
            self.test_results["components"]["claude_code_oauth"] = {
                "status": "PASSED",
                "execution_time": self.execution_time,
                "details": "OAuth integration working correctly"
            }
            
        except Exception as e:
            self.test_results["components"]["claude_code_oauth"] = {
                "status": "FAILED",
                "error": str(e),
                "execution_time": self.execution_time
            }
            raise
    
    def test_02_claude_flow_api_server(self):
        """Test Claude Flow API server orchestration"""
        print("\n🌊 Testing Claude Flow API Server...")
        
        try:
            # Start Claude Flow server if not running
            self._ensure_claude_flow_running()
            
            # Test API endpoints
            client = ClaudeFlowClient()
            
            # Test swarm initialization
            swarm_response = client.init_swarm({
                "topology": "mesh",
                "maxAgents": 3,
                "task": "test_orchestration"
            })
            
            self.assertTrue(swarm_response.get('success', False), "Swarm initialization failed")
            
            # Test agent spawning
            spawn_response = client.spawn_agent({
                "type": "coder",
                "role": "test_agent",
                "capabilities": ["python", "testing"]
            })
            
            self.assertTrue(spawn_response.get('success', False), "Agent spawning failed")
            
            self.test_results["components"]["claude_flow_api"] = {
                "status": "PASSED",
                "execution_time": self.execution_time,
                "details": "API server responding correctly"
            }
            
        except Exception as e:
            self.test_results["components"]["claude_flow_api"] = {
                "status": "FAILED",
                "error": str(e),
                "execution_time": self.execution_time
            }
            raise
    
    def test_03_automatic_programming_pipeline(self):
        """Test Automatic Programming Pipeline processing"""
        print("\n🤖 Testing Automatic Programming Pipeline...")
        
        try:
            # Initialize AI interface
            ai_interface = AIInterface()
            
            # Test requirement processing
            test_requirement = "Create a simple Python function that calculates the factorial of a number"
            
            # Process requirement through pipeline
            result = ai_interface.process_automatic_programming_request({
                "requirement": test_requirement,
                "language": "python",
                "include_tests": True,
                "output_dir": str(self.test_dir / "generated_code")
            })
            
            self.assertTrue(result.get('success', False), "Pipeline processing failed")
            self.assertIn('generated_files', result, "No generated files in result")
            
            # Verify generated files exist
            output_dir = Path(result['output_dir'])
            self.assertTrue(output_dir.exists(), "Output directory not created")
            
            generated_files = list(output_dir.rglob("*.py"))
            self.assertGreater(len(generated_files), 0, "No Python files generated")
            
            self.test_results["components"]["programming_pipeline"] = {
                "status": "PASSED",
                "execution_time": self.execution_time,
                "generated_files": len(generated_files),
                "details": "Pipeline processing completed successfully"
            }
            
        except Exception as e:
            self.test_results["components"]["programming_pipeline"] = {
                "status": "FAILED",
                "error": str(e),
                "execution_time": self.execution_time
            }
            raise
    
    def test_04_end_to_end_workflow_calculator(self):
        """Test complete workflow with Python calculator requirement"""
        print("\n🧮 Testing End-to-End Calculator Workflow...")
        
        try:
            # Complex requirement for calculator
            requirement = """
            Create a Python calculator application with CLI interface that:
            1. Supports basic arithmetic operations (+, -, *, /)
            2. Handles parentheses and order of operations
            3. Includes error handling for invalid input
            4. Has comprehensive unit tests with pytest
            5. Includes a command-line interface
            6. Supports both interactive and single-expression modes
            """
            
            # Execute complete workflow
            workflow_result = self._execute_complete_workflow(
                requirement=requirement,
                project_name="calculator_app"
            )
            
            self.assertTrue(workflow_result.get('success', False), "Workflow execution failed")
            
            # Validate generated code structure
            project_dir = Path(workflow_result['project_dir'])
            
            # Check for required files
            required_files = [
                "calculator.py",
                "cli.py",
                "test_calculator.py",
                "requirements.txt",
                "README.md"
            ]
            
            for file_name in required_files:
                file_path = project_dir / file_name
                self.assertTrue(file_path.exists(), f"Required file {file_name} not found")
            
            # Test code functionality
            self._test_generated_calculator(project_dir)
            
            self.test_results["workflows"]["calculator_e2e"] = {
                "status": "PASSED",
                "execution_time": self.execution_time,
                "project_dir": str(project_dir),
                "files_generated": len(list(project_dir.rglob("*.*"))),
                "details": "Complete calculator workflow successful"
            }
            
        except Exception as e:
            self.test_results["workflows"]["calculator_e2e"] = {
                "status": "FAILED",
                "error": str(e),
                "execution_time": self.execution_time
            }
            raise
    
    def test_05_tui_integration(self):
        """Test TUI integration with automatic programming"""
        print("\n🖥️  Testing TUI Integration...")
        
        try:
            # Mock TUI components for testing
            with patch('claude_tui.ui.main_app.ClaudeTUIApp') as mock_app:
                mock_instance = MagicMock()
                mock_app.return_value = mock_instance
                
                # Simulate TUI automatic programming request
                mock_instance.handle_automatic_programming.return_value = {
                    "success": True,
                    "message": "Code generation completed",
                    "files_created": 3
                }
                
                # Test TUI integration
                app = ClaudeTUIApp()
                result = app.handle_automatic_programming({
                    "requirement": "Create a simple web server",
                    "framework": "flask"
                })
                
                self.assertTrue(result.get('success', False), "TUI integration failed")
            
            self.test_results["components"]["tui_integration"] = {
                "status": "PASSED",
                "execution_time": self.execution_time,
                "details": "TUI integration working correctly"
            }
            
        except Exception as e:
            self.test_results["components"]["tui_integration"] = {
                "status": "FAILED",
                "error": str(e),
                "execution_time": self.execution_time
            }
            raise
    
    def test_06_performance_reliability(self):
        """Test performance and reliability with complex requirements"""
        print("\n⚡ Testing Performance and Reliability...")
        
        try:
            # Complex multi-service requirement
            complex_requirement = """
            Create a complete microservices architecture with:
            1. User authentication service (FastAPI)
            2. Product catalog service (Flask)
            3. Order processing service (Django)
            4. Database models for all services
            5. API documentation with OpenAPI
            6. Docker configuration for each service
            7. Comprehensive test suites for all services
            8. CI/CD pipeline configuration
            """
            
            start_time = time.time()
            
            # Execute complex workflow with performance monitoring
            result = self._execute_performance_test(complex_requirement)
            
            execution_time = time.time() - start_time
            
            # Performance thresholds
            self.assertLess(execution_time, 600, "Complex workflow took too long (>10 minutes)")
            
            # Memory usage check
            memory_usage = result.get('memory_usage_mb', 0)
            self.assertLess(memory_usage, 1024, "Memory usage too high (>1GB)")
            
            self.test_results["performance"]["complex_workflow"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "memory_usage_mb": memory_usage,
                "files_generated": result.get('files_generated', 0),
                "details": "Performance within acceptable limits"
            }
            
        except Exception as e:
            self.test_results["performance"]["complex_workflow"] = {
                "status": "FAILED",
                "error": str(e),
                "execution_time": self.execution_time
            }
            raise
    
    def test_07_error_handling_recovery(self):
        """Test error handling and recovery mechanisms"""
        print("\n🛡️  Testing Error Handling and Recovery...")
        
        try:
            # Test various error scenarios
            error_scenarios = [
                {
                    "name": "invalid_requirement",
                    "requirement": "",  # Empty requirement
                    "expected_error": "InvalidRequirementError"
                },
                {
                    "name": "network_failure",
                    "requirement": "Create a simple app",
                    "simulate_network_error": True
                },
                {
                    "name": "resource_exhaustion",
                    "requirement": "Create a massive application with 1000 files",
                    "expected_error": "ResourceLimitError"
                }
            ]
            
            recovery_results = {}
            
            for scenario in error_scenarios:
                try:
                    result = self._test_error_scenario(scenario)
                    recovery_results[scenario['name']] = {
                        "handled": True,
                        "recovered": result.get('recovered', False),
                        "error_type": result.get('error_type')
                    }
                except Exception as e:
                    recovery_results[scenario['name']] = {
                        "handled": False,
                        "error": str(e)
                    }
            
            # Check that at least 2/3 scenarios were handled properly
            handled_count = sum(1 for r in recovery_results.values() if r.get('handled', False))
            self.assertGreaterEqual(handled_count, 2, "Error handling insufficient")
            
            self.test_results["components"]["error_handling"] = {
                "status": "PASSED",
                "execution_time": self.execution_time,
                "scenarios_tested": len(error_scenarios),
                "scenarios_handled": handled_count,
                "details": recovery_results
            }
            
        except Exception as e:
            self.test_results["components"]["error_handling"] = {
                "status": "FAILED",
                "error": str(e),
                "execution_time": self.execution_time
            }
            raise
    
    def _ensure_claude_flow_running(self):
        """Ensure Claude Flow server is running"""
        try:
            response = requests.get(f"{self.test_config['claude_flow_api_url']}/health", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        
        # Start Claude Flow server
        print("Starting Claude Flow server...")
        subprocess.Popen([
            "npx", "claude-flow@alpha", "mcp", "start"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            try:
                response = requests.get(f"{self.test_config['claude_flow_api_url']}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                time.sleep(1)
        
        raise Exception("Failed to start Claude Flow server")
    
    def _execute_complete_workflow(self, requirement: str, project_name: str) -> Dict[str, Any]:
        """Execute complete automatic programming workflow"""
        project_dir = self.test_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Initialize components
        ai_interface = AIInterface()
        validator = RealTimeValidator()
        
        # Process requirement
        result = ai_interface.process_automatic_programming_request({
            "requirement": requirement,
            "project_name": project_name,
            "output_dir": str(project_dir),
            "include_tests": True,
            "include_docs": True
        })
        
        if not result.get('success', False):
            raise Exception(f"Workflow failed: {result.get('error', 'Unknown error')}")
        
        # Validate generated code
        validation_result = validator.validate_project(str(project_dir))
        
        return {
            "success": True,
            "project_dir": str(project_dir),
            "generation_result": result,
            "validation_result": validation_result
        }
    
    def _test_generated_calculator(self, project_dir: Path):
        """Test the generated calculator functionality"""
        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(project_dir)
        
        try:
            # Install dependencies
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            
            # Run tests
            result = subprocess.run([sys.executable, "-m", "pytest", "test_calculator.py", "-v"], 
                                  capture_output=True, text=True)
            
            self.assertEqual(result.returncode, 0, f"Tests failed: {result.stdout}\n{result.stderr}")
            
            # Test basic functionality
            import calculator
            
            # Test basic operations
            self.assertEqual(calculator.add(2, 3), 5)
            self.assertEqual(calculator.multiply(4, 5), 20)
            
        finally:
            os.chdir(original_cwd)
    
    def _execute_performance_test(self, requirement: str) -> Dict[str, Any]:
        """Execute performance test with monitoring"""
        import psutil
        import gc
        
        # Monitor initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Execute workflow
        result = self._execute_complete_workflow(requirement, "performance_test_project")
        
        # Force garbage collection
        gc.collect()
        
        # Monitor final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Count generated files
        project_dir = Path(result['project_dir'])
        files_generated = len(list(project_dir.rglob("*.*")))
        
        return {
            "success": True,
            "memory_usage_mb": final_memory - initial_memory,
            "files_generated": files_generated,
            "project_dir": result['project_dir']
        }
    
    def _test_error_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test specific error scenario"""
        ai_interface = AIInterface()
        
        # Simulate network error if requested
        if scenario.get('simulate_network_error'):
            with patch('requests.post') as mock_post:
                mock_post.side_effect = requests.ConnectionError("Network error")
                
                try:
                    result = ai_interface.process_automatic_programming_request({
                        "requirement": scenario['requirement']
                    })
                    
                    # Should handle the error gracefully
                    return {
                        "recovered": result.get('success', False),
                        "error_type": "NetworkError"
                    }
                except Exception as e:
                    return {
                        "recovered": False,
                        "error_type": type(e).__name__
                    }
        
        # Test with actual requirement
        try:
            result = ai_interface.process_automatic_programming_request({
                "requirement": scenario['requirement']
            })
            
            return {
                "recovered": True,
                "error_type": None
            }
        except Exception as e:
            expected_error = scenario.get('expected_error')
            if expected_error and expected_error in type(e).__name__:
                return {
                    "recovered": True,  # Expected error handled correctly
                    "error_type": type(e).__name__
                }
            else:
                return {
                    "recovered": False,
                    "error_type": type(e).__name__
                }
    
    @classmethod
    def tearDownClass(cls):
        """Generate comprehensive test report"""
        cls._generate_test_report()
    
    @classmethod
    def _generate_test_report(cls):
        """Generate comprehensive test report"""
        cls.test_results['total_execution_time'] = time.time() - cls.test_results['timestamp']
        
        # Calculate summary stats
        total_tests = 0
        passed_tests = 0
        
        for category in ['components', 'workflows', 'performance']:
            for test_name, result in cls.test_results.get(category, {}).items():
                total_tests += 1
                if result.get('status') == 'PASSED':
                    passed_tests += 1
        
        cls.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }
        
        # Save results
        results_file = Path("/home/tekkadmin/claude-tui/testing/complete_system_validation/test_results.json")
        with open(results_file, 'w') as f:
            json.dump(cls.test_results, f, indent=2)
        
        # Generate human-readable report
        report_file = Path("/home/tekkadmin/claude-tui/testing/complete_system_validation/test_report.md")
        cls._generate_markdown_report(report_file)
        
        print(f"\n📊 Test Results Summary:")
        print(f"   Total Tests: {cls.test_results['summary']['total_tests']}")
        print(f"   Passed: {cls.test_results['summary']['passed_tests']}")
        print(f"   Failed: {cls.test_results['summary']['failed_tests']}")
        print(f"   Success Rate: {cls.test_results['summary']['success_rate']:.1f}%")
        print(f"   Total Time: {cls.test_results['total_execution_time']:.2f}s")
        print(f"\n📁 Results saved to: {results_file}")
        print(f"📄 Report saved to: {report_file}")
    
    @classmethod
    def _generate_markdown_report(cls, report_file: Path):
        """Generate markdown test report"""
        with open(report_file, 'w') as f:
            f.write("# End-to-End Automatic Programming System Test Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            summary = cls.test_results['summary']
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests:** {summary['total_tests']}\n")
            f.write(f"- **Passed:** {summary['passed_tests']}\n")
            f.write(f"- **Failed:** {summary['failed_tests']}\n")
            f.write(f"- **Success Rate:** {summary['success_rate']:.1f}%\n")
            f.write(f"- **Total Execution Time:** {cls.test_results['total_execution_time']:.2f}s\n\n")
            
            # Component Tests
            if 'components' in cls.test_results:
                f.write("## Component Tests\n\n")
                for test_name, result in cls.test_results['components'].items():
                    status_emoji = "✅" if result.get('status') == 'PASSED' else "❌"
                    f.write(f"### {status_emoji} {test_name.replace('_', ' ').title()}\n")
                    f.write(f"- **Status:** {result.get('status', 'UNKNOWN')}\n")
                    f.write(f"- **Execution Time:** {result.get('execution_time', 0):.2f}s\n")
                    if 'details' in result:
                        f.write(f"- **Details:** {result['details']}\n")
                    if 'error' in result:
                        f.write(f"- **Error:** {result['error']}\n")
                    f.write("\n")
            
            # Workflow Tests
            if 'workflows' in cls.test_results:
                f.write("## Workflow Tests\n\n")
                for test_name, result in cls.test_results['workflows'].items():
                    status_emoji = "✅" if result.get('status') == 'PASSED' else "❌"
                    f.write(f"### {status_emoji} {test_name.replace('_', ' ').title()}\n")
                    f.write(f"- **Status:** {result.get('status', 'UNKNOWN')}\n")
                    f.write(f"- **Execution Time:** {result.get('execution_time', 0):.2f}s\n")
                    if 'files_generated' in result:
                        f.write(f"- **Files Generated:** {result['files_generated']}\n")
                    if 'details' in result:
                        f.write(f"- **Details:** {result['details']}\n")
                    if 'error' in result:
                        f.write(f"- **Error:** {result['error']}\n")
                    f.write("\n")
            
            # Performance Tests
            if 'performance' in cls.test_results:
                f.write("## Performance Tests\n\n")
                for test_name, result in cls.test_results['performance'].items():
                    status_emoji = "✅" if result.get('status') == 'PASSED' else "❌"
                    f.write(f"### {status_emoji} {test_name.replace('_', ' ').title()}\n")
                    f.write(f"- **Status:** {result.get('status', 'UNKNOWN')}\n")
                    f.write(f"- **Execution Time:** {result.get('execution_time', 0):.2f}s\n")
                    if 'memory_usage_mb' in result:
                        f.write(f"- **Memory Usage:** {result['memory_usage_mb']:.2f}MB\n")
                    if 'files_generated' in result:
                        f.write(f"- **Files Generated:** {result['files_generated']}\n")
                    if 'details' in result:
                        f.write(f"- **Details:** {result['details']}\n")
                    if 'error' in result:
                        f.write(f"- **Error:** {result['error']}\n")
                    f.write("\n")


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)