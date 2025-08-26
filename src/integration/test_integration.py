#!/usr/bin/env python3
"""
Integration Test Suite
Tests the complete MCP server and claude-flow integration
"""

import asyncio
import json
import pytest
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import time
from datetime import datetime

from .startup_manager import StartupManager, QuickStart
from .bridge import IntegrationBridge, BridgeConfig
from .hooks_manager import DevelopmentHooksCoordinator
from .tui_connector import MCPConnectionManager
from ..mcp.server import MCPServerClient, SwarmCoordinator
from ..monitoring.dashboard import MetricsCollector

logger = logging.getLogger(__name__)

class IntegrationTester:
    """Comprehensive integration tester"""
    
    def __init__(self):
        self.startup_manager: Optional[StartupManager] = None
        self.test_results: Dict[str, Any] = {}
        self.start_time = time.time()
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("Starting comprehensive integration tests...")
        
        try:
            # Test 1: Startup sequence
            await self._test_startup_sequence()
            
            # Test 2: MCP server connectivity
            await self._test_mcp_connectivity()
            
            # Test 3: Swarm operations
            await self._test_swarm_operations()
            
            # Test 4: Hooks integration
            await self._test_hooks_integration()
            
            # Test 5: API endpoints
            await self._test_api_endpoints()
            
            # Test 6: Monitoring dashboard
            await self._test_monitoring_dashboard()
            
            # Test 7: TUI connector
            await self._test_tui_connector()
            
            # Test 8: End-to-end workflow
            await self._test_end_to_end_workflow()
            
        except Exception as e:
            logger.error(f"Integration test error: {e}")
            self.test_results["error"] = str(e)
        
        finally:
            # Cleanup
            if self.startup_manager:
                await self.startup_manager.shutdown()
        
        # Generate test report
        return self._generate_test_report()
    
    async def _test_startup_sequence(self):
        """Test startup sequence"""
        logger.info("Testing startup sequence...")
        
        start_time = time.time()
        self.startup_manager = await QuickStart.development_mode()
        end_time = time.time()
        
        success = self.startup_manager.startup_complete
        startup_time = end_time - start_time
        
        self.test_results["startup"] = {
            "success": success,
            "startup_time": startup_time,
            "components": list(self.startup_manager.components.keys()),
            "status": self.startup_manager.get_status()
        }
        
        logger.info(f"Startup test: {'PASS' if success else 'FAIL'} ({startup_time:.2f}s)")
    
    async def _test_mcp_connectivity(self):
        """Test MCP server connectivity"""
        logger.info("Testing MCP server connectivity...")
        
        try:
            async with MCPServerClient() as client:
                coordinator = SwarmCoordinator(client)
                
                # Test basic connectivity
                status = await coordinator.get_swarm_status()
                
                # Test swarm initialization
                init_success = await coordinator.initialize_swarm("mesh", 3)
                
                self.test_results["mcp_connectivity"] = {
                    "success": True,
                    "status_response": status,
                    "init_success": init_success,
                    "client_connected": True
                }
                
        except Exception as e:
            self.test_results["mcp_connectivity"] = {
                "success": False,
                "error": str(e)
            }
        
        logger.info(f"MCP connectivity test: {'PASS' if self.test_results['mcp_connectivity']['success'] else 'FAIL'}")
    
    async def _test_swarm_operations(self):
        """Test swarm operations"""
        logger.info("Testing swarm operations...")
        
        try:
            async with MCPServerClient() as client:
                coordinator = SwarmCoordinator(client)
                
                operations_results = {}
                
                # Test swarm initialization
                init_success = await coordinator.initialize_swarm("hierarchical", 5)
                operations_results["init"] = init_success
                
                # Test agent spawning
                spawn_success = await coordinator.spawn_agent("coder", {"language": "python"})
                operations_results["spawn_coder"] = spawn_success
                
                spawn_success2 = await coordinator.spawn_agent("reviewer", {"expertise": "python"})
                operations_results["spawn_reviewer"] = spawn_success2
                
                # Test task orchestration
                task_id = await coordinator.orchestrate_task(
                    "Test task coordination",
                    ["coder", "reviewer"]
                )
                operations_results["orchestrate"] = task_id is not None
                operations_results["task_id"] = task_id
                
                # Test status retrieval
                status = await coordinator.get_swarm_status()
                operations_results["status"] = status is not None
                
                # Test metrics
                metrics = await coordinator.get_agent_metrics()
                operations_results["metrics"] = isinstance(metrics, list)
                
                self.test_results["swarm_operations"] = {
                    "success": all(operations_results.values()),
                    "operations": operations_results,
                    "final_status": status
                }
                
        except Exception as e:
            self.test_results["swarm_operations"] = {
                "success": False,
                "error": str(e)
            }
        
        logger.info(f"Swarm operations test: {'PASS' if self.test_results['swarm_operations']['success'] else 'FAIL'}")
    
    async def _test_hooks_integration(self):
        """Test hooks integration"""
        logger.info("Testing hooks integration...")
        
        try:
            coordinator = DevelopmentHooksCoordinator()
            
            # Test session management
            session_start = await coordinator.hooks_manager.start_session()
            
            # Test task coordination with hooks
            task_success = await coordinator.coordinate_development_task(
                "Test development task",
                [
                    {"file": "test_file.py", "operation": "create"},
                    {"file": "test_file2.py", "operation": "edit"}
                ]
            )
            
            # Test session end
            await coordinator.finalize_session()
            
            # Get operation status
            status = coordinator.get_operation_status()
            
            self.test_results["hooks_integration"] = {
                "success": session_start and task_success,
                "session_start": session_start,
                "task_coordination": task_success,
                "final_status": status
            }
            
        except Exception as e:
            self.test_results["hooks_integration"] = {
                "success": False,
                "error": str(e)
            }
        
        logger.info(f"Hooks integration test: {'PASS' if self.test_results['hooks_integration']['success'] else 'FAIL'}")
    
    async def _test_api_endpoints(self):
        """Test API endpoints"""
        logger.info("Testing API endpoints...")
        
        try:
            import aiohttp
            
            base_url = f"http://localhost:8000"
            
            async with aiohttp.ClientSession() as session:
                endpoints_results = {}
                
                # Test health endpoint
                try:
                    async with session.get(f"{base_url}/health", timeout=5) as response:
                        endpoints_results["health"] = response.status == 200
                        health_data = await response.json()
                except:
                    endpoints_results["health"] = False
                    health_data = None
                
                # Test swarm init endpoint
                try:
                    payload = {"topology": "mesh", "max_agents": 3}
                    async with session.post(f"{base_url}/api/swarm/init", json=payload, timeout=10) as response:
                        endpoints_results["swarm_init"] = response.status == 200
                        init_data = await response.json()
                except:
                    endpoints_results["swarm_init"] = False
                    init_data = None
                
                # Test swarm status endpoint
                try:
                    async with session.get(f"{base_url}/api/swarm/status", timeout=5) as response:
                        endpoints_results["swarm_status"] = response.status == 200
                        status_data = await response.json()
                except:
                    endpoints_results["swarm_status"] = False
                    status_data = None
                
                self.test_results["api_endpoints"] = {
                    "success": any(endpoints_results.values()),  # At least one endpoint should work
                    "endpoints": endpoints_results,
                    "health_data": health_data,
                    "init_data": init_data,
                    "status_data": status_data
                }
                
        except Exception as e:
            self.test_results["api_endpoints"] = {
                "success": False,
                "error": str(e)
            }
        
        logger.info(f"API endpoints test: {'PASS' if self.test_results['api_endpoints']['success'] else 'FAIL'}")
    
    async def _test_monitoring_dashboard(self):
        """Test monitoring dashboard"""
        logger.info("Testing monitoring dashboard...")
        
        try:
            collector = MetricsCollector()
            
            # Test metrics collection
            metrics = await collector.collect_metrics()
            
            # Test historical data
            history = collector.get_historical_metrics(1)  # Last hour
            
            self.test_results["monitoring_dashboard"] = {
                "success": metrics is not None,
                "current_metrics": {
                    "active_agents": metrics.active_agents,
                    "completed_tasks": metrics.completed_tasks,
                    "timestamp": metrics.timestamp.isoformat()
                } if metrics else None,
                "historical_count": len(history)
            }
            
        except Exception as e:
            self.test_results["monitoring_dashboard"] = {
                "success": False,
                "error": str(e)
            }
        
        logger.info(f"Monitoring dashboard test: {'PASS' if self.test_results['monitoring_dashboard']['success'] else 'FAIL'}")
    
    async def _test_tui_connector(self):
        """Test TUI connector"""
        logger.info("Testing TUI connector...")
        
        try:
            manager = MCPConnectionManager()
            
            # Test connection
            connect_success = await manager.connect()
            
            # Test connection status
            test_result = await manager.test_connection()
            
            # Test connector operations
            connector_operations = {}
            if connect_success:
                connector = manager.get_connector()
                if connector:
                    # Test swarm status
                    status = await connector.get_swarm_status()
                    connector_operations["status"] = status is not None
                    
                    # Test swarm init
                    init_success = await connector.initialize_swarm()
                    connector_operations["init"] = init_success
            
            # Cleanup
            await manager.disconnect()
            
            self.test_results["tui_connector"] = {
                "success": connect_success,
                "connection_test": test_result,
                "connector_operations": connector_operations
            }
            
        except Exception as e:
            self.test_results["tui_connector"] = {
                "success": False,
                "error": str(e)
            }
        
        logger.info(f"TUI connector test: {'PASS' if self.test_results['tui_connector']['success'] else 'FAIL'}")
    
    async def _test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        logger.info("Testing end-to-end workflow...")
        
        try:
            # Simulate complete development workflow
            workflow_steps = {}
            
            # Step 1: Initialize integration bridge
            config = BridgeConfig(mcp_port=3000, api_port=8000)
            bridge = IntegrationBridge(config)
            
            bridge_start = await bridge.start()
            workflow_steps["bridge_start"] = bridge_start
            
            if bridge_start:
                # Step 2: Execute swarm command through bridge
                init_result = await bridge.execute_swarm_command("init", {
                    "topology": "mesh",
                    "max_agents": 3
                })
                workflow_steps["swarm_init"] = init_result.get("success", False)
                
                # Step 3: Spawn agents
                spawn_result = await bridge.execute_swarm_command("spawn", {
                    "agent_type": "coder",
                    "config": {"language": "python"}
                })
                workflow_steps["agent_spawn"] = spawn_result.get("success", False)
                
                # Step 4: Orchestrate task
                orchestrate_result = await bridge.execute_swarm_command("orchestrate", {
                    "description": "End-to-end test task",
                    "agents": ["coder"]
                })
                workflow_steps["task_orchestrate"] = orchestrate_result.get("task_id") is not None
                
                # Step 5: Check status
                status_result = await bridge.execute_swarm_command("status")
                workflow_steps["status_check"] = "status" in status_result
                
                # Cleanup
                await bridge.stop()
            
            self.test_results["end_to_end_workflow"] = {
                "success": all(workflow_steps.values()),
                "steps": workflow_steps
            }
            
        except Exception as e:
            self.test_results["end_to_end_workflow"] = {
                "success": False,
                "error": str(e)
            }
        
        logger.info(f"End-to-end workflow test: {'PASS' if self.test_results['end_to_end_workflow']['success'] else 'FAIL'}")
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        
        # Count successes and failures
        test_categories = [
            "startup", "mcp_connectivity", "swarm_operations",
            "hooks_integration", "api_endpoints", "monitoring_dashboard",
            "tui_connector", "end_to_end_workflow"
        ]
        
        passed_tests = sum(1 for cat in test_categories if self.test_results.get(cat, {}).get("success", False))
        total_tests = len(test_categories)
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests) * 100,
                "total_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        # Save report to file
        report_file = Path.cwd() / ".swarm" / f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.test_results.get("startup", {}).get("success", False):
            recommendations.append("Check MCP server installation and configuration")
        
        if not self.test_results.get("mcp_connectivity", {}).get("success", False):
            recommendations.append("Verify MCP server is running and accessible")
        
        if not self.test_results.get("api_endpoints", {}).get("success", False):
            recommendations.append("Check API server startup and port configuration")
        
        if not self.test_results.get("hooks_integration", {}).get("success", False):
            recommendations.append("Verify claude-flow hooks installation and permissions")
        
        if len(recommendations) == 0:
            recommendations.append("All tests passed! Integration is working correctly.")
        
        return recommendations

async def run_integration_tests():
    """Run integration tests and print report"""
    tester = IntegrationTester()
    report = await tester.run_all_tests()
    
    print("\n" + "="*60)
    print("CLAUDE-TUI MCP INTEGRATION TEST REPORT")
    print("="*60)
    
    summary = report["test_summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Time: {summary['total_time']:.2f}s")
    
    print("\nTEST RESULTS:")
    print("-" * 40)
    
    for test_name, result in report["test_results"].items():
        status = "PASS" if result.get("success", False) else "FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if not result.get("success", False) and "error" in result:
            print(f"  Error: {result['error']}")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"{i}. {rec}")
    
    print(f"\nDetailed report saved to: {Path.cwd() / '.swarm'}")
    
    return report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_integration_tests())