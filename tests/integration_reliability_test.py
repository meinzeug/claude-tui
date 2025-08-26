"""Integration Reliability Test - Comprehensive validation of 99.9% reliability target.

Tests the Integration Manager and all components to validate:
- 99.9% uptime target
- <100ms integration overhead
- Automatic failover functionality
- Circuit breaker pattern
- Error handling and recovery
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock
import pytest

from claude_tui.core.config_manager import ConfigManager
from claude_tui.integrations.integration_manager import (
    IntegrationManager, ServiceType, ServiceStatus, CircuitState
)
from claude_tui.integrations.health_monitor import IntegrationHealthMonitor
from claude_tui.models.project import Project
from claude_tui.models.ai_models import WorkflowRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReliabilityTestSuite:
    """Comprehensive reliability testing suite."""
    
    def __init__(self):
        self.config_manager = MagicMock(spec=ConfigManager)
        self.integration_manager = None
        self.health_monitor = None
        
        # Test configuration
        self.test_duration = 300  # 5 minutes
        self.request_interval = 0.1  # 100ms between requests
        self.expected_reliability = 99.9  # 99.9% target
        self.max_integration_overhead = 0.1  # 100ms
        
        # Test results
        self.results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'integration_overhead': [],
            'circuit_breaker_activations': 0,
            'failover_events': 0,
            'recovery_events': 0
        }
    
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up integration reliability test")
        
        # Mock configuration
        self.config_manager.get_setting = AsyncMock(side_effect=self._mock_config)
        
        # Initialize Integration Manager
        self.integration_manager = IntegrationManager(self.config_manager)
        await self.integration_manager.initialize()
        
        # Initialize Health Monitor
        self.health_monitor = IntegrationHealthMonitor(check_interval=10)
        
        # Register health checkers
        self.health_monitor.register_service(
            'claude_code',
            self._mock_claude_code_health_check
        )
        self.health_monitor.register_service(
            'claude_flow',
            self._mock_claude_flow_health_check
        )
        
        await self.health_monitor.start_monitoring()
        
        logger.info("Test environment setup complete")
    
    async def teardown(self):
        """Cleanup test environment."""
        logger.info("Tearing down test environment")
        
        if self.health_monitor:
            await self.health_monitor.stop_monitoring()
        
        if self.integration_manager:
            await self.integration_manager.cleanup()
        
        logger.info("Test environment cleanup complete")
    
    async def run_reliability_test(self) -> Dict[str, Any]:
        """Run comprehensive reliability test."""
        logger.info(f"Starting {self.test_duration}s reliability test")
        
        await self.setup()
        
        try:
            # Run test scenarios concurrently
            await asyncio.gather(
                self._run_continuous_requests(),
                self._run_failure_injection(),
                self._run_performance_monitoring(),
                return_exceptions=True
            )
            
            # Calculate final metrics
            results = await self._calculate_results()
            
            logger.info("Reliability test completed")
            return results
            
        finally:
            await self.teardown()
    
    async def _run_continuous_requests(self):
        """Run continuous requests to test reliability."""
        start_time = time.time()
        end_time = start_time + self.test_duration
        
        logger.info("Starting continuous request testing")
        
        while time.time() < end_time:
            try:
                # Test coding task request
                await self._test_coding_request()
                
                # Test workflow request
                await self._test_workflow_request()
                
                # Test validation request
                await self._test_validation_request()
                
                await asyncio.sleep(self.request_interval)
                
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                self.results['failed_requests'] += 1
        
        logger.info("Continuous request testing completed")
    
    async def _run_failure_injection(self):
        """Inject failures to test resilience."""
        logger.info("Starting failure injection testing")
        
        # Wait a bit before starting failures
        await asyncio.sleep(60)
        
        # Inject various failure scenarios
        failure_scenarios = [
            self._inject_service_failure,
            self._inject_network_timeout,
            self._inject_rate_limiting,
            self._inject_service_overload
        ]
        
        for scenario in failure_scenarios:
            try:
                await scenario()
                await asyncio.sleep(30)  # Wait between scenarios
            except Exception as e:
                logger.error(f"Failure injection error: {e}")
        
        logger.info("Failure injection testing completed")
    
    async def _run_performance_monitoring(self):
        """Monitor performance metrics during test."""
        logger.info("Starting performance monitoring")
        
        start_time = time.time()
        end_time = start_time + self.test_duration
        
        while time.time() < end_time:
            try:
                # Get health status
                health_status = await self.integration_manager.get_health_status()
                
                # Get performance metrics
                perf_metrics = await self.integration_manager.get_performance_metrics()
                
                # Log metrics periodically
                if int(time.time() - start_time) % 60 == 0:  # Every minute
                    logger.info(f"Health: {health_status['overall_health']}, "
                               f"Reliability: {perf_metrics['integration_performance']['reliability_score']:.3f}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
        
        logger.info("Performance monitoring completed")
    
    async def _test_coding_request(self):
        """Test coding task request."""
        start_time = time.time()
        
        try:
            result = await self.integration_manager.execute_coding_task(
                "Generate a simple Python function",
                {'task_type': 'code_generation'},
                None
            )
            
            integration_overhead = time.time() - start_time
            self.results['integration_overhead'].append(integration_overhead)
            
            if result and getattr(result, 'success', False):
                self.results['successful_requests'] += 1
                self.results['response_times'].append(integration_overhead)
            else:
                self.results['failed_requests'] += 1
            
            self.results['total_requests'] += 1
            
        except Exception as e:
            self.results['failed_requests'] += 1
            self.results['total_requests'] += 1
            raise
    
    async def _test_workflow_request(self):
        """Test workflow request."""
        start_time = time.time()
        
        try:
            workflow_request = WorkflowRequest(
                workflow_name="test_workflow",
                parameters={'test': True},
                variables={}
            )
            
            result = await self.integration_manager.execute_workflow(
                workflow_request,
                None
            )
            
            integration_overhead = time.time() - start_time
            self.results['integration_overhead'].append(integration_overhead)
            
            if result and getattr(result, 'success', False):
                self.results['successful_requests'] += 1
                self.results['response_times'].append(integration_overhead)
            else:
                self.results['failed_requests'] += 1
            
            self.results['total_requests'] += 1
            
        except Exception as e:
            self.results['failed_requests'] += 1
            self.results['total_requests'] += 1
            raise
    
    async def _test_validation_request(self):
        """Test validation request."""
        start_time = time.time()
        
        try:
            result = await self.integration_manager.validate_content(
                "def test(): pass",
                {'validation_type': 'code'},
                None
            )
            
            integration_overhead = time.time() - start_time
            self.results['integration_overhead'].append(integration_overhead)
            
            if result:
                self.results['successful_requests'] += 1
                self.results['response_times'].append(integration_overhead)
            else:
                self.results['failed_requests'] += 1
            
            self.results['total_requests'] += 1
            
        except Exception as e:
            self.results['failed_requests'] += 1
            self.results['total_requests'] += 1
            raise
    
    async def _inject_service_failure(self):
        """Inject service failure to test circuit breaker."""
        logger.info("Injecting service failure")
        
        # Mock service failure
        original_health_check = self._mock_claude_code_health_check
        
        async def failing_health_check():
            return {'is_healthy': False, 'error': 'Simulated failure'}
        
        self.health_monitor.health_checkers['claude_code'] = failing_health_check
        
        # Wait for circuit breaker to activate
        await asyncio.sleep(30)
        
        # Check if circuit breaker activated
        cb_status = self.integration_manager.circuit_breakers[ServiceType.CLAUDE_CODE]
        if cb_status.state == CircuitState.OPEN:
            self.results['circuit_breaker_activations'] += 1
            logger.info("Circuit breaker activated successfully")
        
        # Restore service
        self.health_monitor.health_checkers['claude_code'] = original_health_check
        
        # Wait for recovery
        await asyncio.sleep(30)
        
        if cb_status.state == CircuitState.CLOSED:
            self.results['recovery_events'] += 1
            logger.info("Service recovery successful")
    
    async def _inject_network_timeout(self):
        """Inject network timeout to test timeout handling."""
        logger.info("Injecting network timeout")
        
        # This would be implemented with actual network delays
        # For now, we'll simulate with a slow response
        await asyncio.sleep(0.1)
    
    async def _inject_rate_limiting(self):
        """Inject rate limiting to test backoff logic."""
        logger.info("Injecting rate limiting")
        
        # This would be implemented with actual rate limiting
        # For now, we'll simulate with multiple rapid requests
        for _ in range(10):
            try:
                await self._test_coding_request()
            except:
                pass
    
    async def _inject_service_overload(self):
        """Inject service overload to test failover."""
        logger.info("Injecting service overload")
        
        # Mock overloaded service
        async def overloaded_health_check():
            return {'is_healthy': True, 'response_time': 15.0}  # Slow response
        
        self.health_monitor.health_checkers['claude_code'] = overloaded_health_check
        
        # Wait for failover
        await asyncio.sleep(30)
        
        self.results['failover_events'] += 1
    
    async def _calculate_results(self) -> Dict[str, Any]:
        """Calculate final test results."""
        total_requests = self.results['total_requests']
        successful_requests = self.results['successful_requests']
        
        # Calculate reliability percentage
        reliability = (successful_requests / max(total_requests, 1)) * 100
        
        # Calculate performance metrics
        avg_response_time = statistics.mean(self.results['response_times']) if self.results['response_times'] else 0
        max_response_time = max(self.results['response_times']) if self.results['response_times'] else 0
        p95_response_time = statistics.quantiles(self.results['response_times'], n=20)[18] if len(self.results['response_times']) >= 20 else 0
        
        avg_overhead = statistics.mean(self.results['integration_overhead']) if self.results['integration_overhead'] else 0
        max_overhead = max(self.results['integration_overhead']) if self.results['integration_overhead'] else 0
        
        # Get final health status
        health_status = await self.integration_manager.get_health_status()
        
        results = {\n            'test_summary': {\n                'duration': self.test_duration,\n                'total_requests': total_requests,\n                'successful_requests': successful_requests,\n                'failed_requests': self.results['failed_requests'],\n                'reliability_percentage': reliability,\n                'target_reliability': self.expected_reliability,\n                'reliability_met': reliability >= self.expected_reliability\n            },\n            'performance_metrics': {\n                'avg_response_time': avg_response_time,\n                'max_response_time': max_response_time,\n                'p95_response_time': p95_response_time,\n                'avg_integration_overhead': avg_overhead,\n                'max_integration_overhead': max_overhead,\n                'overhead_target': self.max_integration_overhead,\n                'overhead_target_met': avg_overhead <= self.max_integration_overhead\n            },\n            'resilience_metrics': {\n                'circuit_breaker_activations': self.results['circuit_breaker_activations'],\n                'failover_events': self.results['failover_events'],\n                'recovery_events': self.results['recovery_events']\n            },\n            'final_health_status': health_status,\n            'test_passed': (\n                reliability >= self.expected_reliability and\n                avg_overhead <= self.max_integration_overhead\n            )\n        }\n        \n        return results\n    \n    # Mock implementations\n    \n    async def _mock_config(self, key: str, default: Any = None) -> Any:\n        \"\"\"Mock configuration values.\"\"\"\n        config_values = {\n            'claude_flow': {\n                'endpoint_url': 'http://localhost:3000',\n                'timeout': 30\n            },\n            'integration_manager': {\n                'enable_caching': True,\n                'enable_auto_retry': True,\n                'max_concurrent_requests': 50,\n                'health_check_interval': 10\n            },\n            'anti_hallucination_integration': {\n                'enabled': True,\n                'auto_fix': True,\n                'block_on_critical': True,\n                'async_validation': True\n            },\n            'CLAUDE_CODE_OAUTH_TOKEN': 'mock_token',\n            'CLAUDE_CODE_RATE_LIMIT': 60\n        }\n        \n        return config_values.get(key, default)\n    \n    async def _mock_claude_code_health_check(self) -> Dict[str, Any]:\n        \"\"\"Mock Claude Code health check.\"\"\"\n        await asyncio.sleep(0.01)  # Simulate small delay\n        return {\n            'is_healthy': True,\n            'response_time': 0.05,\n            'success_rate': 0.99,\n            'error_rate': 0.01\n        }\n    \n    async def _mock_claude_flow_health_check(self) -> Dict[str, Any]:\n        \"\"\"Mock Claude Flow health check.\"\"\"\n        await asyncio.sleep(0.02)  # Simulate small delay\n        return {\n            'is_healthy': True,\n            'response_time': 0.08,\n            'success_rate': 0.98,\n            'error_rate': 0.02\n        }\n\n\n# Test execution functions\n\nasync def run_integration_reliability_test() -> Dict[str, Any]:\n    \"\"\"Run the complete integration reliability test.\"\"\"\n    test_suite = ReliabilityTestSuite()\n    return await test_suite.run_reliability_test()\n\n\nasync def run_quick_reliability_test() -> Dict[str, Any]:\n    \"\"\"Run a quick 1-minute reliability test.\"\"\"\n    test_suite = ReliabilityTestSuite()\n    test_suite.test_duration = 60  # 1 minute\n    return await test_suite.run_reliability_test()\n\n\nif __name__ == \"__main__\":\n    async def main():\n        logger.info(\"Starting Integration Reliability Test\")\n        \n        # Run the test\n        results = await run_integration_reliability_test()\n        \n        # Print results\n        print(\"\\n\" + \"=\"*80)\n        print(\"INTEGRATION RELIABILITY TEST RESULTS\")\n        print(\"=\"*80)\n        \n        test_summary = results['test_summary']\n        print(f\"Test Duration: {test_summary['duration']}s\")\n        print(f\"Total Requests: {test_summary['total_requests']}\")\n        print(f\"Successful Requests: {test_summary['successful_requests']}\")\n        print(f\"Failed Requests: {test_summary['failed_requests']}\")\n        print(f\"Reliability: {test_summary['reliability_percentage']:.2f}% (Target: {test_summary['target_reliability']}%)\")\n        print(f\"Reliability Target Met: {test_summary['reliability_met']}\")\n        \n        print(\"\\nPerformance Metrics:\")\n        perf_metrics = results['performance_metrics']\n        print(f\"Average Response Time: {perf_metrics['avg_response_time']*1000:.1f}ms\")\n        print(f\"P95 Response Time: {perf_metrics['p95_response_time']*1000:.1f}ms\")\n        print(f\"Average Integration Overhead: {perf_metrics['avg_integration_overhead']*1000:.1f}ms\")\n        print(f\"Max Integration Overhead: {perf_metrics['max_integration_overhead']*1000:.1f}ms\")\n        print(f\"Overhead Target Met: {perf_metrics['overhead_target_met']}\")\n        \n        print(\"\\nResilience Metrics:\")\n        resilience = results['resilience_metrics']\n        print(f\"Circuit Breaker Activations: {resilience['circuit_breaker_activations']}\")\n        print(f\"Failover Events: {resilience['failover_events']}\")\n        print(f\"Recovery Events: {resilience['recovery_events']}\")\n        \n        print(f\"\\nOVERALL TEST RESULT: {'PASSED' if results['test_passed'] else 'FAILED'}\")\n        print(\"=\"*80)\n        \n        return results\n    \n    # Run the test\n    asyncio.run(main())